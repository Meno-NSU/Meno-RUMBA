#!/usr/bin/env python3
"""Execution-only replica routing around an unchanged audited RAGU adapter.

Logical endpoints, request bodies, cache keys, sampling and schemas stay fixed.
Only physical HTTP destinations and client concurrency change.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import runpy
import threading
import time

import httpx

AUDIT_LOCK = threading.Lock()

def append_event(path, event, lock):
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open('a') as output:
        output.write(json.dumps(event, ensure_ascii=False) + '\n')


class ReleasingStream(httpx.SyncByteStream):
    def __init__(self, stream, release):
        self.stream, self.release = stream, release
        self.closed = False

    def __iter__(self):
        yield from self.stream

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.stream.close()
            finally:
                self.release()


class ReplicaTransport(httpx.BaseTransport):
    def __init__(self, backends, per_backend, audit, route, delegate=None):
        if not backends or per_backend < 1:
            raise ValueError('A nonempty bounded replica pool is required')
        self.backends = [httpx.URL(url) for url in backends]
        self.per_backend = per_backend
        self.active = [0] * len(backends)
        self.cooldown = [0.0] * len(backends)
        self.cursor = 0
        self.condition = threading.Condition()
        self.audit_lock = AUDIT_LOCK
        self.audit, self.route = Path(audit), route
        self.delegate = delegate or httpx.HTTPTransport(retries=0)

    def acquire(self, excluded):
        with self.condition:
            while True:
                now = time.monotonic()
                choices = [i for i in range(len(self.backends)) if i not in excluded
                           and self.active[i] < self.per_backend and self.cooldown[i] <= now]
                if choices:
                    index = min(choices, key=lambda i: (self.active[i], (i - self.cursor) % len(self.backends)))
                    self.active[index] += 1
                    self.cursor = (index + 1) % len(self.backends)
                    return index
                self.condition.wait(timeout=1)

    def release(self, index):
        with self.condition:
            self.active[index] -= 1
            self.condition.notify_all()

    def handle_request(self, request):
        body = request.read()
        tried = set()
        last_error = None
        for _ in self.backends:
            index = self.acquire(tried)
            target = self.backends[index]
            url = request.url.copy_with(scheme=target.scheme, host=target.host, port=target.port)
            headers = [(k, v) for k, v in request.headers.multi_items() if k.lower() != 'host']
            forwarded = httpx.Request(request.method, url, headers=headers, content=body,
                                     extensions=dict(request.extensions))
            event = {'time': time.time(), 'route': self.route, 'logical_url': str(request.url),
                     'physical_url': str(url), 'body_sha256': hashlib.sha256(body).hexdigest(),
                     'request_body_unchanged': forwarded.read() == body}
            started = time.monotonic()
            try:
                response = self.delegate.handle_request(forwarded)
            except httpx.TransportError as error:
                last_error = error
                self.release(index)
                with self.condition:
                    self.cooldown[index] = time.monotonic() + 30
                tried.add(index)
                append_event(self.audit, {**event, 'seconds': time.monotonic() - started,
                                         'error': type(error).__name__}, self.audit_lock)
                continue
            if response.is_closed or response.is_stream_consumed:
                self.release(index)
            else:
                response.stream = ReleasingStream(response.stream, lambda: self.release(index))
            append_event(self.audit, {**event, 'seconds': time.monotonic() - started,
                                     'status_code': response.status_code}, self.audit_lock)
            return response
        raise last_error

    def close(self):
        self.delegate.close()


def validate_policy(policy):
    if policy['model'] != 'meno-lite-0.2':
        raise ValueError('Only Meno0.2 may use this execution policy')
    if not 1 <= policy['total_concurrency'] <= 64:
        raise ValueError('Concurrency must be bounded')
    ports = set()
    for route, logical in [('native', 9032), ('yarn', 9035)]:
        spec = policy['routes'][route]
        if spec['logical_base_url'] != f'http://127.0.0.1:{logical}/v1':
            raise ValueError('Unexpected logical endpoint')
        if not 1 <= spec['concurrency'] <= len(spec['backends']) * spec['per_backend']:
            raise ValueError('Concurrency exceeds verified replica capacity')
        for backend in spec['backends']:
            url = httpx.URL(backend)
            allowed = {9032, 9033, 9034} if route == 'native' else {9035, 9036, 9037}
            if (url.scheme != 'http' or url.host != '127.0.0.1' or url.port in ports
                    or url.port not in allowed or url.path != '/v1'):
                raise ValueError('Only distinct node-loopback replicas are allowed')
            ports.add(url.port)


def install_policy(policy, audit_dir):
    from openai import OpenAI
    from rumba_bench import pipeline
    from rumba_bench import ragu_harness as harness
    from rumba_bench.ragu_routing import ContextRoutingLLM

    original_client = pipeline.client_for
    original_protocol_init = harness.ProtocolLLM.__init__
    original_routing_init = ContextRoutingLLM.__init__
    original_execute = harness.execute
    original_build = harness.Harness.build
    by_url = {spec['logical_base_url']: (route, spec) for route, spec in policy['routes'].items()}

    def client_for(endpoint, runtime):
        selected = by_url.get(endpoint['base_url'].rstrip('/'))
        if endpoint['model'] != policy['model'] or selected is None:
            return original_client(endpoint, runtime)
        route, spec = selected
        transport = ReplicaTransport(spec['backends'], spec['per_backend'],
                                     audit_dir / f'{route}-physical-requests.jsonl', route)
        return OpenAI(base_url=endpoint['base_url'],
                      api_key=os.getenv(endpoint.get('api_key_env', 'VLLM_API_KEY'), 'EMPTY'),
                      timeout=runtime['timeout_s'], max_retries=runtime['max_retries'],
                      http_client=httpx.Client(transport=transport, trust_env=False))

    def protocol_init(self, endpoint, config, cache, *, answer=False):
        original_protocol_init(self, endpoint, config, cache, answer=answer)
        selected = by_url.get(endpoint['base_url'].rstrip('/'))
        if endpoint['model'] == policy['model'] and selected:
            self.semaphore = asyncio.Semaphore(selected[1]['concurrency'])

    def routing_init(self, native, extended, native_endpoint, config, cache):
        original_routing_init(self, native, extended, native_endpoint, config, cache)
        if native_endpoint['model'] == policy['model']:
            self.semaphore = asyncio.Semaphore(policy['total_concurrency'])

    async def execute(args):
        if args.model != policy['model']:
            raise ValueError('Execution override cannot affect another model')
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=64, thread_name_prefix='meno02-http'))
        return await original_execute(args)

    async def build(self, graph, user, sessions, **kwargs):
        result = await original_build(self, graph, user, sessions, **kwargs)
        from ragu.common.global_parameters import Settings
        (Path(Settings.storage_folder) / 'execution-policy.json').write_text(
            json.dumps(policy, ensure_ascii=False, indent=2) + '\n')
        return result

    pipeline.client_for = client_for
    harness.ProtocolLLM.__init__ = protocol_init
    ContextRoutingLLM.__init__ = routing_init
    harness.execute = execute
    harness.Harness.build = build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--lineage', type=Path, required=True)
    parser.add_argument('worker', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    policy_path = Path(os.environ['MENO_EXECUTION_POLICY'])
    policy = json.loads(policy_path.read_text())
    validate_policy(policy)
    if policy['wrapper_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Execution wrapper differs from audited source')
    adapter = Path(policy['summary_adapter'])
    if hashlib.sha256(adapter.read_bytes()).hexdigest() != policy['summary_adapter_sha256']:
        raise ValueError('Summary adapter changed')
    worker = args.worker[1:] if args.worker[:1] == ['--'] else args.worker
    config_path = Path(worker[worker.index('--config') + 1])
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != policy['config_sha256']:
        raise ValueError('Baseline config changed')
    install_policy(policy, args.audit.parent)
    import sys
    sys.argv = [str(adapter), '--audit', str(args.audit), '--lineage', str(args.lineage), '--', *worker]
    runpy.run_path(str(adapter), run_name='__main__')


if __name__ == '__main__':
    main()
