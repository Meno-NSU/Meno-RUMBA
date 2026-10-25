import importlib.util
import json
import copy
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('replicas', ROOT / 'scripts/launch_meno02_replicas.py')
replicas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replicas)
spec = importlib.util.spec_from_file_location('replica_runtime', ROOT / 'scripts/run_replicated_ragu.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_replica_plan_only_changes_port_and_uses_unused_allocated_gpus():
    common = ['vllm', 'serve', 'weights', '--seed', '42', '--dtype', 'half', '--port']
    services = [{'tag': 'meno-lite-0.2-native', 'port': 9032, 'gpu': '2',
                 'command': common + ['9032', '--max-model-len', '32768']},
                {'tag': 'meno-lite-0.2-long', 'port': 9035, 'gpu': '3',
                 'command': common + ['9035', '--max-model-len', '65536', '--hf-overrides', 'YARN']},
                *[{'tag': 'protected', 'gpu': str(i)} for i in [0, 1, 4, 5]]]
    plan = replicas.replica_plan(services, list(map(str, range(10))))
    assert [x['gpu'] for x in plan] == ['6', '7', '8', '9']
    assert [x['route'] for x in plan] == ['native', 'native', 'yarn', 'yarn']
    for item in plan:
        source = next(s for s in services if s['tag'] == item['source_tag'])
        expected = list(source['command'])
        expected[expected.index('--port') + 1] = str(item['port'])
        assert item['command'] == expected
    assert services[0]['command'][8] == '9032'
    with pytest.raises(ValueError):
        replicas.replica_plan(services, list(map(str, range(8))))


def test_physical_routing_preserves_body_and_round_robins(tmp_path):
    seen = []
    def receive(request):
        seen.append((request.url.port, request.content))
        return httpx.Response(200, json={'ok': True})
    transport = runtime.ReplicaTransport(['http://127.0.0.1:9032/v1', 'http://127.0.0.1:9033/v1'],
                                         8, tmp_path/'audit.jsonl', 'native', httpx.MockTransport(receive))
    body = b'{"model":"meno-lite-0.2","seed":42,"temperature":0,"repetition_penalty":1.03}'
    with httpx.Client(transport=transport) as client:
        for _ in range(4):
            response = client.post('http://127.0.0.1:9032/v1/chat/completions', content=body)
            assert response.json() == {'ok': True}
    assert [x[0] for x in seen] == [9032, 9033, 9032, 9033]
    assert all(x[1] == body for x in seen)
    assert transport.active == [0, 0]
    events = [json.loads(x) for x in (tmp_path/'audit.jsonl').read_text().splitlines()]
    assert all(e['request_body_unchanged'] for e in events)


def test_failed_replica_falls_back_without_changing_payload(tmp_path):
    seen = []
    def receive(request):
        seen.append(request.url.port)
        if request.url.port == 9033:
            raise httpx.ConnectError('offline', request=request)
        assert request.content == b'unchanged'
        return httpx.Response(200, text='ok')
    transport = runtime.ReplicaTransport(['http://127.0.0.1:9033/v1', 'http://127.0.0.1:9032/v1'],
                                         8, tmp_path/'audit.jsonl', 'native', httpx.MockTransport(receive))
    with httpx.Client(transport=transport) as client:
        assert client.post('http://127.0.0.1:9032/v1/chat/completions', content=b'unchanged').text == 'ok'
    assert seen == [9033, 9032] and transport.active == [0, 0]


def test_policy_cannot_mix_native_and_yarn_physical_ports():
    policy = {'model':'meno-lite-0.2','total_concurrency':36,'routes':{
        'native':{'logical_base_url':'http://127.0.0.1:9032/v1','concurrency':24,'per_backend':8,
                  'backends':['http://127.0.0.1:9032/v1','http://127.0.0.1:9033/v1','http://127.0.0.1:9034/v1']},
        'yarn':{'logical_base_url':'http://127.0.0.1:9035/v1','concurrency':12,'per_backend':4,
                'backends':['http://127.0.0.1:9035/v1','http://127.0.0.1:9036/v1','http://127.0.0.1:9037/v1']}}}
    runtime.validate_policy(policy)
    policy['routes']['yarn']['backends'][0] = 'http://127.0.0.1:9032/v1'
    with pytest.raises(ValueError):runtime.validate_policy(policy)


def test_unread_http_response_releases_capacity_when_consumed(tmp_path):
    class Stream(httpx.SyncByteStream):
        def __iter__(self): yield b'finished'
        def close(self): pass
    transport = runtime.ReplicaTransport(['http://127.0.0.1:9032/v1'], 1,
        tmp_path/'audit.jsonl', 'native', httpx.MockTransport(lambda request:httpx.Response(200,stream=Stream())))
    with httpx.Client(transport=transport) as client:
        assert client.post('http://127.0.0.1:9032/v1/chat/completions',content=b'body').text=='finished'
        assert client.post('http://127.0.0.1:9032/v1/chat/completions',content=b'body').text=='finished'
    assert transport.active == [0]


def test_runtime_override_keeps_config_cache_identity_and_other_model_untouched(tmp_path, monkeypatch):
    from rumba_bench import pipeline, ragu_harness as harness
    from rumba_bench.ragu_routing import ContextRoutingLLM
    for obj, name in [(pipeline,'client_for'),(harness.ProtocolLLM,'__init__'),
                      (ContextRoutingLLM,'__init__'),(harness,'execute'),(harness.Harness,'build')]:
        monkeypatch.setattr(obj,name,getattr(obj,name))
    policy={'model':'meno-lite-0.2','total_concurrency':36,'routes':{
        'native':{'logical_base_url':'http://127.0.0.1:9032/v1','concurrency':24,'per_backend':8,
                  'backends':['http://127.0.0.1:9032/v1','http://127.0.0.1:9033/v1','http://127.0.0.1:9034/v1']},
        'yarn':{'logical_base_url':'http://127.0.0.1:9035/v1','concurrency':12,'per_backend':4,
                'backends':['http://127.0.0.1:9035/v1','http://127.0.0.1:9036/v1','http://127.0.0.1:9037/v1']}}}
    config={'runtime':{'concurrency':8,'timeout_s':5,'max_retries':0},'protocol':{
        'temperature':0,'seed':42,'repetition_penalty':1.03,'extraction_max_tokens':4096,
        'answer_max_tokens':256,'answer_stop':['\n'],'context_routing':{
            'native_context_tokens':32768,'extended_context_tokens':65536}}}
    before=copy.deepcopy(config)
    runtime.install_policy(policy,tmp_path)
    endpoint={'model':'meno-lite-0.2','base_url':'http://127.0.0.1:9032/v1'}
    native=harness.ProtocolLLM(endpoint,config,tmp_path/'cache')
    extended=harness.ProtocolLLM({**endpoint,'base_url':'http://127.0.0.1:9035/v1'},config,tmp_path/'cache')
    router=ContextRoutingLLM(native,extended,endpoint,config,tmp_path/'tokens')
    assert (native.semaphore._value,extended.semaphore._value,router.semaphore._value)==(24,12,36)
    assert native.generator.endpoint==endpoint and config==before
    peer=harness.ProtocolLLM({'model':'meno-lite-0.1','base_url':'http://127.0.0.1:9022/v1'},config,tmp_path/'peer')
    assert peer.semaphore._value==8
    assert not isinstance(peer.generator.client._client._transport,runtime.ReplicaTransport)
    peer.close();router.close()
