#!/usr/bin/env python3
"""Add Meno0.2 replicas on unused GPUs of an existing exclusive Slurm job."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def replica_plan(services, allocated):
    native = next(s for s in services if s['tag'] == 'meno-lite-0.2-native')
    extended = next(s for s in services if s['tag'] == 'meno-lite-0.2-long')
    used = {s['gpu'] for s in services}
    free = [gpu for gpu in allocated if gpu not in used]
    if len(free) < 4:
        raise ValueError('Four unused GPUs must belong to the existing allocation')
    plan = []
    for i, (source, port, route) in enumerate([
            (native, 9033, 'native'), (native, 9034, 'native'),
            (extended, 9036, 'yarn'), (extended, 9037, 'yarn')]):
        command = list(source['command'])
        command[command.index('--port') + 1] = str(port)
        plan.append({'tag': f'meno-lite-0.2-{route}-extra-{i + 1}', 'model': 'meno-lite-0.2',
                     'gpu': free[i], 'port': port, 'route': route,
                     'source_tag': source['tag'], 'source_port': source['port'],
                     'command': command})
    return plan


def write_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    current = json.loads((project / 'runs/a800-meno-pair/current.json').read_text())
    if current['job_id'] != os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Use a step in the active Slurm allocation')
    job = Path(current['job_dir'])
    allocation = json.loads((job / 'allocation.json').read_text())
    if 'OverSubscribe=NO' not in allocation['description'] or 'NumNodes=1' not in allocation['description']:
        raise ValueError('An exclusive single-node allocation must be verified')
    services = json.loads((job / 'services.json').read_text())
    plan = replica_plan(services, allocation['devices'])
    paths = json.loads((job / 'paths.json').read_text())
    work = Path(paths['work'])
    folder = job / 'performance-meno02'
    folder.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location('server_process', project / 'scripts/server_process.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    status_path = folder / 'replicas-status.json'
    if status_path.exists():
        saved = json.loads(status_path.read_text())
        proc = helper.process_info(saved.get('pid', 0))
        if helper.alive(proc) and proc['start'] == saved.get('start_ticks'):
            raise ValueError('A replica controller is already alive')
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                        '--format=csv,noheader,nounits'], text=True).splitlines()
    available = {x.split(',')[0].strip(): float(x.split(',')[2]) for x in gpu_rows}
    for service in plan:
        if available.get(service['gpu'], float('inf')) > 128:
            raise ValueError(f"GPU {service['gpu']} is no longer unused")
        helper.require_free_port(service['port'])
    if int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                if x.startswith('MemAvailable:'))) < 24 * 1024**2:
        raise ValueError('Insufficient CPU RAM for safe replica startup')
    write_json(folder / 'replica-plan.json', plan)
    identity = helper.process_info(os.getpid())
    state = {'pid': os.getpid(), 'start_ticks': identity['start'], 'job_id': current['job_id'],
             'step_id': os.environ.get('SLURM_STEP_ID'), 'plan': plan, 'started_at': time.time()}
    def update(phase, **fields):
        state.update(phase=phase, updated_at=time.time(), **fields)
        write_json(status_path, state)
    stopping, running, files = [], [], []
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(number, lambda number, frame: stopping.append(number))
    try:
        update('starting')
        for service in plan:
            source_file = job / (service['source_tag'] + '.pid')
            source_pid = int(source_file.read_text().strip())
            source_info = helper.process_info(source_pid)
            if not helper.alive(source_info) or not helper.expected_process(
                    source_info, source_file, service['source_port'], service['model']):
                raise ValueError('Source server identity changed')
            env = dict(x.split('=', 1) for x in Path(f'/proc/{source_pid}/environ').read_bytes().decode().split('\0') if '=' in x)
            env.update({k: v for k, v in os.environ.items() if k.startswith('SLURM_')})
            env.update(CUDA_VISIBLE_DEVICES=service['gpu'], OMP_NUM_THREADS='1',
                       TORCHINDUCTOR_COMPILE_THREADS='1')
            for key in ('OUTLINES_CACHE_DIR', 'TRITON_CACHE_DIR', 'FLASHINFER_WORKSPACE_BASE'):
                env[key] = str(Path(env[key]).parent / service['tag'])
            output = (folder / (service['tag'] + '.log')).open('a')
            files.append(output)
            child = subprocess.Popen(service['command'], cwd=work, env=env,
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
            running.append((service, child))
            helper.record(folder / (service['tag'] + '.pid'), child.pid, service['port'], service['model'])
            update('starting', servers=[{'tag': s['tag'], 'pid': p.pid} for s, p in running])
        deadline = time.monotonic() + 900
        ready = set()
        while not stopping:
            for service, child in running:
                if child.poll() is not None:
                    raise RuntimeError(f"Replica {service['tag']} exited {child.returncode}")
                if service['port'] in ready:
                    continue
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{service['port']}/v1/models", timeout=2) as response:
                        data = json.load(response)['data']
                    expected = 32768 if service['route'] == 'native' else 65536
                    if [(m['id'], m.get('max_model_len')) for m in data] == [(service['model'], expected)]:
                        ready.add(service['port'])
                except (OSError, ValueError):
                    pass
            if len(ready) != len(plan) and time.monotonic() > deadline:
                raise TimeoutError('Replica startup exceeded15minutes')
            update('ready' if len(ready) == len(plan) else 'starting', ready_ports=sorted(ready))
            time.sleep(5)
        update('stopping')
    except Exception as error:
        update('failed', error=str(error))
        raise
    finally:
        for service, child in reversed(running):
            helper.stop(folder / (service['tag'] + '.pid'), service['port'], service['model'], timeout=60)
        for output in files:
            output.close()
    update('stopped')


if __name__ == '__main__':
    main()
