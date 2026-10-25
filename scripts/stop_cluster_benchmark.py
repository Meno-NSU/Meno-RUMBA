#!/usr/bin/env python3
"""Stop an owned Slurm benchmark, preserve unfinished work, then let its writer finish.

Run inside its existing allocation. The original launcher remains the only
checkpoint writer; this script creates a separate preservation archive.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def process(pid, job):
    p = Path('/proc') / str(pid)
    if not p.exists():
        return None
    if p.stat().st_uid != os.getuid():
        raise ValueError('Refusing another user process')
    stat = (p / 'stat').read_text().split(') ', 1)[1].split()
    if stat[0] == 'Z':
        return {'pid': int(pid), 'state': 'Z', 'birth': stat[19], 'argv': []}
    env = (p / 'environ').read_bytes().split(b'\0')
    if ('SLURM_JOB_ID=' + job).encode() not in env:
        raise ValueError('Process is outside the selected allocation')
    return {'pid': int(pid), 'ppid': int(stat[1]), 'argv': (p / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
            'birth': stat[19], 'state': stat[0], 'cwd': str((p / 'cwd').resolve())}


def send(record, number, job):
    current = process(record['pid'], job)
    if current and current['state'] != 'Z':
        if current['birth'] != record['birth'] or current['argv'] != record['argv']:
            raise ValueError('Process identity changed')
        os.kill(record['pid'], number)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--parent-script', type=Path, required=True)
    parser.add_argument('--model', default='meno-lite-0.1')
    args = parser.parse_args()
    if os.environ.get('SLURM_JOB_ID') != args.job:
        raise ValueError('Use the existing Slurm allocation')
    root = args.project.resolve()
    pointer = read(root / 'runs/a800-meno-pair/current.json')
    if pointer['job_id'] != args.job:
        raise ValueError('Current job changed')
    durable = Path(pointer['durable'])
    paths = read(Path(pointer['job_dir']) / 'paths.json')
    work = Path(paths['work'])
    folder = work / paths['run_relative'] / args.model
    statefile = root / f'runs/cluster-setup/recovery-{args.job}/{args.model}-status.json'
    worker = read(statefile)
    archive = durable / 'stopped-by-user'
    archive.mkdir(exist_ok=True)
    lock = (archive / 'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    statusfile = archive / 'state.json'
    if statusfile.exists() and read(statusfile).get('phase') == 'preserved-parent-checkpointing':
        print(json.dumps(read(statusfile))); return
    wrapper = process(worker['pid'], args.job)
    if not wrapper or wrapper['state'] == 'Z':
        raise ValueError('Recovery wrapper has exited; inspect final checkpoint instead')
    parent = process(wrapper['ppid'], args.job)
    if not parent or parent['pid'] == os.getpid():
        raise ValueError('Missing real checkpoint writer')
    scripts = [Path(parent['cwd']) / value for value in parent['argv'][1:4]
               if value.endswith('.py')]
    if args.parent_script.resolve() not in [p.resolve() for p in scripts]:
        raise ValueError('Worker parent is not the expected checkpoint writer')
    state = {'phase': 'holding-checkpoint-writer', 'time': time.time(), 'job': args.job,
             'model': args.model, 'parent': parent, 'worker_before': worker,
             'archive': str(archive), 'cancelled_by_user': True, 'never_resume': True}
    write(statusfile, state)
    send(parent, signal.SIGSTOP, args.job)
    try:
        records = []
        for key in ['pid', 'child_pid']:
            if not worker.get(key): continue
            record = process(worker[key], args.job)
            if record and record['state'] != 'Z':
                a = record['argv']
                if a[a.index('--model') + 1] != args.model or Path(a[a.index('--config') + 1]) != work / 'config.pair.json':
                    raise ValueError('Worker command does not match the selected run')
                records.append(record)
        if records: send(records[0], signal.SIGTERM, args.job)
        deadline = time.monotonic() + 35
        while any((r := process(p['pid'], args.job)) and r['state'] != 'Z' for p in records):
            if time.monotonic() > deadline:
                for p in records: send(p, signal.SIGKILL, args.job)
                time.sleep(1); break
            time.sleep(.5)
        state.update(phase='preserving-stopped-files', stopped_processes=records)
        write(statusfile, state)
        latest = read(durable / 'checkpoints/latest.json')
        committed = latest['users'].get(args.model, {})
        inventory = []
        def copy(source, relative):
            target = archive / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum = sha(source)
            if not target.exists() or sha(target) != checksum:
                tmp = target.with_name(target.name + '.partial')
                shutil.copy2(source, tmp); tmp.replace(target)
            if sha(target) != checksum: raise ValueError('Preservation checksum mismatch')
            inventory.append({'path': str(relative), 'size': target.stat().st_size, 'sha256': checksum})
            if len(inventory) % 100 == 0:
                state.update(verified_files=len(inventory), updated_at=time.time()); write(statusfile, state)
        if shutil.disk_usage(durable).free < 3 * 1024**3:
            raise ValueError('Insufficient preservation reserve; writer remains held')
        selected = set()
        for graph in (folder / 'graphs').iterdir():
            if not graph.is_dir(): continue
            ready = graph / '.checkpoint-ready.json'
            if not ready.exists() or graph.name not in committed:
                selected.add(graph.name)
                for source in graph.rglob('*'):
                    if source.is_file(): copy(source, Path('uncommitted-model/graphs') / graph.name / source.relative_to(graph))
        for kind in ['answers', 'retrieval']:
            for source in (folder / kind / 'full').glob('*.json'):
                if read(source).get('user_id') in selected or kind == 'retrieval':
                    copy(source, Path('uncommitted-model') / kind / 'full' / source.name)
        for kind in ['generation', 'tokenization']:
            for source in (work / 'runs/cache/ragu' / kind).glob('*'):
                if not source.is_file(): continue
                saved = durable / 'checkpoints/cache' / kind / source.name
                if not saved.exists() or source.stat().st_size != saved.stat().st_size or source.stat().st_mtime_ns != saved.stat().st_mtime_ns:
                    if not saved.exists() or sha(source) != sha(saved):
                        copy(source, Path('raw-delta') / kind / source.name)
        for source, relative in [(work / 'config.pair.json', Path('config.pair.json')),
                                 (folder.parent / 'manifest.json', Path('manifest.json')),
                                 (statefile, Path('worker-after-stop.json'))]:
            copy(source, relative)
        for source in folder.glob('*.json'): copy(source, Path('model-context') / source.name)
        for source in (folder / 'full').glob('*.json'): copy(source, Path('model-context/full') / source.name)
        write(archive / 'files.sha256.json', inventory)
        stopped = read(statefile)
        stopped.update(cancelled_by_user=True, never_restart=True, stopped_at=time.time())
        write(statefile, stopped)
        state.update(phase='preserved-parent-checkpointing', verified_files=len(inventory),
                     verified_bytes=sum(p['size'] for p in inventory),
                     uncommitted_users=sorted(selected), generation_stopped=True)
        write(statusfile, state)
        send(parent, signal.SIGUSR1, args.job)
        send(parent, signal.SIGCONT, args.job)
        print(json.dumps({k: state[k] for k in ['phase', 'verified_files', 'verified_bytes', 'uncommitted_users']}), flush=True)
    except BaseException as error:
        state.update(phase='preservation-failed-parent-held', error=repr(error))
        write(statusfile, state)
        raise


if __name__ == '__main__':
    main()
