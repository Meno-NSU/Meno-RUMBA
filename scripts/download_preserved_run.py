#!/usr/bin/env python3
"""Resume an rsync copy through a fixed-host SSH wrapper, then verify every SHA256.

The source must be a stopped, immutable run directory. Never removes source or
destination files. State, inventory and logs are private local artifacts.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


def write(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''): result.update(chunk)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ssh-wrapper', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--io-timeout', type=int, default=300,
                   help='Allow slow remote filesystem reads without treating them as a lost connection')
    a = p.parse_args()
    if not 30 <= a.io_timeout <= 3600: p.error('--io-timeout must be between30 and3600 seconds')
    a.destination = a.destination.resolve(); a.state = a.state.resolve()
    a.destination.mkdir(parents=True, exist_ok=True); a.state.parent.mkdir(parents=True, exist_ok=True)
    lock = a.state.with_suffix('.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'started_at': time.time(), 'source': a.source,
             'destination': str(a.destination), 'phase': 'starting'}
    def update(phase, **kw):
        state.update(phase=phase, updated_at=time.time(), **kw); write(a.state, state)
    source = str(Path(a.source))
    if not source.startswith('/') or '..' in Path(source).parts: raise ValueError('Use an absolute source directory')
    env = dict(os.environ, RUMBA_SSH_WRAPPER=str(a.ssh_wrapper.resolve()))
    rsh = shlex.join([sys.executable, str(Path(__file__).resolve()), '--rsync-transport'])
    command = ['rsync', '-rltz', '--partial', f'--timeout={a.io_timeout}', '--progress', '--stats', '-e', rsh,
               'benchmark-cluster:' + source.rstrip('/') + '/', str(a.destination) + '/']
    logpath = a.state.with_suffix('.log')
    try:
        for attempt in range(1, 4):
            with logpath.open('a') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                update('copying', child_pid=child.pid, attempt=attempt, log=str(logpath))
                while child.poll() is None:
                    time.sleep(5); update('copying', child_pid=child.pid, attempt=attempt)
            if child.returncode == 0: break
            update('transport-retry', returncode=child.returncode, child_pid=None)
            if attempt == 3: raise RuntimeError('Bounded rsync retries exhausted; rerun to resume')
            time.sleep(15 * attempt)
        update('hashing-source', child_pid=None)
        inventory = a.state.with_name(a.state.stem + '-files.sha256.jsonl')
        remote = '''import hashlib,json,os,sys\nfrom pathlib import Path\nr=Path(sys.argv[1]).resolve();count=size=0\nfor base,dirs,names in os.walk(r):\n dirs.sort()\n for name in sorted(names):\n  p=Path(base)/name\n  if p.is_symlink():raise ValueError("Unexpected symlink in preserved run")\n  if not p.is_file():continue\n  h=hashlib.sha256()\n  with p.open("rb") as f:\n   for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)\n  n=p.stat().st_size;print(json.dumps({"path":str(p.relative_to(r)),"size":n,"sha256":h.hexdigest()}),flush=True);count+=1;size+=n\nprint(json.dumps({"complete":True,"files":count,"bytes":size}),flush=True)\n'''
        remote_command = shlex.join(['python3', '-', source])
        with logpath.open('a') as log, inventory.with_suffix('.partial').open('w') as output:
            child = subprocess.Popen([str(a.ssh_wrapper.resolve()), remote_command], stdin=subprocess.PIPE,
                                     stdout=output, stderr=log, text=True)
            child.stdin.write(remote); child.stdin.close()
            update('hashing-source', child_pid=child.pid)
            while child.poll() is None: time.sleep(5); update('hashing-source', child_pid=child.pid)
            if child.returncode: raise RuntimeError('Source inventory interrupted; resume without deleting partial copy')
        inventory.with_suffix('.partial').replace(inventory)
        entries = [json.loads(line) for line in inventory.read_text().splitlines()]
        complete = entries.pop()
        if not complete.get('complete') or complete['files'] != len(entries): raise ValueError('Incomplete source inventory')
        update('verifying-local', child_pid=None, expected_files=len(entries), expected_bytes=complete['bytes'])
        def verify(entry):
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe inventory path')
            target = a.destination / relative
            return target.is_file() and target.stat().st_size == entry['size'] and sha(target) == entry['sha256']
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for number, valid in enumerate(pool.map(verify, entries), 1):
                if not valid: raise ValueError('Local checksum mismatch; retain data and repair before success')
                if number % 1000 == 0: update('verifying-local', verified_files=number)
        update('verified', child_pid=None, verified_files=len(entries), verified_bytes=complete['bytes'],
               inventory=str(inventory), inventory_sha256=sha(inventory), source_untouched=True)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        update('incomplete-resumable', error=repr(error)); raise


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--rsync-transport':
        if len(sys.argv) < 4 or sys.argv[2] != 'benchmark-cluster': raise SystemExit('Unexpected rsync host')
        wrapper = os.environ['RUMBA_SSH_WRAPPER']
        os.execv(wrapper, [wrapper, *sys.argv[3:]])
    main()
