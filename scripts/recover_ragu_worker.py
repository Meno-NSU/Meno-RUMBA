#!/usr/bin/env python3
"""Resume one stopped worker without interrupting another model's Slurm run.

Only JSON string serialization is repaired: literal control characters become
escapes with identical decoded values. Original wire responses remain archived.
Generation flags, schemas, ontology, scoring and pinned checkout stay unchanged.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def escape_json_string_controls(text):
    """Return unchanged valid JSON, or a lossless serialization-only repair."""
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError as error:
        if not error.msg.startswith("Invalid control character"):
            raise
    original_value = json.loads(text, strict=False)
    inside = escaped = False
    result = []
    for char in text:
        if inside and ord(char) < 32:
            result.append(json.dumps(char)[1:-1])
            escaped = False
            continue
        result.append(char)
        if escaped:
            escaped = False
        elif char == "\\" and inside:
            escaped = True
        elif char == '"':
            inside = not inside
    normalized = "".join(result)
    if json.loads(normalized) != original_value:
        raise ValueError("JSON normalization changed decoded values")
    return normalized


def live_workers(model, work):
    found = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = (proc / 'cmdline').read_bytes().decode().split('\0')
            if 'rumba_bench.ragu_harness' not in args or '--model' not in args:
                continue
            if args[args.index('--model') + 1] == model and (proc / 'cwd').resolve() == work:
                found.append(int(proc.name))
        except (OSError, ValueError, IndexError):
            continue
    return found


def repair_cached_json(cache, model, archive):
    repairs = []
    for path in sorted(cache.glob('*.json')):
        raw = json.loads(path.read_text())
        request = raw.get('request', {})
        if (request.get('model') != model or raw.get('finish_reason') != 'stop'
                or not request.get('extra_body', {}).get('structured_outputs')):
            continue
        try:
            normalized = escape_json_string_controls(raw['text'])
        except (json.JSONDecodeError, KeyError):
            continue
        if normalized == raw['text']:
            continue
        backup = archive / 'wire-responses' / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        event = {'kind': 'escape_literal_json_string_controls', 'wire_backup': str(backup),
                 'original_text_sha256': hashlib.sha256(raw['text'].encode()).hexdigest(),
                 'normalized_text_sha256': hashlib.sha256(normalized.encode()).hexdigest(),
                 'decoded_values_identical': True, 'request_unchanged': True}
        raw['original_wire_text'] = raw['text']
        raw['text'] = normalized
        raw['json_serialization_repair'] = event
        write_json(path, raw)
        repairs.append({'cache_file': str(path), **event})
    return repairs


def archive_unclosed(folder, archive):
    users = []
    for graph in sorted((folder / 'graphs').glob('*')):
        if not graph.is_dir() or (graph / '.checkpoint-ready.json').exists():
            continue
        user = graph.name
        destination = archive / 'unclosed-users' / user
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Move retrieval before its answer, and the graph last. If interrupted,
        # an unclosed graph still identifies every remaining answer to archive.
        for path in (folder / 'answers/full').glob('*.json'):
            if json.loads(path.read_text()).get('user_id') != user:
                continue
            retrieval = folder / 'retrieval/full' / path.name
            if retrieval.exists():
                target = archive / 'retrieval' / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(retrieval), str(target))
            target = archive / 'answers' / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
        # Scratch and persistent logs can be different filesystems. The graph
        # is closed; shutil.move safely falls back to copying before removal.
        shutil.move(str(graph), str(destination))
        users.append(user)
    return users


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'run-folder', 'state', 'log-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--max-attempts', type=int, default=180)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run inside the existing Slurm allocation')
    config = args.config.resolve(); work = config.parent
    folder = args.run_folder.resolve(); logdir = args.log_dir.resolve()
    if folder.name != args.model or not folder.is_relative_to(work) or not (folder.parent / 'manifest.json').exists():
        raise ValueError('Run folder must belong to the selected model and working checkout')
    settings = json.loads(config.read_text())
    if settings.get('judge') is not None or settings['protocol']['temperature'] != 0 or settings['protocol']['seed'] != 42 or settings['protocol']['repetition_penalty'] != 1.03:
        raise ValueError('Recovery requires the authorized unchanged no-judge protocol')
    logdir.mkdir(parents=True, exist_ok=True)
    lock = (logdir / 'recovery.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'job_id': os.environ['SLURM_JOB_ID'],
             'step_id': os.environ.get('SLURM_STEP_ID'), 'model': args.model,
             'config': str(config), 'run_folder': str(folder), 'started_at': time.time(), 'history': []}
    def update(phase, **extra):
        state.update(phase=phase, updated_at=time.time(), **extra)
        write_json(args.state, state)
    cache = work / 'runs/cache/ragu/generation'
    state['cache_repairs_already_present'] = [
        raw['json_serialization_repair'] for path in cache.glob('*.json')
        if (raw := json.loads(path.read_text())).get('request', {}).get('model') == args.model
        and raw.get('json_serialization_repair')]
    for attempt in range(1, args.max_attempts + 1):
        existing = live_workers(args.model, work)
        if existing:
            raise RuntimeError(f'Refusing to archive or duplicate live workers: {existing}')
        archive = logdir / ('attempt-' + str(attempt) + '-' + str(time.time_ns()))
        repairs = repair_cached_json(cache, args.model, archive)
        if attempt > 1 and not repairs:
            update('failed', error='Worker failed without a supported serialization repair', child_pid=None)
            return 1
        archived = archive_unclosed(folder, archive)
        event = {'attempt': attempt, 'repairs': repairs, 'archived_users': archived, 'archive': str(archive)}
        state['history'].append(event)
        command = [sys.executable, '-u', '-m', 'rumba_bench.ragu_harness', 'run',
                   '--config', str(config), '--model', args.model, '--suite', 'full']
        with (logdir / ('benchmark-attempt-' + str(attempt) + '.log')).open('a') as output:
            child = subprocess.Popen(command, cwd=work, env=os.environ.copy(), stdout=output,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            update('running', attempt=attempt, child_pid=child.pid, child_command=command,
                   log=str(logdir / ('benchmark-attempt-' + str(attempt) + '.log')))
            while child.poll() is None:
                update('running', answers=len(list((folder / 'answers/full').glob('*.json'))),
                       closed_users=len(list((folder / 'graphs').glob('*/.checkpoint-ready.json'))))
                time.sleep(15)
        event['exit_code'] = child.returncode
        if child.returncode == 0:
            qc = json.loads((folder / 'full/qc.json').read_text())
            valid = (qc['expected_questions'] == qc['answered_questions'] == 1543
                     and qc['same_question_ids'] and not qc['truncated_answers']
                     and qc['llm_extraction_errors'] == 0 and qc['all_graphs_consistent']
                     and qc['all_source_chunks_stored'] and qc['judge_pending'])
            update('complete' if valid else 'failed-qc', qc=qc, child_pid=None)
            return 0 if valid else 1
        update('retrying', child_pid=None, last_exit_code=child.returncode)
    update('failed', error='Bounded recovery attempts exhausted', child_pid=None)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
