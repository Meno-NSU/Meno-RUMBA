#!/usr/bin/env python3
"""Export only closed, QC-valid answers; keep missing questions explicit.

Can stream a small judge-ready tarball over SSH without creating a server copy.
Raw caches and unfinished graphs remain in the original preservation archive.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile


def read(path):
    return json.loads(path.read_text())


def collect(args):
    questions = {r['id']: r for r in map(json.loads, args.questions.read_text().splitlines())}
    files, metadata = {}, {}
    if args.checkpoints:
        latest = read(args.checkpoints / 'latest.json')
        prefix = Path(latest['run_relative']).name
        manifest = args.checkpoints / 'identity/manifest.json'
    else:
        prefix = args.run_folder.parent.name
        manifest = args.run_folder.parent / 'manifest.json'
    files[f'{prefix}/manifest.json'] = manifest.read_bytes()
    for model in args.model:
        if args.checkpoints:
            users = latest['users'].get(model, {})
            objects = [(u, args.checkpoints / entry['object']) for u, entry in sorted(users.items())]
            groups = [(u, obj / 'graph', obj / 'answers') for u, obj in objects]
            endpoint = args.checkpoints / 'identity' / model / 'endpoint.json'
        else:
            if model != args.run_folder.name: raise ValueError('Model/folder mismatch')
            groups = [(g.name, g, args.run_folder / 'answers/full')
                      for g in sorted((args.run_folder / 'graphs').iterdir()) if g.is_dir()]
            endpoint = args.run_folder / 'endpoint.json'
        included, excluded, seen, closed = [], [], set(), 0
        for user, graph, answers in groups:
            marker = graph / '.checkpoint-ready.json'
            if not marker.exists(): continue
            ready, report = read(marker), read(graph / 'build_report.json')
            if (ready.get('suite') != 'full' or not ready.get('graph_closed') or not report['consistent']
                    or not report['all_source_chunks_stored'] or report['llm_errors']):
                excluded.append({'user': user, 'reason': 'graph-QC-or-scope'}); continue
            batch = []
            for name in ready['answers']:
                if Path(name).name != name: raise ValueError('Unsafe answer filename')
                source = answers / name
                row = read(source)
                if row['id'] not in questions or row['id'] in seen: raise ValueError('Unknown/duplicate question ID')
                q = questions[row['id']]
                if row['user_id'] != user or any(row[k] != q[k] for k in ['question', 'answer', 'question_date']):
                    raise ValueError('Saved answer does not match source question')
                if (row.get('truncated') or row.get('finish_reason') != 'stop'
                        or not row.get('prediction', '').strip() or not row.get('raw_response_hash')):
                    raise ValueError('Closed graph contains an invalid answer')
                batch.append((source, row))
                seen.add(row['id'])
            for source, row in batch:
                files[f'{prefix}/{model}/answers/full/{source.name}'] = source.read_bytes()
                included.append(row['id'])
            for name in ['build_report.json', 'pipeline-repair.json', 'execution-policy.json']:
                if (graph / name).exists(): files[f'{prefix}/{model}/source-graph-qc/{user}/{name}'] = (graph / name).read_bytes()
            closed += 1
        if endpoint.exists(): files[f'{prefix}/{model}/endpoint.json'] = endpoint.read_bytes()
        coverage = {'status': 'partial' if len(included) < len(questions) else 'complete',
                    'cancelled_by_user': True, 'expected_questions': len(questions),
                    'included_questions': len(included), 'included_ids': sorted(included),
                    'missing_ids': sorted(set(questions) - seen), 'excluded_users': excluded,
                    'closed_qc_valid_users': closed, 'raw_response_hashes_present': True,
                    'source_generation_unchanged': True, 'judge_pending': True}
        metadata[model] = coverage
        files[f'{prefix}/{model}/coverage.json'] = (json.dumps(coverage, ensure_ascii=False, indent=2) + '\n').encode()
    receipts = [{'path': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                for name, data in sorted(files.items())]
    files['files.sha256.json'] = (json.dumps(receipts, indent=2) + '\n').encode()
    return files, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--checkpoints', type=Path)
    sources.add_argument('--run-folder', type=Path)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--model', action='append', required=True)
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument('--destination', type=Path)
    outputs.add_argument('--tar-stdout', action='store_true')
    args = parser.parse_args()
    files, metadata = collect(args)
    if args.tar_stdout:
        with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o600
                archive.addfile(info, io.BytesIO(data))
    else:
        args.destination.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            target = args.destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data: raise ValueError('Conflicting existing export')
            target.write_bytes(data)
        print(json.dumps({m: {'answers': v['included_questions'], 'expected': v['expected_questions'], 'status': v['status']}
                          for m, v in metadata.items()}), flush=True)


if __name__ == '__main__':
    main()
