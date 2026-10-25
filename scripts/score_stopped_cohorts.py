#!/usr/bin/env python3
"""Prepare or judge the Meno0.2 anchor, Qwen anchor and independent cohorts.

Scores are cached once per model/question. Run without --score to prepare reports
without contacting any endpoint. The judge itself is started by the operator.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rumba_bench.cohorts import build_reports
from rumba_bench.data import read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folders', type=Path, required=True, help='JSON model label -> saved model folder')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--score', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'scoring.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'started_at': time.time(), 'status': 'preparing',
             'birth': subprocess.check_output(['ps', '-p', str(os.getpid()), '-o', 'lstart='], text=True).strip(),
             'folders': str(args.folders.resolve()), 'output': str(args.output.resolve())}
    def update(status, **values):
        state.update(status=status, updated_at=time.time(), **values)
        write_json(args.output / 'workflow-status.json', state)
    update('preparing')
    folders = {m: Path(v).resolve() for m, v in read_json(args.folders).items()}
    report = build_reports(folders, args.output)
    if args.score:
        if args.config is None: p.error('--score requires --config')
        from rumba_bench.cli import score
        config = read_json(args.config)
        if not config.get('judge', {}).get('model'):
            raise ValueError('Configure the independent judge endpoint')
        if any(not config['models'][m].get('model') for m in folders):
            raise ValueError('models must contain flat native endpoint metadata, not native/yarn containers')
        plan = read_json(args.output / 'cohort-plan.json')
        try:
            for cohort in plan['cohorts']:
                for model in cohort['models']:
                    ids = cohort['ids'] if cohort['paired'] else cohort['ids_by_model'][model]
                    update('running', cohort=cohort['name'], model=model, target_questions=len(ids))
                    score(config, model, 'full', folder_override=folders[model], allow_partial=True, question_ids=ids)
                report = build_reports(folders, args.output)
                update('cohort-complete', cohort=cohort['name'])
                print(json.dumps({'cohort': cohort['name'], 'status': report['cohorts'][cohort['name']]['status']}), flush=True)
        except BaseException as error:
            update('failed', error=repr(error)); raise
        update('complete')
    else:
        update('prepared')
    print(json.dumps({'status': report['status'], 'output': str(args.output.resolve()),
                      'counts': {c: {m: v['overall']['n'] for m, v in d['models'].items()}
                                 for c, d in report['cohorts'].items()}}), flush=True)


if __name__ == '__main__':
    main()
