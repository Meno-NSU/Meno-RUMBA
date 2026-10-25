#!/usr/bin/env python3
"""Supervise both workers while executing the unchanged pinned pair launcher.

Only child-process lifecycle is wrapped. The original checkout, run identity,
generation configuration, graph code, strict QC and checkpoint writer remain
the pinned launcher's responsibility. Every process override is recorded.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import uuid


def supervised_command(command, project, helper, job):
    original = [str(x) for x in command]
    if len(original) < 6 or original[1:5] != ['-u','-m','rumba_bench.ragu_harness','run']:
        return original, None
    def value(flag):
        return original[original.index(flag)+1]
    model = value('--model')
    if model not in ('meno-lite-0.1','meno-lite-0.2') or value('--suite') != 'full':
        raise ValueError('Only the authorized Meno full workers may be supervised')
    current = json.loads((project/'runs/a800-meno-pair/current.json').read_text())
    if current['job_id'] != job:
        raise ValueError('Refusing a stale Slurm current pointer')
    jobdir = Path(current['job_dir'])
    paths = json.loads((jobdir/'paths.json').read_text())
    work = Path(paths['work'])
    config = Path(value('--config'))
    if config.resolve() != (work/'config.pair.json').resolve():
        raise ValueError('Worker config does not belong to the staged work directory')
    state = project/'runs/cluster-setup'/('recovery-'+job)/(model+'-status.json')
    logdir = jobdir/('recovery-'+model)
    wrapped = [original[0],'-u',str(helper),'--config',str(config),'--model',model,
               '--run-folder',str(work/paths['run_relative']/model),
               '--state',str(state),'--log-dir',str(logdir)]
    return wrapped, {'model':model,'original_command':original,'supervised_command':wrapped,
                     'state':str(state),'log_dir':str(logdir),
                     'run_folder':str(work/paths['run_relative']/model),
                     'checkpoint_dir':str(Path(current['durable'])/'checkpoints'),
                     'checkpoint_fingerprint':current['fingerprint']}


def mark_pristine_restored_users(event):
    """The pinned restore removes markers; retain its verified closed objects.

    This runs exactly before the model's first process is spawned, while the
    staged restored databases have never been opened by a benchmark worker.
    """
    checkpoint=Path(event['checkpoint_dir']);latest=checkpoint/'latest.json'
    if not latest.exists():
        return 0
    state=json.loads(latest.read_text())
    if state['fingerprint']!=event['checkpoint_fingerprint']:
        raise ValueError('Restored checkpoint fingerprint changed')
    count=0
    for user,record in state['users'].get(event['model'],{}).items():
        relative=Path(record['object'])
        if relative.is_absolute() or '..' in relative.parts or Path(user).name!=user:
            raise ValueError('Unsafe restored object path')
        source=checkpoint/relative;graph=Path(event['run_folder'])/'graphs'/user
        if (graph/'build_report.json').read_bytes()!=(source/'graph/build_report.json').read_bytes():
            raise ValueError('Restored graph differs from its immutable checkpoint')
        ready=json.loads((source/'object.json').read_text())
        if not ready['graph_closed'] or ready['user']!=user or ready['suite']!='full':
            raise ValueError('Invalid committed closed-user marker')
        marker=graph/'.checkpoint-ready.json';tmp=marker.with_name(marker.name+'.tmp-'+uuid.uuid4().hex)
        tmp.write_text(json.dumps(ready,indent=2)+'\n');tmp.replace(marker);count+=1
    return count


def main():
    project = Path(os.environ.get('MENO_RUMBA_PROJECT', Path.cwd())).resolve()
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        raise ValueError('Use a Slurm allocation')
    launcher = project/'scripts/run_meno_pair.py'
    helper = Path(os.environ.get('MENO_WORKER_RECOVERY_HELPER',Path(__file__).with_name('recover_ragu_worker.py'))).resolve()
    if not launcher.is_file() or not helper.is_file():
        raise ValueError('Pinned launcher and recovery helper must exist')
    expected = os.environ.get('MENO_PAIR_REVISION')
    revision = subprocess.check_output(['git','-C',str(project),'rev-parse','HEAD'],text=True).strip()
    if not expected or expected != revision:
        raise ValueError('An exact pinned MENO_PAIR_REVISION is required')
    original_popen = subprocess.Popen
    def popen(command, *args, **kwargs):
        if not isinstance(command,(list,tuple)):
            return original_popen(command,*args,**kwargs)
        wrapped, event = supervised_command(command,project,helper,job)
        if event:
            event['pristine_restored_users']=mark_pristine_restored_users(event)
            event.update(event='worker_supervision',job_id=job,revision=revision,
                         launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
                         recovery_helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                         supervision_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            target=Path(event['log_dir']);target.mkdir(parents=True,exist_ok=True)
            (target/'supervision.json').write_text(json.dumps(event,indent=2)+'\n')
            print(json.dumps(event),flush=True)
        return original_popen(wrapped,*args,**kwargs)
    subprocess.Popen = popen
    try:
        sys.argv=[str(launcher),*sys.argv[1:]]
        runpy.run_path(str(launcher),run_name='__main__')
    finally:
        subprocess.Popen = original_popen


if __name__ == '__main__':
    main()
