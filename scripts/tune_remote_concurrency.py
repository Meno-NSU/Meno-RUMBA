#!/usr/bin/env python3
"""Quiesce one remote workflow, measure concurrency, resume the same run.

Config changes are restricted to runtime.concurrency; exact manifest identity,
protocol, endpoints, existing answers and successful cache entries are retained.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rumba_bench.data import digest,read_json,write_json
from rumba_bench.ragu_harness import manifest
spec=importlib.util.spec_from_file_location('process_checks',ROOT/'scripts/server_process.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def with_concurrency(config,value):
 import copy
 assert 1<=value<=24
 result=copy.deepcopy(config);result['runtime']['concurrency']=value
 assert {k:v for k,v in result.items() if k!='runtime'}=={k:v for k,v in config.items() if k!='runtime'}
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.config=a.config.resolve();a.state=a.state.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True);lock=(a.output/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=read_json(a.config);original_spec=manifest(cfg);original_sha=digest(original_spec);old=read_json(a.state);assert old['status']=='running' and old['model']==a.model and old['stage']=='full';run=Path(old['run_folder']);assert run.parent.name=='ragu-'+original_sha[:16]
 own=helper.process_info(os.getpid());status={'pid':os.getpid(),'birth':own['start'],'phase':'preparing','started_at':time.time(),'config':str(a.config),'model':a.model,'run_folder':str(run),'manifest_sha256':original_sha,'original_concurrency':cfg['runtime']['concurrency']};statuspath=a.output/'tuning-status.json'
 def update(phase,**values):status.update(phase=phase,updated_at=time.time(),**values);write_json(statuspath,status)
 identities={}
 for key in ['pid','child_pid']:
  info=helper.process_info(old[key]);assert helper.alive(info);args=info['argv'];assert '--model' in args and args[args.index('--model')+1]==a.model
  assert (any('run_remote_ragu_full.py' in s for s in args) if key=='pid' else 'rumba_bench.ragu_harness' in args)
  identities[key]=info
 boundary={}
 for marker in (run/'graphs').glob('*/.checkpoint-ready.json'):
  ready=read_json(marker);report=read_json(marker.parent/'build_report.json');assert report['consistent'] and report['all_source_chunks_stored'] and not report['llm_errors']
  for name in ready['answers']:
   answer=run/'answers'/ready['suite']/name;boundary[str(answer.relative_to(run))]=hashlib.sha256(answer.read_bytes()).hexdigest()
 write_json(a.output/'before.json',{'config':cfg,'manifest_sha256':original_sha,'processes':identities,'boundary_hashes':boundary,'time':time.time()})
 quiesced=False;benchmark=None;chosen=cfg['runtime']['concurrency'];warning=None
 def interrupted(number,frame):raise KeyboardInterrupt('Tuning interrupted; resume original workflow')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  update('quiescing',old_processes=identities);current=helper.process_info(old['pid']);assert current['start']==identities['pid']['start'];os.kill(old['pid'],signal.SIGTERM)
  deadline=time.monotonic()+75
  while any(helper.alive(info:=helper.process_info(old[k])) and info['start']==identities[k]['start'] for k in identities):
   if time.monotonic()>deadline:raise RuntimeError('Old workflow did not stop; refusing a duplicate')
   time.sleep(.5)
  quiesced=True
  command=[sys.executable,'-u',str(ROOT/'scripts/benchmark_concurrency.py'),'--config',str(a.config),'--model',a.model,'--workflow-state',str(a.state),'--output',str(a.output),'--levels','8,16,8']
  with (a.output/'benchmark.log').open('a') as log:
   benchmark=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);update('benchmarking',benchmark_pid=benchmark.pid)
   deadline=time.monotonic()+1800
   while benchmark.poll() is None:
    if time.monotonic()>deadline:raise TimeoutError('Bounded benchmark time exceeded')
    time.sleep(3)
  if benchmark.returncode:raise RuntimeError('Comparison failed; retain original concurrency')
  comparison=read_json(a.output/'comparison.json');chosen=comparison['chosen_concurrency'];update('comparison-complete',comparison=comparison)
 except BaseException as error:
  warning=type(error).__name__+': '+str(error);update('recovering-original-workflow',warning=warning);chosen=cfg['runtime']['concurrency']
 finally:
  if benchmark is not None and benchmark.poll() is None:
   benchmark.terminate()
   try:benchmark.wait(timeout=10)
   except subprocess.TimeoutExpired:benchmark.kill();benchmark.wait()
  if not quiesced:
   update('not-resumed-old-process-still-live',warning=warning);return 1
  assert read_json(a.config)==cfg,'Config changed concurrently; refusing overwrite'
  for name,checksum in boundary.items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==checksum,'A closed answer changed during comparison'
  candidate=with_concurrency(cfg,chosen);assert manifest(candidate)==original_spec,'Runtime change must preserve exact manifest identity'
  if chosen!=cfg['runtime']['concurrency']:write_json(a.config,candidate)
  write_json(a.output/'execution-change.json',{'time':time.time(),'old_concurrency':cfg['runtime']['concurrency'],'new_concurrency':chosen,'manifest_sha256_before':original_sha,'manifest_sha256_after':digest(manifest(candidate)),'generation_endpoints_prompts_unchanged':True,'closed_boundary_hashes_verified':len(boundary),'warning':warning})
  command=[sys.executable,'-u',str(ROOT/'scripts/run_remote_ragu_full.py'),'--config',str(a.config),'--model',a.model,'--state',str(a.state)]
  with (a.output/'resumed-workflow.log').open('a') as log:
   child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  info=helper.process_info(child.pid);update('resumed',chosen_concurrency=chosen,new_supervisor_pid=child.pid,new_supervisor_birth=info['start'],warning=warning)
  print(json.dumps({'phase':'resumed','chosen_concurrency':chosen,'new_supervisor_pid':child.pid,'manifest_preserved':True,'warning':warning}),flush=True)
 return 0

if __name__=='__main__':raise SystemExit(main())
