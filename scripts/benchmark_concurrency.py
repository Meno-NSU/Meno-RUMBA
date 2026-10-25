#!/usr/bin/env python3
"""Compare execution concurrency with identical successful cached requests.

No prompts/sampling are changed. Diagnostic outputs never enter production cache.
Pause the selected workflow before benchmarking shared remote hardware.
"""
import argparse,concurrent.futures,hashlib,json,os,random,threading,time
from pathlib import Path
import httpx
from openai import OpenAI

ROOT=Path(__file__).resolve().parents[1]
def write(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def metrics(config,label):
 result={}
 for name,endpoint in [('native',config['models'][label]),('yarn',config['long_context_models'][label])]:
  base=endpoint['base_url'].rstrip('/').removesuffix('/v1')
  try:
   r=httpx.get(base+'/metrics',timeout=5,trust_env=False);r.raise_for_status();selected={}
   for line in r.text.splitlines():
    if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{','vllm:num_preemptions_total{')):selected[line.split('{',1)[0]]=float(line.rsplit(' ',1)[1])
   result[name]=selected
  except Exception as e:result[name]={'error':type(e).__name__}
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--workflow-state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--levels',default='8,16,8');a=p.parse_args();cfg=json.loads(a.config.read_text());protocol=cfg['protocol'];assert (protocol['temperature'],protocol['seed'],protocol['repetition_penalty'])==(0,42,1.03)
 state=json.loads(a.workflow_state.read_text());assert state['status']!='running','Pause the workflow first'
 # Check known old PIDs; recycled unrelated PIDs are not treated as our worker.
 import importlib.util
 spec=importlib.util.spec_from_file_location('process_checks',ROOT/'scripts/server_process.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 for pid in [state.get('pid'),state.get('child_pid')]:
  if pid:
   info=helper.process_info(pid)
   assert not(helper.alive(info) and ('rumba_bench.ragu_harness' in info['argv'] or any('run_remote_ragu_full.py' in x for x in info['argv']))),'A selected worker is still alive'
 a.output.mkdir(parents=True,exist_ok=True);selected={'native':[],'yarn':[]};limits={'native':20,'yarn':4};endpoint=cfg['models'][a.model]
 for path in sorted((ROOT/'runs/cache/ragu/generation').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True):
  raw=json.loads(path.read_text());q=raw.get('request',{});u=raw.get('usage',{})
  if (q.get('model')!=endpoint['model'] or raw.get('finish_reason')!='stop' or q.get('max_tokens')!=4096 or not q.get('extra_body',{}).get('structured_outputs') or not 350<=u.get('completion_tokens',0)<=1200):continue
  assert q['temperature']==0 and q['seed']==42 and q['extra_body']['repetition_penalty']==1.03
  route='native' if u['prompt_tokens']+q['max_tokens']<=32768 else 'yarn'
  if len(selected[route])<limits[route]:selected[route].append({'cache_key':path.stem,'route':route,'request':q,'request_sha256':hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()})
  if all(len(selected[k])==v for k,v in limits.items()):break
 assert all(len(selected[k])==v for k,v in limits.items()),'Not enough representative cached requests'
 samples=selected['native']+selected['yarn'];random.Random(42).shuffle(samples);write(a.output/'sample-metadata.json',[{k:v for k,v in x.items() if k!='request'} for x in samples])
 deadline=time.monotonic()+120
 while True:
  m=metrics(cfg,a.model)
  if all(not x.get('error') and x.get('vllm:num_requests_running',0)==x.get('vllm:num_requests_waiting',0)==0 for x in m.values()):break
  assert time.monotonic()<deadline,'Endpoints did not become idle; refusing contaminated comparison';time.sleep(2)
 profiles=[]
 for index,level in enumerate(map(int,a.levels.split(',')),1):
  assert 1<=level<=24;name=f'{index}-concurrency-{level}';directory=a.output/name;directory.mkdir(exist_ok=True);clients={route:OpenAI(base_url=(cfg['models'] if route=='native' else cfg['long_context_models'])[a.model]['base_url'],api_key='EMPTY',timeout=cfg['runtime']['timeout_s'],max_retries=0,http_client=httpx.Client(trust_env=False)) for route in selected}
  status={'pid':os.getpid(),'started_at':time.time(),'phase':'benchmarking','profile':name,'concurrency':level,'completed':0,'total':len(samples)};write(a.output/'status.json',status);stop=threading.Event();observations=[]
  def observe():
   while not stop.is_set():observations.append({'time':time.time(),'metrics':metrics(cfg,a.model)});stop.wait(3)
  monitor=threading.Thread(target=observe,daemon=True);monitor.start()
  def call(item):
   started=time.monotonic()
   try:
    response=clients[item['route']].chat.completions.create(**item['request']);choice=response.choices[0];content=choice.message.content or '';json.loads(content)
    write(directory/(item['cache_key']+'.json'),response.model_dump())
    return {'key':item['cache_key'],'route':item['route'],'request_sha256':item['request_sha256'],'seconds':time.monotonic()-started,'finish_reason':choice.finish_reason,'output_chars':len(content),'usage':response.usage.model_dump(),'output_sha256':hashlib.sha256(content.encode()).hexdigest()}
   except Exception as e:return {'key':item['cache_key'],'route':item['route'],'error_type':type(e).__name__,'seconds':time.monotonic()-started}
  start=time.monotonic();results=[]
  with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
   futures=[pool.submit(call,item) for item in samples]
   for future in concurrent.futures.as_completed(futures):results.append(future.result());status.update(completed=len(results),updated_at=time.time());write(a.output/'status.json',status)
  elapsed=time.monotonic()-start;stop.set();monitor.join(timeout=12)
  for client in clients.values():client.close()
  errors=[r for r in results if r.get('error_type') or r.get('finish_reason')!='stop'];profile={'profile':name,'concurrency':level,'seconds':elapsed,'requests':len(results),'errors':errors,'output_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in results),'metrics':observations,'results':results};profiles.append(profile);write(a.output/(name+'.json'),profile)
  print(json.dumps({k:profile[k] for k in ['profile','concurrency','seconds','requests','errors','output_tokens']}),flush=True)
 baselines=[p for p in profiles if p['concurrency']==8 and not p['errors']];assert baselines,'Baseline invalid';baseline=min(baselines,key=lambda p:p['seconds']);candidates=[p for p in profiles if not p['errors']];best=min(candidates,key=lambda p:p['seconds']);chosen=best['concurrency'] if best['seconds']<baseline['seconds']*0.95 else 8
 summary={'phase':'complete','time':time.time(),'chosen_concurrency':chosen,'baseline_seconds':baseline['seconds'],'best_seconds':best['seconds'],'speedup':baseline['seconds']/best['seconds'],'profiles':[{k:p[k] for k in ['profile','concurrency','seconds','requests','errors','output_tokens']} for p in profiles],'same_requests_and_sampling':True,'production_cache_untouched':True};write(a.output/'comparison.json',summary);write(a.output/'status.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
