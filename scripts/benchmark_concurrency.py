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

def choose_profile(profiles, baseline_level, minimum_gain=.05):
 baselines=[p for p in profiles if p['concurrency']==baseline_level and not p['errors']]
 if len(baselines)<2:raise ValueError('Two successful baseline controls are required')
 baseline=min(baselines,key=lambda p:p['seconds']);baseline_tps=baseline['output_tokens']/baseline['seconds']
 drift=max(p['seconds'] for p in baselines)/baseline['seconds']-1
 if drift>.15:return baseline,baseline,'baseline-drift'
 eligible=[]
 for profile in profiles:
  if profile['errors'] or profile['concurrency']==baseline_level:continue
  if profile.get('preemptions',0)>max(p.get('preemptions',0) for p in baselines):continue
  if (profile['seconds']<baseline['seconds']*(1-minimum_gain)
      and profile['output_tokens']/profile['seconds']>baseline_tps*(1+minimum_gain)):
   eligible.append(profile)
 if not eligible:return baseline,baseline,'no-reliable-gain'
 fastest=min(eligible,key=lambda p:p['seconds'])
 chosen=min([p for p in eligible if p['seconds']<=fastest['seconds']*1.03],key=lambda p:p['concurrency'])
 return chosen,baseline,'measured-gain'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--workflow-state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--levels');p.add_argument('--baseline',type=int);p.add_argument('--native-samples',type=int,default=20);p.add_argument('--long-samples',type=int,default=4);a=p.parse_args();cfg=json.loads(a.config.read_text());protocol=cfg['protocol'];assert (protocol['temperature'],protocol['seed'],protocol['repetition_penalty'])==(0,42,1.03)
 baseline_level=a.baseline or cfg['runtime']['concurrency'];levels=list(map(int,(a.levels or f'{baseline_level},{min(24,baseline_level*2)},{baseline_level}').split(',')))
 assert levels[0]==levels[-1]==baseline_level and all(1<=v<=24 for v in levels),'Bracket candidates with the current baseline'
 assert a.native_samples>0 and a.long_samples>0
 state=json.loads(a.workflow_state.read_text());assert state['status']!='running','Pause the workflow first'
 # Check known old PIDs; recycled unrelated PIDs are not treated as our worker.
 import importlib.util
 spec=importlib.util.spec_from_file_location('process_checks',ROOT/'scripts/server_process.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 for pid in [state.get('pid'),state.get('child_pid')]:
  if pid:
   info=helper.process_info(pid)
   assert not(helper.alive(info) and ('rumba_bench.ragu_harness' in info['argv'] or any('run_remote_ragu_full.py' in x for x in info['argv']))),'A selected worker is still alive'
 a.output.mkdir(parents=True,exist_ok=True);selected={'native':[],'yarn':[]};limits={'native':a.native_samples,'yarn':a.long_samples};endpoint=cfg['models'][a.model]
 for path in sorted((ROOT/'runs/cache/ragu/generation').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True):
  raw=json.loads(path.read_text());q=raw.get('request',{});u=raw.get('usage',{})
  if (q.get('model')!=endpoint['model'] or raw.get('finish_reason')!='stop' or q.get('max_tokens')!=4096 or not q.get('extra_body',{}).get('structured_outputs') or not 350<=u.get('completion_tokens',0)<=2000):continue
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
 profiles=[];wire_by_key={};local=threading.local()
 def record_wire(request):local.wire.append(hashlib.sha256(request.read()).hexdigest())
 for index,level in enumerate(levels,1):
  name=f'{index}-concurrency-{level}';directory=a.output/name;directory.mkdir(exist_ok=True);clients={route:OpenAI(base_url=(cfg['models'] if route=='native' else cfg['long_context_models'])[a.model]['base_url'],api_key='EMPTY',timeout=cfg['runtime']['timeout_s'],max_retries=0,http_client=httpx.Client(trust_env=False,event_hooks={'request':[record_wire]})) for route in selected}
  status={'pid':os.getpid(),'started_at':time.time(),'phase':'benchmarking','profile':name,'concurrency':level,'completed':0,'total':len(samples)};write(a.output/'status.json',status);stop=threading.Event();observations=[{'time':time.time(),'metrics':metrics(cfg,a.model)}]
  def observe(stop_event=stop,seen=observations):
   while not stop_event.is_set():seen.append({'time':time.time(),'metrics':metrics(cfg,a.model)});stop_event.wait(3)
  monitor=threading.Thread(target=observe,daemon=True);monitor.start()
  def call(item):
   started=time.monotonic()
   try:
    local.wire=[]
    response=clients[item['route']].chat.completions.create(**item['request']);choice=response.choices[0];content=choice.message.content or '';json.loads(content)
    assert len(local.wire)==1,'Unexpected retry/redirect during comparison'
    assert hashlib.sha256(json.dumps(item['request'],sort_keys=True).encode()).hexdigest()==item['request_sha256'],'Request mutated'
    write(directory/(item['cache_key']+'.json'),response.model_dump())
    return {'key':item['cache_key'],'route':item['route'],'request_sha256':item['request_sha256'],'wire_sha256':local.wire[0],'seconds':time.monotonic()-started,'finish_reason':choice.finish_reason,'output_chars':len(content),'usage':response.usage.model_dump(),'output_sha256':hashlib.sha256(content.encode()).hexdigest()}
   except Exception as e:return {'key':item['cache_key'],'route':item['route'],'error_type':type(e).__name__,'seconds':time.monotonic()-started}
  start=time.monotonic();results=[]
  with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
   futures=[pool.submit(call,item) for item in samples]
   for future in concurrent.futures.as_completed(futures):results.append(future.result());status.update(completed=len(results),updated_at=time.time());write(a.output/'status.json',status)
  elapsed=time.monotonic()-start;stop.set();monitor.join(timeout=12);observations.append({'time':time.time(),'metrics':metrics(cfg,a.model)})
  for client in clients.values():client.close()
  errors=[r for r in results if r.get('error_type') or r.get('finish_reason')!='stop']
  for result in results:
   if 'wire_sha256' in result:
    before=wire_by_key.setdefault(result['key'],result['wire_sha256']);assert before==result['wire_sha256'],'Wire payload changed between profiles'
  preemptions=0;peaks={}
  for route in selected:
   route_observations=[o['metrics'].get(route,{}) for o in observations];counts=[m['vllm:num_preemptions_total'] for m in route_observations if 'vllm:num_preemptions_total' in m]
   preemptions+=max(counts)-min(counts) if counts else 0
   peaks[route]={key:max([m.get(key,0) for m in route_observations],default=0) for key in ['vllm:kv_cache_usage_perc','vllm:num_requests_waiting']}
  latencies=sorted(r['seconds'] for r in results);profile={'profile':name,'concurrency':level,'seconds':elapsed,'requests':len(results),'errors':errors,'output_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in results),'preemptions':preemptions,'p95_request_seconds':latencies[min(len(latencies)-1,int(len(latencies)*.95))],'peaks':peaks,'metrics':observations,'results':results};profiles.append(profile);write(a.output/(name+'.json'),profile)
  print(json.dumps({k:profile[k] for k in ['profile','concurrency','seconds','requests','errors','output_tokens','preemptions','p95_request_seconds']}),flush=True)
 chosen,baseline,reason=choose_profile(profiles,baseline_level)
 summary={'phase':'complete','time':time.time(),'baseline_concurrency':baseline_level,'chosen_concurrency':chosen['concurrency'],'selection_reason':reason,'baseline_seconds':baseline['seconds'],'best_seconds':chosen['seconds'],'speedup':baseline['seconds']/chosen['seconds'],'profiles':[{k:p[k] for k in ['profile','concurrency','seconds','requests','errors','output_tokens','preemptions','p95_request_seconds','peaks']} for p in profiles],'wire_payloads_identical':True,'same_requests_and_sampling':True,'production_cache_untouched':True};write(a.output/'comparison.json',summary);write(a.output/'status.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
