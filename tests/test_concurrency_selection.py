import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('comparison',Path(__file__).resolve().parents[1]/'scripts/benchmark_concurrency.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)

def profile(concurrency,seconds,tokens=1000,errors=None,preemptions=0):
 return {'concurrency':concurrency,'seconds':seconds,'output_tokens':tokens,'errors':errors or [],'preemptions':preemptions}

def test_current16_is_baseline_and_close_candidates_prefer_lower_load():
 chosen,baseline,reason=comparison.choose_profile([profile(16,100),profile(20,90),profile(24,89),profile(16,102)],16)
 assert chosen['concurrency']==20 and baseline['concurrency']==16 and reason=='measured-gain'

def test_shorter_output_alone_does_not_count_as_throughput_improvement():
 chosen,_,reason=comparison.choose_profile([profile(16,100),profile(24,85,tokens=800),profile(16,101)],16)
 assert chosen['concurrency']==16 and reason=='no-reliable-gain'

def test_errors_or_extra_kv_preemptions_prevent_automatic_increase():
 chosen,_,_=comparison.choose_profile([profile(16,100),profile(20,80,errors=['timeout']),profile(24,75,preemptions=1),profile(16,102)],16)
 assert chosen['concurrency']==16

def test_drift_between_control_runs_keeps_existing_setting():
 chosen,_,reason=comparison.choose_profile([profile(16,100),profile(24,80),profile(16,120)],16)
 assert chosen['concurrency']==16 and reason=='baseline-drift'

def test_two_valid_controls_are_required():
 with pytest.raises(ValueError):comparison.choose_profile([profile(16,100),profile(24,80)],16)

def test_multi_profile_run_reuses_requests_not_metrics(tmp_path,monkeypatch):
 import collections,hashlib,json,shutil,sys
 import httpx
 source=Path(__file__).resolve().parents[1]
 (tmp_path/'scripts').mkdir();shutil.copy2(source/'scripts/server_process.py',tmp_path/'scripts/server_process.py')
 cache=tmp_path/'runs/cache/ragu/generation';cache.mkdir(parents=True)
 for i,tokens in enumerate([100,200,30000]):
  q={'model':'test-model','messages':[{'role':'user','content':f'case{i}'}],'temperature':0,'seed':42,'max_tokens':4096,'extra_body':{'repetition_penalty':1.03,'structured_outputs':{'json':{'type':'object'}}}}
  (cache/f'{i}.json').write_text(json.dumps({'request':q,'finish_reason':'stop','usage':{'prompt_tokens':tokens,'completion_tokens':512}}))
 config={'protocol':{'temperature':0,'seed':42,'repetition_penalty':1.03},'runtime':{'timeout_s':5,'concurrency':16},'models':{'test':{'base_url':'http://test.invalid:9000/v1','model':'test-model'}},'long_context_models':{'test':{'base_url':'http://test.invalid:9001/v1','model':'test-model'}}}
 cfg=tmp_path/'config.json';cfg.write_text(json.dumps(config));state=tmp_path/'state.json';state.write_text(json.dumps({'status':'interrupted'}));received=[]
 def receive(self,request):
  body=request.read();received.append(hashlib.sha256(body).hexdigest());q=json.loads(body);assert q['temperature']==0 and q['seed']==42 and q['repetition_penalty']==1.03
  return httpx.Response(200,request=request,json={'id':'test','object':'chat.completion','created':1,'model':'test-model','choices':[{'index':0,'message':{'role':'assistant','content':'{}'},'finish_reason':'stop'}],'usage':{'prompt_tokens':100,'completion_tokens':512,'total_tokens':612}})
 monkeypatch.setattr(comparison,'ROOT',tmp_path);monkeypatch.setattr(comparison,'metrics',lambda *a:{'native':{},'yarn':{}});monkeypatch.setattr(httpx.HTTPTransport,'handle_request',receive)
 out=tmp_path/'output';monkeypatch.setattr(sys,'argv',['benchmark','--config',str(cfg),'--model','test','--workflow-state',str(state),'--output',str(out),'--levels','16,20,24,16','--native-samples','2','--long-samples','1'])
 comparison.main();result=json.loads((out/'comparison.json').read_text())
 assert [p['requests'] for p in result['profiles']]==[3,3,3,3]
 assert result['wire_payloads_identical'] and len(received)==12
 assert sorted(collections.Counter(received).values())==[4,4,4]
