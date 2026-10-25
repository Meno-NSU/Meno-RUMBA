#!/usr/bin/env python3
"""Preserve the cancelled Meno0.2 run and release only its independent replicas.

Original servers belong to the pinned parent: stop them only during a graceful
parent handoff, because that parent treats any server exit as a global failure.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,shutil,signal,time
from pathlib import Path

def save(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
 return h.hexdigest()

def finish_archive(root,ptr,job,target,statefile,state):
 def update(phase,**kw):state.update(phase=phase,updated_at=time.time(),**kw);save(statefile,state)
 summary=json.loads((target/'archive-summary.json').read_text())
 manifest=json.loads((target/'files.sha256.json').read_text())
 assert summary['verified_files']==len(manifest)==state['verified_files']
 assert summary['verified_bytes']==state['bytes']
 update('stopping-independent-replicas',archive_manifest_sha256=sha(target/'files.sha256.json'))
 spec=importlib.util.spec_from_file_location('sp',root/'scripts/server_process.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);perf=job/'performance-meno02';repfile=perf/'replicas-status.json'
 if repfile.exists():
  replicas=json.loads(repfile.read_text());info=helper.process_info(replicas['pid'])
  if helper.alive(info):
   assert info['start']==replicas['start_ticks'] and 'launch_meno02_replicas.py' in ' '.join(info['argv']);os.kill(replicas['pid'],signal.SIGTERM)
   deadline=time.monotonic()+180
   while helper.alive(helper.process_info(replicas['pid'])) and time.monotonic()<deadline:time.sleep(2)
  for service in replicas['plan']:helper.stop(perf/(service['tag']+'.pid'),service['port'],'meno-lite-0.2',timeout=30)
 update('preserved',**summary,independent_replica_gpu_released=['6','7','8','9'],original_servers_shutdown='Only when pinned parent finishes Meno0.1; never stop them individually')
 canonical=root/f"runs/cluster-setup/recovery-{ptr['job_id']}/meno-lite-0.2-status.json";old=json.loads(canonical.read_text());old.update(phase='cancelled-by-user',cancelled_by_user=True,archive=str(target),never_restart=True,updated_at=time.time());save(canonical,old)
 print(json.dumps({k:state[k] for k in ['phase','archive','closed_valid_answers','answer_files','verified_files','verified_bytes','independent_replica_gpu_released']}),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--finish-existing',action='store_true');a=p.parse_args();root=a.project.resolve();ptr=json.loads((root/'runs/a800-meno-pair/current.json').read_text());assert ptr['job_id']==os.environ.get('SLURM_JOB_ID');job=Path(ptr['job_dir']);paths=json.loads((job/'paths.json').read_text());work=Path(paths['work']);folder=work/paths['run_relative']/'meno-lite-0.2';statefile=root/'runs/cluster-setup/meno02-archive-current.json';lock=statefile.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if statefile.exists():
  old=json.loads(statefile.read_text())
  if old.get('phase')=='preserved':print(json.dumps(old));return
  if a.finish_existing:
   assert old['phase']=='stopping-independent-replicas','Resume only a fully verified archive'
   assert Path(old['archive'])==Path(ptr['durable'])/'abandoned/meno-lite-0.2'
   old.update(resumed_by_pid=os.getpid(),resumed_at=time.time())
   finish_archive(root,ptr,job,Path(old['archive']),statefile,old);return
 for proc in Path('/proc').glob('[0-9]*'):
  try:
   if proc.stat().st_uid!=os.getuid():continue
   cmd=(proc/'cmdline').read_bytes().decode().split('\0')
   if 'rumba_bench.ragu_harness' in cmd and '--model' in cmd and cmd[cmd.index('--model')+1]=='meno-lite-0.2':raise RuntimeError('Cancelled model still has a live worker; stop it before archiving')
  except (OSError,ValueError,IndexError):pass
 target=Path(ptr['durable'])/'abandoned/meno-lite-0.2';target.mkdir(parents=True,exist_ok=True);state={'pid':os.getpid(),'birth':Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],'phase':'inventory','archive':str(target),'started_at':time.time(),'cancelled_by_user':True,'never_restart':True}
 def update(phase,**kw):state.update(phase=phase,updated_at=time.time(),**kw);save(statefile,state)
 files=[(src,Path('model')/src.relative_to(folder)) for src in folder.rglob('*') if src.is_file()]
 cache=work/'runs/cache/ragu'
 for src in (cache/'generation').glob('*.json'):
  if json.loads(src.read_text()).get('request',{}).get('model')=='meno-lite-0.2':files.append((src,Path('raw/generation')/src.name))
 files += [(src,Path('raw/tokenization')/src.name) for src in (cache/'tokenization').glob('*.json')]
 size=sum(src.stat().st_size for src,dst in files);assert shutil.disk_usage(root).free>size+6*1024**3,'Insufficient archive reserve';update('copying',files=len(files),bytes=size);manifest=[]
 for n,(src,relative) in enumerate(files,1):
  dst=target/relative;dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():shutil.copy2(src,dst)
  checksum=sha(src);assert sha(dst)==checksum;manifest.append({'path':str(relative),'size':src.stat().st_size,'sha256':checksum})
  if n%250==0:update('copying',verified_files=n)
 rows=[json.loads(p.read_text()) for p in (folder/'answers/full').glob('*.json')];closed=[]
 for marker in (folder/'graphs').glob('*/.checkpoint-ready.json'):
  ready=json.loads(marker.read_text());report=json.loads((marker.parent/'build_report.json').read_text());assert ready['graph_closed'] and report['consistent'] and report['all_source_chunks_stored'] and not report['llm_errors'];closed.extend(ready['answers'])
 save(target/'files.sha256.json',manifest);save(target/'archive-summary.json',{'cancelled_by_user':True,'full_run_complete':False,'closed_valid_answers':len(closed),'answer_files':len(rows),'truncated_answer_ids':[r['id'] for r in rows if r.get('truncated')],'closed_users':len(list((folder/'graphs').glob('*/.checkpoint-ready.json'))),'raw_generation_files':sum(str(v['path']).startswith('raw/generation/') for v in manifest),'verified_files':len(manifest),'verified_bytes':size,'historical_audit_directory':str(job),'original_checkpoints':str(Path(ptr['durable'])/'checkpoints'),'original_data_untouched':True})
 update('stopping-independent-replicas',verified_files=len(manifest));finish_archive(root,ptr,job,target,statefile,state)

if __name__=='__main__':main()
