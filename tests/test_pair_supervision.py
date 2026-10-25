import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('pair_resilient',Path(__file__).resolve().parents[1]/'scripts/run_meno_pair_resilient.py')
pair=importlib.util.module_from_spec(spec);spec.loader.exec_module(pair)


def test_only_authorized_worker_commands_are_wrapped(tmp_path):
    jobdir=tmp_path/'job';work=tmp_path/'work';jobdir.mkdir();work.mkdir()
    pointer=tmp_path/'runs/a800-meno-pair/current.json';pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({'job_id':'123','job_dir':str(jobdir),'fingerprint':'identity','durable':str(tmp_path/'durable')}))
    (jobdir/'paths.json').write_text(json.dumps({'work':str(work),'run_relative':'runs/ragu-pinned'}))
    command=['python','-u','-m','rumba_bench.ragu_harness','run','--config',str(work/'config.pair.json'),'--model','meno-lite-0.2','--suite','full']
    wrapped,event=pair.supervised_command(command,tmp_path,tmp_path/'helper.py','123')
    assert wrapped[0]=='python' and '--model' in wrapped
    assert event['checkpoint_fingerprint']=='identity'
    assert event['original_command']==command
    assert str(work/'runs/ragu-pinned/meno-lite-0.2') in wrapped
    unrelated=['vllm','serve','model','--port','9022']
    assert pair.supervised_command(unrelated,tmp_path,tmp_path/'helper.py','123')==(unrelated,None)
    with pytest.raises(ValueError):pair.supervised_command(command,tmp_path,tmp_path/'helper.py','124')
    command[command.index('meno-lite-0.2')]='Qwen'
    with pytest.raises(ValueError):pair.supervised_command(command,tmp_path,tmp_path/'helper.py','123')


def test_restored_closed_graph_is_kept_instead_of_archived(tmp_path):
    checkpoint=tmp_path/'checkpoint';obj=checkpoint/'objects/meno-lite-0.1/user_0';graph=tmp_path/'run/graphs/user_0'
    (obj/'graph').mkdir(parents=True);graph.mkdir(parents=True)
    report=json.dumps({'consistent':True,'llm_errors':[]})
    (obj/'graph/build_report.json').write_text(report);(graph/'build_report.json').write_text(report)
    ready={'graph_closed':True,'suite':'full','user':'user_0','answers':['0_q0.json']}
    (obj/'object.json').write_text(json.dumps(ready))
    (checkpoint/'latest.json').write_text(json.dumps({'fingerprint':'same','users':{'meno-lite-0.1':{'user_0':{'object':'objects/meno-lite-0.1/user_0'}}}}))
    event={'checkpoint_dir':str(checkpoint),'checkpoint_fingerprint':'same','model':'meno-lite-0.1','run_folder':str(tmp_path/'run')}
    assert pair.mark_pristine_restored_users(event)==1
    assert json.loads((graph/'.checkpoint-ready.json').read_text())==ready
    (graph/'build_report.json').write_text('{}')
    with pytest.raises(ValueError):pair.mark_pristine_restored_users(event)
