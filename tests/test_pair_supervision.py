import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('pair_resilient',Path(__file__).resolve().parents[1]/'scripts/run_meno_pair_resilient.py')
pair=importlib.util.module_from_spec(spec);spec.loader.exec_module(pair)


def test_only_authorized_worker_commands_are_wrapped(tmp_path):
    jobdir=tmp_path/'job';work=tmp_path/'work';jobdir.mkdir();work.mkdir()
    pointer=tmp_path/'runs/a800-meno-pair/current.json';pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({'job_id':'123','job_dir':str(jobdir),'fingerprint':'identity'}))
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
