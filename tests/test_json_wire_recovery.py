import importlib.util
import errno
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('recover_ragu_worker', Path(__file__).resolve().parents[1] / 'scripts/recover_ragu_worker.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_literal_controls_are_escaped_without_changing_decoded_values():
    raw = '{"name":"Описание:\tкнига\nвторая строка","nested":["a\x00b",{"x":"quote: \\\""}]}'
    fixed = recovery.escape_json_string_controls(raw)
    assert json.loads(fixed) == json.loads(raw, strict=False)
    assert '\\t' in fixed and '\\n' in fixed and '\\u0000' in fixed


def test_valid_responses_remain_byte_identical():
    raw = ' { "x" : "escaped \\t and \\\" quotation", "n": 42 }\n'
    assert recovery.escape_json_string_controls(raw) == raw


def test_cache_repair_preserves_wire_request_and_other_models(tmp_path):
    cache = tmp_path / 'node-local-cache'
    cache.mkdir()
    raw = {'request': {'model': 'meno-lite-0.1', 'extra_body': {'structured_outputs': {'json': {}}}},
           'text': '{"entities":[{"description":"a\tb"}]}', 'finish_reason': 'stop'}
    source = cache / 'invalid.json'
    source.write_text(json.dumps(raw, ensure_ascii=False))
    original = source.read_bytes()
    peer = cache / 'peer.json'
    peer.write_text(json.dumps({**raw, 'request': {**raw['request'], 'model': 'meno-lite-0.2'}}))
    peer_original = peer.read_bytes()
    events = recovery.repair_cached_json(cache, 'meno-lite-0.1', tmp_path / 'audit')
    repaired = json.loads(source.read_text())
    assert len(events) == 1
    assert repaired['request'] == raw['request']
    assert repaired['original_wire_text'] == raw['text']
    assert json.loads(repaired['text']) == json.loads(raw['text'], strict=False)
    assert Path(events[0]['wire_backup']).read_bytes() == original
    assert peer.read_bytes() == peer_original


def test_incomplete_or_structurally_invalid_json_is_not_repaired():
    for raw in ['{"x":"unfinished', '{"x":1,}', '{"x":1}\x00']:
        with pytest.raises(json.JSONDecodeError):
            recovery.escape_json_string_controls(raw)


def test_incomplete_retries_keep_wire_and_stop_at_the_bound(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    raw={'request':{'model':'meno-lite-0.2','seed':42,'max_tokens':4096},'text':'{"description":"repeat','finish_reason':'length'}
    path=cache/'request.json';counts={}
    for attempt in range(2):
        path.write_text(json.dumps(raw))
        original=path.read_bytes()
        events=recovery.retry_incomplete_cached(cache,'meno-lite-0.2',tmp_path/str(attempt),counts,2)
        assert len(events)==1 and not path.exists()
        assert Path(events[0]['wire_backup']).read_bytes()==original
        assert events[0]['retry']==attempt+1
    path.write_text(json.dumps(raw))
    assert recovery.retry_incomplete_cached(cache,'meno-lite-0.2',tmp_path/'third',counts,2)==[]
    assert json.loads(path.read_text())==raw


def test_archive_handles_separate_filesystems_and_keeps_closed_users(tmp_path, monkeypatch):
    folder = tmp_path / 'run'
    graph = folder / 'graphs/user_1'
    graph.mkdir(parents=True)
    (graph / 'data').write_text('closed database content')
    good = folder / 'graphs/user_0'
    good.mkdir()
    (good / '.checkpoint-ready.json').write_text('{}')
    for kind in ['answers', 'retrieval']:
        p = folder / kind / 'full/1_q0.json'
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({'user_id': 'user_1'}))
    rename = recovery.os.rename
    def cross_device(source, target, *args, **kwargs):
        if str(source).startswith(str(folder)):
            raise OSError(errno.EXDEV, 'separate filesystem')
        return rename(source, target, *args, **kwargs)
    monkeypatch.setattr(recovery.os, 'rename', cross_device)
    archive = tmp_path / 'persistent'
    assert recovery.archive_unclosed(folder, archive) == ['user_1']
    assert (good / '.checkpoint-ready.json').exists()
    assert not graph.exists()
    assert (archive / 'unclosed-users/user_1/data').read_text() == 'closed database content'
    assert (archive / 'answers/1_q0.json').exists()
    assert (archive / 'retrieval/1_q0.json').exists()
