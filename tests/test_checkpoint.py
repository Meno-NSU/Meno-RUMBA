import json
from pathlib import Path

import pytest

from rumba_bench.checkpoint import ClosedUserCheckpoints, READY, mark_user_closed
from rumba_bench.data import read_json, write_json


def user(work, name, *, closed):
    folder = work / "runs/ragu-test/meno-lite-0.1"
    graph = folder / "graphs" / name
    graph.mkdir(parents=True)
    write_json(graph / "build_report.json", {"consistent": True, "all_source_chunks_stored": True, "llm_errors": []})
    (graph / "storage.sqlite").write_bytes(b"closed database" if closed else b"live database: do not copy")
    question = {"id": name + "|q1", "user_id": name}
    write_json(folder / "answers/full" / (name + "_q1.json"), {
        **question, "prediction": "answer", "answer": "answer", "category": "StaticUser, single, atemporal",
        "finish_reason": "stop", "truncated": False})
    if closed:
        mark_user_closed(folder, name, "full", [question])
    return folder, graph


def test_only_closed_graphs_and_answers_are_restored(tmp_path):
    work = tmp_path / "work"
    _, good = user(work, "user_1", closed=True)
    user(work, "user_2", closed=False)
    write_json(work / "runs/cache/ragu/generation/a.json", {"text": "expensive call for unfinished user"})
    store = ClosedUserCheckpoints(tmp_path / "durable", "fingerprint")
    result = store.commit(work, "runs/ragu-test", ["meno-lite-0.1"], "full")
    assert result["answers"] == {"meno-lite-0.1": 1}
    restored = tmp_path / "new-node"
    assert store.restore(restored, "runs/ragu-test") == {"users": 1, "answers": 1}
    assert (restored / "runs/ragu-test/meno-lite-0.1/graphs/user_1/storage.sqlite").read_bytes() == b"closed database"
    assert not (restored / "runs/ragu-test/meno-lite-0.1/graphs/user_2").exists()
    assert not (restored / "runs/ragu-test/meno-lite-0.1/graphs/user_1" / READY).exists()
    assert read_json(restored / "runs/cache/ragu/generation/a.json")["text"]
    exported = store.export_answers(tmp_path / "results")
    assert (exported / "meno-lite-0.1/answers/full/user_1_q1.json").is_file()
    assert not (exported / "meno-lite-0.1/graphs").exists()
    assert not (tmp_path / "results/runs/cache").exists()


def test_failed_copy_does_not_publish_partial_checkpoint(tmp_path, monkeypatch):
    work = tmp_path / "work"
    user(work, "user_1", closed=True)
    store = ClosedUserCheckpoints(tmp_path / "durable", "fingerprint")
    store.commit(work, "runs/ragu-test", ["meno-lite-0.1"], "full")
    original = store.latest.read_bytes()
    user(work, "user_2", closed=True)
    import rumba_bench.checkpoint as module
    real_copy = module.shutil.copytree
    def fail(source, target, *args, **kwargs):
        if Path(source).name == "user_2": raise OSError("simulated node/disk failure")
        return real_copy(source, target, *args, **kwargs)
    monkeypatch.setattr(module.shutil, "copytree", fail)
    with pytest.raises(OSError):store.commit(work, "runs/ragu-test", ["meno-lite-0.1"], "full")
    assert store.latest.read_bytes() == original
    assert not (store.durable / "objects/meno-lite-0.1/user_2").exists()
    assert not list(store.durable.glob("objects/meno-lite-0.1/*.partial-*"))


def test_checkpoint_identity_and_bad_answers_are_rejected(tmp_path):
    folder, graph = user(tmp_path / "work", "user_1", closed=False)
    row = read_json(folder / "answers/full/user_1_q1.json")
    row["truncated"] = True
    write_json(folder / "answers/full/user_1_q1.json", row)
    with pytest.raises(ValueError):mark_user_closed(folder, "user_1", "full", [{"id": "user_1|q1"}])
    assert not (graph / READY).exists()
    store = ClosedUserCheckpoints(tmp_path / "durable", "original")
    store.commit(tmp_path / "work", "runs/ragu-test", ["meno-lite-0.1"], "full")
    with pytest.raises(ValueError):ClosedUserCheckpoints(store.durable, "different code or weights")
