import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("ragu")
from rumba_bench.data import read_json, write_json

spec = importlib.util.spec_from_file_location("remote_workflow", Path(__file__).resolve().parents[1] / "scripts/run_remote_ragu_full.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def test_resume_archives_only_incomplete_user_and_preserves_raw_cache(tmp_path):
    folder = tmp_path / "model"
    closed = folder / "graphs/user_1"
    active = folder / "graphs/user_2"
    closed.mkdir(parents=True); active.mkdir(parents=True)
    (closed / ".checkpoint-ready.json").write_text("{}")
    (active / "graph.json").write_text('{"partial":true}')
    for suite in ("pilot", "full"):
        for user in ("user_1", "user_2"):
            write_json(folder / "answers" / suite / f"{user}_q1.json", {"user_id": user, "prediction": user})
            write_json(folder / "retrieval" / suite / f"{user}_q1.json", {"source": user})
    raw = tmp_path / "cache/generation/request.json"
    write_json(raw, {"text": "successful expensive extraction"})
    result = workflow.archive_incomplete_users(folder)
    assert len(result) == 1 and result[0]["user"] == "user_2"
    archive = Path(result[0]["archive"])
    assert closed.is_dir() and not active.exists()
    assert (archive / "user_2/graph.json").exists()
    for suite in ("pilot", "full"):
        assert (folder / "answers" / suite / "user_1_q1.json").exists()
        assert not (folder / "answers" / suite / "user_2_q1.json").exists()
        assert read_json(archive / suite / "answers/user_2_q1.json")["prediction"] == "user_2"
        assert read_json(archive / suite / "retrieval/user_2_q1.json")["source"] == "user_2"
    assert read_json(raw)["text"] == "successful expensive extraction"
