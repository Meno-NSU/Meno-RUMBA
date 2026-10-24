#!/usr/bin/env python3
"""Run smoke -> fixed pilot -> full RAGU against existing remote services.

All answers/raw calls remain on the client disk. A failed partial user graph is
archived before resuming; completed graphs and successful raw calls are reused.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rumba_bench.checkpoint import READY
from rumba_bench.data import digest, read_json, write_json
from rumba_bench.ragu_harness import manifest, validate_config


def archive_incomplete_users(folder):
    archived = []
    for graph in sorted((folder / "graphs").glob("*")):
        if not graph.is_dir() or (graph / READY).exists():
            continue
        user = graph.name
        target = folder / "interrupted" / (str(int(time.time())) + "-" + uuid.uuid4().hex[:8])
        target.mkdir(parents=True)
        graph.rename(target / user)
        for suite in ("pilot", "full"):
            for path in (folder / "answers" / suite).glob("*.json"):
                if read_json(path)["user_id"] != user:
                    continue
                destination = target / suite / "answers" / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.rename(destination)
                retrieval = folder / "retrieval" / suite / path.name
                if retrieval.exists():
                    destination = target / suite / "retrieval" / path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    retrieval.rename(destination)
        archived.append({"user": user, "archive": str(target)})
    return archived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    validate_config(config)
    if config.get("judge") is not None:
        raise ValueError("This workflow runs generation only; set judge=null")
    specification = manifest(config)
    folder = ROOT / "runs" / ("ragu-" + digest(specification)[:16]) / args.model
    workflow = folder / "workflow"
    workflow.mkdir(parents=True, exist_ok=True)
    lock = (workflow / "lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    archived = archive_incomplete_users(folder)
    stop = []
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda number, frame: stop.append(number))
    started = time.time()
    state = {"model": args.model, "config": str(args.config.resolve()), "run_folder": str(folder),
             "pid": os.getpid(), "started_at": started, "judge": False, "archived_incomplete_users": archived,
             "decoding": {k: config["protocol"][k] for k in ("seed", "temperature", "top_p", "repetition_penalty")}}
    def update(stage, status, **extra):
        if state.get("stage") != stage and "qc" in state:
            state.setdefault("completed_stage_qc", {})[state["stage"]] = state.pop("qc")
        state.update(stage=stage, status=status, updated_at=time.time(), elapsed_seconds=time.time()-started,
                     answers={suite: len(list((folder / "answers" / suite).glob("*.json"))) for suite in ("pilot", "full")},
                     cached_user_graphs=len(list((folder / "graphs").glob("*/build_report.json"))), **extra)
        write_json(args.state, state)
        write_json(workflow / "status.json", state)
    update("prepare", "running")
    phases = ["pilot", "full"] if (folder / "smoke_result.json").exists() else ["smoke", "pilot", "full"]
    for phase in phases:
        if stop:
            update(phase, "interrupted"); return 75
        if manifest(config) != specification:
            raise ValueError("Runtime changed between stages; refusing to mix experiments")
        command = [sys.executable, "-u", "-m", "rumba_bench.ragu_harness", "smoke" if phase == "smoke" else "run",
                   "--config", str(args.config.resolve()), "--model", args.model]
        if phase != "smoke":
            command += ["--suite", phase]
        log = workflow / (phase + ".log")
        with log.open("a") as output:
            output.write(f"\nWORKFLOW START {time.time()} {phase}\n"); output.flush()
            child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, pass_fds=(lock.fileno(),))
            update(phase, "running", child_pid=child.pid, log=str(log))
            next_stats = 0
            while child.poll() is None:
                if stop:
                    child.terminate()
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                    update(phase, "interrupted"); return 75
                update(phase, "running")
                if time.monotonic() >= next_stats:
                    write_json(workflow / "statistics" / (str(time.time_ns()) + ".json"), state)
                    print({"stage": phase, "answers": state["answers"], "cached_user_graphs": state["cached_user_graphs"]}, flush=True)
                    next_stats = time.monotonic() + 3600
                time.sleep(15)
        if child.returncode:
            update(phase, "failed", exit_code=child.returncode)
            print(log.read_text(errors="replace")[-5000:], flush=True)
            return child.returncode
        if phase == "smoke":
            row = read_json(folder / "smoke_result.json")
            if row.get("truncated") or not row.get("prediction"):
                raise ValueError("Smoke response is empty or truncated")
        else:
            qc = read_json(folder / phase / "qc.json")
            expected = 50 if phase == "pilot" else 1543
            if not (qc["expected_questions"] == qc["answered_questions"] == expected and qc["same_question_ids"]
                    and not qc["truncated_answers"] and qc["llm_extraction_errors"] == 0
                    and qc["all_graphs_consistent"] and qc["all_source_chunks_stored"] and qc["judge_pending"]):
                update(phase, "failed", qc=qc)
                raise ValueError(f"QC failed: {qc}")
            update(phase, "stage_complete", qc=qc)
            print({"stage": phase, "qc": "passed", "metrics": read_json(folder / phase / "report.json")["overall"]}, flush=True)
    update("full", "complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--state" in sys.argv:
            path = Path(sys.argv[sys.argv.index("--state") + 1])
            state = read_json(path) if path.exists() else {}
            state.update(status="failed", error=str(error), error_type=type(error).__name__, updated_at=time.time())
            write_json(path, state)
        raise
