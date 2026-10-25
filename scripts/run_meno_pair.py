#!/usr/bin/env python3
"""Two simultaneous Meno full runs, with cluster-owned logging/checkpoints."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rumba_bench.checkpoint import ClosedUserCheckpoints
from rumba_bench.data import digest, read_json, write_json
from rumba_bench.report import make_report, comparison

LABELS = ("meno-lite-0.1", "meno-lite-0.2")


def log(event, **fields):
    print(json.dumps({"time": time.time(), "event": event, **fields}, ensure_ascii=False), flush=True)


def command(args, cwd=None, env=None):
    return subprocess.check_output([str(x) for x in args], cwd=cwd, env=env, text=True).strip()


def allocated_devices(environment, job_description, visible_count):
    """Do not invent host GPU IDs unless the entire single node is exclusive."""
    if not environment.get("SLURM_JOB_ID"):
        raise ValueError("This workflow requires a Slurm allocation")
    mask = environment.get("CUDA_VISIBLE_DEVICES", "")
    if mask:
        ids = mask.split(",")
        if len(ids) != visible_count or any(not i or "MIG-" in i for i in ids):
            raise ValueError("Invalid scheduler GPU visibility")
    else:
        fields = dict(re.findall(r"(\w+)=([^\s]+)", job_description))
        exclusive = fields.get("Shared") in ("0", "NO") or fields.get("OverSubscribe") in ("NO", "EXCLUSIVE")
        if fields.get("NumNodes") != "1" or not exclusive:
            raise ValueError("GPU GRES/mask absent: an exclusive single-node allocation must be verified")
        ids = [str(i) for i in range(visible_count)]
    if len(ids) < 6:
        raise ValueError("Two native/long pairs plus embedding/reranker require at least six GPUs")
    return ids


def validate_pooling_weights(path):
    """Some official pooling checkpoints provide .bin rather than safetensors."""
    path = Path(path)
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (path / name).is_file() and (path / name).stat().st_size >= 1024:
            return [name]
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if not (path / name).is_file():
            continue
        shards = sorted(set(read_json(path / name)["weight_map"].values()))
        if shards and all((path / f).is_file() and (path / f).stat().st_size >= 1024 for f in shards):
            return shards
    raise ValueError(f"Missing/incomplete pooling weights; configs alone are not sufficient: {path}")


def model_plan(models, server, devices):
    common = ["--host", "127.0.0.1", "--dtype", "half", "--seed", "42",
              "--gpu-memory-utilization", "0.90", "--enable-prefix-caching", "--enable-chunked-prefill",
              "--structured-outputs-config", '{"backend":"xgrammar","disable_any_whitespace":true}',
              "--generation-config", "vllm", "--override-generation-config",
              '{"temperature":0,"top_p":1,"top_k":-1,"min_p":0,"repetition_penalty":1.03}']
    services = []
    identities = {}
    for offset, label in enumerate(LABELS):
        path = Path(models) / label
        config = read_json(path / "config.json")
        rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
        if (config.get("model_type") != "qwen2" or config.get("max_position_embeddings") != 32768
                or rope.get("rope_type", rope.get("type", "default")) != "default"):
            raise ValueError(f"Expected native qwen2/32768 checkpoint: {path}")
        theta = rope.get("rope_theta", config.get("rope_theta"))
        if theta is None:
            raise ValueError("Missing checkpoint rope_theta")
        index = read_json(path / "model.safetensors.index.json")
        shards = sorted(set(index["weight_map"].values()))
        if not (path / "tokenizer.json").is_file() or not (path / "tokenizer_config.json").is_file():
            raise ValueError(f"Missing tokenizer: {path}")
        sizes = {}
        for name in shards:
            p = path / name
            if not p.is_file() or p.stat().st_size < 1024:
                raise ValueError(f"Missing/truncated checkpoint shard: {p}")
            sizes[name] = {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        identities[label] = {"path": str(path.resolve()), "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
                             "index_sha256": hashlib.sha256((path / "model.safetensors.index.json").read_bytes()).hexdigest(), "shards": sizes}
        for long in (False, True):
            port = 9022 + offset * 10 + (3 if long else 0)
            seqs, batch = (4, 8192) if long else (16, 16384)
            args = [str(server), "serve", str(path), "--served-model-name", label, *common,
                    "--port", str(port), "--max-model-len", "65536" if long else "32768",
                    "--max-num-seqs", str(seqs), "--max-cudagraph-capture-size", str(seqs),
                    "--max-num-batched-tokens", str(batch)]
            if long:
                args += ["--hf-overrides", json.dumps({"rope_parameters": {"rope_type": "yarn", "factor": 2.0,
                          "original_max_position_embeddings": 32768, "rope_theta": theta}})]
            services.append({"tag": label + ("-long" if long else "-native"), "model": label,
                             "port": port, "gpu": devices[offset * 2 + int(long)], "command": args})
    for tag, alias, port, gpu in (("embedder", "bge-m3", 9023, devices[4]),
                                   ("reranker", "bge-reranker-v2-m3", 9024, devices[5])):
        path = Path(models) / alias
        if not (path / "config.json").is_file():
            raise ValueError(f"Missing local pooling model: {path}")
        validate_pooling_weights(path)
        args = [str(server), "serve", str(path), "--served-model-name", alias, "--runner", "pooling",
                "--host", "127.0.0.1", "--port", str(port), "--dtype", "half", "--max-model-len", "8192",
                "--max-num-seqs", "8", "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", "0.90"]
        if tag == "embedder":
            args += ["--pooler-config", '{"task":"embed","pooling_type":"CLS","use_activation":true}']
        services.append({"tag": tag, "model": alias, "port": port, "gpu": gpu, "command": args})
    return services, identities


def stage_repository(project, target, revision):
    if target.exists():
        raise ValueError("Use a fresh node-local working directory")
    command(["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout", project, target])
    command(["git", "checkout", "--quiet", "--detach", revision], cwd=target)
    submodule = target / "third_party/RAGU"
    command(["git", "clone", "--quiet", "--no-hardlinks", project / "third_party/RAGU", submodule])
    pinned = command(["git", "rev-parse", "HEAD"], cwd=project / "third_party/RAGU")
    command(["git", "checkout", "--quiet", "--detach", pinned], cwd=submodule)
    shutil.copytree(project / "data", target / "data")
    shutil.copytree(project / ".cache/fastembed", target / ".cache/fastembed")


def metrics(work, run_relative, children):
    result = {}
    for label in LABELS:
        folder = work / run_relative / label
        rows = [read_json(p) for p in (folder / "answers/full").glob("*.json")]
        result[label] = {"answers": len(rows), "expected": 1543,
                         "closed_users": len(list((folder / "graphs").glob("*/.checkpoint-ready.json"))),
                         "truncated": sum(bool(r.get("truncated")) for r in rows),
                         "worker_exit_code": children[label].poll() if label in children else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--vllm-env", type=Path, required=True)
    parser.add_argument("--local-scratch", type=Path, required=True)
    parser.add_argument("--checkpoint-seconds", type=int, default=3600)
    parser.add_argument("--statistics-seconds", type=int, default=3600)
    args = parser.parse_args()
    if min(args.checkpoint_seconds, args.statistics_seconds) < 30:
        parser.error("Checkpoint/statistics intervals must be at least 30 seconds")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Submit through Slurm; do not run on a login node")
    server = args.vllm_env.resolve() / "bin/vllm"
    server_python = args.vllm_env.resolve() / "bin/python"
    gpu_info = json.loads(command([server_python, "-c", "import torch,json;print(json.dumps([{'name':torch.cuda.get_device_properties(i).name,'bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]))"]))
    if any(g["bytes"] < 72 * 2**30 or not any(n in g["name"] for n in ("A100", "A800")) for g in gpu_info):
        raise ValueError("This profile requires full A100/A800 80 GB GPUs")
    job = os.environ["SLURM_JOB_ID"]
    allocation = command(["scontrol", "show", "job", "-o", job])
    devices = allocated_devices(os.environ, allocation, len(gpu_info))
    services, weights = model_plan(args.models_dir.resolve(), server, devices)
    revision = command(["git", "rev-parse", "HEAD"], cwd=ROOT)
    requested_revision = os.environ.get("MENO_PAIR_REVISION")
    if requested_revision and requested_revision != revision:
        raise ValueError("Project changed since submission; restore the pinned MENO_PAIR_REVISION before resuming")
    config = read_json(ROOT / "config.ragu.example.json")
    config["models"] = {}; config["long_context_models"] = {}
    for offset, label in enumerate(LABELS):
        config["models"][label] = {"model": label, "base_url": f"http://127.0.0.1:{9022 + 10 * offset}/v1",
                                  "weights": "operator-provided checkpoint; see weights.json"}
        config["long_context_models"][label] = {**config["models"][label], "base_url": f"http://127.0.0.1:{9025 + 10 * offset}/v1",
                                               "max_model_len": 65536, "rope_profile": "yarn-2x-preserve-checkpoint-rope-theta"}
    config["embedding"]["base_url"] = "http://127.0.0.1:9023/v1"
    config["reranker"]["base_url"] = "http://127.0.0.1:9024/v1"
    config["runtime"]["concurrency"] = 8
    config["judge"] = None
    identity = {"version": 1, "revision": revision, "config": config, "weights": weights}
    fingerprint = digest(identity)[:20]
    durable = ROOT / "runs/a800-meno-pair" / fingerprint
    durable.mkdir(parents=True, exist_ok=True)
    lock = (durable / "workflow.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    job_dir = durable / "jobs" / (job + "-" + os.environ.get("SLURM_RESTART_COUNT", "0"))
    job_dir.mkdir(parents=True, exist_ok=True)
    write_json(durable / "identity.json", identity)
    write_json(job_dir / "allocation.json", {"description": allocation, "gpu_inventory": gpu_info, "devices": devices})
    write_json(job_dir / "services.json", services)
    args.local_scratch.mkdir(parents=True, exist_ok=True)
    filesystem = command(["stat", "-f", "-c", "%T", args.local_scratch]).lower()
    if any(kind in filesystem for kind in ("nfs", "cifs", "lustre", "ceph", "smb")):
        raise ValueError("Live Qdrant/SQLite stores need node-local scratch, not a network filesystem")
    if shutil.disk_usage(args.local_scratch).free < 15 * 2**30:
        raise ValueError("At least 15 GiB of free node-local scratch is required")
    if shutil.disk_usage(durable).free < 12 * 2**30:
        raise ValueError("At least 12 GiB free is required for durable checkpoints")
    work = args.local_scratch / "work"
    stage_repository(ROOT, work, revision)
    environment = {**os.environ, "PYTHONPATH": str(work / "third_party/RAGU") + os.pathsep + str(work)}
    config_path = work / "config.pair.json"
    write_json(config_path, config)
    manifest_code = "from pathlib import Path;from rumba_bench.data import digest,read_json;from rumba_bench.ragu_harness import manifest;print('runs/ragu-'+digest(manifest(read_json(Path('config.pair.json'))))[:16])"
    run_relative = Path(command([sys.executable, "-c", manifest_code], cwd=work, env=environment).splitlines()[-1])
    checkpoints = ClosedUserCheckpoints(durable / "checkpoints", fingerprint)
    log("restore", **checkpoints.restore(work, run_relative))
    write_json(job_dir / "paths.json", {"work": str(work), "run_relative": str(run_relative), "durable": str(durable)})
    write_json(ROOT / "runs/a800-meno-pair/current.json", {"job_id": job, "job_dir": str(job_dir), "durable": str(durable),
                                                         "fingerprint": fingerprint, "models": list(LABELS), "suite": "full", "judge": False})
    spec = importlib.util.spec_from_file_location("server_process", ROOT / "scripts/server_process.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    running = []; children = {}; files = []; stopping = []; failures = []
    def stop_signal(number, frame):
        stopping.append(number)
        log("signal", number=number, action="checkpoint and stop; resubmit the same script to resume")
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(number, stop_signal)
    def status(phase, **extra):
        value = {"time": time.time(), "job_id": job, "hostname": socket.gethostname(), "phase": phase,
                 "models": metrics(work, run_relative, children), "failures": failures, **extra}
        write_json(durable / "status.json", value)
        return value
    def checkpoint(final=False):
        status("checkpointing")
        value = checkpoints.commit(work, run_relative, LABELS, "full", final=final)
        write_json(job_dir / "checkpoint-last.json", {"time": time.time(), **value})
        log("checkpoint", **value)
        return value
    complete = False
    try:
        for service in services:
            if stopping: raise InterruptedError("Interrupted during startup")
            helper.require_free_port(service["port"])
            output = (job_dir / (service["tag"] + ".log")).open("a"); files.append(output)
            env = {**environment, "CUDA_VISIBLE_DEVICES": service["gpu"]}
            # Build caches for independent vLLM executors must not overwrite one
            # another; retain every required cache variable from the sbatch file.
            for key in ("OUTLINES_CACHE_DIR", "TRITON_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE"):
                env[key] = str(Path(environment[key]) / service["tag"])
            log("service_start", **service)
            proc = subprocess.Popen(service["command"], env=env, stdout=output, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, cwd=work)
            running.append((service, proc))
            helper.record(job_dir / (service["tag"] + ".pid"), proc.pid, service["port"], service["model"])
            for _ in range(600):
                if stopping: raise InterruptedError("Interrupted during service readiness")
                if proc.poll() is not None: raise RuntimeError(f"Service {service['tag']} exited: {proc.returncode}")
                try:
                    with urlopen(f"http://127.0.0.1:{service['port']}/v1/models", timeout=2) as r:
                        names = [x["id"] for x in json.load(r)["data"]]
                    if service["model"] in names: break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            else: raise TimeoutError(f"Service {service['tag']} readiness timeout")
        for label in LABELS:
            output = (job_dir / (label + "-benchmark.log")).open("a"); files.append(output)
            args_run = [sys.executable, "-u", "-m", "rumba_bench.ragu_harness", "run", "--config", str(config_path),
                        "--model", label, "--suite", "full"]
            log("benchmark_start", model=label, command=args_run)
            children[label] = subprocess.Popen(args_run, cwd=work, env=environment, stdout=output, stderr=subprocess.STDOUT)
        next_stats = next_checkpoint = next_heartbeat = 0
        finished = set()
        while True:
            now = time.monotonic()
            for service, proc in running:
                if proc.poll() is not None: raise RuntimeError(f"Service {service['tag']} exited: {proc.returncode}")
            for label, proc in children.items():
                if proc.poll() is not None and label not in finished:
                    finished.add(label)
                    if proc.returncode:
                        failures.append({"model": label, "exit_code": proc.returncode})
                    else:
                        qc = read_json(work / run_relative / label / "full/qc.json")
                        valid = (qc["expected_questions"] == qc["answered_questions"] == 1543 and qc["same_question_ids"]
                                 and not qc["truncated_answers"] and qc["llm_extraction_errors"] == 0
                                 and qc["all_graphs_consistent"] and qc["all_source_chunks_stored"] and qc["judge_pending"])
                        if not valid: failures.append({"model": label, "error": "final QC failed", "qc": qc})
                    log("benchmark_exit", model=label, exit_code=proc.returncode)
                    checkpoint()
            if now >= next_heartbeat:
                status("running"); next_heartbeat = now + 60
            if now >= next_stats:
                value = status("running")
                value["gpu_csv"] = command(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw", "--format=csv,noheader"])
                write_json(durable / "statistics" / (str(time.time_ns()) + ".json"), value)
                log("hourly_statistics", models=value["models"])
                next_stats = now + args.statistics_seconds
            if now >= next_checkpoint:
                checkpoint(); next_checkpoint = now + args.checkpoint_seconds
            if stopping or len(finished) == len(LABELS): break
            time.sleep(5)
        complete = not stopping and not failures and len(finished) == len(LABELS)
    except Exception as error:
        failures.append({"error": str(error), "type": type(error).__name__})
        log("workflow_error", error=str(error), type=type(error).__name__)
    finally:
        for proc in children.values():
            if proc.poll() is None: proc.terminate()
        for proc in children.values():
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        try:
            checkpoint(final=complete)
            # Reports can be reconstructed entirely from committed answer files.
            export = checkpoints.export_answers(durable / "results")
            for label in LABELS: make_report(export / label, "full")
            comparison(export, "full", LABELS)
        except Exception as error:
            complete = False; failures.append({"checkpoint_error": str(error)}); log("checkpoint_error", error=str(error))
        for service, proc in reversed(running):
            try: helper.stop(job_dir / (service["tag"] + ".pid"), service["port"], service["model"])
            except Exception as error: log("service_stop_error", service=service["tag"], error=str(error))
        status("complete" if complete else "interrupted" if stopping else "failed")
        for handle in files: handle.close()
    return 0 if complete else 75 if stopping else 1


if __name__ == "__main__":
    raise SystemExit(main())
