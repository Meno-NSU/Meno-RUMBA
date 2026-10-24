#!/usr/bin/env bash
# One-node A100 workflow; only child services of this invocation are stopped.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PHASE=${1:?Usage: cluster_a100.sh inference MODEL | judge RUN_ROOT MODEL [MODEL ...]}
shift
LAYOUT=${A100_LAYOUT:-single80}
SUITE=${BENCH_SUITE:-full}
case "$PHASE" in inference|judge) ;; *) echo "Unknown phase: $PHASE" >&2; exit 1;; esac
case "$LAYOUT" in single80|dual40) ;; *) echo "A100_LAYOUT must be single80 or dual40" >&2; exit 1;; esac
case "$SUITE" in pilot|full) ;; *) echo "BENCH_SUITE must be pilot or full" >&2; exit 1;; esac
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  : "${CUDA_VISIBLE_DEVICES:?Slurm must expose the allocated GPUs}"
else
  export CUDA_VISIBLE_DEVICES=1
  [[ $LAYOUT == single80 ]] || { echo "dual40 requires a two-GPU Slurm allocation; manual launches use GPU 1" >&2; exit 1; }
fi
[[ $CUDA_VISIBLE_DEVICES != *MIG-* ]] || { echo "This profile requires full GPUs, not MIG instances" >&2; exit 1; }
CLIENT=$ROOT/.venv-ragu/bin/python
SERVER_ENV=${CLUSTER_VLLM_ENV:-$ROOT/.venv-vllm}
SERVER=$SERVER_ENV/bin/vllm
MODELS=${MODELS_DIR:-$ROOT/../models}
PROCESS=$ROOT/scripts/server_process.py
JOB_DIR=$ROOT/runs/cluster-jobs/${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%S)-$$}
mkdir -p "$JOB_DIR"
exec > >(tee -a "$JOB_DIR/workflow.log") 2>&1
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
log "phase=$PHASE suite=$SUITE layout=$LAYOUT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
[[ -x $CLIENT && -x $SERVER && -x $SERVER_ENV/bin/python ]] || { echo "Install both Python environments first" >&2; exit 1; }
[[ -f data/manifest.json ]] || { echo "Prepare RUMBA on a network-enabled node first" >&2; exit 1; }

# Inspect only the scheduler's visible devices (or manual GPU 1), not host IDs.
"$SERVER_ENV/bin/python" - "$LAYOUT" <<'PY'
import sys, torch
layout = sys.argv[1]
count = 1 if layout == "single80" else 2
minimum_gib = 72 if layout == "single80" else 36
if torch.cuda.device_count() != count:
    raise SystemExit(f"Expected {count} visible GPU(s) for {layout}")
for i in range(count):
    p = torch.cuda.get_device_properties(i)
    print(f"Visible GPU {i}: {p.name}; {p.total_memory / 2**30:.1f} GiB", flush=True)
    if "A100" not in p.name or p.total_memory < minimum_gib * 2**30:
        raise SystemExit("This profile requires full A100 devices of the selected memory size")
PY
IFS=, read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
NATIVE_GPU=${gpu_ids[0]}
LONG_GPU=$NATIVE_GPU
NATIVE_MEMORY=0.40; LONG_MEMORY=0.45; EMBED_MEMORY=0.04; RERANK_MEMORY=0.06
JUDGE_TP=1
if [[ $LAYOUT == dual40 ]]; then
  [[ ${#gpu_ids[@]} == 2 ]] || { echo "Expected two allocated CUDA device IDs or UUIDs" >&2; exit 1; }
  LONG_GPU=${gpu_ids[1]}
  NATIVE_MEMORY=0.78; LONG_MEMORY=0.78; EMBED_MEMORY=0.08; RERANK_MEMORY=0.12
  JUDGE_TP=2
fi
CONFIG=$JOB_DIR/config.json
"$CLIENT" - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
c = json.loads(Path("config.ragu.example.json").read_text())
for label, name, weights in (("meno-lite-0.1", "meno-lite-0.1", "bond005/meno-lite-0.1"),
                             ("meno-lite-0.2", "meno-lite-0.2", "bond005/meno-lite-0.2"),
                             ("qwen3.5-9b", "Qwen3.5-9B", "Qwen/Qwen3.5-9B")):
    c["models"][label] = {"base_url": "http://127.0.0.1:9022/v1", "model": name, "weights": weights}
    c["long_context_models"][label] = {**c["models"][label], "base_url": "http://127.0.0.1:9025/v1",
        "rope_profile": "native-65536-no-scaling" if label == "qwen3.5-9b" else "yarn-2x-preserve-checkpoint-rope-theta", "max_model_len": 65536}
    if label == "qwen3.5-9b":
        for section in ("models", "long_context_models"):
            c[section][label]["chat_template_kwargs"] = {"enable_thinking": False}
c["embedding"]["base_url"] = "http://127.0.0.1:9023/v1"
c["reranker"]["base_url"] = "http://127.0.0.1:9024/v1"
c["runtime"]["concurrency"] = 4
Path(sys.argv[1]).write_text(json.dumps(c, ensure_ascii=False, indent=2) + "\n")
PY

pid_files=(); ports=(); aliases=()
cleanup() {
  local code=$? i
  trap - EXIT INT TERM
  log "cleanup: stop only services recorded in $JOB_DIR; exit=$code"
  if [[ ${#pid_files[@]} -gt 0 ]]; then
    for ((i=${#pid_files[@]}-1; i>=0; i--)); do
      "$CLIENT" "$PROCESS" stop --pid-file "${pid_files[$i]}" --port "${ports[$i]}" --model "${aliases[$i]}" || true
    done
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
start() {
  local tag=$1 mask=$2 port=$3 alias=$4 pid file
  shift 4
  file=$JOB_DIR/$tag.pid
  "$CLIENT" "$PROCESS" free --port "$port"
  log "start $tag: GPU mask=$mask port=$port log=$JOB_DIR/$tag.log"
  printf 'COMMAND:'; printf ' %q' env "CUDA_VISIBLE_DEVICES=$mask" "$SERVER" "$@"; printf '\n'
  env CUDA_VISIBLE_DEVICES="$mask" "$SERVER" "$@" > "$JOB_DIR/$tag.log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$file"
  pid_files+=("$file"); ports+=("$port"); aliases+=("$alias")
  "$CLIENT" "$PROCESS" record --pid "$pid" --pid-file "$file" --port "$port" --model "$alias"
  for _ in {1..300}; do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then return; fi
    if ! "$CLIENT" "$PROCESS" alive --pid "$pid"; then
      tail -n 80 "$JOB_DIR/$tag.log"; return 1
    fi
    sleep 2
  done
  tail -n 80 "$JOB_DIR/$tag.log"; echo "$tag did not become ready" >&2; return 1
}

if [[ $PHASE == inference ]]; then
  [[ $# == 1 ]] || { echo "Pass exactly one candidate model label" >&2; exit 1; }
  LABEL=$1
  case "$LABEL" in meno-lite-0.1|meno-lite-0.2) NAME=$LABEL;; qwen3.5-9b) NAME=Qwen3.5-9B;; *) echo "Unsupported candidate: $LABEL" >&2; exit 1;; esac
  MODEL=$MODELS/$NAME
  # Use exactly the same checkpoint/RoPE validation as the established profile.
  "$SERVER_ENV/bin/python" - "$MODEL" "$JOB_DIR/rope.json" <<'PY'
import json, sys
from transformers import AutoConfig
c = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)
c = c.get_text_config() if hasattr(c, "get_text_config") else c
r = getattr(c, "rope_parameters", None) or getattr(c, "rope_scaling", None) or {}
if r.get("rope_type", r.get("type", "default")) != "default":
    raise SystemExit("Restore the checkpoint's native RoPE config before benchmarking")
if c.model_type == "qwen3_5_text" and c.max_position_embeddings >= 65536:
    result = {}
elif c.model_type == "qwen2" and c.max_position_embeddings == 32768:
    theta = r.get("rope_theta", getattr(c, "rope_theta", None))
    if theta is None: raise SystemExit("Missing rope_theta")
    result = {"rope_parameters": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32768, "rope_theta": theta}}
else:
    raise SystemExit("Unsupported native context/model architecture")
open(sys.argv[2], "w").write(json.dumps(result))
PY
  common=(--served-model-name "$NAME" --host 127.0.0.1 --dtype half --seed 42
    --max-num-seqs 4 --max-num-batched-tokens 8192 --max-cudagraph-capture-size 4
    --enable-prefix-caching --enable-chunked-prefill
    --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
    --generation-config vllm --override-generation-config '{"temperature":0,"top_p":1,"top_k":-1,"min_p":0,"repetition_penalty":1.03}')
  if [[ $NAME == Qwen3.5-9B ]]; then
    common+=(--language-model-only --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking":false}')
  fi
  long_args=("${common[@]}")
  if [[ $(cat "$JOB_DIR/rope.json") != '{}' ]]; then long_args+=(--hf-overrides "$(cat "$JOB_DIR/rope.json")"); fi
  start embedder "$NATIVE_GPU" 9023 bge-m3 serve "$MODELS/bge-m3" --served-model-name bge-m3 --runner pooling \
    --pooler-config '{"task":"embed","pooling_type":"CLS","use_activation":true}' \
    --host 127.0.0.1 --port 9023 --dtype half --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 8192 --gpu-memory-utilization "$EMBED_MEMORY"
  start reranker "$LONG_GPU" 9024 bge-reranker-v2-m3 serve "$MODELS/bge-reranker-v2-m3" --served-model-name bge-reranker-v2-m3 \
    --runner pooling --host 127.0.0.1 --port 9024 --dtype half --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 8192 --gpu-memory-utilization "$RERANK_MEMORY"
  start native "$NATIVE_GPU" 9022 "$NAME" serve "$MODEL" "${common[@]}" --port 9022 --max-model-len 32768 --gpu-memory-utilization "$NATIVE_MEMORY"
  start long "$LONG_GPU" 9025 "$NAME" serve "$MODEL" "${long_args[@]}" --port 9025 --max-model-len 65536 --gpu-memory-utilization "$LONG_MEMORY"
  "$CLIENT" - "$CONFIG" "$LABEL" "$JOB_DIR/run-folder.txt" <<'PY'
import sys
from pathlib import Path
from rumba_bench.data import ROOT, digest, read_json
from rumba_bench.ragu_harness import manifest
folder = ROOT / "runs" / ("ragu-" + digest(manifest(read_json(Path(sys.argv[1]))))[:16]) / sys.argv[2]
Path(sys.argv[3]).write_text(str(folder) + "\n")
print(f"RUN_FOLDER={folder}", flush=True)
PY
  "$CLIENT" -u -m rumba_bench.ragu_harness preflight --config "$CONFIG" --model "$LABEL"
  "$CLIENT" -u -m rumba_bench.ragu_harness run --config "$CONFIG" --model "$LABEL" --suite "$SUITE"
  "$CLIENT" - "$(cat "$JOB_DIR/run-folder.txt")" "$SUITE" <<'PY'
import sys
from pathlib import Path
from rumba_bench.data import read_json
q = read_json(Path(sys.argv[1]) / sys.argv[2] / "qc.json")
expected = 1543 if sys.argv[2] == "full" else 50
if not (q["expected_questions"] == q["answered_questions"] == expected and q["same_question_ids"]
        and not q["truncated_answers"] and q["llm_extraction_errors"] == 0
        and q["all_graphs_consistent"] and q["all_source_chunks_stored"] and q["judge_pending"]):
    raise SystemExit(f"Inference QC failed; inspect saved artifacts: {q}")
print(f"INFERENCE QC PASSED: {expected} answers, judge_pending", flush=True)
PY
else
  RUN_ROOT=${1:?Pass the saved runs/ragu-HASH directory, followed by model labels}
  shift
  [[ $# -gt 0 ]] || { echo "Pass explicit model labels to judge" >&2; exit 1; }
  "$CLIENT" - "$CONFIG" "$RUN_ROOT" "$SUITE" "$@" <<'PY'
import sys
from pathlib import Path
from rumba_bench.data import read_json, write_json
from rumba_bench.cli import selected_questions
config = read_json(Path(sys.argv[1])); root = Path(sys.argv[2]); suite = sys.argv[3]
if not (root / "manifest.json").exists(): raise SystemExit("Missing saved RAGU manifest")
expected = {r["id"] for r in selected_questions(suite)}
for label in sys.argv[4:]:
    if label not in config["models"]: raise SystemExit(f"Unknown model label: {label}")
    paths = list((root / label / "answers" / suite).glob("*.json"))
    if len(paths) != len(expected) or {read_json(p)["id"] for p in paths} != expected:
        raise SystemExit(f"Incomplete saved answers for {label}")
config["judge"] = {"base_url": "http://127.0.0.1:9026/v1", "model": "pollux-judge-32b", "source_model": "ai-forever/pollux-judge-32b"}
write_json(Path(sys.argv[1]), config)
PY
  "$SERVER_ENV/bin/python" - "$MODELS/pollux-judge-32b" <<'PY'
import json, sys
from pathlib import Path
c = json.loads((Path(sys.argv[1]) / "config.json").read_text())
r = c.get("rope_parameters") or c.get("rope_scaling") or {}
if not (c.get("architectures") == ["Qwen2ForCausalLM"] and c.get("model_type") == "qwen2"
        and c.get("max_position_embeddings", 0) >= 8192
        and r.get("rope_type", r.get("type", "default")) == "default"):
    raise SystemExit("Expected native generative Pollux checkpoint, not a regression variant")
PY
  start judge "$CUDA_VISIBLE_DEVICES" 9026 pollux-judge-32b serve "$MODELS/pollux-judge-32b" \
    --served-model-name pollux-judge-32b --host 127.0.0.1 --port 9026 --dtype bfloat16 --seed 42 \
    --tensor-parallel-size "$JUDGE_TP" --max-model-len 8192 --gpu-memory-utilization 0.92 \
    --max-num-seqs 4 --max-num-batched-tokens 8192 --max-cudagraph-capture-size 4 \
    --enable-prefix-caching --enable-chunked-prefill --generation-config vllm \
    --override-generation-config '{"temperature":0,"top_p":1,"top_k":-1,"min_p":0,"repetition_penalty":1}'
  "$CLIENT" - "$CONFIG" "$RUN_ROOT" "$SUITE" "$JOB_DIR/judge-smoke.json" "$@" <<'PY'
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rumba_bench.data import category, read_json, timestamp, write_json
from rumba_bench.pipeline import Generation
from rumba_bench.upstream import load_official
c = read_json(Path(sys.argv[1])); root = Path(sys.argv[2]); suite = sys.argv[3]
official = load_official(); jobs = []
for label in sys.argv[5:]:
    paths = sorted((root / label / "answers" / suite).glob("*.json"))
    groups = {}
    for path in paths:
        _, scope, temporal = category(read_json(path))
        groups.setdefault((temporal, scope), path)
    selected = set(groups.values())
    for path in paths:
        if len(selected) >= 5: break
        selected.add(path)
    jobs.extend((label, path) for path in sorted(selected))
def check(job):
    label, path = job; row = read_json(path)
    g = Generation(c["judge"], c["runtime"], root / label / "judge-requests")
    try:
        question = f"{row['question']} (текущая дата: {timestamp(row['question_date']).date()})"
        raw = g.call(official["judge_template"](question, row["prediction"], row["answer"]), c["protocol"]["judge_max_tokens"])
        reason, verdict = official["extract_judge_feedback"](raw["text"])
        if not reason or verdict not in (0, 1, 2, 3): raise ValueError(f"Invalid Pollux verdict for {label}/{row['id']}")
        return {"model": label, "id": row["id"], "verdict": verdict}
    finally:
        g.client.close()
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(check, jobs))
write_json(Path(sys.argv[4]), {"status": "passed", "n": len(results), "results": results})
print(f"JUDGE SMOKE PASSED: {len(results)} saved answers", flush=True)
PY
  for LABEL in "$@"; do
    "$CLIENT" -u -m rumba_bench.ragu_harness score --config "$CONFIG" --model "$LABEL" --suite "$SUITE" --run-folder "$RUN_ROOT/$LABEL"
    "$CLIENT" - "$RUN_ROOT/$LABEL" "$SUITE" <<'PY'
import sys
from pathlib import Path
from rumba_bench.data import read_json, write_json
folder = Path(sys.argv[1]); suite = sys.argv[2]
r = read_json(folder / suite / "report.json")
raw = [read_json(p) for p in (folder / "judge-requests").glob("*.json")]
if not (r["status"] == "scored" and r["overall"]["judged"] == r["overall"]["n"]
        and r["judge_parse_errors"] == 0 and all(x["finish_reason"] == "stop" for x in raw)):
    raise SystemExit(f"Judge QC failed: {r}")
qc_path = folder / suite / "qc.json"
if qc_path.exists():
    qc = read_json(qc_path); qc.update(judge_pending=False, judged_questions=r["overall"]["judged"], judge_parse_errors=0)
    write_json(qc_path, qc)
print(f"JUDGE QC PASSED: {r['overall']}", flush=True)
PY
  done
fi
log "COMPLETE; artifacts=$JOB_DIR"
