#!/usr/bin/env bash
# Sourced by launchers; no environment dump or model credentials are logged.
launcher_log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

launcher_stage() {
  SERVER_LAUNCH_STAGE=$*
  launcher_log "STAGE: $SERVER_LAUNCH_STAGE"
}

launcher_command() {
  printf '[%s] COMMAND:' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf ' %q' "$@"
  printf '\n'
}

launcher_snapshot() {
  launcher_log "pid=$$ cwd=$PWD CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i 1 --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  fi
}

launcher_process_failed() {
  local pid=$1 file=$2 state_code child_code=0
  if python "$PROCESS_HELPER" alive --pid "$pid"; then
    launcher_log "ERROR: process $pid is still alive; inspect the preceding PID-recording error"
  else
    state_code=$?
    if [[ $state_code == 1 ]]; then
      wait "$pid" || child_code=$?
      launcher_log "ERROR: vLLM process $pid exit code=$child_code stage=${SERVER_LAUNCH_STAGE:-startup}"
    else
      launcher_log "ERROR: could not inspect process $pid (inspection exit=$state_code)"
    fi
  fi
  if [[ -f "$file" ]]; then
    launcher_log "Last 60 lines of $file:"
    tail -n 60 "$file" || true
  fi
}

launcher_error() {
  local code=$1 line=$2 command=$3
  if [[ ${SERVER_LAUNCH_ERROR_REPORTED:-0} == 1 ]]; then return; fi
  SERVER_LAUNCH_ERROR_REPORTED=1
  launcher_log "ERROR: exit=$code stage=${SERVER_LAUNCH_STAGE:-initialization} line=$line command=$command"
  launcher_snapshot
  local file
  for file in "${NATIVE_LOG:-}" "${YARN_LOG:-}" "${LOG_FILE:-}"; do
    if [[ -n "$file" && -f "$file" ]]; then
      launcher_log "Last 60 lines of $file:"
      tail -n 60 "$file" || true
    fi
  done
}

launcher_exit() {
  local code=$1 line=$2 command=$3
  if [[ $code != 0 ]]; then launcher_error "$code" "$line" "$command"; fi
  launcher_log "EXIT: code=$code stage=${SERVER_LAUNCH_STAGE:-initialization} log=$LAUNCH_LOG"
}

launcher_logging_init() {
  LAUNCH_LOG=$1
  export LAUNCH_LOG
  mkdir -p "$(dirname -- "$LAUNCH_LOG")"
  if [[ ${SERVER_LAUNCH_LOG_ACTIVE:-} != "$LAUNCH_LOG" ]]; then
    exec > >(tee -a "$LAUNCH_LOG") 2>&1
    export SERVER_LAUNCH_LOG_ACTIVE=$LAUNCH_LOG
  fi
  set -E
  trap 'launcher_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
  trap 'launcher_exit "$?" "$LINENO" "$BASH_COMMAND"' EXIT
  launcher_log "START: script=$0"
  launcher_snapshot
}
