#!/bin/sh
# Run from any directory. / Запуск из любой директории.
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_ROOT"
exec "$PROJECT_ROOT/.venv/bin/python" -m rumba_bench.cli "$@"
