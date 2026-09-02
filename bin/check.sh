#!/usr/bin/env bash
# Run lint, type checks and tests for all three sub-projects.
#
#   bin/check.sh          # everything
#   bin/check.sh lint     # ruff only
#   bin/check.sh types    # mypy only
#   bin/check.sh test     # pytest only
#
# Each sub-project has its own uv environment; `uv sync` them first
# (see README.md for the CTG_ML torch setup).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECTS=("$ROOT" "$ROOT/CTG_preprocess" "$ROOT/CTG_ML")
WHAT="${1:-all}"

status=0

run_lint() {
    echo "==> ruff check"
    (cd "$ROOT" && uvx ruff check .) || status=1
    echo "==> ruff format --check"
    (cd "$ROOT" && uvx ruff format --check .) || status=1
}

run_types() {
    for project in "${PROJECTS[@]}"; do
        echo "==> mypy: $(basename "$project")"
        (cd "$project" && uv run --no-sync mypy) || status=1
    done
}

run_tests() {
    for project in "${PROJECTS[@]}"; do
        echo "==> pytest: $(basename "$project")"
        (cd "$project" && uv run --no-sync pytest -q) || status=1
    done
}

case "$WHAT" in
    all)   run_lint; run_types; run_tests ;;
    lint)  run_lint ;;
    types) run_types ;;
    test)  run_tests ;;
    *)     echo "usage: $0 [all|lint|types|test]" >&2; exit 2 ;;
esac

if [ "$status" -ne 0 ]; then
    echo "FAILED" >&2
fi
exit "$status"
