#!/usr/bin/env bash
# Daily JobScout run. Point your scheduler at this file.
set -euo pipefail

# Resolve the project root (this script lives in jobscout/scripts/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Use the venv if present.
[ -d ".venv" ] && source .venv/bin/activate

# Load .env if present (for ANTHROPIC_API_KEY etc.).
[ -f ".env" ] && set -a && . ./.env && set +a

echo "=== JobScout run $(date '+%Y-%m-%d %H:%M:%S') ==="
python -m jobscout.cli run --tiered
