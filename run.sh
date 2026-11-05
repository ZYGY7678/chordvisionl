#!/usr/bin/env bash
# Start ChordVision. Builds the frontend if it has not been built yet,
# then serves the API and the app from one process on port 8000.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "No .env found. Creating one from .env.example with a generated secret."
  cp .env.example .env
  secret="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
  python3 - "$secret" <<'PY'
import sys, pathlib
p = pathlib.Path(".env")
p.write_text(p.read_text().replace("CHORDVISION_SECRET_KEY=", f"CHORDVISION_SECRET_KEY={sys.argv[1]}"))
PY
fi

if [ ! -d frontend/dist ]; then
  echo "Building the frontend..."
  (cd frontend && npm install && npm run build)
fi

set -a; . ./.env; set +a
exec python3 -m uvicorn app.main:app \
  --app-dir backend --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
