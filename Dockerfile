# syntax=docker/dockerfile:1
#
# ChordVision for Render (or any Docker host).
#
# Two stages: Node builds the React frontend, then a slim Python image runs
# the API and serves the built frontend from the same process on one port.
# ffmpeg is installed because every upload is decoded with it. The active
# recognizer also uses librosa/Numba for harmonic separation, CQT and NNLS.

# ---- stage 1: build the frontend ------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- stage 2: the runtime image -------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes do not reinstall them. constraints.txt
# pins the exact versions the tests were run against.
COPY backend/requirements.txt backend/constraints.txt backend/
RUN pip install -r backend/requirements.txt -c backend/constraints.txt

COPY backend/ backend/
COPY engine/ engine/
COPY --from=frontend /build/frontend/dist frontend/dist

# Defaults for production. render.yaml sets the same values explicitly; they
# are repeated here so the image also behaves if it is deployed by hand.
# /data is a plain folder in the container. On Render's free plan it is
# wiped on every sleep, restart and deploy; on a paid plan, mount a disk here.
ENV CHORDVISION_ENVIRONMENT=production \
    CHORDVISION_STORAGE_DIR=/data/storage \
    CHORDVISION_DATABASE_URL=sqlite:////data/chordvision.db

EXPOSE 10000

# One uvicorn process on purpose: the job queue and the rate limiter live in
# the process, so a second worker would not see the first one's jobs.
# Render provides $PORT (10000 by default).
CMD ["sh", "-c", "mkdir -p \"$CHORDVISION_STORAGE_DIR\" && exec python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port \"${PORT:-10000}\" --proxy-headers --forwarded-allow-ips='*' --timeout-keep-alive 75"]
