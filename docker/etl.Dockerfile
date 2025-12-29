# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && pip install --no-cache-dir uv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY uv.lock* ./
RUN uv sync --frozen --no-dev || uv sync --no-dev
ENV PATH="/app/.venv/bin:${PATH}"

COPY . .

ENTRYPOINT ["uv", "run", "python", "-m", "pipeline.etl"]
