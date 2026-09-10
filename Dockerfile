# Optional self-contained runtime for the public reproduction workflow.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--no-sync", "causal-observation-run"]
