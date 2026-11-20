# Une seule image pour l'API et pour le serveur MLflow (mêmes versions, figées par uv.lock)
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

WORKDIR /app
ENV UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

# 1. Les librairies d'abord : cette couche reste en cache tant que uv.lock ne change pas
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

# 2. Puis le code (le modèle n'est pas dans l'image : il est chargé depuis le Registry par alias)
COPY src ./src
COPY api ./api
COPY params.yaml ./

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
