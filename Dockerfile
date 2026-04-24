FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    ENV=prod

RUN groupadd --system app && useradd --system --gid app --home /home/app --create-home app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app streamlit_app.py ./

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/_stcore/health').read()" || exit 1

CMD streamlit run streamlit_app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
