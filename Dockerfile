FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/ml

WORKDIR /app

COPY ml/requirements.txt /app/ml/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/ml/requirements.txt

COPY ml /app/ml
COPY README_ML.md /app/README_ML.md

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.infer.api:app", "--host", "0.0.0.0", "--port", "8000"]
