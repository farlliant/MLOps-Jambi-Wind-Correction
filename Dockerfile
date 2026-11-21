FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt requirements-lk09.txt ./
RUN python -m pip install --no-cache-dir -r requirements-lk09.txt

COPY train.py ./train.py
COPY src/ ./src/

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.lk09_api:app", "--host", "0.0.0.0", "--port", "8000"]
