FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY fiberseg ./fiberseg
COPY configs ./configs

RUN pip install --upgrade pip && pip install -e .

CMD ["python", "-m", "fiberseg.train", "--config", "configs/example.yaml"]
