FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    fluidsynth \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir ".[ml]"

COPY . .

RUN pip install --no-cache-dir -e "." --no-deps

EXPOSE 8000
CMD ["uvicorn", "harmonia.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
