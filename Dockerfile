FROM python:3.12-slim

# ffmpeg (pour pydub/speech_recognition) + locales de base
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code + des assets (inclut public/, samples, favicon, etc.)
COPY . .

# Uvicorn expose sur $PORT pour Cloud Run
ENV PORT=8080
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8080","--proxy-headers","--forwarded-allow-ips=*"]
