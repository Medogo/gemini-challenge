# ---- Image Python légère + ffmpeg (pour audio/vidéo) ----
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + fichiers publics
COPY . .

# Cloud Run attend que l’app écoute sur 8080
ENV PORT=8080

# Démarrage de FastAPI (ton app est app.main:app)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
