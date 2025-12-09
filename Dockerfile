# Dockerfile
# Light python image
FROM python:3.13-slim

# Container work dir
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY main.py .
COPY model_artefacts /app/model_artefacts

# Exposer le port par défaut de Uvicorn/FastAPI
EXPOSE 8000

# Commande de démarrage : Utiliser Uvicorn (serveur ASGI pour FastAPI)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]