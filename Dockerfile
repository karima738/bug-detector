# ============================================
# Stage 1: Builder (compilation et dépendances)
# ============================================
FROM python:3.10-slim as builder

# Métadonnées
LABEL maintainer="ezzaimsaloua@... / erremytykarima@gmail.com"
LABEL description="Bug Predictor - Système de prédiction de bugs"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Installer dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Créer dossier de travail
WORKDIR /build

# Copier requirements
COPY requirements.txt .

# Installer dépendances Python dans un venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ============================================
# Stage 2: Runtime (image finale légère)
# ============================================
FROM python:3.10-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Installer dépendances runtime minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Créer utilisateur non-root (sécurité)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/models /app/data /app/results && \
    chown -R appuser:appuser /app

# Copier venv depuis builder
COPY --from=builder /opt/venv /opt/venv

# Définir dossier de travail
WORKDIR /app

# Copier le code de l'application
COPY --chown=appuser:appuser . .

# Passer à l'utilisateur non-root
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')" || exit 1

# Exposer le port Streamlit
EXPOSE 8501

# Point d'entrée
ENTRYPOINT ["streamlit", "run"]

# Commande par défaut
CMD ["app_simple.py", "--server.address", "0.0.0.0", "--server.port", "8501"]