# Dockerfile para el organizador de documentos
FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de requirements
COPY requirements.txt .
COPY setup.py .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Copiar código fuente
COPY organizador_documentos/ ./organizador_documentos/
COPY config.yml .

# Crear directorios para datos
RUN mkdir -p /app/data/origen /app/data/destino /app/logs

# Variables de entorno
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Volúmenes para datos persistentes
VOLUME ["/app/data", "/app/logs"]

# Comando por defecto
ENTRYPOINT ["python", "-m", "organizador_documentos"]
CMD ["--help"]