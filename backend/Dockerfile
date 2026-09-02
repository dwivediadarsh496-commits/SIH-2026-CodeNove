FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies for GDAL/Rasterio and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .

# Install PyTorch CPU using --extra-index-url so all dependencies can be resolved from PyPI
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway/Cloud dynamic port support (defaults to 8000)
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
