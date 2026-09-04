FROM python:3.10-slim

# Install system dependencies for GDAL/Rasterio and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000 (required by HF Spaces)
RUN useradd -m -u 1000 user

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Upgrade pip, setuptools, wheel (as root, before switching user)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY --chown=user requirements.txt .

# Install PyTorch CPU using --extra-index-url so all dependencies can be resolved from PyPI
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Switch to the non-root user before copying app code
USER user

COPY --chown=user . .

# HF Spaces expects the app to listen on port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
