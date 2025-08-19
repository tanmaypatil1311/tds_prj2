# Base image with Python + Playwright + Chromium preinstalled
FROM mcr.microsoft.com/playwright/python:v1.54.0-jammy

# Set working dir
WORKDIR /app

# Copy everything
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start Gunicorn server (Render provides $PORT at runtime)
CMD bash -c 'gunicorn -w 2 --timeout 180 -b 0.0.0.0:$PORT api.index:app'
