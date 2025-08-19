# Base image with Python + Playwright + Chromium preinstalled
FROM mcr.microsoft.com/playwright/python:v1.54.0-jammy

# Set working dir
WORKDIR /app

# Copy everything
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Render sets $PORT automatically)
ENV PORT=10000
EXPOSE $PORT

# Start Gunicorn server
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:$PORT", "api.index:app"]
