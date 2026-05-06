# Use a stable, slim Python base
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /

# Install system dependencies (build-essential is required for numpy/scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the project structure
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY data/ ./data/

# Create necessary directories for models and cache
RUN mkdir -p /app/artifacts /app/data /app/cache

# Expose FastAPI port
EXPOSE 8000



# Start the application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]


