# Start from an official, lightweight Python 3.13 image
FROM python:3.13-slim

# All following commands run inside the /app folder in the container
WORKDIR /app

# Do not create .pyc cache files, and show print() output immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies first (Docker caches this step, so rebuilds are faster)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY . .

# Default action when the container starts: run the analysis
CMD ["python", "analysis.py"]