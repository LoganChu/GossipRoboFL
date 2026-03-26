FROM python:3.11-slim

# System dependencies for OpenCV / imageio / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create results directory
RUN mkdir -p results data

# Default: run the main gossip experiment
ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "gossip"]
