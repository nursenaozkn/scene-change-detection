FROM python:3.9-slim

WORKDIR /app

# Install the CPU-only build of PyTorch first to keep the image small,
# then the remaining dependencies from PyPI.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project source and trained checkpoints into the image.
COPY . .

# Default command: run inference + change visualization on sample pairs.
# The dataset is expected to be mounted at runtime (see README).
CMD ["python", "src/inference/predict.py"]
