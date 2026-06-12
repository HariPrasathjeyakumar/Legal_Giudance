FROM python:3.10-slim

# Install system compiler tools needed for FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face explicitly listens to 7860
EXPOSE 7860

# This line opens up the Flask app securely to the Hugging Face proxy link
CMD ["gunicorn", "--workers", "2", "--timeout", "180", "-b", "0.0.0.0:7860", "app:app"]
