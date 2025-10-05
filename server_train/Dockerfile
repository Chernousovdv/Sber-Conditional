FROM python:3.12.3

WORKDIR /app

COPY requirements.txt .

# Install system dependencies
RUN apt-get update && pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the ingestion script
ENTRYPOINT ["python", "main.py"]