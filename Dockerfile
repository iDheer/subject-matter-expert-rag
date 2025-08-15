FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt requirements_api.txt ./
RUN pip install -r requirements.txt -r requirements_api.txt

# Fix uvloop conflict by removing uvloop
RUN pip uninstall -y uvloop || true

# Copy your code
COPY api_server.py SME_2_query_elasticsearch_system.py ./

# Create directories
RUN mkdir -p data_large elasticsearch_storage_v2

# Copy your data and database
COPY data_large/ ./data_large/
COPY elasticsearch_storage_v2/ ./elasticsearch_storage_v2/

EXPOSE 8000

CMD ["python", "api_server.py"]