# Deployment Guide

## Overview

This guide covers deployment options for the Meera RAG chatbot system, from local development to production environments.

## Local Development Deployment

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM
- 2GB+ free disk space
- Internet connection for model downloads

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd LLM_RAG_Agent

# Install dependencies
pip install -r requirements.txt

# Set API key
export GROQ_API_KEY="your_api_key_here"

# Run application
python main.py
```

### Development Environment Setup

#### Using Virtual Environment
```bash
# Create virtual environment
python -m venv meera_env

# Activate environment
# Windows
meera_env\Scripts\activate
# Linux/Mac
source meera_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Using Conda
```bash
# Create conda environment
conda create -n meera python=3.9
conda activate meera

# Install dependencies
pip install -r requirements.txt
```

### Configuration for Development
```python
# main.py - Development settings
HARDCODED_FOLDER_PATH = "./Insurance PDFs"
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.doc', '.md'}

# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create documents directory
RUN mkdir -p /app/Insurance\ PDFs

# Expose port
EXPOSE 7860

# Set environment variables
ENV GROQ_API_KEY=""
ENV PYTHONPATH=/app

# Run application
CMD ["python", "main.py"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  meera:
    build: .
    ports:
      - "7860:7860"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
    volumes:
      - ./Insurance PDFs:/app/Insurance PDFs
      - ./data:/app/data
    restart: unless-stopped
```

### Docker Commands
```bash
# Build image
docker build -t meera-rag .

# Run container
docker run -p 7860:7860 \
  -e GROQ_API_KEY="your_api_key" \
  -v $(pwd)/Insurance\ PDFs:/app/Insurance\ PDFs \
  meera-rag

# Using docker-compose
docker-compose up -d
```

## Cloud Deployment

### AWS Deployment

#### EC2 Instance Setup
```bash
# Launch EC2 instance (t3.medium or larger)
# Ubuntu 20.04 LTS recommended

# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv git

# Clone repository
git clone <repository-url>
cd LLM_RAG_Agent

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY="your_api_key"

# Run application
python main.py
```

#### Using AWS ECS
```yaml
# task-definition.json
{
  "family": "meera-rag",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "meera",
      "image": "your-account.dkr.ecr.region.amazonaws.com/meera:latest",
      "portMappings": [
        {
          "containerPort": 7860,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "GROQ_API_KEY",
          "value": "your_api_key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/meera",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Platform

#### App Engine Deployment
```yaml
# app.yaml
runtime: python39
entrypoint: gunicorn -b :$PORT main:app

env_variables:
  GROQ_API_KEY: "your_api_key"

automatic_scaling:
  target_cpu_utilization: 0.6
  min_instances: 1
  max_instances: 10

resources:
  cpu: 1
  memory_gb: 2
  disk_size_gb: 10
```

#### Cloud Run Deployment
```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/meera
gcloud run deploy meera \
  --image gcr.io/PROJECT_ID/meera \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GROQ_API_KEY="your_api_key"
```

### Azure Deployment

#### Azure Container Instances
```bash
# Build and push to Azure Container Registry
az acr build --registry your-registry --image meera:latest .

# Deploy to Container Instances
az container create \
  --resource-group your-rg \
  --name meera-container \
  --image your-registry.azurecr.io/meera:latest \
  --dns-name-label meera-app \
  --ports 7860 \
  --environment-variables GROQ_API_KEY="your_api_key"
```

## Production Deployment

### Production Configuration

#### Environment Variables
```bash
# Production environment variables
export GROQ_API_KEY="your_production_api_key"
export ENVIRONMENT="production"
export LOG_LEVEL="INFO"
export MAX_WORKERS=4
export CHUNK_SIZE=500
export CHUNK_OVERLAP=50
```

#### Production Settings
```python
# production_config.py
import os

# Production configuration
PRODUCTION_CONFIG = {
    "chunk_size": int(os.getenv("CHUNK_SIZE", 500)),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 50)),
    "max_workers": int(os.getenv("MAX_WORKERS", 4)),
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
    "enable_caching": True,
    "cache_ttl": 3600,  # 1 hour
    "rate_limit": 100,  # requests per minute
}
```

### Load Balancing

#### Using Nginx
```nginx
# nginx.conf
upstream meera_backend {
    server 127.0.0.1:7860;
    server 127.0.0.1:7861;
    server 127.0.0.1:7862;
    server 127.0.0.1:7863;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://meera_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Using HAProxy
```conf
# haproxy.cfg
global
    daemon

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend meera_frontend
    bind *:80
    default_backend meera_backend

backend meera_backend
    balance roundrobin
    server meera1 127.0.0.1:7860 check
    server meera2 127.0.0.1:7861 check
    server meera3 127.0.0.1:7862 check
```

### Monitoring and Logging

#### Application Monitoring
```python
# monitoring.py
import logging
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('meera.log'),
        logging.StreamHandler()
    ]
)

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper
```

#### Health Check Endpoint
```python
# health_check.py
from flask import Flask, jsonify
import psutil
import os

app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'memory_usage': psutil.virtual_memory().percent,
        'cpu_usage': psutil.cpu_percent(),
        'disk_usage': psutil.disk_usage('/').percent
    })

@app.route('/ready')
def readiness_check():
    # Check if vector database is loaded
    if db is None:
        return jsonify({'status': 'not_ready', 'reason': 'Database not initialized'}), 503
    return jsonify({'status': 'ready'})
```

### Security Considerations

#### API Key Management
```python
# security.py
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        self.key = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
        self.cipher = Fernet(self.key)
    
    def encrypt_api_key(self, api_key):
        return self.cipher.encrypt(api_key.encode()).decode()
    
    def decrypt_api_key(self, encrypted_key):
        return self.cipher.decrypt(encrypted_key.encode()).decode()
```

#### Rate Limiting
```python
# rate_limiter.py
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id):
        now = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests
        client_requests[:] = [req_time for req_time in client_requests 
                            if now - req_time < self.window_seconds]
        
        if len(client_requests) >= self.max_requests:
            return False
        
        client_requests.append(now)
        return True
```

## Scaling Strategies

### Horizontal Scaling
```python
# scaling.py
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def run_worker(worker_id):
    """Run a worker process"""
    os.environ['WORKER_ID'] = str(worker_id)
    # Initialize database for this worker
    initialize_db()
    # Start Gradio interface on different port
    port = 7860 + worker_id
    gr.ChatInterface(fn=ques_responses).launch(server_port=port)

def start_workers(num_workers=4):
    """Start multiple worker processes"""
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(run_worker, range(num_workers))
```

### Caching Strategy
```python
# caching.py
import redis
import json
import hashlib

class ResponseCache:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hour
    
    def get_cache_key(self, question, context_hash):
        """Generate cache key for question and context"""
        content = f"{question}:{context_hash}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, question, context_hash):
        """Get cached response"""
        key = self.get_cache_key(question, context_hash)
        cached = self.redis_client.get(key)
        return json.loads(cached) if cached else None
    
    def set(self, question, context_hash, response):
        """Cache response"""
        key = self.get_cache_key(question, context_hash)
        self.redis_client.setex(key, self.ttl, json.dumps(response))
```

## Backup and Recovery

### Database Backup
```python
# backup.py
import shutil
import os
from datetime import datetime

def backup_vector_database():
    """Backup FAISS vector database"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backups/{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup vector database files
    if os.path.exists("faiss_index"):
        shutil.copytree("faiss_index", f"{backup_dir}/faiss_index")
    
    # Backup documents
    shutil.copytree("Insurance PDFs", f"{backup_dir}/documents")
    
    print(f"Backup created: {backup_dir}")

def restore_vector_database(backup_path):
    """Restore from backup"""
    if os.path.exists(f"{backup_path}/faiss_index"):
        shutil.rmtree("faiss_index", ignore_errors=True)
        shutil.copytree(f"{backup_path}/faiss_index", "faiss_index")
    
    print(f"Restored from backup: {backup_path}")
```

## Performance Optimization

### Memory Optimization
```python
# memory_optimization.py
import gc
import psutil

def optimize_memory():
    """Optimize memory usage"""
    # Force garbage collection
    gc.collect()
    
    # Monitor memory usage
    memory_info = psutil.virtual_memory()
    if memory_info.percent > 80:
        # Clear cache if memory usage is high
        clear_cache()
        gc.collect()

def clear_cache():
    """Clear application cache"""
    global memory
    memory.clear()
    # Clear other caches as needed
```

### Response Time Optimization
```python
# performance.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_ques_responses(question, history, system_prompt, token_limit):
    """Async version of ques_responses for better performance"""
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor() as executor:
        response = await loop.run_in_executor(
            executor, 
            ques_responses, 
            question, history, system_prompt, token_limit
        )
    
    return response
```

## Troubleshooting Deployment

### Common Issues

#### Port Conflicts
```bash
# Check if port is in use
netstat -tulpn | grep :7860

# Kill process using port
sudo kill -9 $(lsof -t -i:7860)
```

#### Memory Issues
```bash
# Monitor memory usage
htop
free -h

# Increase swap space if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### API Key Issues
```bash
# Verify API key is set
echo $GROQ_API_KEY

# Test API key
curl -H "Authorization: Bearer $GROQ_API_KEY" \
  https://api.groq.com/openai/v1/models
```

### Log Analysis
```bash
# View application logs
tail -f meera.log

# Search for errors
grep -i error meera.log

# Monitor real-time logs
journalctl -u meera -f
```
