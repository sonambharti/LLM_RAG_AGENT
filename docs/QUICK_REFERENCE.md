# Quick Reference Guide

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_api_key"
python main.py
```

### Configuration
```python
# main.py
HARDCODED_FOLDER_PATH = "./Insurance PDFs"  # Document folder
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.doc', '.md'}
```

## 📁 File Structure
```
LLM_RAG_Agent/
├── main.py                 # Main application
├── requirements.txt        # Dependencies
├── README.md              # Main documentation
├── docs/                  # Documentation folder
│   ├── ARCHITECTURE.md    # Technical architecture
│   ├── API_REFERENCE.md   # API documentation
│   ├── DEPLOYMENT.md      # Deployment guide
│   ├── USER_GUIDE.md      # User guide
│   └── QUICK_REFERENCE.md # This file
└── Insurance PDFs/        # Document storage
    ├── policy_1.pdf
    ├── policy_2.docx
    └── ...
```

## 🔧 Core Functions

### Document Loading
```python
# Load PDF
documents = load_pdf("file.pdf")

# Load Word document
documents = load_docx("file.docx")

# Load text file
documents = load_txt("file.txt")

# Load all from folder
documents = load_documents_from_folder("./Insurance PDFs")
```

### Vector Database
```python
# Build vector store
vectorstore = build_vectorstore(documents)

# Search similar documents
results = vectorstore.similarity_search("query", k=3)
```

### Query Processing
```python
# Process user query
response = ques_responses(
    question="What is the deductible?",
    history=[],
    system_prompt="You are Meera, an insurance expert.",
    token_limit=150
)
```

## ⚙️ Configuration Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GROQ_API_KEY` | str | Required | Groq API key |
| `HARDCODED_FOLDER_PATH` | str | `"./Insurance PDFs"` | Document folder |
| `SUPPORTED_EXTENSIONS` | set | `{'.pdf', '.docx', '.txt', '.doc', '.md'}` | File types |

## 🐳 Docker Commands

### Build and Run
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

## ☁️ Cloud Deployment

### AWS ECS
```yaml
# task-definition.json
{
  "family": "meera-rag",
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [{
    "name": "meera",
    "image": "your-image:latest",
    "portMappings": [{"containerPort": 7860}],
    "environment": [{"name": "GROQ_API_KEY", "value": "your_key"}]
  }]
}
```

### Google Cloud Run
```bash
gcloud run deploy meera \
  --image gcr.io/PROJECT_ID/meera \
  --set-env-vars GROQ_API_KEY="your_api_key" \
  --allow-unauthenticated
```

## 🔍 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| API Key Error | Verify `GROQ_API_KEY` environment variable |
| No Documents Found | Check `HARDCODED_FOLDER_PATH` and file extensions |
| Memory Issues | Reduce chunk size or increase system RAM |
| Port Conflicts | Change port: `gr.ChatInterface(...).launch(server_port=7861)` |
| Model Download Fail | Clear cache: `rm -rf ~/.cache/huggingface` |

### Performance Tuning
```python
# Reduce memory usage
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Smaller chunks
    chunk_overlap=30
)

# Faster responses
retriever_docs = db.similarity_search(question, k=2)  # Fewer results
```

## 📊 Monitoring

### Health Check
```python
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'memory_usage': psutil.virtual_memory().percent,
        'cpu_usage': psutil.cpu_percent()
    })
```

### Logging
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('meera.log')]
)
```

## 🔐 Security

### API Key Management
```python
# Environment variable (recommended)
export GROQ_API_KEY="your_key"

# Encrypted storage
from cryptography.fernet import Fernet
cipher = Fernet(key)
encrypted_key = cipher.encrypt(api_key.encode())
```

### Rate Limiting
```python
class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
```

## 📈 Scaling

### Horizontal Scaling
```python
def start_workers(num_workers=4):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(run_worker, range(num_workers))
```

### Load Balancing (Nginx)
```nginx
upstream meera_backend {
    server 127.0.0.1:7860;
    server 127.0.0.1:7861;
    server 127.0.0.1:7862;
}

server {
    listen 80;
    location / {
        proxy_pass http://meera_backend;
    }
}
```

## 💾 Backup

### Database Backup
```python
def backup_vector_database():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backups/{timestamp}"
    
    if os.path.exists("faiss_index"):
        shutil.copytree("faiss_index", f"{backup_dir}/faiss_index")
    
    shutil.copytree("Insurance PDFs", f"{backup_dir}/documents")
```

## 🔄 Updates

### Adding New File Types
```python
def load_custom_format(path: str) -> List[Document]:
    # Implementation here
    return [Document(page_content=content, metadata={"source": path})]

# Add to supported extensions
SUPPORTED_EXTENSIONS.add('.custom')
```

### Custom Prompts
```python
custom_template = """
You are an insurance expert. Answer based on this context:

Context: {context}
Question: {question}

Answer:
"""

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=custom_template
)
```

## 📞 Support

### Error Reporting
- **Technical Issues**: Check logs in `meera.log`
- **API Issues**: Verify Groq API key and credits
- **Performance**: Monitor memory and CPU usage
- **Documentation**: See full docs in `docs/` folder

### Useful Commands
```bash
# Check system resources
htop
free -h
df -h

# Monitor logs
tail -f meera.log
grep -i error meera.log

# Test API key
curl -H "Authorization: Bearer $GROQ_API_KEY" \
  https://api.groq.com/openai/v1/models
```

## 📋 Checklist

### Before Deployment
- [ ] API key configured
- [ ] Documents in correct folder
- [ ] Dependencies installed
- [ ] Port 7860 available
- [ ] Sufficient memory (4GB+)

### Production Checklist
- [ ] Environment variables set
- [ ] Logging configured
- [ ] Monitoring enabled
- [ ] Backup strategy in place
- [ ] Security measures implemented
- [ ] Load balancing configured
- [ ] Health checks working

### Maintenance
- [ ] Regular backups
- [ ] Log rotation
- [ ] Performance monitoring
- [ ] Security updates
- [ ] Document updates

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**For full documentation**: See `docs/` folder
