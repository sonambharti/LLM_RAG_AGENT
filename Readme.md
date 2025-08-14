# Meera - Insurance Document RAG Chatbot

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Meera is an intelligent Retrieval-Augmented Generation (RAG) chatbot designed to provide accurate answers to questions about insurance documents. The system leverages advanced natural language processing and vector search capabilities to retrieve relevant information from a collection of insurance PDFs and other documents, then generates contextual responses using the Groq LLM API.

### Key Capabilities
- **Multi-format Document Support**: Processes PDF, DOCX, TXT, DOC, and MD files
- **Semantic Search**: Uses advanced embeddings for contextually relevant document retrieval
- **Conversation Memory**: Maintains conversation history for contextual responses
- **Web-based Interface**: User-friendly Gradio chat interface
- **Real-time Processing**: Instant responses with configurable token limits

## ✨ Features

### Document Processing
- **Automatic Document Loading**: Recursively scans folders for supported file types
- **Intelligent Text Chunking**: Splits documents into optimal chunks for retrieval
- **Metadata Preservation**: Maintains source file information for traceability

### AI-Powered Responses
- **Context-Aware Answers**: Retrieves relevant document sections before generating responses
- **Conversation Continuity**: Remembers previous interactions for coherent conversations
- **Customizable System Prompts**: Configurable AI personality and behavior
- **Token Management**: Adjustable response length limits

### User Interface
- **Modern Web Interface**: Clean, responsive Gradio-based chat UI
- **Real-time Chat**: Instant message exchange with typing indicators
- **Parameter Controls**: Adjustable system prompts and token limits
- **Error Handling**: Graceful error messages and recovery

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Text          │    │   Vector        │
│   Loaders       │───▶│   Chunking      │───▶│   Database      │
│   (PDF/DOCX)    │    │   (500 chars)   │    │   (FAISS)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Similarity    │───▶│   LLM Chain     │
│   (Gradio UI)   │    │   Search        │    │   (Groq API)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Response      │◀───│   Memory        │◀───│   Generated     │
│   Display       │    │   Buffer        │    │   Answer        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow
1. **Document Ingestion**: Files are loaded and converted to LangChain Document objects
2. **Text Processing**: Documents are split into chunks with 500-character size and 50-character overlap
3. **Vectorization**: Text chunks are converted to embeddings using sentence-transformers
4. **Storage**: Embeddings are stored in FAISS vector database for fast similarity search
5. **Query Processing**: User queries are processed through the same embedding pipeline
6. **Retrieval**: Most similar document chunks are retrieved (top 3)
7. **Generation**: Context and query are sent to Groq LLM for response generation
8. **Memory**: Conversation history is maintained for context continuity

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free space for dependencies and vector database
- **Internet**: Required for downloading models and API access

### API Keys
- **Groq API Key**: Required for LLM inference
  - Sign up at [Groq Console](https://console.groq.com/)
  - Generate API key from dashboard
  - Set as environment variable or in code

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd LLM_RAG_Agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Set your Groq API key in one of these ways:

**Option A: Environment Variable (Recommended)**
```bash
export GROQ_API_KEY="your_api_key_here"
```

**Option B: Direct in Code**
Edit `main.py` line 18:
```python
os.environ["GROQ_API_KEY"] = "your_actual_api_key_here"
```

### 4. Prepare Documents
Place your insurance documents in the `Insurance PDFs/` folder:
```
Insurance PDFs/
├── policy_document_1.pdf
├── policy_document_2.docx
├── terms_and_conditions.txt
└── ...
```

## ⚙️ Configuration

### Document Path Configuration
Edit the `HARDCODED_FOLDER_PATH` variable in `main.py`:
```python
HARDCODED_FOLDER_PATH = "./Insurance PDFs"  # Change to your folder path
```

### Supported File Types
The system supports the following file formats:
- **PDF** (.pdf) - Using PyMuPDF
- **Word Documents** (.docx, .doc) - Using python-docx
- **Text Files** (.txt) - UTF-8 encoded
- **Markdown** (.md) - Treated as text files

### Model Configuration
- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`
- **LLM Provider**: Groq (Llama3-8b-8192 model)
- **Chunk Size**: 500 characters with 50-character overlap
- **Retrieval Count**: Top 3 most similar documents

## 💻 Usage

### Starting the Application
```bash
python main.py
```

The application will:
1. Load and process all documents in the configured folder
2. Build the vector database
3. Launch the Gradio web interface
4. Display the local URL (typically `http://127.0.0.1:7860`)

### Using the Chat Interface

1. **Open the Web Interface**: Navigate to the displayed URL in your browser
2. **Configure Parameters**:
   - **System Prompt**: Define the AI's personality and behavior
   - **Max Tokens**: Set response length limit (10-300 tokens)
3. **Start Chatting**: Type your questions about insurance documents
4. **View Responses**: Get contextual answers based on document content

### Example Interactions

**Question**: "What is the deductible for the Gold plan?"
**Response**: Based on retrieved document chunks, provides specific deductible information.

**Question**: "Can you summarize the coverage benefits?"
**Response**: Generates a concise summary of coverage benefits from relevant documents.

**Question**: "What are the exclusions in the policy?"
**Response**: Lists policy exclusions found in the document collection.

## 📚 API Reference

### Core Functions

#### `load_pdf(path: str) -> List[Document]`
Loads and extracts text from PDF files.

**Parameters:**
- `path` (str): Path to the PDF file

**Returns:**
- `List[Document]`: List of LangChain Document objects

#### `load_docx(path: str) -> List[Document]`
Loads and extracts text from Word documents.

**Parameters:**
- `path` (str): Path to the DOCX/DOC file

**Returns:**
- `List[Document]`: List of LangChain Document objects

#### `load_documents_from_folder(folder_path: str) -> List[Document]`
Recursively loads all supported documents from a folder.

**Parameters:**
- `folder_path` (str): Path to the folder containing documents

**Returns:**
- `List[Document]`: List of all loaded documents

#### `build_vectorstore(documents: List[Document])`
Creates a FAISS vector database from document chunks.

**Parameters:**
- `documents` (List[Document]): List of documents to process

**Returns:**
- `FAISS`: Vector database object

#### `ques_responses(question, history, system_prompt, token_limit)`
Main function for processing user queries and generating responses.

**Parameters:**
- `question` (str): User's question
- `history` (list): Conversation history
- `system_prompt` (str): AI system prompt
- `token_limit` (int): Maximum response length

**Returns:**
- `str`: Generated response

### Configuration Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GROQ_API_KEY` | str | Required | Groq API key for LLM access |
| `HARDCODED_FOLDER_PATH` | str | `"./Insurance PDFs"` | Path to document folder |
| `SUPPORTED_EXTENSIONS` | set | `{'.pdf', '.docx', '.txt', '.doc', '.md'}` | Supported file types |

## 🔧 Troubleshooting

### Common Issues

#### 1. API Key Errors
**Error**: "Invalid API key" or authentication failures
**Solution**: 
- Verify your Groq API key is correct
- Check that the key has sufficient credits
- Ensure the key is properly set in environment variables

#### 2. Document Loading Failures
**Error**: "No supported files found in folder"
**Solution**:
- Verify the folder path in `HARDCODED_FOLDER_PATH`
- Ensure documents have supported file extensions
- Check file permissions and accessibility

#### 3. Memory Issues
**Error**: Out of memory errors during processing
**Solution**:
- Reduce chunk size in `RecursiveCharacterTextSplitter`
- Process documents in smaller batches
- Increase system RAM or use cloud resources

#### 4. Model Download Issues
**Error**: Failed to download embedding model
**Solution**:
- Check internet connectivity
- Clear HuggingFace cache: `rm -rf ~/.cache/huggingface`
- Use VPN if in restricted network

#### 5. Gradio Interface Not Loading
**Error**: Cannot access web interface
**Solution**:
- Check if port 7860 is available
- Try different port: `gr.ChatInterface(...).launch(server_port=7861)`
- Verify firewall settings

### Performance Optimization

#### For Large Document Collections
1. **Increase Chunk Size**: Modify `chunk_size` parameter for fewer, larger chunks
2. **Reduce Overlap**: Decrease `chunk_overlap` to save memory
3. **Batch Processing**: Process documents in smaller batches
4. **Use GPU**: Install CUDA-enabled PyTorch for faster embeddings

#### For Better Response Quality
1. **Adjust Retrieval Count**: Increase `k` parameter in `similarity_search`
2. **Fine-tune Prompts**: Modify system prompt for specific use cases
3. **Add Filters**: Implement document filtering based on metadata
4. **Use Different Models**: Experiment with different embedding models

## 🤝 Contributing

We welcome contributions to improve Meera! Here's how you can help:

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Areas for Improvement
- **Additional File Formats**: Support for more document types
- **Advanced Search**: Implement hybrid search (keyword + semantic)
- **User Authentication**: Add user management and access controls
- **Analytics**: Track usage patterns and response quality
- **Export Features**: Save conversations and document summaries

### Code Style
- Follow PEP 8 Python style guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write clear commit messages


## 🙏 Acknowledgments

- **LangChain**: For the RAG framework and utilities
- **Groq**: For providing fast LLM inference
- **HuggingFace**: For the sentence transformers library
- **FAISS**: For efficient vector similarity search
- **Gradio**: For the web interface framework

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the documentation for common solutions

---

**Version**: 1.0.0  
**Last Updated**: 6 August 2025
**Maintainer**: Shaikh Faizan Ahemad