# API Reference Documentation

## Overview

This document provides a comprehensive reference for all functions, classes, and configuration options in the Meera RAG chatbot system.

## Core Functions

### Document Loading Functions

#### `load_pdf(path: str) -> List[Document]`

Loads and extracts text content from PDF files using PyMuPDF.

**Parameters:**
- `path` (str): Absolute or relative path to the PDF file

**Returns:**
- `List[Document]`: List of LangChain Document objects, one per page

**Raises:**
- `FileNotFoundError`: If the PDF file doesn't exist
- `fitz.FileDataError`: If the PDF file is corrupted or invalid

**Example:**
```python
from main import load_pdf

# Load a PDF file
documents = load_pdf("./Insurance PDFs/policy_document.pdf")
print(f"Loaded {len(documents)} pages")
```

#### `load_docx(path: str) -> List[Document]`

Loads and extracts text content from Microsoft Word documents.

**Parameters:**
- `path` (str): Absolute or relative path to the DOCX/DOC file

**Returns:**
- `List[Document]`: List containing a single LangChain Document object

**Raises:**
- `FileNotFoundError`: If the Word file doesn't exist
- `ValueError`: If the file format is not supported

**Example:**
```python
from main import load_docx

# Load a Word document
documents = load_docx("./Insurance PDFs/terms_and_conditions.docx")
print(f"Document content length: {len(documents[0].page_content)}")
```

#### `load_txt(path: str) -> List[Document]`

Loads text content from plain text files.

**Parameters:**
- `path` (str): Absolute or relative path to the text file

**Returns:**
- `List[Document]`: List containing a single LangChain Document object

**Raises:**
- `FileNotFoundError`: If the text file doesn't exist
- `UnicodeDecodeError`: If the file encoding is not UTF-8

**Example:**
```python
from main import load_txt

# Load a text file
documents = load_txt("./Insurance PDFs/notes.txt")
print(f"Text content: {documents[0].page_content[:100]}...")
```

#### `load_file(path: str) -> List[Document]`

Universal file loader that automatically detects file type and calls appropriate loader.

**Parameters:**
- `path` (str): Absolute or relative path to the file

**Returns:**
- `List[Document]`: List of LangChain Document objects

**Supported Formats:**
- `.pdf` → `load_pdf()`
- `.docx`, `.doc` → `load_docx()`
- `.txt`, `.md` → `load_txt()`

**Example:**
```python
from main import load_file

# Load any supported file type
documents = load_file("./Insurance PDFs/document.pdf")
documents = load_file("./Insurance PDFs/document.docx")
documents = load_file("./Insurance PDFs/document.txt")
```

#### `load_documents_from_folder(folder_path: str) -> List[Document]`

Recursively loads all supported documents from a folder and its subfolders.

**Parameters:**
- `folder_path` (str): Path to the folder containing documents

**Returns:**
- `List[Document]`: List of all loaded documents from the folder

**Raises:**
- `ValueError`: If the folder doesn't exist or contains no supported files

**Example:**
```python
from main import load_documents_from_folder

# Load all documents from a folder
documents = load_documents_from_folder("./Insurance PDFs")
print(f"Loaded {len(documents)} total documents")
```

### Vector Database Functions

#### `build_vectorstore(documents: List[Document])`

Creates a FAISS vector database from document chunks using sentence transformers.

**Parameters:**
- `documents` (List[Document]): List of LangChain Document objects

**Returns:**
- `FAISS`: FAISS vector database object

**Configuration:**
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`

**Example:**
```python
from main import build_vectorstore, load_documents_from_folder

# Load documents and build vector store
documents = load_documents_from_folder("./Insurance PDFs")
vectorstore = build_vectorstore(documents)
print("Vector database created successfully")
```

### Query Processing Functions

#### `ques_responses(question: str, history: list, system_prompt: str, token_limit: int) -> str`

Main function for processing user queries and generating AI responses.

**Parameters:**
- `question` (str): User's question or query
- `history` (list): Previous conversation history
- `system_prompt` (str): AI system prompt defining behavior
- `token_limit` (int): Maximum number of tokens in response (10-300)

**Returns:**
- `str`: Generated AI response

**Raises:**
- `ValueError`: If no documents are loaded
- `Exception`: If LLM API call fails

**Example:**
```python
from main import ques_responses

# Process a query
response = ques_responses(
    question="What is the deductible for the Gold plan?",
    history=[],
    system_prompt="You are Meera, an insurance expert assistant.",
    token_limit=150
)
print(response)
```

### Initialization Functions

#### `initialize_db()`

Initializes the global vector database by loading documents and building the FAISS index.

**Raises:**
- `ValueError`: If folder path is invalid or no documents found

**Example:**
```python
from main import initialize_db

# Initialize the database
try:
    initialize_db()
    print("Database initialized successfully")
except ValueError as e:
    print(f"Initialization failed: {e}")
```

## Configuration Variables

### Global Configuration

#### `GROQ_API_KEY`
**Type:** `str`  
**Required:** Yes  
**Description:** Groq API key for LLM inference  
**Default:** `"Your_GROQ_API_KEY_Here"`  
**Environment Variable:** `GROQ_API_KEY`

**Example:**
```python
import os
os.environ["GROQ_API_KEY"] = "your_actual_api_key_here"
```

#### `HARDCODED_FOLDER_PATH`
**Type:** `str`  
**Required:** Yes  
**Description:** Path to folder containing insurance documents  
**Default:** `"./Insurance PDFs"`

**Example:**
```python
HARDCODED_FOLDER_PATH = "/path/to/your/documents"
```

#### `SUPPORTED_EXTENSIONS`
**Type:** `set`  
**Required:** No  
**Description:** Set of supported file extensions  
**Default:** `{'.pdf', '.docx', '.txt', '.doc', '.md'}`

**Example:**
```python
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.doc', '.md', '.rtf'}
```

### Global State Variables

#### `db`
**Type:** `FAISS`  
**Description:** Global vector database instance  
**Initialization:** Set by `initialize_db()`

#### `memory`
**Type:** `ConversationBufferMemory`  
**Description:** Global conversation memory buffer  
**Configuration:**
- `memory_key="history"`
- `input_key="question"`
- `return_messages=True`

## LangChain Components

### Document Class
**Import:** `from langchain.docstore.document import Document`

**Attributes:**
- `page_content` (str): The text content of the document
- `metadata` (dict): Document metadata including source file path

**Example:**
```python
from langchain.docstore.document import Document

document = Document(
    page_content="This is the document content...",
    metadata={"source": "./file.pdf", "page": 1}
)
```

### RecursiveCharacterTextSplitter
**Import:** `from langchain.text_splitter import RecursiveCharacterTextSplitter`

**Configuration:**
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Characters per chunk
    chunk_overlap=50     # Overlap between chunks
)
```

### HuggingFaceEmbeddings
**Import:** `from langchain_huggingface import HuggingFaceEmbeddings`

**Configuration:**
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

### ConversationBufferMemory
**Import:** `from langchain.memory import ConversationBufferMemory`

**Configuration:**
```python
memory = ConversationBufferMemory(
    memory_key="history",
    input_key="question",
    return_messages=True
)
```

## Gradio Interface

### ChatInterface Configuration
**Import:** `import gradio as gr`

**Parameters:**
- `fn`: Function to handle chat (ques_responses)
- `additional_inputs`: List of additional input components
- `title`: Interface title
- `description`: Interface description
- `theme`: UI theme

**Example:**
```python
gr.ChatInterface(
    fn=ques_responses,
    additional_inputs=[
        gr.Textbox("You are Meera, an assistant AI chatbot.", label="System Prompt"),
        gr.Slider(10, 300, step=10, value=100, label="Max Tokens")
    ],
    title="💬 Ask Meera",
    description="Chat with documents from: `./Insurance PDFs`",
    theme="soft"
).launch()
```

## Error Handling

### Common Exceptions

#### `FileNotFoundError`
**Cause:** File doesn't exist at specified path  
**Solution:** Verify file path and permissions

#### `ValueError`
**Cause:** Invalid folder path or no supported files  
**Solution:** Check folder path and file extensions

#### `fitz.FileDataError`
**Cause:** Corrupted or invalid PDF file  
**Solution:** Verify PDF file integrity

#### `UnicodeDecodeError`
**Cause:** Text file not UTF-8 encoded  
**Solution:** Convert file to UTF-8 encoding

### Error Recovery

```python
try:
    documents = load_documents_from_folder("./Insurance PDFs")
    vectorstore = build_vectorstore(documents)
except ValueError as e:
    print(f"Error loading documents: {e}")
    # Handle error gracefully
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log error and continue
```

## Performance Optimization

### Memory Management
```python
# Clear memory buffer if needed
memory.clear()

# Process documents in batches
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    # Process batch
```

### Response Time Optimization
```python
# Reduce chunk size for faster processing
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Smaller chunks
    chunk_overlap=30
)

# Reduce retrieval count for faster search
retriever_docs = db.similarity_search(question, k=2)  # Fewer results
```

## Integration Examples

### Custom Document Loader
```python
def load_custom_format(path: str) -> List[Document]:
    """Custom loader for specific file format"""
    # Implementation here
    return [Document(page_content=content, metadata={"source": path})]

# Add to supported extensions
SUPPORTED_EXTENSIONS.add('.custom')
```

### Custom Prompt Template
```python
from langchain.prompts import PromptTemplate

custom_template = """
You are an insurance expert. Answer questions based on the following context:

Context: {context}

Question: {question}

Provide a detailed answer:
"""

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=custom_template
)
```

### Batch Processing
```python
def process_documents_batch(folder_path: str, batch_size: int = 50):
    """Process documents in batches to manage memory"""
    all_documents = []
    
    for root, _, files in os.walk(folder_path):
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            batch_docs = []
            
            for file in batch_files:
                if os.path.splitext(file)[-1].lower() in SUPPORTED_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    batch_docs.extend(load_file(full_path))
            
            all_documents.extend(batch_docs)
    
    return all_documents
```
