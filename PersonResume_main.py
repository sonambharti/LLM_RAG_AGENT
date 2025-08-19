"""
Recruiter Resume Assistant  
- Recruiter can upload multiple resumes (PDF/DOCX/TXT/CSV).  
- RAG model indexes all resumes.  
- When recruiter types "hi", bot greets politely and asks:  
  👉 "Whose resume would you like to explore?"  
- Recruiter gives a name (e.g., "John Doe")  
- Bot fetches details from that resume and answers politely.  
"""

import os
import csv
import time
import re
from typing import List, Dict, Tuple, Optional
import requests, json

import fitz
from docx import Document as DocxDocument
from dotenv import load_dotenv
import logging

# LangChain / RAG
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain_community.document_loaders import CSVLoader
import weaviate
from langchain_community.vectorstores import Weaviate

# UI
import gradio as gr

# ================== LOGGING ==================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================== ENV ==================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ================== CONFIG ==================
HARDCODED_FOLDER_PATH = "./Resumes"
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.doc', '.md', '.csv'}

# ================== LOADERS ==================
def extract_candidate_name(filename: str) -> str:
    """Extract candidate name from filename (e.g., 'Sonam_Resume.pdf' -> 'Sonam')."""
    base = os.path.splitext(filename)[0]
    # Remove words like 'resume', 'cv' etc. for cleaner names
    cleaned = re.sub(r'(?i)\b(resume|cv|profile)\b', '', base)
    return cleaned.strip().replace("_", " ").replace("-", " ").title()


def load_pdf(path: str) -> List[Document]:
    try:
        doc = fitz.open(path)
        documents = []
        candidate_name = extract_candidate_name(os.path.basename(path))
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(path),
                        "page": page_num + 1,
                        "candidate_name": candidate_name
                    }
                ))
        doc.close()
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF {path}: {e}")
        return []

def load_docx(path: str) -> List[Document]:
    try:
        doc = DocxDocument(path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        candidate_name = extract_candidate_name(os.path.basename(path))
        return [Document(page_content=text, metadata={
            "source": os.path.basename(path),
            "candidate_name": candidate_name
        })]
    except Exception as e:
        logger.error(f"Error loading DOCX {path}: {e}")
        return []

def load_txt(path: str) -> List[Document]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        candidate_name = extract_candidate_name(os.path.basename(path))
        return [Document(page_content=text, metadata={
            "source": os.path.basename(path),
            "candidate_name": candidate_name
        })]
    except Exception as e:
        logger.error(f"Error loading text file {path}: {e}")
        return []

def load_csv_generic(path: str) -> List[Document]:
    try:
        loader = CSVLoader(file_path=path)
        text = loader.load()
        candidate_name = extract_candidate_name(os.path.basename(path))
        return [Document(page_content=text, metadata={
            "source": os.path.basename(path),
            "candidate_name": candidate_name
        })]
    except Exception as e:
        logger.error(f"Error loading CSV {path}: {e}")
        return []

def load_file(path: str) -> List[Document]:
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.pdf':
        return load_pdf(path)
    elif ext in {'.docx', '.doc'}:
        return load_docx(path)
    elif ext in {'.txt', '.md'}:
        return load_txt(path)
    elif ext == '.csv':
        return load_csv_generic(path)
    return []

def load_documents_from_folder(folder_path: str) -> Tuple[List[Document], int]:
    # Doc loader
    start_loader = time.perf_counter()
    docs, file_count = [], 0
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")
    logger.info(f"📁 Scanning folder: {folder_path}")
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[-1].lower() in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(root, file)
                logger.info(f"  📄 Loading file: {file}")
                try:
                    file_docs = load_file(file_path)
                    docs.extend(file_docs)
                    file_count += 1
                except Exception as e:
                    logger.error(f"    ❌ Error loading {file}: {e}")
    end_loader = time.perf_counter()
    logger.info(f"📊 Total documents loaded: {len(docs)} from {file_count} files")
    print(f"Time taken by Data Loader: {end_loader - start_loader:.2f} seconds")
    return docs, file_count


# ================== VECTORSTORE ==================
def build_vectorstore(documents: List[Document]) -> Weaviate:
    # starting chunking and splitting
    start_chunk = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    end_chunk = time.perf_counter()
    print(f"⏱ Chunking & Splitting: {end_chunk - start_chunk:.2f} seconds")

    # Embedding
    start_embed = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    end_embed = time.perf_counter()
    print(f"⏱ Embedding model load: {end_embed - start_embed:.2f} seconds")

    # Vectorization
    start_vector = time.perf_counter()
    client = weaviate.Client("http://localhost:8080")
    vectorstore = Weaviate.from_documents(chunks, embeddings, client=client)
    end_vector = time.perf_counter()
    print(f"⏱ Vectorization (Weaviate build): {end_vector - start_vector:.2f} seconds \n\n")
    logger.info(f"💾 Stored {len(chunks)} chunks in Weaviate (local)")
    return vectorstore


# ================== GLOBALS ==================
qa_chain = None
vectorstore = None

# ================== RAG INIT ==================
def initialize_rag_chain():
    global qa_chain, vectorstore
    logger.info("🚀 Initializing RAG Chain...")
    docs, _ = load_documents_from_folder(HARDCODED_FOLDER_PATH)
    if not docs:
        raise ValueError(f"No resumes found in '{HARDCODED_FOLDER_PATH}'. Please add files.")
    vectorstore = build_vectorstore(docs)
    

    


# ================== HANDLER ==================
def recruiter_responses(question: str, system_prompt, history: list) -> str:
    global qa_chain, vectorstore
    if vectorstore is None:
        return "❌ RAG chain is not initialized. Please check server logs."
    
    # Retriever
    start_retrieval = time.perf_counter()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(question)  
    end_retrieval = time.perf_counter()
    print(f"⏱ Retrieval: {end_retrieval - start_retrieval:.2f} seconds")
    print(f"Retriever: \n {retriever}")
    print(f"Retrieved Document: {docs}")
    
    context = "\n".join([
        f"Candidate: {doc.metadata.get('candidate_name', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    ])
    
    memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=True)

    instruction = f"""You are a polite and professional recruiter assistant.

            STRICT RULES:
            1. Always greet recruiter warmly and politely.
            2. If recruiter says "hi" or "hello", respond with:
            "Hello! Whose resume would you like me to summarize or answer questions about?"
            3. Once the recruiter provides a name:
            - Treat that candidate as the ACTIVE_CANDIDATE.
            - Answer ALL future questions only about ACTIVE_CANDIDATE unless a new name is explicitly given.
            4. If recruiter asks without giving a name AND history has an ACTIVE_CANDIDATE, use that candidate’s details.
            5. NEVER assume or invent candidate names/details not present in context or history.
            6. If the recruiter asks about multiple candidates, clarify politely which one they want to discuss first.
            7. Keep answers professional, concise, and helpful. Highlight strengths, skills, and ATS score only if they exist in context.
            8. If information is missing from resumes, respond with:
            "I could not find that information in the provided resume."
        """
    template = f"""
            You are assisting a recruiter in reviewing candidate resumes.

            SYSTEM INSTRUCTION:
            {system_prompt}

            ACTIVE CONTEXT (Resume Chunks):
            {context}

            CONVERSATION HISTORY:
            {history}

            CURRENT QUESTION FROM RECRUITER:
            {question}

            TASK:
            - Always follow STRICT RULES provided above.
            - Use history to remember the last mentioned ACTIVE_CANDIDATE if no new name is given.
            - Only provide details that exist in the resumes (from CONTEXT).
            - If unsure, explicitly say you don’t know instead of guessing.

            Final Answer:
        """

    prompt = PromptTemplate(input_variables=["system_prompt", "context", "history", "question"], template=template)

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    qa_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
    )
    
    logger.info("✅ RAG Chain initialized successfully!")
    
    response = qa_chain.predict(question=question, context=context, instruction=instruction, system_prompt=system_prompt)
    return response


# ================== MAIN ==================
def main():
    try:
        initialize_rag_chain()
        interface = gr.ChatInterface(
            fn=recruiter_responses, 
            additional_inputs=[
                gr.Textbox("You are Rekha, an assistant AI chatbot", label="System Prompt"),
            ],
            title="💼 Resume Recruiter Assistant",
            description="Upload resumes in ./Resumes and ask questions about candidates.",
            theme="soft",
            examples=[
                ["hi"],
                ["Show me work experience"],
                ["What are the key skills of Sonam?"],
                ["Summarize education details of Rahul Kumar"]
            ],
        )
        logger.info("🚀 Launching Gradio interface...")
        interface.queue().launch(share=False, show_error=True)
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise

if __name__ == "__main__":
    main()

"""
Time taken by Data Loader: 0.42 seconds
⏱ Chunking & Splitting: 0.00 seconds
⏱ Embedding model load: 16.13 seconds
⏱ Vectorization (Weaviate build): 43.98 seconds 
⏱ Retrieval: 0.32 seconds
Retriever: 
tags=['Weaviate', 'HuggingFaceEmbeddings'] vectorstore=<langchain_community.vectorstores.weaviate.Weaviate object at 0x0000021BEAB7CD70> search_kwargs={'k': 3}
"""