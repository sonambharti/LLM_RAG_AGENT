import time
import os
import fitz
from docx import Document as DocxDocument
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from typing import List
import gradio as gr
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ----------- Config -----------
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

HARDCODED_FOLDER_PATH = "./Insurance PDFs"
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.doc', '.md'}

# ----------- Loaders -----------
def load_pdf(path: str) -> List[Document]:
    doc = fitz.open(path)
    return [Document(page_content=page.get_text(), metadata={"source": path}) for page in doc]

def load_docx(path: str) -> List[Document]:
    doc = DocxDocument(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return [Document(page_content=text, metadata={"source": path})]

def load_txt(path: str) -> List[Document]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": path})]

def load_file(path: str) -> List[Document]:
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.pdf':
        return load_pdf(path)
    elif ext in {'.docx', '.doc'}:
        return load_docx(path)
    elif ext in {'.txt', '.md'}:
        return load_txt(path)
    return []

def load_documents_from_folder(folder_path: str) -> List[Document]:
    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[-1].lower() in SUPPORTED_EXTENSIONS:
                try:
                    docs.extend(load_file(os.path.join(root, file)))
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    return docs

def build_vectorstore(documents: List[Document]):
    # Chunking
    start_chunk = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    end_chunk = time.perf_counter()
    print(f"⏱ Chunking & Splitting: {end_chunk - start_chunk:.2f} seconds")

    # Embedding
    start_embed = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    end_embed = time.perf_counter()
    print(f"⏱ Embedding model load: {end_embed - start_embed:.2f} seconds")

    # Vectorization
    start_vector = time.perf_counter()
    db = FAISS.from_documents(chunks, embeddings)
    end_vector = time.perf_counter()
    print(f"⏱ Vectorization (FAISS build): {end_vector - start_vector:.2f} seconds")

    return db

# ----------- Global State -----------
db = None
memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=True)

# ----------- RAG Logic -----------
def ques_responses(question, history, system_prompt, token_limit):
    global db
    if db is None:
        return "❌ No documents loaded."

    start_retrieval = time.perf_counter()
    retriever_docs = db.similarity_search(question, k=3)
    end_retrieval = time.perf_counter()
    print(f"⏱ Retrieval: {end_retrieval - start_retrieval:.2f} seconds")

    context = "\n".join([doc.page_content for doc in retriever_docs])
    instruction = f"""
        1. Answer the question: {question} based on the provided context.
        2. If you do not find any relevant info, respond with "Sorry, I don't know."
        3. Summarize if asked.
        4. Explain simply if asked.
    """

    template = """
    {system_prompt}
    {context}
    {instruction}
    Conversation history:
    {history}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["system_prompt", "context", "instruction", "question", "history"],
        template=template,
    )

    llm = init_chat_model("llama3-8b-8192", model_provider="groq")
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    return chain.predict(
        question=question,
        context=context,
        instruction=instruction,
        system_prompt=system_prompt,
    )

# ----------- Load at Startup -----------
def initialize_db():
    global db
    print(f"📁 Loading documents from: {HARDCODED_FOLDER_PATH}")

    start_loader = time.perf_counter()
    docs = load_documents_from_folder(HARDCODED_FOLDER_PATH)
    end_loader = time.perf_counter()
    print(f"⏱ Data Loader: {end_loader - start_loader:.2f} seconds")

    if not docs:
        raise ValueError("No supported files found.")
    db = build_vectorstore(docs)
    print(f"✅ {len(docs)} documents loaded.")

initialize_db()

# ----------- Gradio Chat UI -----------
gr.ChatInterface(
    fn=ques_responses,
    additional_inputs=[
        gr.Textbox("You are Meera, an assistant AI chatbot.", label="System Prompt"),
        gr.Slider(10, 300, step=10, value=100, label="Max Tokens")
    ],
    title="💬 Ask Meera",
    description=f"Chat with documents from: `{HARDCODED_FOLDER_PATH}`",
    theme="soft",
).queue().launch()



"""
📁 Loading documents from: ./Insurance PDFs
⏱ Data Loader: 0.39 seconds
⏱ Chunking & Splitting: 0.01 seconds
⏱ Embedding model load: 26.83 seconds
⏱ Vectorization (FAISS build): 51.98 seconds

✅ 25 documents loaded.

⏱ Retrieval: 1.32 seconds

⏱ Retrieval: 0.14 seconds
⏱ Retrieval: 0.46 seconds

"""