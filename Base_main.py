import os
import time
import fitz
from docx import Document as DocxDocument
from typing import List, Dict, Tuple
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
import gradio as gr
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ================== CONFIG ==================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

HARDCODED_FOLDER_PATH = "./SK_Finance_SOP"
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.doc', '.md', '.csv'}
PINECONE_INDEX_NAME = "sk-finance-sop-index"

# ================== TIMING CLASS ==================
class ComponentTimer:
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start_timer(self, component_name: str):
        """Start timing a component"""
        self.start_times[component_name] = time.time()
        logger.info(f"⏱️ Starting {component_name}...")
    
    def end_timer(self, component_name: str) -> float:
        """End timing a component and return elapsed time"""
        if component_name in self.start_times:
            elapsed = time.time() - self.start_times[component_name]
            self.timings[component_name] = elapsed
            logger.info(f"✅ {component_name} completed in {elapsed:.2f} seconds")
            return elapsed
        return 0.0
    
    def get_timings(self) -> Dict[str, float]:
        """Get all component timings"""
        return self.timings
    
    def print_summary(self):
        """Print timing summary for all components"""
        logger.info("=" * 50)
        logger.info("📊 COMPONENT TIMING SUMMARY")
        logger.info("=" * 50)
        total_time = sum(self.timings.values())
        for component, timing in self.timings.items():
            percentage = (timing / total_time * 100) if total_time > 0 else 0
            logger.info(f"{component:25} | {timing:8.2f}s | {percentage:5.1f}%")
        logger.info("=" * 50)
        logger.info(f"Total Execution Time: {total_time:.2f} seconds")
        logger.info("=" * 50)

# Global timer instance
timer = ComponentTimer()

# ================== LOADERS ==================
def load_pdf(path: str) -> List[Document]:
    """Load PDF file and return list of documents"""
    try:
        doc = fitz.open(path)
        documents = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():  # Only add non-empty pages
                documents.append(Document(
                    page_content=text, 
                    metadata={"source": os.path.basename(path), "page": page_num + 1}
                ))
        doc.close()
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF {path}: {e}")
        return []

def load_docx(path: str) -> List[Document]:
    """Load DOCX file and return list of documents"""
    try:
        doc = DocxDocument(path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [Document(page_content=text, metadata={"source": os.path.basename(path)})]
    except Exception as e:
        logger.error(f"Error loading DOCX {path}: {e}")
        return []

def load_txt(path: str) -> List[Document]:
    """Load text file and return list of documents"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": os.path.basename(path)})]
    except Exception as e:
        logger.error(f"Error loading text file {path}: {e}")
        return []

def load_csv(path: str) -> List[Document]:
    """Load CSV file and return list of documents"""
    try:
        loader = CSVLoader(file_path=path)
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading CSV {path}: {e}")
        return []

def load_file(path: str) -> List[Document]:
    """Load file based on extension"""
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.pdf':
        return load_pdf(path)
    elif ext in {'.docx', '.doc'}:
        return load_docx(path)
    elif ext in {'.txt', '.md'}:
        return load_txt(path)
    elif ext == '.csv':
        return load_csv(path)
    return []

def load_documents_from_folder(folder_path: str) -> Tuple[List[Document], int]:
    """Load all documents from folder with timing"""
    timer.start_timer("Data Loader")
    
    docs = []
    file_count = 0
    
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
                    logger.info(f"    ✅ Loaded {len(file_docs)} document(s) from {file}")
                except Exception as e:
                    logger.error(f"    ❌ Error loading {file}: {e}")
    
    timer.end_timer("Data Loader")
    logger.info(f"📊 Total documents loaded: {len(docs)} from {file_count} files")
    return docs, file_count

# ================== VECTOR DB BUILD ==================
def build_vectorstore(documents: List[Document]) -> PineconeVectorStore:
    """Build vector store with timing for each component"""
    
    # Component 1: Chunking & Splitting
    timer.start_timer("Chunking & Splitting")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    timer.end_timer("Chunking & Splitting")
    logger.info(f"📝 Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Component 2: Embedding
    timer.start_timer("Embedding")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}  # Use CPU for compatibility
    )
    timer.end_timer("Embedding")
    
    # Component 3: Vector Indexing & Storing
    timer.start_timer("Vector Indexing & Storing")
    
    try:
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = pinecone.list_indexes()
        index_exists = PINECONE_INDEX_NAME in existing_indexes.names()
        
        if not index_exists:
            logger.info(f"🔨 Creating new Pinecone index: {PINECONE_INDEX_NAME}")
            pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info("⏳ Waiting for index to be ready...")
            while not pinecone.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
            logger.info("✅ Index is ready")
        else:
            logger.info(f"📋 Index '{PINECONE_INDEX_NAME}' already exists")
        
        # Store documents
        vectorstore = PineconeVectorStore.from_documents(
            chunks,
            embeddings,
            index_name=PINECONE_INDEX_NAME
        )
        
        timer.end_timer("Vector Indexing & Storing")
        logger.info(f"💾 Successfully stored {len(chunks)} chunks in vector database")
        
        return vectorstore
        
    except Exception as e:
        timer.end_timer("Vector Indexing & Storing")
        logger.error(f"❌ Error building vector store: {e}")
        raise

# ================== GLOBALS ==================
qa_chain = None
vectorstore = None

# ================== RAG LOGIC ==================
def initialize_rag_chain():
    """Initialize the RAG chain with comprehensive error handling"""
    global qa_chain, vectorstore
    
    try:
        logger.info("🚀 Initializing RAG Chain...")
        
        # Load documents
        docs, file_count = load_documents_from_folder(HARDCODED_FOLDER_PATH)
        
        if not docs:
            raise ValueError(f"No supported documents found in '{HARDCODED_FOLDER_PATH}'. Please add your files.")
        
        # Build vector store
        vectorstore = build_vectorstore(docs)
        
        # Create retriever for context retrieval
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        
        # Create memory for conversation tracking
        memory = ConversationBufferMemory(
            memory_key="history", 
            input_key="query", 
            return_messages=True
        )
        
        # Create prompt template for script following
        template = """
        {system_prompt}

        You are Rekha, a loan verification assistant from SK Finance. You MUST follow the verification script step-by-step.

        IMPORTANT RULES:
        1. Follow the script EXACTLY as written - no deviations or additions
        2. Speak ONE step at a time and wait for user response
        3. Use conversation history to track which step you're on
        4. If user gives expected response, move to next step
        5. If user asks unrelated questions, redirect back to script
        6. Always speak in the same language the user uses

        SCRIPT CONTEXT:
        {context}

        CONVERSATION HISTORY:
        {history}

        USER'S LATEST RESPONSE:
        {query}

        TASK: Based on the conversation history and script, determine the NEXT step to say.
        - If no history, start with Step 6.1 (greeting)
        - If user confirmed identity, move to Step 6.2
        - If user gave consent, move to Step 7.1
        - Continue following the script sequence

        YOUR NEXT SCRIPT LINE (ONLY the next line from the script, nothing else):
        """
        
        prompt = PromptTemplate(
            input_variables=["system_prompt", "context", "history", "query"],
            template=template,
        )
        
        # Create LLM
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            memory=memory,
            chain_type_kwargs={"prompt": prompt, "verbose": False}
        )
        
        # Print timing summary
        timer.print_summary()
        
        logger.info("✅ RAG Chain initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG chain: {e}")
        raise

# ================== SCRIPT FOLLOWING LOGIC ==================
class ScriptManager:
    """Manages script execution and conversation flow"""
    
    def __init__(self):
        self.script_steps = {
            "start": "नमस्कार, क्या मेरी बात [Guarantor Name] जी से हो रही है?",
            "intro": "मैं Rekha, SK Finance Limited से बात कर रही हूँ। यह कॉल [Applicant Name] द्वारा SK Finance में किए गए लोन आवेदन से संबंधित है। आपके दस्तावेज़ हमारे पास गारंटर के रूप में प्राप्त हुए हैं। क्या मैं आपसे दो मिनट बात कर सकती हूँ?",
            "recording": "धन्यवाद। अब मैं आपकी लोन से जुड़ी, कुछ जानकारी की पुष्टि करना चाहूँगी। आपकी यह कॉल ट्रेनिंग और क्वालिटी पर्पस के लिए रिकॉर्ड किया जाएगा।",
            "applicant_knowledge": "अब कृपया बताएं — क्या आप [Applicant Name] जी को जानते हैं?",
            "relationship": "आप [Applicant Name] जी को कैसे जानते हैं?",
            "dob": "कृपया अपना जन्म तिथि बताएं।",
            "father_name": "कृपया आप अपना पिता का नाम बताएं।",
            "documents": "क्या दस्तावेज़ आपके द्वारा ही प्रस्तुत किए गए हैं?",
            "final_info": "अंत में एक महत्वपूर्ण सूचना: अगर भविष्य में आपके पते, मोबाइल नंबर या ईमेल ID में कोई भी बदलाव होता है, तो कृपया हमें तुरंत सूचित करें। आप यह अपडेट हमारे टोल-फ्री नंबर एक आठ शून्य शून्य एक शून्य तीन नौ शून्य तीन नौ, WhatsApp, या निकटतम SK Finance शाखा में जाकर करवा सकते हैं।",
            "closing": "SK Finance Limited से जुड़ने के लिए धन्यवाद। आपका दिन शुभ हो।"
        }
        
        self.step_sequence = [
            "start", "intro", "recording", "applicant_knowledge", 
            "relationship", "dob", "father_name", "documents", 
            "final_info", "closing"
        ]
        
        self.current_step_index = 0
        self.conversation_progress = []
    
    def get_next_step(self, conversation_history: list, user_response: str) -> str:
        """Determine the next script step based on conversation history and user response"""
        
        # Update conversation progress
        self.conversation_progress.append({
            'step': self.step_sequence[self.current_step_index] if self.current_step_index < len(self.step_sequence) else 'closing',
            'user_response': user_response,
            'timestamp': time.time()
        })
        
        # Determine if we should move to next step
        if self._should_advance_step(user_response, conversation_history):
            self.current_step_index += 1
        
        # Ensure we don't go beyond the script
        if self.current_step_index >= len(self.step_sequence):
            self.current_step_index = len(self.step_sequence) - 1
        
        # Get current step
        current_step = self.step_sequence[self.current_step_index]
        response = self.script_steps[current_step]
        
        # Log progress
        logger.info(f"📝 Script Progress: Step {self.current_step_index + 1}/{len(self.step_sequence)} - {current_step}")
        logger.info(f"💬 User Response: {user_response[:100]}...")
        logger.info(f"🤖 Bot Response: {response[:100]}...")
        
        return response
    
    def _should_advance_step(self, user_response: str, history: list) -> bool:
        """Determine if we should advance to the next step based on user response"""
        
        user_response_lower = user_response.lower()
        
        # Positive responses that indicate we should move forward
        positive_indicators = [
            'yes', 'हाँ', 'हां', 'ठीक है', 'बिलकुल', 'सही', 'हूं', 'है',
            'जानता हूं', 'जानती हूं', 'पता है', 'मालूम है'
        ]
        
        # Negative responses that need special handling
        negative_indicators = ['no', 'नहीं', 'नही', 'नहीं है', 'नहीं हूं']
        
        # Check if user response indicates we should advance
        if any(indicator in user_response_lower for indicator in positive_indicators):
            return True
        
        # Special handling for negative responses
        if any(indicator in user_response_lower for indicator in negative_indicators):
            if self.current_step_index == 1:  # After intro
                return False  # Don't advance, handle rescheduling
            else:
                return True  # Advance for other negative responses
        
        # For other responses (like providing information), advance
        return True
    
    def get_conversation_summary(self) -> dict:
        """Get a summary of the conversation progress"""
        return {
            'current_step': self.step_sequence[self.current_step_index] if self.current_step_index < len(self.step_sequence) else 'closing',
            'step_number': self.current_step_index + 1,
            'total_steps': len(self.step_sequence),
            'progress_percentage': ((self.current_step_index + 1) / len(self.step_sequence)) * 100,
            'conversation_progress': self.conversation_progress
        }
    
    def reset_conversation(self):
        """Reset the conversation to start"""
        self.current_step_index = 0
        self.conversation_progress = []
        logger.info("🔄 Conversation reset to beginning")

# Global script manager
script_manager = ScriptManager()

# ================== FUNCTION UPDATED ==================
def ques_responses(question: str, history: list, system_prompt: str) -> str:
    """Handle question responses with script following logic"""
    global qa_chain, script_manager
    
    if qa_chain is None:
        return "❌ RAG chain is not initialized. Please check server logs."
    
    try:
        # Component 4: Retrieval
        timer.start_timer("Retrieval")
        
        # Handle special commands
        if question.lower() in ['reset', 'restart', 'start over', 'नया शुरू करें']:
            script_manager.reset_conversation()
            timer.end_timer("Retrieval")
            return "🔄 Conversation reset. Starting fresh verification process.\n\n" + script_manager.script_steps["start"]
        
        if question.lower() in ['progress', 'status', 'कहाँ हैं हम']:
            summary = script_manager.get_conversation_summary()
            timer.end_timer("Retrieval")
            return f"📊 **Conversation Progress:**\nStep {summary['step_number']}/{summary['total_steps']} ({summary['progress_percentage']:.1f}%)\nCurrent: {summary['current_step']}"
        
        if question.lower() in ['help', 'सहायता', 'मदद']:
            timer.end_timer("Retrieval")
            return get_help_text()
        
        # Get the next script step based on conversation flow
        response = script_manager.get_next_step(history, question)
        
        # Handle special cases for negative responses
        if any(word in question.lower() for word in ['no', 'नहीं', 'नही', 'नहीं है']):
            if script_manager.current_step_index == 1:  # After intro
                response = "हम आपकी सुविधा के अनुसार इस कॉल को दोबारा शेड्यूल कर सकते हैं। क्या आप कृपया बता सकते हैं, कि आपको किस समय कॉल करना सुविधाजनक रहेगा?"
        
        timer.end_timer("Retrieval")
        
        return response
        
    except Exception as e:
        timer.end_timer("Retrieval")
        logger.error(f"❌ Error in question response: {e}")
        return f"❌ Error processing your question: {str(e)}"

# ================== GRADIO INTERFACE FUNCTIONS ==================
def get_conversation_progress():
    """Get current conversation progress for display"""
    global script_manager
    if script_manager:
        summary = script_manager.get_conversation_summary()
        return f"**Step {summary['step_number']}/{summary['total_steps']}** - {summary['current_step'].replace('_', ' ').title()}"
    return "Initializing..."

def get_help_text():
    """Get help text for users"""
    return """
**🤖 How to use this verification assistant:**

1. **Start**: Type 'hello', 'hi', or 'नमस्ते' to begin
2. **Follow the script**: The agent will guide you through each verification step
3. **Respond naturally**: Answer questions as you would in a real call
4. **Special commands**:
   - Type 'progress' or 'status' to see where you are in the process
   - Type 'reset' or 'restart' to start over
   - Type 'help' to see this message again

**📋 Verification Steps:**
1. Greeting & Identity confirmation
2. Introduction & Purpose explanation  
3. Call recording disclosure
4. Applicant knowledge verification
5. Relationship verification
6. Date of birth verification
7. Father's name verification
8. Document submission verification
9. Final instructions
10. Call closing

**💡 Tips:**
- The agent follows a strict script and cannot deviate
- Each step must be completed before moving to the next
- You can respond in Hindi, English, or other supported languages
- The conversation is tracked step-by-step for quality assurance
"""

# ================== MAIN EXECUTION ==================
def main():
    """Main function to run the application"""
    try:
        # Initialize RAG chain
        initialize_rag_chain()
        
        # Initialize script manager with placeholder values
        global script_manager
        script_manager.script_steps = {
            "start": "नमस्कार, क्या मेरी बात [Guarantor Name] जी से हो रही है?",
            "intro": "मैं Rekha, SK Finance Limited से बात कर रही हूँ। यह कॉल [Applicant Name] द्वारा SK Finance में किए गए लोन आवेदन से संबंधित है। आपके दस्तावेज़ हमारे पास गारंटर के रूप में प्राप्त हुए हैं। क्या मैं आपसे दो मिनट बात कर सकती हूँ?",
            "recording": "धन्यवाद। अब मैं आपकी लोन से जुड़ी, कुछ जानकारी की पुष्टि करना चाहूँगी। आपकी यह कॉल ट्रेनिंग और क्वालिटी पर्पस के लिए रिकॉर्ड किया जाएगा।",
            "applicant_knowledge": "अब कृपया बताएं — क्या आप [Applicant Name] जी को जानते हैं?",
            "relationship": "आप [Applicant Name] जी को कैसे जानते हैं?",
            "dob": "कृपया अपना जन्म तिथि बताएं।",
            "father_name": "कृपया आप अपना पिता का नाम बताएं।",
            "documents": "क्या दस्तावेज़ आपके द्वारा ही प्रस्तुत किए गए हैं?",
            "final_info": "अंत में एक महत्वपूर्ण सूचना: अगर भविष्य में आपके पते, मोबाइल नंबर या ईमेल ID में कोई भी बदलाव होता है, तो कृपया हमें तुरंत सूचित करें। आप यह अपडेट हमारे टोल-फ्री नंबर एक आठ शून्य शून्य एक शून्य तीन नौ शून्य तीन नौ, WhatsApp, या निकटतम SK Finance शाखा में जाकर करवा सकते हैं।",
            "closing": "SK Finance Limited से जुड़ने के लिए धन्यवाद। आपका दिन शुभ हो।"
        }
        
        # Create and launch Gradio interface
        interface = gr.ChatInterface(
            fn=ques_responses,
            additional_inputs=[
                gr.Textbox(
                    """# IDENTITY
                    You are Rekha, a friendly and professional loan verification assistant from SK Finance.
                    
                    # CORE RULES
                    - You MUST follow the verification script step-by-step
                    - Speak ONE step at a time and wait for user response
                    - Use conversation history to track your place in the script
                    - If user gives expected response, move to next step
                    - If user asks unrelated questions, redirect back to script
                    - Always speak in the same language the user uses
                    
                    # SCRIPT FLOW
                    1. Greeting & Identity confirmation
                    2. Introduction & Purpose explanation
                    3. Call recording disclosure
                    4. Applicant knowledge verification
                    5. Relationship verification
                    6. Date of birth verification
                    7. Father's name verification
                    8. Document submission verification
                    9. Final instructions
                    10. Call closing""",
                    label="System Prompt",
                    lines=12
                ),
            ],
            title="💬 Rekha | SK Finance Verification Assistant",
            description=f"""
            🤖 **AI-Powered Loan Verification Assistant**
            
            This agent follows a strict verification script step-by-step, just like a real verification call.
            
            **🚀 Quick Start:**
            - Type 'hello' or 'hi' to begin the verification process
            - Type 'help' for detailed instructions
            - Type 'progress' to see your current step
            - Type 'reset' to start over
            
            **📋 What to expect:**
            The agent will guide you through 10 verification steps, asking one question at a time and waiting for your response before proceeding.
            
            **🌐 Language Support:**
            Respond in Hindi, English, or other supported languages - the agent will match your language preference.
            """,
            theme="soft",
            examples=[
                ["hello"],
                ["hi"],
                ["नमस्ते"],
                ["yes"],
                ["हाँ"],
                ["help"],
                ["progress"],
                ["reset"]
            ]
        )
        
        logger.info("🚀 Launching Gradio interface...")
        interface.queue().launch(share=False, show_error=True)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize the application: {e}")
        logger.error("Please check your environment variables and folder structure.")
        raise

if __name__ == "__main__":
    main()