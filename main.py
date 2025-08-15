import os
import csv
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
llm_simple = None

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
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        # Lightweight LLM for controlled step generation
        global llm_simple
        llm_simple = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
        
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

# ================== USER DATA MANAGEMENT ==================
class UserDataManager:
    """Manages user data loading and variable replacement"""
    
    def __init__(self, user_data_folder: str = "./User_Data"):
        self.user_data_folder = user_data_folder
        self.user_data = {}
        self.current_user = None
        self.load_user_data()
    
    def load_user_data(self):
        """Load all user data from CSV files.
        Supports a single CSV containing multiple users (one per row) with a 'User ID' column.
        """
        try:
            if not os.path.exists(self.user_data_folder):
                logger.warning(f"User data folder '{self.user_data_folder}' not found")
                return

            csv_files = [f for f in os.listdir(self.user_data_folder) if f.endswith('.csv')]
            logger.info(f"📁 Found {len(csv_files)} user data files")

            for csv_file in csv_files:
                file_path = os.path.join(self.user_data_folder, csv_file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        row_count = 0
                        for row in reader:
                            # Normalize keys and values (strip whitespace)
                            cleaned_row = { (k or '').strip(): (v or '').strip() for k, v in row.items() }
                            user_id = cleaned_row.get('User ID') or cleaned_row.get('user_id') or cleaned_row.get('id')
                            if not user_id:
                                # Fallback: synthesize a user id from filename and row number
                                user_id = f"{os.path.splitext(csv_file)[0]}_{row_count+1}"

                            self.user_data[user_id] = cleaned_row
                            row_count += 1

                        logger.info(f"✅ Loaded {row_count} user record(s) from {csv_file}")

                except Exception as e:
                    logger.error(f"❌ Error loading user data from {csv_file}: {e}")

            # Set first user as default if available and none selected yet
            if self.user_data and not self.current_user:
                first_user = list(self.user_data.keys())[0]
                self.set_current_user(first_user)
                logger.info(f"🎯 Set default user: {first_user}")

        except Exception as e:
            logger.error(f"❌ Error loading user data: {e}")
    
    def set_current_user(self, user_id: str):
        """Set the current user for the conversation"""
        if user_id in self.user_data:
            self.current_user = user_id
            logger.info(f"👤 Switched to user: {user_id} - {self.user_data[user_id].get('Applicant Name', 'Unknown')}")
            return True
        else:
            logger.error(f"❌ User {user_id} not found")
            return False
    
    def get_current_user_data(self) -> dict:
        """Get data for the current user"""
        if self.current_user and self.current_user in self.user_data:
            return self.user_data[self.current_user]
        return {}
    
    def get_all_users(self) -> list:
        """Get list of all available users"""
        return list(self.user_data.keys())
    
    def replace_variables_in_text(self, text: str) -> str:
        """Replace placeholder variables with actual user data"""
        if not self.current_user:
            return text
        
        user_data = self.user_data[self.current_user]
        
        # Define variable mappings
        variable_mappings = {
            '[User ID]': user_data.get('User ID', '[User ID]'),
            '[Applicant Name]': user_data.get('Applicant Name', '[Applicant Name]'),
            '[Guarantor Name]': user_data.get('Guarantor Name', '[Guarantor Name]'),
            '[Guarantor Relation]': user_data.get('Guarantor Relation with Applicant', '[Guarantor Relation]'),
            '[Guarantor DOB]': user_data.get('Guarantor DOB', '[Guarantor DOB]'),
            '[Guarantor Father Name]': user_data.get("Guarantor Father's Name", '[Guarantor Father Name]'),
            '[Guarantor Mobile]': user_data.get("Guarantor's Mobile", '[Guarantor Mobile]'),
            '[Guarantor Place]': user_data.get("Guarantor's Place", '[Guarantor Place]'),
            '[Guarantor Language]': user_data.get("Guarantor's Language", '[Guarantor Language]'),
            '[Applicant DOB]': user_data.get('DOB', '[Applicant DOB]'),
            '[Applicant Mobile]': user_data.get('Mobile', '[Applicant Mobile]'),
            '[Applicant Place]': user_data.get('Place', '[Applicant Place]')
        }
        
        # Replace variables
        for placeholder, value in variable_mappings.items():
            text = text.replace(placeholder, value)
        
        return text
    
    def verify_user_response(self, step: str, user_response: str) -> tuple[bool, str]:
        """Verify user response against loaded data and provide feedback"""
        if not self.current_user:
            return False, "No user data loaded"
        
        user_data = self.user_data[self.current_user]
        user_response_lower = user_response.lower().strip()
        
        if step == "dob":
            expected_dob = user_data.get('Guarantor DOB', '').lower().strip()
            if expected_dob and expected_dob in user_response_lower:
                return True, f"✅ Correct! Your date of birth is {user_data.get('Guarantor DOB')}. Thank you for confirming."
            elif expected_dob:
                return False, f"❌ That doesn't match our records. According to our data, your date of birth is {user_data.get('Guarantor DOB')}. Please confirm."
            else:
                return True, "Thank you for providing your date of birth."
        
        elif step == "father_name":
            expected_father = user_data.get("Guarantor Father's Name", '').lower().strip()
            if expected_father and expected_father in user_response_lower:
                return True, f"✅ Correct! Your father's name is {user_data.get("Guarantor Father's Name")}. Thank you for confirming."
            elif expected_father:
                return False, f"❌ That doesn't match our records. According to our data, your father's name is {user_data.get("Guarantor Father's Name")}. Please confirm."
            else:
                return True, "Thank you for providing your father's name."
        
        elif step == "applicant_knowledge":
            expected_applicant = user_data.get('Applicant Name', '').lower().strip()
            positive_indicators = [
                'yes', 'हाँ', 'हां', 'जी', 'ok', 'okay', 'haan', 'bilkul', 'sure', 'confirm'
            ]
            negative_indicators = ['no', 'नहीं', 'नहि']
            if any(ind in user_response_lower for ind in positive_indicators):
                return True, "✅ Thank you for confirming that you know the applicant."
            if any(ind in user_response_lower for ind in negative_indicators):
                return False, f"❌ Please confirm that you know {user_data.get('Applicant Name')}. Your documents were submitted as a guarantor for their loan application."
            if expected_applicant and expected_applicant in user_response_lower:
                return True, "✅ Thank you for confirming that you know the applicant."
            return True, "Thank you for confirming your knowledge of the applicant."
        
        elif step == "relationship":
            expected_relation = user_data.get('Guarantor Relation with Applicant', '').lower().strip()
            positive_indicators = ['yes', 'हाँ', 'हां', 'जी', 'ok', 'okay']
            # Accept if they provide any non-empty relation or a positive confirmation
            if any(ind in user_response_lower for ind in positive_indicators):
                return True, f"✅ Thank you. Noted your relationship with the applicant as {user_data.get('Guarantor Relation with Applicant', 'provided')}"
            if expected_relation and (expected_relation in user_response_lower):
                return True, f"✅ Correct! Your relationship with the applicant is {user_data.get('Guarantor Relation with Applicant')}. Thank you for confirming."
            if user_response_lower.strip():
                return True, "✅ Thank you for providing your relationship with the applicant."
            if expected_relation:
                return False, f"❌ That doesn't match our records. According to our data, your relationship with the applicant is {user_data.get('Guarantor Relation with Applicant')}. Please confirm."
            return True, "Thank you for providing your relationship with the applicant."
        
        # For other steps, just acknowledge the response
        return True, "Thank you for your response."

# Global user data manager
user_data_manager = UserDataManager()

# ================== SCRIPT FOLLOWING LOGIC ==================
class ScriptManager:
    """Manages script execution and conversation flow"""
    
    def __init__(self):
        # Multi-language script templates
        self.script_templates = {
            "Hindi": {
                "identify_user": "कृपया अपना User ID बताइए (उदाहरण: user_1)।",
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
            },
            "English": {
                "identify_user": "Please provide your User ID (e.g., user_1).",
                "start": "Hello, am I speaking with [Guarantor Name]?",
                "intro": "I am Rekha from SK Finance Limited. This call is related to a loan application made by [Applicant Name] at SK Finance. Your documents have been received with us as a guarantor. May I speak with you for two minutes?",
                "recording": "Thank you. Now I would like to verify some information related to your loan. This call will be recorded for training and quality purposes.",
                "applicant_knowledge": "Now please tell me — do you know [Applicant Name]?",
                "relationship": "How do you know [Applicant Name]?",
                "dob": "Please tell me your date of birth.",
                "father_name": "Please tell me your father's name.",
                "documents": "Were the documents submitted by you personally?",
                "final_info": "Finally, an important notice: If there are any changes in your address, mobile number, or email ID in the future, please inform us immediately. You can update this by calling our toll-free number 18001039039, WhatsApp, or by visiting the nearest SK Finance branch.",
                "closing": "Thank you for connecting with SK Finance Limited. Have a good day."
            }
        }
        
        self.step_sequence = [
            "identify_user", "start", "intro", "recording", "applicant_knowledge", 
            "relationship", "dob", "father_name", "documents", 
            "final_info", "closing"
        ]
        
        self.current_step_index = 0
        self.conversation_progress = []
        self.current_language = "Hindi"  # Default language
        self.language_switched = False
        self.verification_results = {"applicant_knowledge": None, "relationship": None, "dob": None, "father_name": None}
    
    def get_script_text(self, step: str) -> str:
        """Get script text in current language with variables replaced"""
        if step not in self.script_templates[self.current_language]:
            return f"Step {step} not found in {self.current_language}"
        
        script_text = self.script_templates[self.current_language][step]
        # Replace variables with actual user data
        return user_data_manager.replace_variables_in_text(script_text)
    
    def set_language(self, language: str):
        """Set the conversation language"""
        if language in self.script_templates:
            self.current_language = language
            self.language_switched = True
            logger.info(f"🌐 Language switched to: {language}")
            return True
        return False
    
    def get_next_step(self, conversation_history: list, user_response: str) -> str:
        """Determine the next script step based on conversation history and user response"""
        
        # Handle language switching
        if self._is_language_switch_request(user_response):
            return self._handle_language_switch(user_response)
        
        # Update conversation progress
        self.conversation_progress.append({
            'step': self.step_sequence[self.current_step_index] if self.current_step_index < len(self.step_sequence) else 'closing',
            'user_response': user_response,
            'timestamp': time.time()
        })
        
        # Get current step name
        current_step = self.step_sequence[self.current_step_index]
        
        # Handle verification steps
        if current_step in ["applicant_knowledge", "relationship", "dob", "father_name"]:
            return self._handle_verification_step(current_step, user_response)
        
        # Handle regular steps
        return self._handle_regular_step(current_step, user_response)
    
    def _is_language_switch_request(self, user_response: str) -> bool:
        """Check if user wants to switch language"""
        language_indicators = [
            "english", "अंग्रेजी", "eng", "english mein", "in english",
            "hindi", "हिंदी", "हिन्दी", "hindi mein", "in hindi"
        ]
        return any(indicator in user_response.lower() for indicator in language_indicators)
    
    def _handle_language_switch(self, user_response: str) -> str:
        """Handle language switching request"""
        user_response_lower = user_response.lower()
        
        if any(word in user_response_lower for word in ["english", "अंग्रेजी", "eng"]):
            self.set_language("English")
            return "🌐 Swit ched to English. " + self.get_script_text(self.step_sequence[self.current_step_index])
        elif any(word in user_response_lower for word in ["hindi", "हिंदी", "हिन्दी"]):
            self.set_language("Hindi")
            return "🌐 हिंदी में बदल गया। " + self.get_script_text(self.step_sequence[self.current_step_index])
        else:
            return "🌐 Please specify the language: 'English' or 'Hindi' / 'अंग्रेजी' या 'हिंदी'"
    
    def _handle_verification_step(self, step: str, user_response: str) -> str:
        """Handle verification steps with proper progression"""
        logger.info(f"🔍 Processing verification step: {step} with response: {user_response[:50]}...")
        
        # Verify user response
        is_correct, verification_message = user_data_manager.verify_user_response(step, user_response)
        # Track verification outcomes without revealing stored values
        if step in self.verification_results:
            self.verification_results[step] = bool(is_correct)
        
        if is_correct:
            # Move to next step
            old_index = self.current_step_index
            self.current_step_index += 1
            logger.info(f"✅ Verification passed for step {step}, moving from index {old_index} to {self.current_step_index}")
            
            # Get next step text
            if self.current_step_index < len(self.step_sequence):
                next_step = self.step_sequence[self.current_step_index]
                next_step_text = self._generate_output(next_step, user_response)
                logger.info(f"📝 Next step: {next_step}")
                return f"{verification_message}\n\n{next_step_text}"
            else:
                logger.info("🏁 Reached end of script")
                summary_text = self._log_conversation_summary()
                return f"{verification_message}\n\n{self.get_script_text('closing')}\n\n{summary_text}"
        else:
            # Stay on same step, show error and repeat question
            logger.info(f"❌ Verification failed for step {step}: {verification_message}")
            current_step_text = self._generate_output(step, user_response)
            return f"{verification_message}\n\n{current_step_text}"
    
    def _handle_regular_step(self, step: str, user_response: str) -> str:
        """Handle regular steps (non-verification)"""
        logger.info(f"🔍 Processing regular step: {step} with response: {user_response[:50]}...")
        
        # Special handling for user identification step
        if step == "identify_user":
            user_response_clean = (user_response or "").strip().lower()
            available_users = user_data_manager.get_all_users()
            matched_user = None
            for uid in available_users:
                if uid.lower() in user_response_clean:
                    matched_user = uid
                    break
            if matched_user:
                user_data_manager.set_current_user(matched_user)
                # Set language preference
                self._set_guarantor_language()
                # Move to next step and generate output
                self.current_step_index += 1
                next_step = self.step_sequence[self.current_step_index]
                next_line = self._generate_output(next_step, user_response)
                return f"✅ User '{matched_user}' set.\n\n{next_line}"
            else:
                prompt_line = self.get_script_text("identify_user")
                helper = "\n\nType 'users' to list available IDs." if available_users else ""
                return f"{prompt_line}{helper}"

        # Check if we should advance
        should_advance = self._should_advance_step(user_response)
        logger.info(f"🤔 Should advance step? {should_advance}")
        
        if should_advance:
            old_index = self.current_step_index
            self.current_step_index += 1
            logger.info(f"✅ Moving to next step from {step}, index {old_index} to {self.current_step_index}")
        
        # Ensure we don't go beyond the script
        if self.current_step_index >= len(self.step_sequence):
            self.current_step_index = len(self.step_sequence) - 1
            logger.info(f"⚠️ Adjusted step index to {self.current_step_index} (end of script)")
        
        # Get current step text
        current_step = self.step_sequence[self.current_step_index]
        response = self._generate_output(current_step, user_response)
        
        # Log progress
        logger.info(f"📝 Script Progress: Step {self.current_step_index + 1}/{len(self.step_sequence)} - {current_step}")
        logger.info(f"💬 User Response: {user_response[:100]}...")
        logger.info(f"🤖 Bot Response: {response[:100]}...")

        # If we are at closing, log summary
        if current_step == "closing":
            self._log_conversation_summary()
        
        return response
    
    def _should_advance_step(self, user_response: str) -> bool:
        """Determine if we should advance to the next step based on user response"""
        user_response_lower = user_response.lower()
        
        # Positive responses that indicate we should move forward
        positive_indicators = [
            'yes', 'हाँ', 'हां', 'ठीक है', 'बिलकुल', 'सही', 'हूं', 'है',
            'जानता हूं', 'जानती हूं', 'पता है', 'मालूम है', 'ok', 'okay'
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
            'conversation_progress': self.conversation_progress,
            'current_language': self.current_language,
            'verification_results': self.verification_results
        }
    
    def reset_conversation(self):
        """Reset the conversation to start"""
        self.current_step_index = 0
        self.conversation_progress = []
        # Reset to guarantor's preferred language
        self._set_guarantor_language()
        logger.info("🔄 Conversation reset to beginning")
    
    def _set_guarantor_language(self):
        """Set language based on guarantor's preference"""
        user_data = user_data_manager.get_current_user_data()
        if user_data:
            preferred_language = user_data.get("Guarantor's Language", "Hindi")
            if preferred_language in self.script_templates:
                self.current_language = preferred_language
                logger.info(f"🌐 Set language to guarantor's preference: {preferred_language}")

    def _generate_output(self, step: str, user_response: str) -> str:
        """Use a lightweight LLM to produce the next line while enforcing no data leakage"""
        base_line = self.get_script_text(step)
        if llm_simple is None:
            return base_line
        try:
            from langchain.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are 'Rekha', a compliant loan verification assistant. Speak exactly ONE short sentence for the next step. Do not reveal any stored user data such as date of birth, father's name, phone number, address, place, or any values that are not directly stated by the user. If the step asks for such information, only ask the user to provide it. Keep the message in the specified language."),
                ("human", "Language: {language}\nStep: {step}\nBase line to follow: {base_line}\nUser said: {user_response}\nOutput only the single next sentence.")
            ])
            chain = prompt | llm_simple
            return chain.invoke({
                "language": self.current_language,
                "step": step,
                "base_line": base_line,
                "user_response": user_response or ""
            }).content.strip()
        except Exception as e:
            logger.warning(f"LLM generation failed, falling back to base line. Error: {e}")
            return base_line

    def _log_conversation_summary(self) -> str:
        """Log a concise summary stating if verification is successful or not"""
        user_id = user_data_manager.current_user or "unknown"
        user_info = user_data_manager.get_current_user_data() or {}
        all_verification_steps = [k for k in self.verification_results.keys()]
        passed = all(self.verification_results.get(k) is True for k in all_verification_steps)
        summary = {
            "user_id": user_id,
            "applicant": user_info.get('Applicant Name', 'N/A'),
            "guarantor": user_info.get('Guarantor Name', 'N/A'),
            "language": self.current_language,
            "verification_results": self.verification_results,
            "verification_successful": passed,
            "steps_completed": self.current_step_index + 1,
            "total_steps": len(self.step_sequence)
        }
        logger.info(f"📘 Conversation Summary: {summary}")
        status_line = "✅ Verification successful." if passed else "❌ Verification not successful."
        return status_line
    
    def get_user_info_display(self) -> str:
        """Get formatted user information for display"""
        user_data = user_data_manager.get_current_user_data()
        if not user_data:
            return "No user data loaded"
        
        return f"""
        **👤 Current User Information:**
        - **Applicant:** {user_data.get('Applicant Name', 'N/A')}
        - **Guarantor:** {user_data.get('Guarantor Name', 'N/A')}
        - **Relation:** {user_data.get('Guarantor Relation with Applicant', 'N/A')}
        - **Guarantor DOB:** {user_data.get('Guarantor DOB', 'N/A')}
        - **Guarantor Father:** {user_data.get("Guarantor Father's Name", 'N/A')}
        - **Guarantor Mobile:** {user_data.get("Guarantor's Mobile", 'N/A')}
        - **Guarantor Place:** {user_data.get("Guarantor's Place", 'N/A')}
        - **Preferred Language:** {user_data.get("Guarantor's Language", 'N/A')}
        - **Current Language:** {self.current_language}
        """

# Global script manager
script_manager = ScriptManager()

# ================== FUNCTION UPDATED ==================
def ques_responses(question: str, history: list, system_prompt: str) -> str:
    """Handle question responses with script following logic"""
    global qa_chain, script_manager, user_data_manager
    
    if qa_chain is None:
        return "❌ RAG chain is not initialized. Please check server logs."

    try:
        # Component 4: Retrieval
        timer.start_timer("Retrieval")
        
        # Handle special commands
        if question.lower() in ['reset', 'restart', 'start over', 'नया शुरू करें']:
            script_manager.reset_conversation()
            timer.end_timer("Retrieval")
            return "🔄 Conversation reset. Starting fresh verification process.\n\n" + script_manager.get_script_text('identify_user')
        
        if question.lower() in ['progress', 'status', 'कहाँ हैं हम']:
            summary = script_manager.get_conversation_summary()
            timer.end_timer("Retrieval")
            return f"📊 **Conversation Progress:**\nStep {summary['step_number']}/{summary['total_steps']} ({summary['progress_percentage']:.1f}%)\nCurrent: {summary['current_step']}"
        
        if question.lower() in ['help', 'सहायता', 'मदद']:
            timer.end_timer("Retrieval")
            return get_help_text()
        
        # New user management commands
        if question.lower() in ['user info', 'userinfo', 'user details', 'user']:
            timer.end_timer("Retrieval")
            return script_manager.get_user_info_display()
        
        if question.lower().startswith('switch user '):
            user_id = question.lower().replace('switch user ', '').strip()
            if user_data_manager.set_current_user(user_id):
                script_manager.reset_conversation()
                timer.end_timer("Retrieval")
                return f"✅ Switched to user: {user_id}\n🔄 Conversation reset for new user.\n\n" + script_manager.get_script_text('identify_user')
            else:
                timer.end_timer("Retrieval")
                available_users = user_data_manager.get_all_users()
                return f"❌ User '{user_id}' not found.\n\n**Available users:**\n" + "\n".join([f"- {u}" for u in available_users])
        
        if question.lower() in ['list users', 'users', 'available users']:
            available_users = user_data_manager.get_all_users()
            current_user = user_data_manager.current_user
            timer.end_timer("Retrieval")
            user_list = "\n".join([f"- {u} {'(Current)' if u == current_user else ''}" for u in available_users])
            return f"**👥 Available Users:**\n{user_list}\n\n**To switch users:** Type 'switch user [user_id]'"
        
        # Debug command
        if question.lower() in ['debug', 'debug step', 'step debug']:
            timer.end_timer("Retrieval")
            summary = script_manager.get_conversation_summary()
            debug_info = f"""
            **🐛 DEBUG INFORMATION:**
            - Current Step Index: {script_manager.current_step_index}
            - Current Step: {summary['current_step']}
            - Step Number: {summary['step_number']}/{summary['total_steps']}
            - Progress: {summary['progress_percentage']:.1f}%
            - Current Language: {summary['current_language']}
            - Conversation History Length: {len(history)}

            **📝 Step Sequence:**
            {chr(10).join([f"{i+1}. {step} {'← CURRENT' if i == script_manager.current_step_index else ''}" for i, step in enumerate(script_manager.step_sequence)])}

            **💬 Last User Response:**
            "{question}"
            """
            return debug_info
        
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

    **👤 User Management Commands:**
    - Type 'user info' to see current user details
    - Type 'list users' to see all available users
    - Type 'switch user [user_id]' to change to a different user
    - Type 'users' to see available users

    **🌐 Language Support:**
    - The bot automatically starts in the guarantor's preferred language
    - Type 'English' or 'अंग्रेजी' to switch to English
    - Type 'Hindi' or 'हिंदी' to switch to Hindi
    - You can switch languages at any time during the conversation

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
    - User data is automatically loaded from CSV files
    - Script variables are replaced with actual user information
    - Language preference is automatically set based on guarantor's data
    """

# ================== MAIN EXECUTION ==================
def main():
    """Main function to run the application"""
    try:
        # Initialize RAG chain
        initialize_rag_chain()
        
        # Initialize script manager (no need to set script_steps manually anymore)
        global script_manager
        
        # Set the language based on guarantor's preference
        script_manager._set_guarantor_language()
        
        # Display loaded user information
        logger.info("📊 User Data Summary:")
        if user_data_manager.user_data:
            for user_id, user_info in user_data_manager.user_data.items():
                logger.info(f"  👤 {user_id}: {user_info.get('Applicant Name', 'Unknown')} -> {user_info.get('Guarantor Name', 'Unknown')}")
                logger.info(f"     🌐 Preferred Language: {user_info.get("Guarantor's Language", 'Hindi')}")
        else:
            logger.warning("  ⚠️ No user data loaded")
        
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
                    - Always speak in the guarantor's preferred language initially
                    - Allow language switching on user request
                    
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
                    10. Call closing
                    
                    # LANGUAGE HANDLING
                    - Start in guarantor's preferred language from CSV
                    - Support Hindi and English
                    - Allow language switching during conversation
                    - Maintain conversation flow in chosen language""",
                    label="System Prompt",
                    lines=16
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
            
            **👤 User Management:**
            - Type 'user info' to see current user details
            - Type 'list users' to see all available users
            - Type 'switch user [user_id]' to change users
            
            **🌐 Language Support:**
            - Automatically starts in guarantor's preferred language
            - Type 'English' or 'Hindi' to switch languages
            - Supports both Hindi and English throughout the conversation
            
            **📋 What to expect:**
            The agent will guide you through 10 verification steps, asking one question at a time and waiting for your response before proceeding.
            
            **📊 Data Integration:**
            User data is automatically loaded from CSV files and script variables are replaced with actual information.
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
                ["reset"],
                ["user info"],
                ["list users"],
                ["switch user user_1"],
                ["English"],
                ["Hindi"]
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