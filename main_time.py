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
from langchain_community.document_loaders import CSVLoader
import weaviate
# import weaviate.classes.init as wvc
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


# Connect to local open-source Weaviate
# client = weaviate.connect_to_local(
#     host="localhost",
#     port=8080,       # disables gRPC (REST only)
#     skip_init_checks=True
# )

# ================== CONFIG ==================
HARDCODED_FOLDER_PATH = "./SK_Finance_SOP"
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.doc', '.md', '.csv'}

# ================== LOADERS ==================

def load_pdf(path: str) -> List[Document]:
    try:
        doc = fitz.open(path)
        documents = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": os.path.basename(path), "page": page_num+1}))
        doc.close()
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF {path}: {e}")
        return []

def load_docx(path: str) -> List[Document]:
    try:
        doc = DocxDocument(path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [Document(page_content=text, metadata={"source": os.path.basename(path)})]
    except Exception as e:
        logger.error(f"Error loading DOCX {path}: {e}")
        return []

def load_txt(path: str) -> List[Document]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": os.path.basename(path)})]
    except Exception as e:
        logger.error(f"Error loading text file {path}: {e}")
        return []

def load_csv_generic(path: str) -> List[Document]:
    try:
        loader = CSVLoader(file_path=path)
        return loader.load()
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
        # SOP CSVs only (NOT user PII). User PII is loaded separately via UserDataManager.
        return load_csv_generic(path)
    return []

def load_documents_from_folder(folder_path: str) -> Tuple[List[Document], int]:
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
                    logger.info(f"    ✅ Loaded {len(file_docs)} document(s) from {file}")
                except Exception as e:
                    logger.error(f"    ❌ Error loading {file}: {e}")
    end_loader = time.perf_counter()
    logger.info(f"📊 Total documents loaded: {len(docs)} from {file_count} files")
    print(f"Time taken by Data Loader: {end_loader - start_loader:.2f} seconds")
    return docs, file_count


# ================== VECTORSTORE (OPEN SOURCE WEAVIATE) ==================

def build_vectorstore(documents: List[Document]) -> Weaviate:
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
    print(f"⏱ Vectorization (FAISS build): {end_vector - start_vector:.2f} seconds \n\n")
    logger.info(f"💾 Stored {len(chunks)} chunks in Weaviate (open source, local)")
    return vectorstore

# ================== GLOBALS ==================
qa_chain = None
vectorstore = None
llm_simple = None

# ================== RAG INIT ==================

def initialize_rag_chain():
    global qa_chain, vectorstore, llm_simple
    logger.info("🚀 Initializing RAG Chain...")
    docs, _ = load_documents_from_folder(HARDCODED_FOLDER_PATH)
    if not docs:
        raise ValueError(f"No supported documents found in '{HARDCODED_FOLDER_PATH}'. Please add your files.")
    vectorstore = build_vectorstore(docs)
    
    #
    start_retrieval = time.perf_counter()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    end_retrieval = time.perf_counter()
    print(f"⏱ Retrieval: {end_retrieval - start_retrieval:.2f} seconds")
    
    memory = ConversationBufferMemory(memory_key="history", input_key="query", return_messages=True)

    # Prompt enforces script-following without leaking PII.
    template = (
        """
        {system_prompt}

        ROLE: You are Rekha from SK Finance. Follow the scripted verification flow **step-by-step**.
        HARD RULES (PRIVACY-FIRST):
        - NEVER reveal or read out any value from internal records/CSVs.
        - When confirming user data, **echo only the user's own words** (e.g., "You said your DOB is '{{user_value}}'. Is this correct?").
        - Do NOT include names, DOB, phone numbers, addresses, or any other personal values unless the user has just said them, and only to confirm.
        - Keep messages short, single-step, and in the user's chosen language.

        CONTEXT FROM SOP (for agent guidance only):
        {context}

        HISTORY:
        {history}

        USER:
        {query}

        TASK: Output **only the next line** of the script (one sentence). No extra commentary.
        """
    )

    prompt = PromptTemplate(input_variables=["system_prompt", "context", "history", "query"], template=template)

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    llm_simple = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)  # faster, cheaper for short generations

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        memory=memory,
        chain_type_kwargs={"prompt": prompt, "verbose": False},
    )

    
    logger.info("✅ RAG Chain initialized successfully!")

# ================== USER DATA (PRIVATE, NO LEAK) ==================

class UserDataManager:
    """Loads user records from CSV files but **never** exposes values to the agent output.
    It only compares user utterances to internal records using semantic/fuzzy checks.
    """
    def __init__(self, folder: str = "./User_Data"):
        self.folder = folder
        self.user_data: Dict[str, Dict[str, str]] = {}
        self.current_user: Optional[str] = None
        self.load_user_data()

    def load_user_data(self):
        if not os.path.exists(self.folder):
            logger.warning(f"User data folder '{self.folder}' not found")
            return
        csv_files = [f for f in os.listdir(self.folder) if f.lower().endswith('.csv')]
        for csv_file in csv_files:
            fp = os.path.join(self.folder, csv_file)
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        cleaned = {(k or '').strip(): (v or '').strip() for k, v in row.items()}
                        user_id = cleaned.get('User ID') or cleaned.get('user_id') or cleaned.get('id') or f"{os.path.splitext(csv_file)[0]}_{i+1}"
                        self.user_data[user_id] = cleaned
                logger.info(f"✅ Loaded user records from {csv_file}")
            except Exception as e:
                logger.error(f"❌ Error loading {csv_file}: {e}")
        if self.user_data and not self.current_user:
            self.current_user = list(self.user_data.keys())[0]
            logger.info(f"🎯 Default user set: {self.current_user}")

    def set_current_user(self, user_id: str) -> bool:
        if user_id in self.user_data:
            self.current_user = user_id
            logger.info(f"👤 Switched to user: {user_id}")
            return True
        return False

    # ---------- Semantic normalization helpers (no external libs) ----------
    MONTHS = {
        'january': 1, 'jan': 1, 'jan.': 1,
        'february': 2, 'feb': 2, 'feb.': 2, 'febuary': 2, 'febuaray': 2, 'fev': 2, 'fib': 2,
        'march': 3, 'mar': 3, 'mar.': 3,
        'april': 4, 'apr': 4, 'apr.': 4,
        'may': 5,
        'june': 6, 'jun': 6, 'jun.': 6,
        'july': 7, 'jul': 7, 'jul.': 7,
        'august': 8, 'aug': 8, 'aug.': 8,
        'september': 9, 'sep': 9, 'sept': 9, 'sept.': 9,
        'october': 10, 'oct': 10, 'oct.': 10,
        'november': 11, 'nov': 11, 'nov.': 11,
        'december': 12, 'dec': 12, 'dec.': 12,
    }

    def _normalize_name(self, s: str) -> str:
        s = re.sub(r"[^a-zA-Z\s]", "", s or "").lower().strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def _normalize_relation(self, s: str) -> str:
        s = self._normalize_name(s)
        synonyms = {
            'father': ['father', 'dad', 'pita', 'pitaji', 'baba'],
            'mother': ['mother', 'mom', 'maa', 'mataji'],
            'brother': ['brother', 'bhai', 'bhrata'],
            'sister': ['sister', 'behen'],
            'spouse': ['wife', 'husband', 'pati', 'patni', 'spouse']
        }
        for canon, arr in synonyms.items():
            for a in arr:
                if a in s:
                    return canon
        return s

    def _normalize_date(self, s: str) -> Optional[Tuple[int, int, int]]:
        """Try to parse many noisy DOB formats like '12 feb 1990', '12-02-1990', '1990/2/12'."""
        if not s:
            return None
        s = s.replace(',', ' ').replace('\n', ' ').lower()
        s = re.sub(r"\s+", " ", s)
        # Replace textual months with numbers
        tokens = []
        for tok in re.split(r"[\s/\-]", s):
            if tok in self.MONTHS:
                tokens.append(str(self.MONTHS[tok]))
            else:
                tokens.append(tok)
        s2 = "-".join(tokens)
        # candidates: dd-mm-yyyy, yyyy-mm-dd, mm-dd-yyyy
        m = re.findall(r"(\d{1,4})-(\d{1,2})-(\d{1,4})", s2)
        for a, b, c in m:
            a, b, c = int(a), int(b), int(c)
            # Heuristic choose which is year: the 4-digit number >= 1900
            y, mth, d = None, None, None
            if a > 31 and a >= 1900:
                y, mth, d = a, b, c
            elif c > 31 and c >= 1900:
                y, mth, d = c, a, b
            else:
                # fallback assume middle is month if plausible
                if 1 <= b <= 12:
                    y, mth, d = (a if a >= 1900 else c), b, (c if a >= 1900 else a)
            if y and 1 <= mth <= 12 and 1 <= d <= 31:
                return (y, mth, d)
        return None

    def _ratio(self, a: str, b: str) -> float:
        # quick similarity without external libs
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()

    def compare_field(self, field: str, user_text: str) -> Tuple[bool, float]:
        """Semantic/fuzzy compare **without** exposing stored value."""
        if not self.current_user or self.current_user not in self.user_data:
            return False, 0.0
        record = self.user_data[self.current_user]
        expected_raw = record.get(field, "")
        if not expected_raw:
            return True, 0.5  # no record to compare against

        # Field-specific handling
        if 'dob' in field.lower():
            exp = self._normalize_date(expected_raw)
            got = self._normalize_date(user_text)
            if exp and got:
                match = (exp == got)
                return match, 1.0 if match else 0.0
            # if parsing failed, fall back to fuzzy
            return (self._ratio(expected_raw.lower(), user_text.lower()) > 0.8, 0.6)

        if 'relation' in field.lower():
            exp = self._normalize_relation(expected_raw)
            got = self._normalize_relation(user_text)
            return (exp == got or self._ratio(exp, got) > 0.85, 0.9)

        if "father" in field.lower():
            exp = self._normalize_name(expected_raw)
            got = self._normalize_name(user_text)
            return (self._ratio(exp, got) > 0.9, 0.9)

        # default name/short text
        exp = self._normalize_name(expected_raw)
        got = self._normalize_name(user_text)
        return (self._ratio(exp, got) > 0.9, 0.8)

# Global user data manager
user_data_manager = UserDataManager()

# ================== SCRIPT / FLOW (PRIVACY-SAFE) ==================

class ScriptManager:
    """Conversation state machine with **no PII substitution**.
    The agent asks user for info and only echoes the user's own words for confirmation.
    """
    def __init__(self):
        self.script_templates = {
            "Hindi": {
                "identify_user": "कृपया अपना यूज़र आईडी बताइए (उदाहरण: user_1)।",
                "start": "नमस्कार, क्या मैं {{Guarantor Name}} से बात कर रही हूँ?",
                "intro": "मैं Rekha, SK Finance Limited से बोल रही हूँ। यह कॉल आपकी गारंटर वेरिफ़िकेशन से संबंधित है। क्या मैं दो मिनट बात कर सकती हूँ?",
                "recording": "धन्यवाद। यह कॉल ट्रेनिंग और क्वालिटी पर्पज़ के लिए रिकॉर्ड किया जा रहा है।",
                "applicant_knowledge": "क्या आप {{Applicant Name}} को जानते हैं?",
                "relationship": "आपका {{Applicant Name}} से क्या संबंध है?",
                "dob": "कृपया अपनी जन्म तिथि बताइए।",
                "father_name": "कृपया अपने पिता का नाम बताइए।",
                "documents": "क्या दस्तावेज़ आपने स्वयं जमा किए थे?",
                "final_info": "भविष्य में यदि पते, मोबाइल नंबर या ईमेल में कोई बदलाव हो तो कृपया हमें सूचित करें।",
                "any_query": "क्या आपको कोई प्रश्न है जिसमें मैं मदद कर सकती हूँ?",
                "closing": "धन्यवाद! SK Finance Limited से जुड़ने के लिए आभार। आपका दिन शुभ हो।",
                "confirm_echo": "आपने कहा: '{value}'. क्या यह सही है?"
            },
            "English": {
                "identify_user": "Please provide your User ID (e.g., user_1).",
                "start": "Hello, am I speaking with the {Guarantor Name}?",
                "intro": "I am Rekha from SK Finance Limited. This call is for your guarantor verification. May I speak with you for two minutes?",
                "recording": "Thank you. This call will be recorded for training and quality purposes.",
                "applicant_knowledge": "Do you know the {Applicant Name}?",
                "relationship": "What is your relationship to the {Applicant Name}?",
                "dob": "Please tell me your date of birth.",
                "father_name": "Please tell me your father's name.",
                "documents": "Were the documents submitted by you personally?",
                "final_info": "If your address, mobile number, or email changes in the future, please inform us.",
                "any_query": "Do you have any question I can help with?",
                "closing": "Thank you for your time. Have a good day!",
                "confirm_echo": "You said: '{value}'. Is this correct?"
            }
        }
        self.step_sequence = [
            "identify_user", "start", "intro", "recording",
            "applicant_knowledge", "relationship", "dob", "father_name", "documents",
            "final_info", "any_query", "closing"
        ]
        self.current_step_index = 0
        self.conversation_progress = []
        self.current_language = "Hindi"
        self.language_switched = False
        self.verification_results = {"applicant_knowledge": None, "relationship": None, "dob": None, "father_name": None}

    def _t(self, key: str, **kwargs) -> str:
        text = self.script_templates[self.current_language].get(key, key)
        if kwargs:
            return text.format(**kwargs)
        return text

    def set_language(self, language: str) -> bool:
        if language in self.script_templates:
            self.current_language = language
            self.language_switched = True
            logger.info(f"🌐 Language switched to: {language}")
            return True
        return False

    def _is_language_switch(self, txt: str) -> bool:
        return any(w in (txt or '').lower() for w in ["english", "अंग्रेजी", "eng", "hindi", "हिंदी", "हिन्दी"]) 

    def _handle_language_switch(self, txt: str) -> str:
        low = (txt or '').lower()
        if any(w in low for w in ["english", "अंग्रेजी", "eng"]):
            self.set_language("English")
            return "🌐 Switched to English. " + self._t(self.step_sequence[self.current_step_index])
        if any(w in low for w in ["hindi", "हिंदी", "हिन्दी"]):
            self.set_language("Hindi")
            return "🌐 हिंदी में स्विच किया गया। " + self._t(self.step_sequence[self.current_step_index])
        return self._t("any_query")

    def get_next_step(self, conversation_history: list, user_response: str) -> str:
        if self._is_language_switch(user_response):
            return self._handle_language_switch(user_response)
        # Log progress
        self.conversation_progress.append({
            'step': self.step_sequence[self.current_step_index] if self.current_step_index < len(self.step_sequence) else 'closing',
            'user_response': user_response,
            'timestamp': time.time()
        })
        current_step = self.step_sequence[self.current_step_index]

        # Verification steps use semantic comparison but only echo user's words
        if current_step in ["applicant_knowledge", "relationship", "dob", "father_name"]:
            return self._handle_verification_step(current_step, user_response)
        else:
            return self._handle_regular_step(current_step, user_response)

    def _advance(self):
        self.current_step_index = min(self.current_step_index + 1, len(self.step_sequence) - 1)

    def _handle_regular_step(self, step: str, user_response: str) -> str:
        # identify_user special: set current user by id without disclosing any fields
        if step == "identify_user":
            user_response_clean = (user_response or "").strip().lower()
            available_users = list(user_data_manager.user_data.keys())
            matched_user = None
            for uid in available_users:
                if uid.lower() in user_response_clean:
                    matched_user = uid
                    break
            if matched_user and user_data_manager.set_current_user(matched_user):
                self._advance()
                return "✅ User set.\n\n" + self._t(self.step_sequence[self.current_step_index])
            helper = "\n\nType 'users' to list available IDs." if available_users else ""
            return self._t("identify_user") + helper

        # simple advance on positive/neutral responses
        self._advance()
        # If we just advanced to closing, also generate summary log internally
        if self.step_sequence[self.current_step_index] == "closing":
            self._log_conversation_summary()
        return self._t(self.step_sequence[self.current_step_index])

    def _handle_verification_step(self, step: str, user_response: str) -> str:
        # Compare semantics against CSV **privately**
        field_map = {
            "applicant_knowledge": "Applicant Name",  # loose check
            "relationship": "Guarantor Relation with Applicant",
            "dob": "Guarantor DOB",
            "father_name": "Guarantor Father's Name",
        }
        field = field_map[step]
        ok, conf = user_data_manager.compare_field(field, user_response or "")
        self.verification_results[step] = bool(ok)

        # Always echo the user's own words only
        echo = self._t("confirm_echo", value=(user_response or "").strip())

        # Move to next step regardless; confirmation question is part of the reply
        self._advance()
        next_line = self._t(self.step_sequence[self.current_step_index])
        return f"{echo}\n\n{next_line}"

    def get_conversation_summary(self) -> dict:
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
        self.current_step_index = 0
        self.conversation_progress = []
        logger.info("🔄 Conversation reset to beginning")

    def _log_conversation_summary(self) -> str:
        # Produce a private summary (no PII). Semantic match used already in verification flags.
        passed = all(self.verification_results.get(k) is True for k in self.verification_results.keys())
        status_line = "✅ Verification successful." if passed else "❌ Verification not fully successful."
        logger.info(f"📘 Conversation Summary: status={status_line}, results={self.verification_results}")
        return status_line

# Global script manager
script_manager = ScriptManager()

# ================== HANDLER ==================

def ques_responses(question: str, history: list, system_prompt: str) -> str:
    global qa_chain, script_manager
    if qa_chain is None:
        return "❌ RAG chain is not initialized. Please check server logs."

    # Shortcuts / commands
    ql = (question or "").strip().lower()
    if ql in ['reset', 'restart', 'start over', 'नया शुरू करें']:
        script_manager.reset_conversation()
        return "🔄 Conversation reset. Starting fresh verification process.\n\n" + script_manager._t('identify_user')

    if ql in ['progress', 'status', 'कहाँ हैं हम']:
        summary = script_manager.get_conversation_summary()
        return f"📊 **Conversation Progress:**\nStep {summary['step_number']}/{summary['total_steps']} ({summary['progress_percentage']:.1f}%)\nCurrent: {summary['current_step']}"

    if ql in ['help', 'सहायता', 'मदद']:
        return get_help_text()

    if ql in ['list users', 'users', 'available users']:
        users = list(user_data_manager.user_data.keys())
        current = user_data_manager.current_user
        user_list = "\n".join([f"- {u} {'(Current)' if u == current else ''}" for u in users]) if users else "(none)"
        return f"**👥 Available Users:**\n{user_list}\n\n**To switch users:** Type 'switch user [user_id]'"

    if ql.startswith('switch user '):
        uid = ql.replace('switch user ', '').strip()
        if user_data_manager.set_current_user(uid):
            script_manager.reset_conversation()
            return f"✅ Switched to user: {uid}\n🔄 Conversation reset for new user.\n\n" + script_manager._t('identify_user')
        else:
            users = list(user_data_manager.user_data.keys())
            return f"❌ User '{uid}' not found.\n\n**Available users:**\n" + "\n".join([f"- {u}" for u in users])

    # Default flow: compute next step output locally (no CSV values are ever spoken)
    return script_manager.get_next_step(history, question)

# ================== UI HELP ==================

def get_help_text():
    return (
        """
        **🤖 How to use this verification assistant (Privacy-First):**

        1. **Start**: Type 'hello', 'hi', or 'नमस्ते' to begin.
        2. **Follow the steps**: One question at a time; the agent never reads your records aloud.
        3. **Confirmations**: The agent will only echo *your* words to confirm (e.g., "You said: '12 Feb 1990'. Is this correct?").
        4. **Commands**:
           - 'progress' → show current step
           - 'reset' → restart
           - 'users' → list available user IDs (no personal data)
           - 'switch user <id>' → switch current user
        5. **Language**: Say 'English' or 'Hindi' anytime to switch.
        """
    )

# ================== MAIN ==================

def main():
    try:
        initialize_rag_chain()
        global script_manager
        interface = gr.ChatInterface(
            fn=ques_responses,
            additional_inputs=[
                gr.Textbox(
                    """# IDENTITY\nYou are Rekha, a friendly and professional loan verification assistant from SK Finance.\n\n# CORE RULES (PRIVACY-FIRST)\n- Never reveal or read any CSV/record values.\n- Confirm using only the user's own words (echo + "Is this correct?").\n- One short step at a time; follow the script strictly.\n- Use the current language (English/Hindi) and allow switching.\n\n# SCRIPT FLOW\n1. Greeting & Identity confirmation\n2. Introduction & Purpose\n3. Recording disclosure\n4. Applicant knowledge\n5. Relationship\n6. Date of birth\n7. Father's name\n8. Document submission\n9. Final instructions\n10. Ask if user has any query\n11. Closing\n""",
                    label="System Prompt",
                    lines=16,
                ),
            ],
            title="💬 Rekha | SK Finance Verification Assistant (Privacy-First)",
            description=(
                """
                🤖 **AI-Powered Loan Verification Assistant** (local Weaviate index for privacy)\n\n
                **🚀 Quick Start:**\n- Type 'hello' or 'hi' to begin\n- Type 'help' for instructions\n - Type 'progress' to see your current step\n- Type 'reset' to start over\n\n**👤 User Management:**\n- Type 'users' to list available user IDs\n- Type 'switch user [user_id]' to change users\n\n**🔏 Data Protection:**\n- No personal data from CSV is ever read aloud or shown in the UI\n- All comparisons are semantic and privacy-preserving\n- Vector search uses **local Weaviate** (no external vector DB)\n"""
            ),
            theme="soft",
            examples=[["hello"], ["hi"], ["नमस्ते"], ["yes"], ["हाँ"], ["help"], ["progress"], ["reset"], ["users"], ["switch user user_1"], ["English"], ["Hindi"]],
        )
        logger.info("🚀 Launching Gradio interface...")
        interface.queue().launch(share=False, show_error=True)
    except Exception as e:
        logger.error(f"❌ Failed to initialize the application: {e}")
        raise

if __name__ == "__main__":
    main()