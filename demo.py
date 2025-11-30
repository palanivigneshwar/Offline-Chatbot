import os
import shutil
import logging
from datetime import datetime
import warnings
import gradio as gr
from pathlib import Path
import hashlib

# Core message & prompt objects
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Community integrations
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp

# Chain constructors - FIXED IMPORTS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PDF_FOLDER = os.path.join(os.getcwd(), "pdf_uploads")
DB_FAISS_PATH = os.path.join(os.getcwd(), "vectorstores")
MODEL_PATH = r"models\models--microsoft--Phi-3-mini-4k-instruct-GGUF\snapshots\999f761fe19e26cf1a339a5ec5f9f201301cbb83\Phi-3-mini-4k-instruct-q4.gguf"
PROCESSED_LOG = os.path.join(os.getcwd(), "processed_pdfs.txt")
ACCESS_LOG = os.path.join(os.getcwd(), "access_log.txt")

# Create directories
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(DB_FAISS_PATH, exist_ok=True)

# LLM Configuration
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.1,
    n_threads=8,
    max_tokens=2048,
    n_ctx=4096,
    verbose=False,
    repeat_penalty=1.1,
    top_p=0.9,
    top_k=40
)

def log_event(event):
    """Log events with timestamp"""
    try:
        with open(ACCESS_LOG, "a", encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} - {event}\n")
    except Exception as e:
        logger.error(f"Error logging event: {e}")

# Embeddings Model
embeddings = HuggingFaceEmbeddings(
    model_name="./Embeddings",
    model_kwargs={"device": "cpu"}
)

# Load or initialize vector store
vector_store = None
if os.path.exists(os.path.join(DB_FAISS_PATH, "index.faiss")):
    try:
        vector_store = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")

retriever_chain = None

def get_processed_pdfs():
    """Get set of already processed PDFs with their hashes"""
    if os.path.exists(PROCESSED_LOG):
        try:
            with open(PROCESSED_LOG, "r", encoding='utf-8') as f:
                return {line.strip().split('|')[0]: line.strip().split('|')[1] 
                        for line in f.readlines() if '|' in line}
        except Exception as e:
            logger.error(f"Error reading processed PDFs: {e}")
            return {}
    return {}

def calculate_file_hash(filepath):
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {filepath}: {e}")
        return None

def mark_pdf_as_processed(pdf_name, file_hash):
    """Mark a PDF as processed with its hash"""
    try:
        with open(PROCESSED_LOG, "a", encoding='utf-8') as f:
            f.write(f"{pdf_name}|{file_hash}\n")
    except Exception as e:
        logger.error(f"Error marking PDF as processed: {e}")

def create_retriever_chain(vector_store):
    """Create retriever chain with history awareness"""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Contextualize question prompt for history-aware retrieval
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer generation prompt
    qa_system_prompt = """You are a defence knowledge assistant specialized in the Coast Guard Act and related documents. 
Answer questions ONLY and STRICTLY based on the provided context from the uploaded documents.

IMPORTANT RULES:
1. Use ONLY information from the context below
2. Do NOT use external knowledge or general information
3. If the answer is not in the context, say: "The information is not available in the uploaded documents."
4. Cite specific sections or page numbers when possible
5. Be concise but complete
6. If asked about something ambiguous, ask for clarification

Context from documents:
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Create document chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create full RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def scan_and_ingest_pdfs():
    """Scan PDF folder and ingest new or modified PDFs"""
    global vector_store, retriever_chain

    processed_pdfs = get_processed_pdfs()
    
    try:
        pdf_files = [f for f in os.listdir(PDF_FOLDER) 
                    if f.lower().endswith('.pdf')]
    except Exception as e:
        logger.error(f"Error listing PDF folder: {e}")
        return f"Error accessing PDF folder: {str(e)}", list(processed_pdfs.keys())

    if not pdf_files:
        return "No PDF files found in the folder.", list(processed_pdfs.keys())

    # Check for new or modified PDFs
    pdfs_to_process = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        current_hash = calculate_file_hash(pdf_path)
        
        if current_hash is None:
            continue
            
        # Check if new or modified
        if pdf_file not in processed_pdfs or processed_pdfs[pdf_file] != current_hash:
            pdfs_to_process.append((pdf_file, current_hash))

    if not pdfs_to_process:
        return "No new or modified PDFs found.", list(processed_pdfs.keys())

    # Process PDFs
    all_documents = []
    newly_processed = []
    errors = []

    for pdf_file, file_hash in pdfs_to_process:
        try:
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            if not documents:
                errors.append(f"{pdf_file}: No content extracted")
                continue

            # Add metadata
            for doc in documents:
                doc.metadata['source'] = pdf_file
                doc.metadata['processed_date'] = datetime.now().isoformat()

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            all_documents.extend(texts)

            mark_pdf_as_processed(pdf_file, file_hash)
            newly_processed.append(pdf_file)
            log_event(f"Successfully ingested: {pdf_file}")
            logger.info(f"Successfully processed: {pdf_file} ({len(texts)} chunks)")

        except Exception as e:
            error_msg = f"{pdf_file}: {str(e)}"
            errors.append(error_msg)
            logger.error(f"Error processing {pdf_file}: {e}")

    if not all_documents:
        error_summary = "\n".join(errors) if errors else "No documents could be processed"
        return f"Error: {error_summary}", list(processed_pdfs.keys())

    # Create or update vector store
    try:
        if vector_store is None:
            vector_store = FAISS.from_documents(all_documents, embeddings)
            logger.info("Created new vector store")
        else:
            # Add to existing vector store
            new_vectorstore = FAISS.from_documents(all_documents, embeddings)
            vector_store.merge_from(new_vectorstore)
            logger.info("Merged with existing vector store")

        # Save vector store
        vector_store.save_local(DB_FAISS_PATH)
        
        # Create retriever chain
        retriever_chain = create_retriever_chain(vector_store)

        message = f"✅ Successfully ingested {len(newly_processed)} PDF(s): {', '.join(newly_processed)}"
        if errors:
            message += f"\n⚠️ Errors: {', '.join(errors)}"

        # Get updated list
        updated_processed = get_processed_pdfs()
        return message, list(updated_processed.keys())

    except Exception as e:
        logger.error(f"Error creating/updating vector store: {e}")
        return f"Error updating vector store: {str(e)}", list(processed_pdfs.keys())

def initialize_from_folder():
    """Initialize vector store from existing PDFs on startup"""
    global vector_store, retriever_chain

    logger.info("Initializing from PDF folder...")

    # Check if vector store exists
    if os.path.exists(os.path.join(DB_FAISS_PATH, "index.faiss")):
        try:
            vector_store = FAISS.load_local(
                DB_FAISS_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            retriever_chain = create_retriever_chain(vector_store)
            logger.info("Loaded existing vector store")
            return
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")

    # If no vector store, process PDFs
    try:
        pdf_files = [f for f in os.listdir(PDF_FOLDER) 
                    if f.lower().endswith('.pdf')]
        if pdf_files:
            logger.info(f"Found {len(pdf_files)} PDFs, processing...")
            scan_and_ingest_pdfs()
    except Exception as e:
        logger.error(f"Error during initialization: {e}")

# Initialize on startup
initialize_from_folder()

# ============= GRADIO APP =============

with gr.Blocks(theme=gr.themes.Soft(), title="Medical Chatbot") as demo:
    gr.Markdown("""
    # 🏥 Medical Document Chatbot
    ### Coast Guard Act Knowledge Assistant
    """)
    
    gr.Markdown(f"""
    **PDF Upload Folder:** `{PDF_FOLDER}`  
    Place PDF files in the folder and click 'Scan for New PDFs'
    """)

    # Authentication Section
    with gr.Row():
        with gr.Column(scale=1):
            username = gr.Textbox(label="Username", placeholder="Enter username")
            password = gr.Textbox(label="Password", type="password", placeholder="Enter password")
            login_btn = gr.Button("🔐 Login", variant="primary")
            auth_message = gr.Textbox(label="Status", interactive=False)

    auth_flag = gr.State(False)

    def authenticate(user, pwd):
        # TODO: Replace with secure authentication
        if user == "admin" and pwd == "secure_password_123":
            log_event(f"Login successful: {user}")
            return "✅ Login successful.", True
        else:
            log_event(f"Login failed: {user}")
            return "❌ Invalid credentials.", False

    login_btn.click(
        authenticate,
        inputs=[username, password],
        outputs=[auth_message, auth_flag]
    )

    gr.Markdown("---")

    # PDF Management Section
    with gr.Row():
        scan_btn = gr.Button("🔄 Scan for New PDFs", variant="primary", size="lg")
        clear_db_btn = gr.Button("🗑️ Clear Database", variant="stop", size="lg")

    with gr.Row():
        with gr.Column():
            ingest_status = gr.Textbox(
                label="📊 Ingestion Status",
                interactive=False,
                lines=3
            )
        with gr.Column():
            processed_list = gr.Textbox(
                label="📁 Processed PDFs",
                interactive=False,
                value="\n".join(get_processed_pdfs().keys()),
                lines=3
            )

    gr.Markdown("---")

    # Chat Section
    gr.Markdown("### 💬 Chat with Documents")
    
    chatbot = gr.Chatbot(
        label="Conversation",
        height=400,
        show_label=True
    )
    
    with gr.Row():
        user_input = gr.Textbox(
            label="Your Question",
            placeholder="Ask about the Coast Guard Act or uploaded documents...",
            scale=4
        )
        send_btn = gr.Button("📤 Send", variant="primary", scale=1)
    
    processing_status = gr.Textbox(
        label="⚙️ Processing Status",
        interactive=False,
        value="Ready"
    )

    chat_history_state = gr.State([])

    # ============= EVENT HANDLERS =============

    def handle_scan(is_auth):
        """Handle PDF scanning"""
        if not is_auth:
            return "❌ Unauthorized. Please login first.", ""
        try:
            message, processed = scan_and_ingest_pdfs()
            return message, "\n".join(processed)
        except Exception as e:
            logger.error(f"Scan error: {e}")
            return f"❌ Error during scan: {str(e)}", ""

    scan_btn.click(
        handle_scan,
        inputs=[auth_flag],
        outputs=[ingest_status, processed_list]
    )

    def handle_clear_db(is_auth):
        """Clear database and processed log"""
        global vector_store, retriever_chain
        
        if not is_auth:
            return "❌ Unauthorized. Please login first.", ""

        try:
            # Clear vector store
            if os.path.exists(DB_FAISS_PATH):
                shutil.rmtree(DB_FAISS_PATH)
                os.makedirs(DB_FAISS_PATH, exist_ok=True)

            # Clear processed log
            if os.path.exists(PROCESSED_LOG):
                os.remove(PROCESSED_LOG)

            vector_store = None
            retriever_chain = None
            
            log_event("Database cleared by user")
            logger.info("Database cleared successfully")

            return "✅ Database cleared. You can now re-scan PDFs.", ""
            
        except Exception as e:
            logger.error(f"Clear DB error: {e}")
            return f"❌ Error: {str(e)}", ""

    clear_db_btn.click(
        handle_clear_db,
        inputs=[auth_flag],
        outputs=[ingest_status, processed_list]
    )

    def handle_chat(message, chat_history, is_auth):
        """Handle chat interaction with streaming status"""
        if not is_auth:
            yield [[message, "❌ Please login to use the chat."]], chat_history, "Authentication required"
            return

        if not message or not message.strip():
            yield chat_history, chat_history, "Ready"
            return

        if vector_store is None or retriever_chain is None:
            yield [[message, "⚠️ No documents loaded. Please scan PDFs first."]], chat_history, "No documents"
            return

        # Add user message to history
        chat_history.append([message, ""])
        
        try:
            # Update status
            yield chat_history, chat_history, "🔍 Retrieving documents..."

            # Convert to LangChain format
            lc_history = []
            for human, ai in chat_history[:-1]:
                if human:
                    lc_history.append(HumanMessage(content=human))
                if ai:
                    lc_history.append(AIMessage(content=ai))

            # Invoke chain
            yield chat_history, chat_history, "🤖 Generating response..."
            
            result = retriever_chain.invoke({
                "chat_history": lc_history,
                "input": message
            })

            response = result.get("answer", "No answer generated.")
            
            # Update with response
            chat_history[-1][1] = response
            
            log_event(f"Chat query: {message[:50]}...")
            yield chat_history, chat_history, "✅ Complete"

        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_msg = f"❌ Error: {str(e)}"
            chat_history[-1][1] = error_msg
            yield chat_history, chat_history, "Error occurred"

    send_btn.click(
        handle_chat,
        inputs=[user_input, chatbot, auth_flag],
        outputs=[chatbot, chat_history_state, processing_status]
    ).then(
        lambda: "",  # Clear input after sending
        None,
        user_input
    )

    # Allow Enter key to send
    user_input.submit(
        handle_chat,
        inputs=[user_input, chatbot, auth_flag],
        outputs=[chatbot, chat_history_state, processing_status]
    ).then(
        lambda: "",
        None,
        user_input
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        debug=True,
        share=False
    )