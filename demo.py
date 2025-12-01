import os
import shutil
import logging
from datetime import datetime
import warnings
import gradio as gr
from pathlib import Path
import hashlib
import re

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

# LLM Configuration - FULLY OPTIMIZED WITH STOP TOKENS
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.05,
    n_threads=8,
    max_tokens=256,
    n_ctx=4096,
    verbose=False,
    repeat_penalty=1.15,
    top_p=0.85,
    top_k=30,
    stop=[
        "\n\nHuman:",
        "\n\nQuestion:",
        "\n\nQ:",
        "\nLabel",
        "* [response]:",
        "\n\n\n",
        "---",
        "Assistant:",
        "\nUser:",
        "\n\n#"
    ]
)

def log_event(event):
    """Log events with timestamp"""
    try:
        with open(ACCESS_LOG, "a", encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} - {event}\n")
    except Exception as e:
        logger.error(f"Error logging event: {e}")

# Improved clean function (keeps your original behavior plus fixes contradictory suffixes)
def clean_llm_response(response: str) -> str:
    """Clean LLM artifacts and force a single short line answer."""
    if not response:
        return response

    resp = response.strip()

    # Remove trailing structural artifacts
    for pat in [
        '\n\nHuman:', '\n\nQuestion:', '\n\nQ:', '\nLabel',
        '* [response]:', '\n---\n', 'Assistant:', '\n\n\n', '\nUser:'
    ]:
        if pat in resp:
            resp = resp.split(pat)[0].strip()

    # If model appended the 'context' refusal in same output, standardize:
    if 'The context does not provide this information' in resp:
        # If there is a quoted answer before that, extract it and prefer it
        m = re.search(r'(["\'])(.+?)\1', resp)
        if m:
            resp = m.group(0).strip()
        else:
            resp = "The context does not provide this information."

    # Keep only first non-empty line
    lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    if not lines:
        return "Unable to generate a valid response."
    first = lines[0]

    # Limit to up to two sentences
    sentences = re.split(r'(?<=[.!?])\s+', first)
    if len(sentences) > 2:
        first = ' '.join(sentences[:2]).strip()
        if first and first[-1] not in '.!?':
            first += '.'

    # Collapse obvious exact repetition
    if len(first) > 40:
        half = len(first) // 2
        if first[:half].strip() == first[-half:].strip():
            first = first[:half].strip()

    # Safety: if result is too short / nonsense, return refusal
    if len(re.sub(r'[^A-Za-z0-9]', '', first)) < 5:
        return "The context does not provide this information."

    return first

def post_process_answer(ans: str) -> str:
    """Guarantee a final one-line 'According to ...' style response or the refusal."""
    if not ans:
        return "The context does not provide this information."

    # If it's the explicit refusal, keep it exactly
    if ans.strip() == "The context does not provide this information.":
        return ans

    # Try to find a section reference like "Section 17" or "Sec. 17" etc.
    m = re.search(r'(Section|Sec(?:tion)?|S\.?)\s*\.?\s*([0-9]+(?:-[0-9]+)?)', ans, re.IGNORECASE)
    if m:
        sec = m.group(2)
        q = re.search(r'(["\'])(.+?)\1', ans)
        if q:
            quote = q.group(2)
            # Ensure short quote
            sentence = re.split(r'(?<=[.!?])\s+', quote)[0]
            return f'According to Section {sec}: "{sentence.strip()}"'
        # otherwise take first sentence of ans
        sentence = re.split(r'(?<=[.!?])\s+', ans)[0]
        return f'According to Section {sec}: "{sentence.strip()}"'

    sentence = re.split(r'(?<=[.!?])\s+', ans)[0]
    return sentence.strip()

# Compatibility helper for retriever API differences (same as yours)
def safe_retrieve_documents(retriever, query, **kwargs):
    """
    Backwards-compatible retriever caller.

    Tries common method names and invocation shapes. If run_manager is required,
    supplies run_manager=None. Returns list of documents on success or raises
    AttributeError with a helpful message.
    """
    method_candidates = [
        "get_relevant_documents",
        "_get_relevant_documents",
        "retrieve",
        "_retrieve",
        "get_documents",
        "get_relevant",
    ]

    last_exc = None
    for name in method_candidates:
        method = getattr(retriever, name, None)
        if not callable(method):
            continue

        try:
            return method(query, **kwargs)
        except TypeError as e:
            last_exc = e
            err_msg = str(e).lower()

            if "run_manager" in err_msg or "required keyword-only argument: 'run_manager'" in err_msg:
                try:
                    return method(query, run_manager=None, **kwargs)
                except Exception as e2:
                    last_exc = e2

            # try alternative invocation shapes
            try:
                return method([query], **kwargs)
            except Exception as e2:
                last_exc = e2

            try:
                return method(**kwargs)
            except Exception as e2:
                last_exc = e2

        except Exception as e:
            last_exc = e

    raise AttributeError(
        f"No supported retrieve method found on retriever. Tried: {method_candidates}. "
        f"Last error: {repr(last_exc)}"
    )

# Robust chain invocation helper
def robust_invoke_chain(chain, inputs):
    """
    Attempt multiple common invocation styles for LangChain-like chain objects,
    normalize the output to a string, and return (raw_result, normalized_string).
    Raises RuntimeError on total failure.
    """
    logger.debug(f"robust_invoke_chain inputs keys: {list(inputs.keys()) if isinstance(inputs, dict) else 'non-dict'}")
    raw = None
    last_exc = None

    attempts = [
        lambda: chain.invoke(inputs) if hasattr(chain, "invoke") else None,
        lambda: chain.invoke(**inputs) if hasattr(chain, "invoke") and isinstance(inputs, dict) else None,
        lambda: chain.run(**inputs) if hasattr(chain, "run") and isinstance(inputs, dict) else None,
        lambda: chain.run(inputs) if hasattr(chain, "run") else None,
        lambda: chain.__call__(**inputs) if hasattr(chain, "__call__") and isinstance(inputs, dict) else None,
        lambda: chain.__call__(inputs) if hasattr(chain, "__call__") else None,
        lambda: chain(inputs) if callable(chain) else None,
    ]

    for attempt in attempts:
        try:
            candidate = attempt()
            if candidate is not None:
                raw = candidate
                break
        except TypeError as te:
            last_exc = te
            logger.debug(f"Invocation TypeError: {te}")
            continue
        except Exception as e:
            last_exc = e
            logger.debug(f"Invocation Exception: {repr(e)}")
            continue

    if raw is None:
        raise RuntimeError(f"Failed to invoke chain. Last error: {repr(last_exc)}")

    normalized = ""
    try:
        if isinstance(raw, dict):
            normalized = (
                raw.get("answer")
                or raw.get("output_text")
                or raw.get("text")
                or raw.get("result")
                or raw.get("response")
                or raw.get("output")
                or next((v for v in raw.values() if isinstance(v, str)), None)
                or str(raw)
            )
        elif isinstance(raw, str):
            normalized = raw
        else:
            normalized = str(raw)
    except Exception as e:
        logger.debug(f"Normalization error: {e}")
        normalized = str(raw)

    logger.info(f"Chain raw output type: {type(raw)}, normalized_len: {len(normalized) if normalized else 0}")
    return raw, normalized or ""

# Embeddings Model
embeddings = HuggingFaceEmbeddings(model_name="./Embeddings")

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
    """Create retriever chain with optimized prompts (prompt MUST accept {context})."""
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2}
    )

    # System prompt stays static (do NOT include {context} here)
    contextualize_q_system_prompt = (
        "Given chat history and a question, reformulate the question as a standalone query. "
        "Do NOT answer the question here."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Keep system instructions here; do NOT put {context} in the system text.
    qa_system_prompt = """
    You are a helpful assistant that answers questions strictly from the provided context.
    Give factual and to the point answers. If not found, say "The context does not provide this information."
    """

    # THIS human message MUST include {context} and {input}
    # The combiner will substitute {context} with the retrieved chunks.
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "Context:\n{context}\n\nQuestion: {input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def extract_answer_fallback(query: str, vector_store, llm) -> str:
    """Direct extraction with ultra-simple prompt"""
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        try:
            docs = retriever._get_relevant_documents(query, run_manager=None)
        except Exception as e:
            try:
                docs = retriever.get_relevant_documents(query)
            except Exception as e2:
                logger.error(f"Retriever API mismatch both attempts: {e} / {e2}")
                return "Error: Retriever API mismatch. Check LangChain / vectorstore versions."

        if not docs:
            return "No relevant documents found."

        # Very short context
        context = "\n\n".join([
            f"[Page {doc.metadata.get('page', '?')}] {doc.page_content[:400]}"
            for doc in docs
        ])

        # ULTRA-SIMPLE prompt
        prompt = f"""
Use ONLY the text in the context to answer the question.

Context:
{context}

Question: {query}

Provide a short answer (1–3 sentences). If the answer is not in the context, say so.
Answer:
"""
        response = llm.invoke(prompt)
        return clean_llm_response(response)

    except Exception as e:
        logger.error(f"Fallback error: {e}")
        return f"Error: {str(e)}"

def scan_and_ingest_pdfs():
    """Scan PDF folder and ingest new or modified PDFs — safer ordering & batching"""
    global vector_store, retriever_chain

    processed_pdfs = get_processed_pdfs()

    try:
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    except Exception as e:
        logger.error(f"Error listing PDF folder: {e}")
        return f"Error accessing PDF folder: {str(e)}", list(processed_pdfs.keys())

    if not pdf_files:
        return "No PDF files found in the folder.", list(processed_pdfs.keys())

    pdfs_to_process = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        current_hash = calculate_file_hash(pdf_path)
        if current_hash is None:
            continue
        if pdf_file not in processed_pdfs or processed_pdfs[pdf_file] != current_hash:
            pdfs_to_process.append((pdf_file, current_hash))

    if not pdfs_to_process:
        return "No new or modified PDFs found.", list(processed_pdfs.keys())

    all_documents = []
    newly_processed = []
    errors = []

    for pdf_file, file_hash in pdfs_to_process:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            if not documents:
                errors.append(f"{pdf_file}: No content extracted")
                logger.warning(f"No content extracted from {pdf_file}")
                continue

            for doc in documents:
                doc.metadata['source'] = pdf_file
                doc.metadata['processed_date'] = datetime.now().isoformat()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                length_function=len,
                separators=[
                    "\n\n",                    # big breaks
                    "\n",                      # lines
                    ". ",                      # sentence ends
                ],
                keep_separator=True
            )

            texts = text_splitter.split_documents(documents)
            if texts:
                all_documents.extend(texts)
                newly_processed.append((pdf_file, file_hash))
                logger.info(f"Prepared {pdf_file} ({len(texts)} chunks)")
            else:
                errors.append(f"{pdf_file}: No chunks generated")
        except Exception as e:
            err = f"{pdf_file}: {str(e)}"
            errors.append(err)
            logger.error(f"Error processing {pdf_file}: {e}")

    if not all_documents:
        error_summary = "\n".join(errors) if errors else "No documents could be processed"
        return f"Error: {error_summary}", list(processed_pdfs.keys())

    try:
        new_vs = FAISS.from_documents(all_documents, embeddings)

        if vector_store is None:
            vector_store = new_vs
            logger.info("Created new vector store (initial)")
        else:
            vector_store.merge_from(new_vs)
            logger.info("Merged new documents into existing vector store")

        vector_store.save_local(DB_FAISS_PATH)

        retriever_chain = create_retriever_chain(vector_store)

        for pdf_file, file_hash in newly_processed:
            mark_pdf_as_processed(pdf_file, file_hash)
            log_event(f"Successfully ingested: {pdf_file}")

        updated_processed = get_processed_pdfs()

        message = f"✅ Successfully ingested {len(newly_processed)} PDF(s): {', '.join([p for p,_ in newly_processed])}"
        if errors:
            message += f"\n⚠️ Errors: {', '.join(errors)}"
        return message, list(updated_processed.keys())

    except Exception as e:
        logger.error(f"Error creating/updating vector store: {e}")
        return f"Error updating vector store: {str(e)}", list(processed_pdfs.keys())

def initialize_from_folder():
    """Initialize vector store from existing PDFs on startup"""
    global vector_store, retriever_chain

    logger.info("Initializing from PDF folder...")

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

    try:
        pdf_files = [f for f in os.listdir(PDF_FOLDER)
                    if f.lower().endswith('.pdf')]
        if pdf_files:
            logger.info(f"Found {len(pdf_files)} PDFs, processing...")
            scan_and_ingest_pdfs()
    except Exception as e:
        logger.error(f"Error during initialization: {e}")

initialize_from_folder()

# ============= GRADIO APP =============

with gr.Blocks(theme=gr.themes.Soft(), title="Coast Guard Act Chatbot") as demo:
    gr.Markdown("""
    # ⚖️ Coast Guard Act Assistant
    ### Optimized for Concise, Accurate Answers
    """)

    gr.Markdown(f"""
    **Config:** Chunks=1200, Retrieval=4, MaxTokens=256 (with stop sequences)  
    **PDF Folder:** `{PDF_FOLDER}`
    """)

    # Authentication
    with gr.Row():
        with gr.Column(scale=1):
            username = gr.Textbox(label="Username", placeholder="Enter username")
            password = gr.Textbox(label="Password", type="password", placeholder="Enter password")
            login_btn = gr.Button("🔐 Login", variant="primary")
            auth_message = gr.Textbox(label="Status", interactive=False)

    auth_flag = gr.State(False)

    def authenticate(user, pwd):
        if user == "admin" and pwd == "secure_password_123":
            log_event(f"Login successful: {user}")
            return "✅ Login successful.", True
        else:
            log_event(f"Login failed: {user}")
            return "❌ Invalid credentials.", False

    login_btn.click(authenticate, [username, password], [auth_message, auth_flag])

    gr.Markdown("---")

    # PDF Management
    with gr.Row():
        scan_btn = gr.Button("🔄 Scan PDFs", variant="primary", size="lg")
        clear_db_btn = gr.Button("🗑️ Clear DB", variant="stop", size="lg")

    with gr.Row():
        ingest_status = gr.Textbox(label="📊 Status", interactive=False, lines=3)
        processed_list = gr.Textbox(
            label="📁 Processed",
            interactive=False,
            value="\n".join(get_processed_pdfs().keys()),
            lines=3
        )

    gr.Markdown("---")

    # Chat
    gr.Markdown("### 💬 Ask Questions")

    chatbot = gr.Chatbot(label="Q&A", height=400)

    with gr.Row():
        user_input = gr.Textbox(
            label="Question",
            placeholder="e.g., What is the punishment for Mutiny?",
            scale=4
        )
        send_btn = gr.Button("📤 Send", variant="primary", scale=1)

    processing_status = gr.Textbox(label="Status", interactive=False, value="Ready")

    with gr.Row():
        use_fallback = gr.Checkbox(label="Use Fallback (if verbose)", value=False)

    chat_history_state = gr.State([])

    # Handlers

    def handle_scan(is_auth):
        if not is_auth:
            return "❌ Login required", ""
        try:
            message, processed = scan_and_ingest_pdfs()
            return message, "\n".join(processed)
        except Exception as e:
            return f"❌ {e}", ""

    scan_btn.click(handle_scan, [auth_flag], [ingest_status, processed_list])

    def handle_clear_db(is_auth):
        global vector_store, retriever_chain

        if not is_auth:
            return "❌ Login required", ""

        try:
            if os.path.exists(DB_FAISS_PATH):
                shutil.rmtree(DB_FAISS_PATH)
                os.makedirs(DB_FAISS_PATH, exist_ok=True)
            if os.path.exists(PROCESSED_LOG):
                os.remove(PROCESSED_LOG)

            vector_store = None
            retriever_chain = None
            log_event("DB cleared")
            return "✅ Cleared", ""
        except Exception as e:
            return f"❌ {e}", ""

    clear_db_btn.click(handle_clear_db, [auth_flag], [ingest_status, processed_list])

    def handle_chat(message, chat_history, is_auth, use_manual):
        if not is_auth:
            yield [[message, "❌ Login required"]], chat_history, "Auth error"
            return

        if not message.strip():
            yield chat_history, chat_history, "Ready"
            return

        if not vector_store or not retriever_chain:
            yield [[message, "⚠️ Scan PDFs first"]], chat_history, "No docs"
            return

        chat_history.append([message, ""])

        try:
            if use_manual:
                yield chat_history, chat_history, "🔍 Fallback extraction..."
                response = extract_answer_fallback(message, vector_store, llm)
                response = post_process_answer(response)
            else:
                yield chat_history, chat_history, "🔍 Retrieving..."

                lc_history = []
                for human, ai in chat_history[:-1]:
                    if human:
                        lc_history.append(HumanMessage(content=human))
                    if ai:
                        lc_history.append(AIMessage(content=ai))

                yield chat_history, chat_history, "🤖 Generating..."

                # Use robust invoker and pass both 'input' and 'query' to satisfy templates expecting either
                try:
                    raw_result, normalized = robust_invoke_chain(
                        retriever_chain,
                        {"chat_history": lc_history, "input": message, "query": message}
                    )
                    logger.debug(f"robust_invoke_chain raw type: {type(raw_result)}")
                    response = clean_llm_response(normalized)
                    response = post_process_answer(response)
                except Exception as e:
                    logger.error(f"Chain invocation failed: {e}")
                    yield chat_history, chat_history, "🔄 Trying fallback..."
                    response = extract_answer_fallback(message, vector_store, llm)
                    response = post_process_answer(response)

                # Auto-fallback if still poor (very short or the explicit refusal)
                if not response or len(response) < 20 or response == "The context does not provide this information.":
                    yield chat_history, chat_history, "🔄 Trying fallback..."
                    response = extract_answer_fallback(message, vector_store, llm)
                    response = post_process_answer(response)

            if not response or len(response) < 10:
                response = "Unable to generate answer. Try enabling fallback."

            chat_history[-1][1] = response
            log_event(f"Q: {message[:50]}")
            yield chat_history, chat_history, "✅ Done"

        except Exception as e:
            logger.error(f"Chat error: {e}")
            try:
                yield chat_history, chat_history, "🔄 Error, trying fallback..."
                fallback = extract_answer_fallback(message, vector_store, llm)
                chat_history[-1][1] = post_process_answer(fallback)
                yield chat_history, chat_history, "✅ Done (fallback)"
            except Exception as inner:
                logger.error(f"Fallback also failed: {inner}")
                chat_history[-1][1] = f"❌ Error: {str(e)}"
                yield chat_history, chat_history, "Error"

    send_btn.click(
        handle_chat,
        [user_input, chatbot, auth_flag, use_fallback],
        [chatbot, chat_history_state, processing_status]
    ).then(lambda: "", None, user_input)

    user_input.submit(
        handle_chat,
        [user_input, chatbot, auth_flag, use_fallback],
        [chatbot, chat_history_state, processing_status]
    ).then(lambda: "", None, user_input)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, debug=True, share=False)
