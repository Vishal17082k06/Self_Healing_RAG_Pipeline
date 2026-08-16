import os

# Must be set before torch/onnxruntime get imported (via chromadb, sentence-transformers,
# mlflow's numpy/scipy stack, etc.) — on Windows, loading more than one library's bundled
# OpenMP runtime DLL in the same process crashes with "DLL initialization routine failed".
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# On Windows, importing mlflow before torch corrupts torch's DLL loading (its bundled
# numpy/scipy MKL DLLs conflict with torch's own) and crashes with "DLL initialization
# routine failed" the first time HuggingFaceEmbeddings pulls torch in later. Importing
# torch first, before mlflow, avoids the conflict.
import torch  # noqa: F401

import time
import shutil
import threading
import uuid
import mlflow
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from feedback_store import init_db, insert_feedback

load_dotenv()

from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

tracer_provider = register(
    project_name="SelfHealingRAG",
    endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces"),
    batch=True,
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


# --- MLflow setup ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("self-healing-rag")

app = FastAPI(title="Event AI Assistant")

# Validate API keys
groq_key = os.getenv("GROQ_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if not groq_key and not openai_key:
    raise ValueError("Either GROQ_API_KEY or OPENAI_API_KEY must be set in environment variables.")

# Config values pulled out so they're loggable, not buried as magic numbers
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, runs locally via HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
PRIMARY_LLM_PROVIDER = "groq"
PRIMARY_LLM_MODEL = "llama-3.3-70b-versatile"  # Groq's fast model
FALLBACK_LLM_MODEL = "gpt-3.5-turbo"  # OpenAI fallback

CHROMA_BASE_DIR = "./chroma_db"
CURRENT_VERSION_FILE = os.path.join(CHROMA_BASE_DIR, "current_version.txt")
MAX_INDEX_VERSIONS = 2
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

_retriever_lock = threading.Lock()

def get_llm():
    """Get LLM with Groq as primary and OpenAI as fallback"""
    if PRIMARY_LLM_PROVIDER == "groq" and groq_key:
        try:
            print(f"Using Groq with model: {PRIMARY_LLM_MODEL}")
            return ChatGroq(model=PRIMARY_LLM_MODEL, temperature=0, api_key=groq_key)
        except Exception as e:
            print(f"Groq initialization failed: {e}. Falling back to OpenAI...")
            if openai_key:
                print(f"Using OpenAI fallback with model: {FALLBACK_LLM_MODEL}")
                return ChatOpenAI(model=FALLBACK_LLM_MODEL, temperature=0)
            raise ValueError("Groq failed and no OpenAI key available for fallback")
    elif openai_key:
        print(f"Using OpenAI with model: {FALLBACK_LLM_MODEL}")
        return ChatOpenAI(model=FALLBACK_LLM_MODEL, temperature=0)
    else:
        raise ValueError("No valid API keys available")

# Sentinel for a pre-existing flat (non-versioned) ./chroma_db, e.g. one restored via `dvc pull`
# from before versioned hot-swapping was introduced. Its data is loaded in place, never migrated.
LEGACY_VERSION = "legacy"

def _version_dir(version):
    if version == LEGACY_VERSION:
        return CHROMA_BASE_DIR
    return os.path.join(CHROMA_BASE_DIR, version)

def _read_current_version():
    if os.path.exists(CURRENT_VERSION_FILE):
        with open(CURRENT_VERSION_FILE) as f:
            version = f.read().strip()
            if version:
                return version
    if os.path.exists(os.path.join(CHROMA_BASE_DIR, "chroma.sqlite3")):
        return LEGACY_VERSION
    return None

def _write_current_version(version):
    os.makedirs(CHROMA_BASE_DIR, exist_ok=True)
    with open(CURRENT_VERSION_FILE, "w") as f:
        f.write(version)

def _prune_old_versions(keep_version):
    if not os.path.exists(CHROMA_BASE_DIR):
        return
    versions = sorted(
        (d for d in os.listdir(CHROMA_BASE_DIR) if d.startswith("v_")),
        reverse=True,
    )
    for old in versions[MAX_INDEX_VERSIONS:]:
        if old == keep_version:
            continue
        shutil.rmtree(os.path.join(CHROMA_BASE_DIR, old), ignore_errors=True)

def build_vector_db(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, force_rebuild=False):
    """Builds (or loads) a versioned Chroma index under ./chroma_db/v_<timestamp>/.
    Never deletes the currently-serving version in place; the pointer file is only
    updated once a build succeeds, so a crash mid-build can't take the live index down."""
    current_version = _read_current_version()

    if current_version and not force_rebuild:
        version_dir = _version_dir(current_version)
        if os.path.exists(version_dir):
            print(f"Loading existing ChromaDB version {current_version}...")
            return Chroma(persist_directory=version_dir, embedding_function=get_embeddings())

    new_version = f"v_{int(time.time())}"
    version_dir = os.path.join(CHROMA_BASE_DIR, new_version)
    print(f"Building ChromaDB version {new_version} (chunk_size={chunk_size}, overlap={chunk_overlap})...")

    with mlflow.start_run(run_name="ingestion"):
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("embedding_model", EMBEDDING_MODEL)
        mlflow.log_param("triggered_by", "self_healing" if force_rebuild else "initial_build")
        mlflow.log_param("index_version", new_version)

        start = time.time()
        loader = TextLoader("data/event_schedule.md")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        new_vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=version_dir
        )
        ingestion_time = time.time() - start

        mlflow.log_metric("num_chunks", len(chunks))
        mlflow.log_metric("ingestion_time_seconds", ingestion_time)

    _write_current_version(new_version)
    _prune_old_versions(new_version)
    return new_vectorstore

def swap_retriever(new_vectorstore):
    """Atomically swaps the live retriever. In-flight requests keep using the
    reference they already hold; new requests see the swap immediately."""
    global vectorstore, retriever
    with _retriever_lock:
        vectorstore = new_vectorstore
        retriever = new_vectorstore.as_retriever()

vectorstore = build_vector_db()
retriever = vectorstore.as_retriever()
init_db()

llm = get_llm()

system_prompt = """You are an assistant for a college event website. 
Use the following retrieved context to answer the user's question. 
If you don't know the answer, say that you don't know.

Context: {context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(system_prompt)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: str
    tags: list[str] = []
    comment: str | None = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    try:
        with mlflow.start_run(run_name="query", nested=False):
            mlflow.log_param("question", request.question)
            mlflow.log_param("session_id", session_id)
            mlflow.log_param("message_id", message_id)
            mlflow.log_param("llm_provider", PRIMARY_LLM_PROVIDER)
            mlflow.log_param("llm_model", PRIMARY_LLM_MODEL if PRIMARY_LLM_PROVIDER == "groq" else FALLBACK_LLM_MODEL)

            start = time.time()
            retrieved_docs = retriever.invoke(request.question)
            retrieval_time = time.time() - start

            context = format_docs(retrieved_docs)

            start = time.time()
            try:
                # Try with current LLM
                response = (prompt | llm | StrOutputParser()).invoke(
                    {"context": context, "question": request.question}
                )
            except Exception as llm_error:
                # If Groq fails, try OpenAI fallback
                if PRIMARY_LLM_PROVIDER == "groq" and openai_key:
                    print(f"Primary LLM failed: {llm_error}. Attempting OpenAI fallback...")
                    mlflow.log_param("fallback_triggered", True)
                    fallback_llm = ChatOpenAI(model=FALLBACK_LLM_MODEL, temperature=0)
                    response = (prompt | fallback_llm | StrOutputParser()).invoke(
                        {"context": context, "question": request.question}
                    )
                    mlflow.log_param("actual_llm_used", "openai_fallback")
                else:
                    raise llm_error
                    
            generation_time = time.time() - start

            mlflow.log_metric("num_docs_retrieved", len(retrieved_docs))
            mlflow.log_metric("retrieval_time_seconds", retrieval_time)
            mlflow.log_metric("generation_time_seconds", generation_time)
            mlflow.log_metric("total_response_time_seconds", retrieval_time + generation_time)

        return {
            "answer": response,
            "contexts": [doc.page_content for doc in retrieved_docs],
            "session_id": session_id,
            "message_id": message_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    insert_feedback(
        session_id=request.session_id,
        message_id=request.message_id,
        question=None,
        rating=request.rating,
        tags=request.tags,
        comment=request.comment,
    )
    return {"status": "recorded"}

@app.post("/admin/reload-index")
async def reload_index(x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing admin token")

    current_version = _read_current_version()
    if not current_version:
        raise HTTPException(status_code=404, detail="No index version available")

    version_dir = _version_dir(current_version)
    if not os.path.exists(version_dir):
        raise HTTPException(status_code=404, detail=f"Index version {current_version} not found on disk")

    new_vectorstore = Chroma(persist_directory=version_dir, embedding_function=get_embeddings())
    swap_retriever(new_vectorstore)
    return {"status": "reloaded", "version": current_version}