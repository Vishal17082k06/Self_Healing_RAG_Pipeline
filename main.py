import os
import time
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

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

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Config values pulled out so they're loggable, not buried as magic numbers
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-ada-002"  # default OpenAIEmbeddings model
LLM_MODEL = "gpt-3.5-turbo"

def build_vector_db(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, force_rebuild=False):
    if os.path.exists("./chroma_db") and not force_rebuild:
        print("Loading existing ChromaDB...")
        return Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

    if force_rebuild and os.path.exists("./chroma_db"):
        import shutil
        print(f"Rebuilding ChromaDB (chunk_size={chunk_size}, overlap={chunk_overlap})...")
        shutil.rmtree("./chroma_db")

    with mlflow.start_run(run_name="ingestion"):
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("embedding_model", EMBEDDING_MODEL)
        mlflow.log_param("triggered_by", "self_healing" if force_rebuild else "initial_build")

        start = time.time()
        loader = TextLoader("data/event_schedule.md")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory="./chroma_db"
        )
        ingestion_time = time.time() - start

        mlflow.log_metric("num_chunks", len(chunks))
        mlflow.log_metric("ingestion_time_seconds", ingestion_time)

    return vectorstore

vectorstore = build_vector_db()
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

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

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        with mlflow.start_run(run_name="query", nested=False):
            mlflow.log_param("question", request.question)

            start = time.time()
            retrieved_docs = retriever.invoke(request.question)
            retrieval_time = time.time() - start

            context = format_docs(retrieved_docs)

            start = time.time()
            response = (prompt | llm | StrOutputParser()).invoke(
                {"context": context, "question": request.question}
            )
            generation_time = time.time() - start

            mlflow.log_metric("num_docs_retrieved", len(retrieved_docs))
            mlflow.log_metric("retrieval_time_seconds", retrieval_time)
            mlflow.log_metric("generation_time_seconds", generation_time)
            mlflow.log_metric("total_response_time_seconds", retrieval_time + generation_time)

        return {
            "answer": response,
            "contexts": [doc.page_content for doc in retrieved_docs] 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))