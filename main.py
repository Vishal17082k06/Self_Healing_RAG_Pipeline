import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Initialize the FastAPI app
app = FastAPI(title="Event AI Assistant")

# Verify OpenAI API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.") 

# 2. Setup the RAG Pipeline (Runs when the server starts)
def initialize_vector_db():
    print("Loading event data into ChromaDB...")
    loader = TextLoader("data/event_schedule.md")
    documents = loader.load()
    
    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # Create the vector database on your local hard drive
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )
    return vectorstore

vectorstore = initialize_vector_db()
retriever = vectorstore.as_retriever()

# 3. Create the AI Brain using LCEL (LangChain Expression Language)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

system_prompt = """You are an assistant for a college event website. 
Use the following retrieved context to answer the user's question. 
If you don't know the answer, say that you don't know.

Context: {context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(system_prompt)

# Format documents for context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. The API Endpoint
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = rag_chain.invoke(request.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))