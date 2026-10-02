from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv

from document_processor import process_document
from vector_store import VectorStore
from llm_service import generate_answer

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI-Powered Document Question-Answering RAG API")

# Initialize Vector Store (will be instantiated once on startup)
vector_store = None

@app.on_event("startup")
def startup_event():
    global vector_store
    # We initialize the vector store once during startup so it doesn't reload the DB continuously
    vector_store = VectorStore()

@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Ingests a document, extracts text, chunks it, embeds it, and stores it in the vector database.
    """
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized.")

    # 1. Validate File Extension
    allowed_extensions = [".txt", ".md", ".pdf"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Unsupported file format. Please upload .txt, .md, or .pdf"}
        )

    # 2. Extract Text and Chunk (document_processor)
    try:
        chunks = await process_document(file, file_ext)
    except ValueError as e:
        # e.g., Scanned PDF with no extractable text
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    # 3. Store in Vector Database (vector_store)
    try:
        vector_store.add_chunks(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing vectors: {str(e)}")

    return {"message": "File uploaded and indexed successfully."}

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_document(request: QueryRequest):
    """
    Accepts a question, performs semantic search to retrieve context, and queries the LLM.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized.")

    # Prevent querying if DB is empty
    if vector_store.is_empty():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet."
        )

    # 1. Semantic Search (vector_store)
    try:
        retrieved_chunks = vector_store.search(request.question, top_k=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving context: {str(e)}")

    # 2. LLM Generation (llm_service)
    try:
        answer = generate_answer(request.question, retrieved_chunks)
    except Exception as e:
        # If external LLM fails, we return a 502 Bad Gateway
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API failure: {str(e)}"
        )

    # 3. Format and Return
    return {
        "answer": answer,
        "sources": retrieved_chunks
    }

@app.get("/report")
async def get_report():
    """
    Returns hardcoded mock evaluation metrics.
    """
    return {
        "context_precision": 0.90,
        "faithfulness": 0.85,
        "system_status": "healthy"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
