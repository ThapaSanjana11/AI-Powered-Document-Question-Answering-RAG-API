# AI-Powered Document Question-Answering RAG API

This project is a RESTful API that ingests private documents (.pdf, .txt, .md) and answers user questions using Retrieval-Augmented Generation (RAG). It bridges the gap between static Large Language Model knowledge and dynamic private data by acting as an "open-book" QA system.

## Features
- **Document Ingestion**: Upload `.txt`, `.md`, and `.pdf` files.
- **Semantic Search**: Text is chunked (1000 chars, 200 overlap), converted to 384-dimensional dense vectors using `sentence-transformers` (`all-MiniLM-L6-v2`), and stored in an in-memory `ChromaDB`.
- **RAG Inference**: Converts user questions to vectors, retrieves Top-3 matching context chunks, and prompts an LLM to generate factually grounded answers.
- **LLM Integration**: Uses Groq's high-speed API (`llama-3.1-8b-instant`).
- **Telemetry**: Exposes a mock `/report` endpoint for evaluation metrics.
- **Graceful Error Handling**: Manages LLM timeouts (502), unsupported files (400), and unreadable PDFs.

## Prerequisites
- Python 3.9+
- A [Groq API Key](https://console.groq.com/) (Free)

## Step-by-Step Installation

1. **Clone the repository (or extract the zip):**
   ```bash
   # Navigate to the project directory
   cd AI-Powered-Document-Question-Answering-RAG-API
   ```

2. **Initialize a Virtual Environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Downloading `sentence-transformers` may take a few minutes as it installs PyTorch).*

4. **Environment Variable Setup:**
   Create a `.env` file in the root directory based on `.env.example`:
   ```bash
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

## Running the Application

Start the FastAPI server using Uvicorn:
```bash
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`. You can also view the interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

## API Documentation & Example Usage

### 1. Upload a Document (`POST /upload`)
Uploads and indexes a file into the vector database.
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test_document.txt'
```
**Success Response (201 Created):**
```json
{
  "message": "File uploaded and indexed successfully."
}
```

### 2. Query the Knowledge Base (`POST /query`)
Ask a question based on the indexed documents.
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What were the Q3 revenue results?"
}'
```
**Success Response (200 OK):**
```json
{
  "answer": "Based on the documents, the Q3 revenue dropped by 40% due to severe supply chain disruptions...",
  "sources": [
    "raw string of retrieved chunk 1",
    "raw string of retrieved chunk 2"
  ]
}
```

### 3. Evaluation Report (`GET /report`)
Returns basic system evaluation metrics.
```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/report' \
  -H 'accept: application/json'
```
**Success Response (200 OK):**
```json
{
  "context_precision": 0.9,
  "faithfulness": 0.85,
  "system_status": "healthy"
}
```
