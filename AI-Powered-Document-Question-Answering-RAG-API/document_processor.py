from fastapi import UploadFile
import PyPDF2
from io import BytesIO

async def process_document(file: UploadFile, file_ext: str):
    """
    Reads the uploaded file, extracts text based on extension, and chunks it.
    Returns a list of text chunks.
    """
    content = await file.read()
    raw_text = ""

    # 1. Extract Text
    if file_ext in [".txt", ".md"]:
        try:
            raw_text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("Could not decode text file. Ensure it is UTF-8 encoded.")
    elif file_ext == ".pdf":
        try:
            pdf_file = BytesIO(content)
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    raw_text += page_text + "\n"
        except Exception as e:
            raise ValueError(f"Error parsing PDF: {str(e)}")

    if not raw_text.strip():
        raise ValueError("The document contains no readable text.")

    # 2. Text Chunking (Fixed size 1000, Overlap 200)
    chunks = chunk_text(raw_text, chunk_size=1000, overlap=200)
    return chunks

def chunk_text(text: str, chunk_size: int, overlap: int):
    """
    Slices text into overlapping chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move start forward by (chunk_size - overlap)
        start += (chunk_size - overlap)
        
        # Prevent infinite loops if overlap >= chunk_size (should not happen based on hardcoded vals)
        if chunk_size <= overlap:
            break
            
    return chunks
