import chromadb
from sentence_transformers import SentenceTransformer
import uuid

class VectorStore:
    def __init__(self):
        # 1. Initialize embedding model (runs locally)
        # all-MiniLM-L6-v2 maps sentences to a 384-dimensional dense vector space
        print("Initializing sentence-transformers model (this may take a moment on first run)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Initialize ChromaDB (In-memory for testing purposes as per PRD)
        print("Initializing ChromaDB in-memory client...")
        self.chroma_client = chromadb.Client()
        
        # Create or get a collection to store our documents
        self.collection = self.chroma_client.get_or_create_collection(name="rag_documents")

    def add_chunks(self, chunks: list[str]):
        """
        Generates embeddings for the text chunks and stores them in ChromaDB.
        """
        if not chunks:
            return

        # Generate vectors
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        # Generate unique IDs for each chunk
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        # Store in ChromaDB
        # Chroma expects documents (raw text), embeddings, and ids.
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
        print(f"Successfully added {len(chunks)} chunks to Vector DB.")

    def search(self, query: str, top_k: int = 3):
        """
        Embeds the query and performs a similarity search to find Top-K matching chunks.
        """
        # Encode the question using the exact same model
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Query ChromaDB (calculates Cosine Similarity under the hood for matching)
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # results["documents"] is a list of lists: [['chunk1', 'chunk2', ...]]
        if results and "documents" in results and results["documents"]:
            return results["documents"][0]
        
        return []

    def is_empty(self):
        """
        Checks if the collection has any documents indexed.
        """
        return self.collection.count() == 0
