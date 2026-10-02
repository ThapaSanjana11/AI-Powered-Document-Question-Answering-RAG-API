import os
from groq import Groq

def generate_answer(question: str, retrieved_chunks: list[str]) -> str:
    """
    Constructs a prompt with context and calls the Groq API to generate an answer.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing.")

    client = Groq(api_key=api_key)

    # 1. Construct the Prompt Template
    context_str = "\n".join(retrieved_chunks)
    
    prompt = f"""You are a helpful assistant answering questions based on the provided document context.
ONLY use the following context. Under no circumstances use outside knowledge.

Context Information:
---------------------
{context_str}
---------------------
Given the context information and no prior knowledge, answer the following user question.
If the answer is not contained in the context, explicitly state "I cannot find the answer in the provided documents."

Question: {question}
Answer:"""

    # 2. Call the LLM API
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0, # Low temperature for factual retrieval
            max_tokens=500,
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        # We catch any network/API errors here to gracefully degrade
        raise Exception(f"Groq API Error: {str(e)}")
