# StudyBot LLM Configuration
# Model: llama-3.3-70b-versatile (updated April 2026)
# Old model llama3-70b-8192 was decommissioned
# Always check console.groq.com/docs/models for current models

# src/chain.py
# Handles: RAG chain, conversation memory, source citations

def load_retriever(vectorstore):
    """
    Create a retriever from the ChromaDB vectorstore.
    
    Args:
        vectorstore: Chroma vectorstore from ingest_pdf()
        
    Returns:
        Retriever configured to return top 4 chunks
    """
    pass


def build_chain(retriever):
    """
    Build the conversational RAG chain.
    Connects: retriever → Groq Llama3 → memory → output
    
    Args:
        retriever: Retriever from load_retriever()
        
    Returns:
        Configured ConversationalRetrievalChain
    """
    pass


def get_answer(chain, question: str, chat_history: list) -> dict:
    """
    Get an answer from the RAG chain.
    
    Args:
        chain: Built chain from build_chain()
        question: User's question string
        chat_history: List of previous (question, answer) tuples
        
    Returns:
        Dict with keys: 'answer' (str) and 'sources' (list of dicts)
    """
    pass