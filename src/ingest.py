# src/ingest.py
# Handles: PDF loading, chunking, embedding, storing in ChromaDB

from typing import List


def load_pdf(file_path: str) -> str:
    """
    Load a PDF file and extract all text content.
    
    Args:
        file_path: Path to the PDF file on disk
        
    Returns:
        Single string containing all extracted text
    """
    pass


def split_into_chunks(text: str) -> List[str]:
    """
    Split extracted text into overlapping chunks.
    
    Args:
        text: Full text extracted from PDF
        
    Returns:
        List of text chunks, each ~1000 characters
    """
    pass


def create_embeddings():
    """
    Initialize the HuggingFace embedding model.
    
    Returns:
        HuggingFaceEmbeddings object using all-MiniLM-L6-v2
    """
    pass


def store_in_chroma(chunks: List[str], embeddings):
    """
    Store text chunks and their embeddings in ChromaDB.
    
    Args:
        chunks: List of text chunks from split_into_chunks()
        embeddings: Embedding model from create_embeddings()
        
    Returns:
        Chroma vectorstore object
    """
    pass


def ingest_pdf(file_path: str):
    """
    Master function — runs the full ingestion pipeline.
    Calls load_pdf → split_into_chunks → create_embeddings → store_in_chroma
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Chroma vectorstore ready for querying
    """
    pass