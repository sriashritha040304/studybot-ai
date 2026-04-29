# src/ingest.py
# Handles: PDF loading, chunking, embedding, storing in ChromaDB

import os
from typing import List
from dotenv import load_dotenv

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()


def load_pdf(file_path: str) -> str:
    """
    Load a PDF file and extract all text content.
    
    Args:
        file_path: Path to the PDF file on disk
        
    Returns:
        Single string containing all extracted text
    """
    print(f"Loading PDF: {file_path}")
    
    # Open the PDF file in binary read mode
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        
        # Track total pages for logging
        total_pages = len(reader.pages)
        print(f"Total pages found: {total_pages}")
        
        # Extract text from every page
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
    print(f"Total characters extracted: {len(text)}")
    return text


def split_into_chunks(text: str) -> List[str]:
    """
    Split extracted text into overlapping chunks.
    
    Args:
        text: Full text extracted from PDF
        
    Returns:
        List of text chunks, each ~1000 characters
    """
    print("Splitting text into chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # Target size of each chunk in characters
        chunk_overlap=200,      # Overlap between consecutive chunks
        length_function=len,    # How to measure chunk size
        separators=["\n\n", "\n", ". ", " ", ""]  # Split priority order
    )
    
    chunks = splitter.split_text(text)
    
    print(f"Total chunks created: {len(chunks)}")
    print(f"Average chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars")
    
    return chunks


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