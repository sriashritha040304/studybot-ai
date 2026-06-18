# src/ingest.py
# Handles: PDF loading, chunking, embedding, storing in ChromaDB

import os
from typing import List
from dotenv import load_dotenv

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()


def load_pdf(file_path: str) -> str:
    print(f"Loading PDF: {file_path}")
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)
        print(f"Total pages found: {total_pages}")
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    print(f"Total characters extracted: {len(text)}")
    return text


def split_into_chunks(text: str) -> List[str]:
    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"Total chunks created: {len(chunks)}")
    print(f"Average chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars")
    return chunks


def create_embeddings():
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Embedding model loaded successfully")
    return embeddings


def store_in_chroma(chunks: List[str], embeddings):
    print(f"Storing {len(chunks)} chunks in ChromaDB...")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print(f"Stored successfully in ./chroma_db/")
    return vectorstore


def ingest_pdf(file_path: str):
    print("\n" + "="*50)
    print("STARTING INGESTION PIPELINE")
    print("="*50)
    text = load_pdf(file_path)
    chunks = split_into_chunks(text)
    embeddings = create_embeddings()
    vectorstore = store_in_chroma(chunks, embeddings)
    print(f"\nINGESTION COMPLETE — {len(chunks)} chunks stored")
    print("="*50 + "\n")
    return vectorstore