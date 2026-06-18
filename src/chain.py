# src/chain.py
# Handles: RAG chain, conversation memory, source citations

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"


def load_retriever(vectorstore):
    """
    Create a retriever from the ChromaDB vectorstore.
    Returns top 4 most relevant chunks for any query.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    print("Retriever created — will return top 4 chunks")
    return retriever


def build_chain(retriever):
    """
    Build the RAG chain using modern LangChain v1.2+ syntax.
    """
    print("Building RAG chain...")

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0.1,
        api_key=os.environ.get("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful course tutor. \
Answer the question using ONLY the context provided below.
If the answer is not in the context, say \
'I don't know based on the provided material.'
Always be clear and concise.

Context: {context}"""),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chain built successfully")
    return chain, retriever


def get_answer(chain_tuple, question: str, chat_history: list) -> dict:
    """
    Get an answer from the RAG chain with source citations.
    
    Args:
        chain_tuple: (chain, retriever) from build_chain()
        question: User's question string
        chat_history: List of previous (question, answer) tuples
        
    Returns:
        Dict with keys: 'answer' (str) and 'sources' (list of dicts)
    """
    chain, retriever = chain_tuple
    print(f"Processing question: {question}")

    # Get answer from chain
    answer = chain.invoke(question)

    # Get source documents separately for citations
    docs = retriever.invoke(question)
    sources = []
    for doc in docs:
        sources.append({
            "content": doc.page_content[:200],
            "metadata": doc.metadata
        })

    return {
        "answer": answer,
        "sources": sources
    }