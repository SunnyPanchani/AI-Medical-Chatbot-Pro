


"""
Medical Chatbot Helper Functions
Optimized for LangChain 1.0+ with Groq and Pinecone
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ==========================================
# DOCUMENT PROCESSING FUNCTIONS
# ==========================================

def load_pdf_files(data_path: str):
    """
    Load all PDF files from directory
    
    Args:
        data_path: Path to directory containing PDF files
        
    Returns:
        List of loaded documents
    """
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents from {data_path}")
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Keep only essential metadata (source) from documents
    
    Args:
        docs: List of documents with full metadata
        
    Returns:
        List of documents with minimal metadata
    """
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get('source')
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={'source': src})
        )
    print(f"✅ Filtered to {len(minimal_docs)} minimal documents")
    return minimal_docs


def text_split(documents, chunk_size: int = 500, chunk_overlap: int = 20):
    """
    Split documents into smaller chunks for better retrieval
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk (default: 500)
        chunk_overlap: Overlap between chunks (default: 20)
        
    Returns:
        List of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts_chunk = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(texts_chunk)} chunks")
    return texts_chunk


def download_embeddings():
    """
    Initialize HuggingFace embeddings model
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print(f"✅ Embeddings model loaded: {model_name}")
    return embeddings


# ==========================================
# PINECONE VECTOR STORE FUNCTIONS
# ==========================================

def initialize_pinecone(index_name: str = "medical-chatbot"):
    """
    Initialize Pinecone client and create index if needed
    
    Args:
        index_name: Name of the Pinecone index
        
    Returns:
        Pinecone client instance
    """
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    if not PINECONE_API_KEY:
        raise ValueError("❌ PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if doesn't exist
    if not pc.has_index(index_name):
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,  # all-MiniLM-L6-v2 dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("✅ Index created")
    else:
        print(f"✅ Index '{index_name}' already exists")
    
    return pc


def get_vector_count(pc, index_name: str) -> int:
    """
    Get current vector count in Pinecone index
    
    Args:
        pc: Pinecone client
        index_name: Name of the index
        
    Returns:
        Number of vectors in index
    """
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        return stats.get('total_vector_count', 0)
    except Exception as e:
        print(f"❌ Error getting vector count: {e}")
        return 0


def create_vector_store(
    texts_chunk,
    embeddings,
    index_name: str = "medical-chatbot",
    force_recreate: bool = False
):
    """
    Create or load Pinecone vector store intelligently
    
    Args:
        texts_chunk: Document chunks to store
        embeddings: Embedding model
        index_name: Name of the index
        force_recreate: If True, clears existing vectors and recreates
        
    Returns:
        PineconeVectorStore instance
    """
    # Initialize Pinecone
    pc = initialize_pinecone(index_name)
    
    # Check current vector count
    current_count = get_vector_count(pc, index_name)
    print(f"Current vectors in index: {current_count}")
    
    if current_count > 0:
        if force_recreate:
            print("⚠️  Force recreate enabled. Clearing existing vectors...")
            pc.Index(index_name).delete(delete_all=True)
            time.sleep(2)  # Wait for deletion to complete
            print("Creating new vector store...")
            docsearch = PineconeVectorStore.from_documents(
                documents=texts_chunk,
                embedding=embeddings,
                index_name=index_name
            )
            print(f"✅ Vector store created with {len(texts_chunk)} new documents")
        else:
            print("Vector store already has data. Loading existing...")
            docsearch = PineconeVectorStore.from_existing_index(
                embedding=embeddings,
                index_name=index_name
            )
            print("✅ Loaded existing vector store")
    else:
        print("Index is empty. Creating new vector store...")
        docsearch = PineconeVectorStore.from_documents(
            documents=texts_chunk,
            embedding=embeddings,
            index_name=index_name
        )
        print(f"✅ Vector store created with {len(texts_chunk)} documents")
    
    # Verify final count
    final_count = get_vector_count(pc, index_name)
    print(f"Final vector count: {final_count}")
    
    return docsearch


def load_vector_store(embeddings, index_name: str = "medical-chatbot"):
    """
    Load existing Pinecone vector store
    
    Args:
        embeddings: Embedding model
        index_name: Name of the index
        
    Returns:
        PineconeVectorStore instance
    """
    docsearch = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name=index_name
    )
    print("✅ Loaded existing vector store")
    return docsearch


# ==========================================
# LLM AND RAG CHAIN FUNCTIONS
# ==========================================

def initialize_groq_llm(
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
    max_tokens: int = 1024
):
    """
    Initialize Groq LLM with optimized settings
    
    Args:
        model_name: Groq model to use
        temperature: Creativity level (0-1)
        max_tokens: Maximum response length
        
    Returns:
        ChatGroq instance
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        raise ValueError("❌ GROQ_API_KEY not found in environment variables")
    
    chatModel = ChatGroq(
        model_name=model_name,
        groq_api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    print(f"✅ Groq LLM initialized")
    print(f"   Model: {model_name}")
    print(f"   Temperature: {temperature}")
    print(f"   Max tokens: {max_tokens}")
    
    return chatModel


def format_docs(docs):
    """
    Format documents with clear structure for context
    
    Args:
        docs: List of documents
        
    Returns:
        Formatted string
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content.strip()
        formatted.append(f"[Document {i} - {source}]:\n{content}")
    return "\n\n" + "="*60 + "\n\n".join(formatted)


def create_rag_chain(docsearch, chatModel, num_documents: int = 5):
    """
    Create optimized RAG chain for medical Q&A
    
    Args:
        docsearch: PineconeVectorStore instance
        chatModel: ChatGroq instance
        num_documents: Number of documents to retrieve (default: 5)
        
    Returns:
        Complete RAG chain
    """
    # Create retriever
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": num_documents}
    )
    
    # Enhanced system prompt
    system_prompt = """You are an expert Medical Assistant with comprehensive knowledge of medical conditions, treatments, and healthcare.

Your role is to provide detailed, accurate, and helpful answers based on the medical literature provided in the context.

Guidelines for your responses:
1. **Be Comprehensive**: Provide thorough explanations covering all relevant aspects
2. **Be Structured**: Organize information logically (definition, causes, symptoms, treatment, etc.)
3. **Be Clear**: Explain medical terms in understandable language
4. **Be Accurate**: Only use information from the provided context
5. **Be Helpful**: Anticipate follow-up questions and address them
6. **Be Honest**: If information is missing from context, clearly state it

Context from medical literature:
{context}

Provide a detailed, well-structured answer to the following question:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | chatModel
        | StrOutputParser()
    )
    
    print("✅ RAG chain created")
    print(f"   Retrieving top {num_documents} documents per query")
    
    return rag_chain, retriever


# ==========================================
# QUESTION ANSWERING FUNCTIONS
# ==========================================

def ask_question(
    rag_chain,
    question: str,
    retriever=None,
    show_sources: bool = True,
    show_timing: bool = True
) -> Dict:
    """
    Ask a comprehensive medical question
    
    Args:
        rag_chain: RAG chain instance
        question: Medical question to ask
        retriever: Optional retriever for source display
        show_sources: Display sources used
        show_timing: Show response time
        
    Returns:
        Dictionary with answer and metadata
    """
    start_time = time.time()
    
    # Header
    print("\n" + "="*80)
    print(f"📋 QUESTION: {question}")
    print("="*80 + "\n")
    
    # Get answer
    print("🤖 Generating comprehensive answer...\n")
    response = rag_chain.invoke(question)
    
    elapsed_time = time.time() - start_time
    
    # Display answer
    print("💡 DETAILED ANSWER:")
    print("-" * 80)
    print(response)
    print("-" * 80)
    
    # Show sources
    if show_sources and retriever:
        retrieved_docs = retriever.invoke(question)
        print(f"\n📚 SOURCES CONSULTED ({len(retrieved_docs)} documents):")
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            preview = doc.page_content[:200].replace('\n', ' ').strip()
            print(f"\n  [{i}] {source}")
            print(f"      Preview: {preview}...")
    
    # Show timing
    if show_timing:
        print(f"\n⏱️  Response time: {elapsed_time:.2f} seconds")
    
    print("="*80 + "\n")
    
    return {
        "question": question,
        "answer": response,
        "time": elapsed_time
    }


# ==========================================
# COMPLETE SETUP FUNCTION
# ==========================================

def setup_medical_chatbot(
    data_path: str = "data",
    index_name: str = "medical-chatbot",
    force_recreate: bool = False,
    num_documents: int = 5
):
    """
    Complete setup for medical chatbot system
    
    Args:
        data_path: Path to PDF files
        index_name: Name of Pinecone index
        force_recreate: Clear and recreate vectors
        num_documents: Number of documents to retrieve per query
        
    Returns:
        Tuple of (rag_chain, retriever, docsearch)
    """
    print("\n" + "="*80)
    print("MEDICAL CHATBOT SETUP")
    print("="*80 + "\n")
    
    # Load and process documents
    print("📄 Loading documents...")
    extracted_docs = load_pdf_files(data_path)
    minimal_docs = filter_to_minimal_docs(extracted_docs)
    texts_chunk = text_split(minimal_docs)
    
    # Initialize embeddings
    print("\n🔤 Initializing embeddings...")
    embeddings = download_embeddings()
    
    # Create/load vector store
    print("\n🗄️  Setting up vector store...")
    docsearch = create_vector_store(
        texts_chunk,
        embeddings,
        index_name,
        force_recreate=force_recreate
    )
    
    # Initialize LLM
    print("\n🤖 Initializing Groq LLM...")
    chatModel = initialize_groq_llm()
    
    # Create RAG chain
    print("\n⛓️  Creating RAG chain...")
    rag_chain, retriever = create_rag_chain(docsearch, chatModel, num_documents)
    
    print("\n" + "="*80)
    print("✅ MEDICAL CHATBOT READY!")
    print("="*80 + "\n")
    
    return rag_chain, retriever, docsearch