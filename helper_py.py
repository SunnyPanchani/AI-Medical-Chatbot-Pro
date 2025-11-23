from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec
from typing import List
from langchain.schema import Document
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Extract PDF files from directory
def load_pdf_files(data_path):
    """Load PDF files from specified directory"""
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Filter documents to minimal metadata
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep only essential metadata (source) from documents"""
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get('source')
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={'source': src})
        )
    return minimal_docs

# Split text into chunks
def text_split(documents):
    """Split documents into smaller chunks for better retrieval"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    texts_chunk = text_splitter.split_documents(documents)
    return texts_chunk

# Download and initialize embeddings
def download_embeddings():
    """Initialize HuggingFace embeddings model"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

# Initialize Pinecone vector store
def initialize_pinecone_index(index_name="medical-chatbot"):
    """Create or connect to Pinecone index"""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Create index if it doesn't exist
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,  # Dimension for all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    return pc.Index(index_name)

# Create vector store from documents
def create_vector_store(texts_chunk, embeddings, index_name="medical-chatbot"):
    """Create Pinecone vector store from document chunks"""
    docsearch = PineconeVectorStore.from_documents(
        documents=texts_chunk,
        embedding=embeddings,
        index_name=index_name
    )
    return docsearch

# Load existing vector store
def load_vector_store(embeddings, index_name="medical-chatbot"):
    """Load existing Pinecone vector store"""
    docsearch = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name=index_name
    )
    return docsearch

# Initialize LLM - OPTION 1: Groq (Recommended - Fast & Free)
def initialize_groq_llm():
    """Initialize Groq LLM (Free and Fast)"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file. Get one from https://console.groq.com")
    
    chatModel = ChatGroq(
        model_name="llama3-70b-8192",
        groq_api_key=groq_api_key,
        temperature=0.7
    )
    return chatModel

# Initialize LLM - OPTION 2: OpenRouter (Multiple Free Models)
def initialize_openrouter_llm():
    """Initialize OpenRouter LLM (Multiple free models available)"""
    from langchain_openai import ChatOpenAI
    
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file. Get one from https://openrouter.ai")
    
    chatModel = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        model="deepseek/deepseek-chat-v3-0324:free",  # Free model
        default_headers={
            "HTTP-Referer": "http://localhost:8080",
            "X-Title": "Medical Chatbot"
        }
    )
    return chatModel

# Initialize LLM - OPTION 3: Google Gemini (Free Tier)
def initialize_gemini_llm():
    """Initialize Google Gemini LLM (Free tier available)"""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file. Get one from https://ai.google.dev")
    
    chatModel = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0.7
    )
    return chatModel

# Create RAG chain
def create_rag_chain(chatModel, retriever, system_prompt=None):
    """Create Retrieval-Augmented Generation chain"""
    
    if system_prompt is None:
        system_prompt = (
            "You are a Medical assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Main function to set up the entire pipeline
def setup_medical_chatbot(
    data_path="data",
    index_name="medical-chatbot",
    llm_provider="groq"  # Options: "groq", "openrouter", "gemini"
):
    """
    Complete setup for medical chatbot
    
    Args:
        data_path: Path to PDF files
        index_name: Name of Pinecone index
        llm_provider: LLM provider to use ("groq", "openrouter", or "gemini")
    
    Returns:
        rag_chain: Ready-to-use RAG chain
    """
    
    # Load and process documents
    print("Loading PDF files...")
    documents = load_pdf_files(data_path)
    
    print("Processing documents...")
    minimal_docs = filter_to_minimal_docs(documents)
    texts_chunk = text_split(minimal_docs)
    
    print("Initializing embeddings...")
    embeddings = download_embeddings()
    
    print("Setting up Pinecone...")
    initialize_pinecone_index(index_name)
    
    # Try to load existing vector store, create if doesn't exist
    try:
        print("Loading existing vector store...")
        docsearch = load_vector_store(embeddings, index_name)
    except:
        print("Creating new vector store...")
        docsearch = create_vector_store(texts_chunk, embeddings, index_name)
    
    # Create retriever
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Initialize LLM based on provider
    print(f"Initializing {llm_provider} LLM...")
    if llm_provider == "groq":
        chatModel = initialize_groq_llm()
    elif llm_provider == "openrouter":
        chatModel = initialize_openrouter_llm()
    elif llm_provider == "gemini":
        chatModel = initialize_gemini_llm()
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")
    
    # Create RAG chain
    print("Creating RAG chain...")
    rag_chain = create_rag_chain(chatModel, retriever)
    
    print("Medical chatbot setup complete!")
    return rag_chain
