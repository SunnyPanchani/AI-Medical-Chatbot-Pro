"""
Medical Chatbot Flask Application
Optimized with enhanced RAG system
"""

from flask import Flask, render_template, request, jsonify
from src.helper import (
    download_embeddings,
    load_vector_store,
    initialize_groq_llm,
    create_rag_chain
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Global variables for chatbot components
rag_chain = None
retriever = None

def initialize_chatbot():
    """Initialize the chatbot on startup"""
    global rag_chain, retriever
    
    print("\n" + "="*80)
    print("INITIALIZING MEDICAL CHATBOT")
    print("="*80 + "\n")
    
    try:
        # Check API keys
        if not os.getenv("PINECONE_API_KEY"):
            raise ValueError("PINECONE_API_KEY not found in .env")
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in .env")
        
        print("✅ API keys found")
        
        # Load embeddings
        print("Loading embeddings...")
        embeddings = download_embeddings()
        
        # Load vector store
        print("Loading vector store...")
        docsearch = load_vector_store(embeddings, index_name="medical-chatbot")
        
        # Initialize LLM
        print("Initializing Groq LLM...")
        chatModel = initialize_groq_llm(
            model_name="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024
        )
        
        # Create RAG chain
        print("Creating RAG chain...")
        rag_chain, retriever = create_rag_chain(
            docsearch,
            chatModel,
            num_documents=5
        )
        
        print("\n" + "="*80)
        print("✅ MEDICAL CHATBOT READY!")
        print("="*80 + "\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error initializing chatbot: {e}")
        print("\nTroubleshooting:")
        print("  • Run: python store_index.py (create vector store)")
        print("  • Check .env file has valid API keys")
        print("  • Ensure data/ folder has PDF files")
        return False

# Initialize on startup
chatbot_ready = initialize_chatbot()

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', chatbot_ready=chatbot_ready)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if chatbot_ready and rag_chain else 'unhealthy',
        'chatbot_ready': chatbot_ready,
        'model': 'llama-3.3-70b-versatile' if chatbot_ready else None
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint - answer medical questions"""
    
    if not chatbot_ready or not rag_chain:
        return jsonify({
            'error': 'Chatbot not initialized. Please check logs and restart.',
            'suggestion': 'Run: python store_index.py'
        }), 500
    
    try:
        # Get question from request
        data = request.get_json()
        question = data.get('message', '').strip()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Get answer from RAG chain
        print(f"\n📋 Question: {question}")
        answer = rag_chain.invoke(question)
        
        # Get source documents
        sources = []
        if retriever:
            retrieved_docs = retriever.invoke(question)
            sources = [
                {
                    'source': doc.metadata.get('source', 'Unknown'),
                    'preview': doc.page_content[:200].replace('\n', ' ').strip()
                }
                for doc in retrieved_docs
            ]
        
        print(f"✅ Answer generated ({len(answer)} chars)")
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'question': question
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            'error': f'Error processing question: {str(e)}'
        }), 500

@app.route('/quick-chat', methods=['POST'])
def quick_chat():
    """Quick chat endpoint - brief answers"""
    
    if not chatbot_ready or not rag_chain:
        return jsonify({'error': 'Chatbot not ready'}), 500
    
    try:
        data = request.get_json()
        question = data.get('message', '').strip()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Create quick answer chain (2-3 sentences)
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from src.helper import format_docs
        
        quick_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical assistant. Answer briefly in 2-3 sentences.\n\nContext: {context}"),
            ("human", "{input}"),
        ])
        
        quick_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | quick_prompt
            | initialize_groq_llm(temperature=0.2, max_tokens=256)
            | StrOutputParser()
        )
        
        answer = quick_chain.invoke(question)
        
        return jsonify({
            'answer': answer,
            'question': question,
            'mode': 'quick'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not chatbot_ready:
        print("\n⚠️  Chatbot not ready. Please fix initialization errors.")
        print("Run: python store_index.py")
    else:
        print("\n🚀 Starting Flask server...")
        print("Access the chatbot at: http://localhost:8080")
        print("Press Ctrl+C to stop\n")
        app.run(debug=True, host='0.0.0.0', port=8080)