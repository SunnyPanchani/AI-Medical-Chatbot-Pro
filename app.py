"""
Medical Chatbot Flask Application
With Streaming Responses
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from src.helper import (
    download_embeddings,
    load_vector_store,
    initialize_groq_llm,
    create_rag_chain
)
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Global variables for chatbot components
rag_chain = None
retriever = None
chatModel = None

def initialize_chatbot():
    """Initialize the chatbot on startup"""
    global rag_chain, retriever, chatModel
    
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
    """Chat endpoint - answer medical questions with streaming"""
    
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
        
        # Get answer from RAG chain (non-streaming for now)
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

@app.route('/chat-stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint - shows answer as it's generated"""
    
    if not chatbot_ready or not rag_chain or not chatModel:
        return jsonify({
            'error': 'Chatbot not initialized'
        }), 500
    
    try:
        data = request.get_json()
        question = data.get('message', '').strip()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        def generate():
            """Generator function for streaming response"""
            try:
                # Get relevant documents first
                retrieved_docs = retriever.invoke(question)
                
                # Format context
                from src.helper import format_docs
                context = format_docs(retrieved_docs)
                
                # Create prompt
                from langchain_core.prompts import ChatPromptTemplate
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
                
                # Format the prompt
                formatted_prompt = prompt.format(context=context, input=question)
                
                # Stream the response
                for chunk in chatModel.stream(formatted_prompt):
                    if chunk.content:
                        # Send each chunk as JSON
                        yield f"data: {json.dumps({'chunk': chunk.content})}\n\n"
                
                # Send sources at the end
                sources = [
                    {
                        'source': doc.metadata.get('source', 'Unknown'),
                        'preview': doc.page_content[:200].replace('\n', ' ').strip()
                    }
                    for doc in retrieved_docs
                ]
                
                yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
    
    except Exception as e:
        print(f"❌ Error: {e}")
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