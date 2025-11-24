"""
AI Medical Chatbot Pro - Streamlit Version
Optimized for Render Deployment
"""

import streamlit as st
import os
from dotenv import load_dotenv
from src.helper import (
    download_embeddings,
    load_vector_store,
    initialize_groq_llm,
    create_rag_chain
)
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Medical Chatbot Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat messages */
    .stChatMessage {
        background: white;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Bot message */
    .stChatMessage[data-testid="assistant-message"] {
        background: white;
        color: #2d3748;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Input box */
    .stTextInput input {
        border-radius: 20px;
        border: 2px solid #667eea;
        padding: 10px 20px;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Source badges */
    .source-badge {
        background: #f7fafc;
        border-left: 4px solid #667eea;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        font-size: 12px;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 10px 0;
    }
    
    .status-ready {
        background: rgba(72, 187, 120, 0.2);
        color: #48bb78;
        border: 1px solid rgba(72, 187, 120, 0.4);
    }
    
    .status-error {
        background: rgba(245, 101, 101, 0.2);
        color: #f56565;
        border: 1px solid rgba(245, 101, 101, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chatbot_ready' not in st.session_state:
    st.session_state.chatbot_ready = False

if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

@st.cache_resource(show_spinner=False)
def initialize_chatbot():
    """Initialize the chatbot (cached to run only once)"""
    try:
        # Check API keys
        pinecone_key = os.getenv("PINECONE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        
        if not pinecone_key or not groq_key:
            return None, None, False, "API keys not found in environment"
        
        # Load embeddings
        with st.spinner("Loading embeddings model..."):
            embeddings = download_embeddings()
        
        # Load vector store
        with st.spinner("Connecting to vector database..."):
            docsearch = load_vector_store(embeddings, index_name="medical-chatbot")
        
        # Initialize LLM
        with st.spinner("Initializing AI model..."):
            chatModel = initialize_groq_llm(
                model_name="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1024
            )
        
        # Create RAG chain
        with st.spinner("Setting up RAG system..."):
            rag_chain, retriever = create_rag_chain(
                docsearch,
                chatModel,
                num_documents=5
            )
        
        return rag_chain, retriever, True, "System ready"
        
    except Exception as e:
        return None, None, False, str(e)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 AI Medical Chatbot Pro</h1>
    <p>Powered by Llama 3.3 70B & Advanced RAG System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Initialize button
    if st.button("🔄 Initialize Chatbot", use_container_width=True):
        with st.spinner("Initializing chatbot..."):
            rag_chain, retriever, ready, message = initialize_chatbot()
            
            if ready:
                st.session_state.rag_chain = rag_chain
                st.session_state.retriever = retriever
                st.session_state.chatbot_ready = True
                st.success("✅ Chatbot initialized successfully!")
            else:
                st.error(f"❌ Error: {message}")
    
    # Status indicator
    if st.session_state.chatbot_ready:
        st.markdown('<div class="status-badge status-ready">● System Ready</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-error">● System Not Ready</div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Information
    st.markdown("""
    ### 📋 About
    This is an AI-powered medical chatbot that provides detailed information 
    about medical conditions, symptoms, and treatments.
    
    ### 🔒 Privacy
    - Your conversations are not stored
    - Data is processed securely
    - Sources are cited for transparency
    
    ### ⚠️ Disclaimer
    This chatbot provides information only. Always consult healthcare 
    professionals for medical advice.
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info("""
    **Model:** Llama 3.3 70B  
    **Provider:** Groq  
    **Documents:** 5 per query  
    **Temperature:** 0.3
    """)

# Main chat area
st.markdown("### 💬 Chat")

# Display welcome message if no messages
if len(st.session_state.messages) == 0:
    st.info("""
    👋 **Welcome to your AI Medical Assistant!**
    
    Ask me anything about medical conditions, symptoms, treatments, and healthcare. 
    I provide detailed, evidence-based information from trusted medical sources.
    
    **Example questions:**
    - What is diabetes mellitus? Explain its types and management.
    - What are the symptoms and treatment options for hypertension?
    - Describe asthma, its causes, and how it can be managed.
    """)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    filename = os.path.basename(source["source"])
                    st.markdown(f"""
                    <div class="source-badge">
                        <strong>{i}.</strong> {filename}
                    </div>
                    """, unsafe_allow_html=True)


# Chat input
if prompt := st.chat_input("Ask a medical question...", disabled=not st.session_state.chatbot_ready):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Show thinking indicator
        with st.spinner("Thinking..."):
            try:
                # Get response (streaming effect)
                response_text = st.session_state.rag_chain.invoke(prompt)
                
                # Simulate streaming effect
                for chunk in response_text.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▊")
                    time.sleep(0.02)  # Adjust speed here
                
                # Final response without cursor
                message_placeholder.markdown(full_response)
                
                # Get sources
                sources = []
                if st.session_state.retriever:
                    retrieved_docs = st.session_state.retriever.invoke(prompt)
                    sources = [
                        {
                            'source': doc.metadata.get('source', 'Unknown'),
                            'preview': doc.page_content[:200]
                        }
                        for doc in retrieved_docs
                    ]
                
                # Display sources
                if sources:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources, 1):
                            filename = os.path.basename(source["source"])
                            st.markdown(f"""
                            <div class="source-badge">
                                <strong>{i}.</strong> {filename}
                            </div>
                            """, unsafe_allow_html=True)

                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Tip: Make sure the chatbot is initialized and your API keys are correct.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; font-size: 12px; padding: 20px;">
    <p>🏥 <strong>AI Medical Chatbot Pro</strong> | Powered by Llama 3.3 70B</p>
    <p>⚠️ <em>This chatbot provides information only. Always consult healthcare professionals for medical advice.</em></p>
</div>
""", unsafe_allow_html=True)