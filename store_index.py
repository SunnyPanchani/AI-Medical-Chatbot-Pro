


"""
Store Index Script - Create/Update Pinecone Vector Store
Run this script to process PDFs and create the vector database
"""

from dotenv import load_dotenv
import os
from src.helper import (
    load_pdf_files,
    filter_to_minimal_docs,
    text_split,
    download_embeddings,
    initialize_pinecone,
    create_vector_store,
    get_vector_count
)
import sys

# Load environment variables
load_dotenv()

# Get API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Fixed typo: was GROK

# Validate API keys
if not PINECONE_API_KEY:
    print("❌ Error: PINECONE_API_KEY not found in .env file")
    print("Please add your Pinecone API key to .env file")
    sys.exit(1)

if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY not found in .env file")
    print("Please add your Groq API key to .env file")
    sys.exit(1)

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

print("\n" + "="*80)
print("MEDICAL CHATBOT - VECTOR STORE CREATION")
print("="*80 + "\n")

# Configuration
DATA_PATH = "data"
INDEX_NAME = "medical-chatbot"
FORCE_RECREATE = False  # Set to True to clear and recreate all vectors

print("Configuration:")
print(f"  • Data path: {DATA_PATH}")
print(f"  • Index name: {INDEX_NAME}")
print(f"  • Force recreate: {FORCE_RECREATE}")
print()

# Check if data directory exists
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: Data directory '{DATA_PATH}' not found")
    print(f"Please create the directory and add your PDF files")
    sys.exit(1)

# Check if there are PDF files
pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
if not pdf_files:
    print(f"❌ Error: No PDF files found in '{DATA_PATH}'")
    print(f"Please add medical PDF files to the directory")
    sys.exit(1)

print(f"Found {len(pdf_files)} PDF file(s):")
for pdf in pdf_files:
    print(f"  • {pdf}")
print()

try:
    # Step 1: Load PDFs
    print("Step 1: Loading PDF files...")
    print("-" * 80)
    extracted_data = load_pdf_files(DATA_PATH)
    print()
    
    # Step 2: Filter documents
    print("Step 2: Filtering documents...")
    print("-" * 80)
    filtered_data = filter_to_minimal_docs(extracted_data)
    print()
    
    # Step 3: Split into chunks
    print("Step 3: Splitting into chunks...")
    print("-" * 80)
    text_chunks = text_split(filtered_data)
    print()
    
    # Step 4: Initialize embeddings
    print("Step 4: Initializing embeddings...")
    print("-" * 80)
    embeddings = download_embeddings()
    print()
    
    # Step 5: Initialize Pinecone
    print("Step 5: Connecting to Pinecone...")
    print("-" * 80)
    pc = initialize_pinecone(INDEX_NAME)
    print()
    
    # Step 6: Check current state
    print("Step 6: Checking current vector store state...")
    print("-" * 80)
    current_count = get_vector_count(pc, INDEX_NAME)
    print(f"Current vector count: {current_count}")
    print()
    
    # Step 7: Decision logic
    if current_count > 0 and not FORCE_RECREATE:
        print("⚠️  Vector store already contains data!")
        print()
        print("Options:")
        print("  1. Keep existing data (do nothing)")
        print("  2. Clear and recreate (set FORCE_RECREATE = True)")
        print()
        
        response = input("Do you want to CLEAR and RECREATE? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            FORCE_RECREATE = True
            print("✅ Will clear and recreate vector store")
        else:
            print("✅ Keeping existing vector store")
            print("\nVector store is ready to use!")
            print("Run your application: python app.py")
            sys.exit(0)
    
    # Step 8: Create/Update vector store
    print("\nStep 7: Creating/Updating vector store...")
    print("-" * 80)
    print("⏳ This may take several minutes depending on document size...")
    print()
    
    doc_search = create_vector_store(
        text_chunks,
        embeddings,
        INDEX_NAME,
        force_recreate=FORCE_RECREATE
    )
    
    # Verify creation
    final_count = get_vector_count(pc, INDEX_NAME)
    
    print("\n" + "="*80)
    print("✅ VECTOR STORE CREATION COMPLETE!")
    print("="*80)
    print()
    print("Summary:")
    print(f"  • PDF files processed: {len(pdf_files)}")
    print(f"  • Documents loaded: {len(extracted_data)}")
    print(f"  • Text chunks created: {len(text_chunks)}")
    print(f"  • Vectors in database: {final_count}")
    print()
    print("Next steps:")
    print("  1. Test with: python test_chatbot.py")
    print("  2. Run application: python app.py")
    print()

except Exception as e:
    print("\n" + "="*80)
    print("❌ ERROR OCCURRED")
    print("="*80)
    print(f"\nError: {str(e)}")
    print("\nTroubleshooting:")
    print("  • Check your API keys in .env file")
    print("  • Ensure PDF files are valid and readable")
    print("  • Check your internet connection")
    print("  • Verify Pinecone account is active")
    print()
    sys.exit(1)