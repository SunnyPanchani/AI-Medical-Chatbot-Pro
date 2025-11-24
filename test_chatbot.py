"""
Test Chatbot Script
Quick test to verify your medical chatbot setup
"""

from src.helper import setup_medical_chatbot, ask_question
import sys

def main():
    print("\n" + "="*80)
    print("MEDICAL CHATBOT TEST SCRIPT")
    print("="*80 + "\n")
    
    try:
        # Setup the chatbot
        print("🚀 Initializing medical chatbot...\n")
        rag_chain, retriever, docsearch = setup_medical_chatbot(
            data_path="data",
            index_name="medical-chatbot",
            force_recreate=False,
            num_documents=5
        )
        
        # Test questions
        test_questions = [
            "What is diabetes mellitus? Explain its types and symptoms.",
            "What are the causes and treatment options for hypertension?",
            "Describe acne, its causes, and how it can be treated."
        ]
        
        print("\n" + "="*80)
        print("🧪 RUNNING TEST QUESTIONS")
        print("="*80 + "\n")
        
        results = []
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*80}")
            print(f"TEST {i}/{len(test_questions)}")
            print(f"{'='*80}\n")
            
            result = ask_question(
                rag_chain,
                question,
                retriever=retriever,
                show_sources=True,
                show_timing=True
            )
            results.append(result)
        
        # Summary
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80 + "\n")
        
        total_time = sum(r['time'] for r in results)
        avg_time = total_time / len(results)
        
        print(f"✅ All {len(test_questions)} tests completed successfully!")
        print(f"\nPerformance:")
        print(f"  • Total time: {total_time:.2f} seconds")
        print(f"  • Average time per question: {avg_time:.2f} seconds")
        print(f"  • Questions answered: {len(results)}")
        
        print("\n" + "="*80)
        print("✅ CHATBOT IS WORKING PERFECTLY!")
        print("="*80)
        print("\nYou can now:")
        print("  1. Run the Flask app: python app.py")
        print("  2. Use the chatbot in your notebooks")
        print("  3. Integrate into your application")
        print()
        
    except FileNotFoundError as e:
        print("\n❌ Error: Required files not found")
        print(f"Details: {e}")
        print("\nPlease ensure:")
        print("  • data/ folder exists with PDF files")
        print("  • .env file contains valid API keys")
        sys.exit(1)
        
    except ValueError as e:
        print("\n❌ Error: Configuration issue")
        print(f"Details: {e}")
        print("\nPlease check:")
        print("  • PINECONE_API_KEY in .env file")
        print("  • GROQ_API_KEY in .env file")
        sys.exit(1)
        
    except Exception as e:
        print("\n❌ Unexpected Error")
        print(f"Details: {e}")
        print("\nTroubleshooting:")
        print("  • Run: python store_index.py (to create vector store)")
        print("  • Check your internet connection")
        print("  • Verify API keys are valid")
        sys.exit(1)

if __name__ == "__main__":
    main()