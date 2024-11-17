from verba_rag import VerbaRAG
import os

def test_rag():
    # Initialize RAG system
    rag = VerbaRAG()
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # You can place your test PDF in the current directory
    pdf_path = os.path.join(current_dir, "test.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"Please place a test PDF file named 'test.pdf' in {current_dir}")
        return
    
    try:
        # Ingest document
        print("Ingesting document...")
        vectorstore = rag.ingest_documents([pdf_path])
        
        # Create QA chain
        print("Creating QA chain...")
        qa_chain = rag.create_qa_chain(vectorstore)
        
        # Test query
        print("\nTesting query...")
        query = "What is this document about?"
        result = rag.query(qa_chain, query)
        
        print(f"\nQuestion: {query}")
        print(f"Answer: {result['answer']}")
        print("\nSources:")
        for doc in result['source_documents']:
            print(f"- {doc.metadata['source']}")
            
        print("\nRAG system test completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_rag()
