import os
from dotenv import load_dotenv
from basic_rag_with_guardrails import RAGWithGuardrails

def main():
    # Load environment variables
    load_dotenv()
    
    print("NeMo Guardrails RAG Interactive Demo")
    print("====================================")
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\nError: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key in the .env file or as an environment variable.")
        return
    
    # Initialize the RAG system
    print("\nInitializing RAG system...")
    try:
        rag = RAGWithGuardrails("config", openai_api_key=api_key)
    except Exception as e:
        print(f"\nError initializing RAG system: {str(e)}")
        return
    
    # Load documents
    print("Loading sample documents...")
    try:
        documents = ["data/sample1.txt", "data/sample2.txt"]
        rag.load_documents(documents)
        print("Documents loaded successfully!")
    except Exception as e:
        print(f"\nError loading documents: {str(e)}")
        return
    
    print("\nYou can now ask questions about AI Safety and Guardrails implementation.")
    print("Type 'exit' to quit.\n")
    
    while True:
        # Get user input
        user_input = input("\nYour question: ")
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using the demo!")
            break
            
        try:
            # Get response
            response = rag.query(user_input)
            print("\nResponse:", response)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try another question or check your OpenAI API key.")

if __name__ == "__main__":
    main()
