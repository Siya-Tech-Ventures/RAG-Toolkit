import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from phoenix.otel import register
import glob
from typing import List, Optional
import sys
from openinference.instrumentation.openai import OpenAIInstrumentor

# Load environment variables
load_dotenv()

# Initialize Phoenix tracer
tracer_provider = register(
    project_name="phoenix-rag-demo",
    endpoint="https://app.phoenix.arize.com/v1/traces"
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PhoenixRAG:
    def __init__(self, persist_directory: Optional[str] = None):
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize vector store with persistence
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            collection_name="phoenix_docs",
            persist_directory=persist_directory
        )

    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Load all text files from a directory."""
        documents = []
        # Get all text files in the directory
        for file_path in glob.glob(os.path.join(directory_path, "*.txt")):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Create document with metadata including source file
                    doc = Document(
                        page_content=content,
                        metadata={"source": file_path}
                    )
                    documents.append(doc)
                print(f"Loaded document: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        return documents

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        try:
            self.vector_store.add_documents(documents)
            print(f"Added {len(documents)} documents to the vector store")
        except Exception as e:
            print(f"Error adding documents: {str(e)}")

    def query(self, question: str) -> str:
        """Query the RAG system with a question."""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.similarity_search(question, k=3)
            
            # Create context from retrieved documents with source information
            context_parts = []
            for doc in retrieved_docs:
                source = doc.metadata.get('source', 'Unknown source')
                content = doc.page_content
                context_parts.append(f"Source: {source}\nContent: {content}")
            
            context = "\n\n".join(context_parts)
            
            # Create prompt
            prompt = f"""Answer the question based on the following context. If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
            
            # Generate response using OpenAI directly
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing query: {str(e)}"

def main():
    # Set up the RAG system with persistence
    persist_dir = "chroma_db"
    rag = PhoenixRAG(persist_directory=persist_dir)
    
    # Process command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--load":
        # Load documents from the data directory
        print("Loading documents from the data directory...")
        documents = rag.load_documents_from_directory("data")
        if documents:
            rag.add_documents(documents)
            print("Documents loaded successfully!")
        else:
            print("No documents found in the data directory.")
        return

    # Interactive query loop
    print("\nWelcome to the Phoenix RAG System!")
    print("Type 'exit' to quit the program.")
    print("Type 'reload' to reload documents from the data directory.")
    print("-" * 50)

    while True:
        try:
            # Get user input
            question = input("\nEnter your question: ").strip()
            
            # Check for exit command
            if question.lower() == 'exit':
                print("Goodbye!")
                break
            
            # Check for reload command
            if question.lower() == 'reload':
                documents = rag.load_documents_from_directory("data")
                if documents:
                    rag.add_documents(documents)
                continue
            
            # Skip empty questions
            if not question:
                continue
            
            # Get and print the answer
            print("\nThinking...")
            answer = rag.query(question)
            print("\nAnswer:", answer)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
