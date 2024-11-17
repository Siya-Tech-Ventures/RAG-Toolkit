import os
from pathlib import Path
from typing import List
import sys
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.preprocessors import DocumentSplitter
from dotenv import load_dotenv

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """Load all text files from a folder and convert them to Documents"""
    documents = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Folder {folder_path} does not exist!")
        return documents
    
    for file_path in folder.glob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Create a Document with the file content and metadata
                doc = Document(
                    content=content,
                )
                documents.append(doc)
                print(f"Loaded document: {file_path.name}")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return documents

def initialize_pipeline(documents: List[Document]):
    """Initialize and return the RAG pipeline"""
    # Initialize document store and write documents
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)
    
    # Create prompt template
    prompt_template = """
    Given these documents, answer the question. If the answer cannot be found in the documents, say so.
    
    Documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}
    
    Question: {{question}}
    
    Answer:
    """
    
    # Initialize components
    retriever = InMemoryBM25Retriever(document_store=document_store)
    prompt_builder = PromptBuilder(template=prompt_template)
    llm = OpenAIGenerator(api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")))
    
    # Build pipeline
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    
    return rag_pipeline

def process_documents(documents: List[Document], preprocessor: DocumentSplitter) -> List[Document]:
    """Process documents using the preprocessor"""
    if not documents:
        print("No documents to process!")
        return []
    
    print(f"\nProcessing {len(documents)} documents...")
    # processed_docs = preprocessor.process(documents)
    # print(f"Created {len(processed_docs)} document chunks")
    return documents

def interactive_query(pipeline: Pipeline):
    """Interactive query loop"""
    print("\nEnter your questions (type 'exit' to quit):")
    
    while True:
        try:
            query = input("\nQuestion: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Run the pipeline
            results = pipeline.run(
                {
                    "retriever": {"query": query},
                    "prompt_builder": {"question": query},
                }
            )
            
            print("\nAnswer:", results["llm"]["replies"][0])
                        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    # Create document splitter for preprocessing
    preprocessor = DocumentSplitter(
        split_by="word",
        split_length=500,
        split_overlap=50
    )
    
    # Load and process documents
    documents = load_documents_from_folder("data")
    processed_docs = process_documents(documents, preprocessor)
    
    if not processed_docs:
        print("No documents to work with. Please add some .txt files to the data directory.")
        return
    
    # Initialize the pipeline
    pipeline = initialize_pipeline(processed_docs)
    
    # Start interactive query session
    interactive_query(pipeline)

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY in the .env file")
        sys.exit(1)
    
    main()
