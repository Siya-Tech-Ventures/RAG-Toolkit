import os
from typing import List
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

class MultiPDFRAGSystem:
    def __init__(self, pdf_folder: str):
        """Initialize the RAG system with a folder containing PDF documents."""
        self.pdf_folder = pdf_folder
        self.embeddings = OpenAIEmbeddings()
        self.chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.vector_store = None
        self.qa_chain = None
        
    def load_and_process_pdfs(self):
        """Load and process all PDF documents from the folder."""
        all_documents = []
        
        # Get all PDF files from the folder
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.pdf_folder}")
        
        print(f"Found {len(pdf_files)} PDF files. Processing...")
        
        # Process each PDF file
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            print(f"Processing {pdf_file}...")
            
            try:
                # Load PDF
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                all_documents.extend(documents)
                print(f"Successfully processed {pdf_file}")
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                continue
        
        if not all_documents:
            raise ValueError("No documents were successfully processed")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(all_documents)
        
        print(f"Created {len(splits)} text chunks from {len(pdf_files)} PDFs")
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        
        # Create QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.chat_model,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
        )
        
    def ask_question(self, question: str, chat_history: List = None) -> str:
        """Ask a question about the PDF content."""
        if chat_history is None:
            chat_history = []
            
        if not self.qa_chain:
            raise ValueError("Please load and process PDFs first using load_and_process_pdfs()")
            
        response = self.qa_chain({"question": question, "chat_history": chat_history})
        return response["answer"]

def main():
    # Example usage
    pdf_folder = "data"  # Replace with your PDF folder path
    
    # Initialize and load PDFs
    rag_system = MultiPDFRAGSystem(pdf_folder)
    rag_system.load_and_process_pdfs()
    
    # Interactive question-answering loop
    print("\nRAG System ready! Type 'quit' to exit.")
    chat_history = []
    
    while True:
        question = input("\nEnter your question: ").strip()
        if question.lower() == 'quit':
            break
            
        try:
            answer = rag_system.ask_question(question, chat_history)
            print(f"\nAnswer: {answer}")
            chat_history.append((question, answer))
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
