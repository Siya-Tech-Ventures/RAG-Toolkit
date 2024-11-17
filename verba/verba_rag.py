import os
from typing import List
import weaviate
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

class VerbaRAG:
    def __init__(self):
        self.client = weaviate.Client(
            url=os.getenv("WEAVIATE_URL"),
            auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),
        )
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def ingest_documents(self, pdf_paths: List[str]):
        """
        Ingest PDF documents into the Weaviate vector store
        """
        documents = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        
        # Split documents into chunks
        splits = self.text_splitter.split_documents(documents)
        
        # Create Weaviate vector store
        vectorstore = Weaviate.from_documents(
            documents=splits,
            embedding=self.embeddings,
            client=self.client,
            by_text=False
        )
        
        return vectorstore

    def create_qa_chain(self, vectorstore):
        """
        Create a conversational retrieval chain
        """
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
        )
        
        return qa_chain

    def query(self, qa_chain, query: str, chat_history: List = None):
        """
        Query the RAG system
        """
        if chat_history is None:
            chat_history = []
            
        result = qa_chain({"question": query, "chat_history": chat_history})
        
        return {
            "answer": result["answer"],
            "source_documents": result["source_documents"],
        }

def main():
    # Initialize RAG system
    rag = VerbaRAG()
    
    # Example usage
    pdf_paths = ["test.pdf"]  # Replace with your PDF paths
    vectorstore = rag.ingest_documents(pdf_paths)
    qa_chain = rag.create_qa_chain(vectorstore)
    
    # Example query
    query = "What is this document about?"
    result = rag.query(qa_chain, query)
    print(f"Question: {query}")
    print(f"Answer: {result['answer']}")
    print("\nSources:")
    for doc in result['source_documents']:
        print(f"- {doc.metadata['source']}")

if __name__ == "__main__":
    main()
