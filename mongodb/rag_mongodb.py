import os
import logging
from typing import List, Dict, Optional, Union
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import magic
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Retrieval Augmented Generation system using MongoDB Atlas Vector Search
    """
    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        db_name: str = "rag_demo",
        collection_name: str = "vector_search",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
    ):
        # Load environment variables
        load_dotenv()
        
        # Initialize MongoDB connection
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_ATLAS_CLUSTER_URI")
        if not self.mongodb_uri:
            raise ValueError("MongoDB URI is required")
        
        self.client = MongoClient(self.mongodb_uri)
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection = self.client[db_name][collection_name]
        
        # Initialize OpenAI
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize vector store
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name="default"
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=temperature,
            model=model_name,
            openai_api_key=self.openai_api_key
        )
        
        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
        )
        
        logger.info(f"Initialized RAG system with database '{db_name}' and collection '{collection_name}'")

    def _get_file_type(self, file_path: str) -> str:
        """
        Determine the file type using python-magic
        """
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        return file_type

    def _get_loader(self, file_path: str):
        """
        Get the appropriate document loader based on file type
        """
        file_type = self._get_file_type(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_type == "text/plain" or file_extension == ".txt":
            return TextLoader(file_path)
        elif file_type == "application/pdf" or file_extension == ".pdf":
            return PyPDFLoader(file_path)
        elif file_type == "text/csv" or file_extension == ".csv":
            return CSVLoader(file_path)
        elif file_type == "text/html" or file_extension in [".html", ".htm"]:
            return UnstructuredHTMLLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def ingest_document(self, file_path: str) -> Dict[str, Union[str, int]]:
        """
        Ingest a document into the vector store
        """
        try:
            logger.info(f"Ingesting document: {file_path}")
            
            # Load the document
            loader = self._get_loader(file_path)
            documents = loader.load()
            
            # Split text into chunks
            texts = self.text_splitter.split_documents(documents)
            
            # Convert documents to dictionary format for MongoDB
            docs_as_dicts = [{
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "embedding": self.embeddings.embed_query(doc.page_content)
            } for doc in texts]
            
            # Insert documents into MongoDB
            result = self.collection.insert_many(docs_as_dicts)
            
            logger.info(f"Successfully inserted {len(docs_as_dicts)} documents")
            return {
                "status": "success",
                "documents_inserted": len(docs_as_dicts),
                "message": f"Inserted {len(docs_as_dicts)} documents into MongoDB"
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            return {
                "status": "error",
                "message": f"Error ingesting document: {str(e)}"
            }

    def batch_ingest_documents(self, directory_path: str) -> List[Dict[str, Union[str, int]]]:
        """
        Ingest all supported documents in a directory
        """
        results = []
        try:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        result = self.ingest_document(file_path)
                        results.append({
                            "file": file_path,
                            **result
                        })
                    except Exception as e:
                        results.append({
                            "file": file_path,
                            "status": "error",
                            "message": str(e)
                        })
            return results
        except Exception as e:
            logger.error(f"Error in batch ingestion: {str(e)}")
            return [{
                "status": "error",
                "message": f"Error in batch ingestion: {str(e)}"
            }]

    def query(
        self,
        question: str,
        search_kwargs: Optional[Dict] = None
    ) -> Dict[str, Union[str, List[Dict]]]:
        """
        Query the RAG system with a question
        """
        try:
            # Update search parameters if provided
            if search_kwargs:
                self.qa_chain.retriever.search_kwargs.update(search_kwargs)
            
            # Get response
            response = self.qa_chain.invoke({
                "query": question
            })
            
            # Get source documents
            source_docs = [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "score": doc.metadata.get("score", None)
                }
                for doc in response.get("source_documents", [])
            ]
            
            return {
                "status": "success",
                "answer": response["result"],
                "source_documents": source_docs
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }

    def clear_database(self) -> Dict[str, str]:
        """
        Clear all documents from the collection
        """
        try:
            result = self.collection.delete_many({})
            logger.info(f"Deleted {result.deleted_count} documents")
            return {
                "status": "success",
                "message": f"Deleted {result.deleted_count} documents"
            }
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return {
                "status": "error",
                "message": f"Error clearing database: {str(e)}"
            }

    def get_collection_stats(self) -> Dict[str, Union[int, List[str]]]:
        """
        Get statistics about the collection
        """
        try:
            doc_count = self.collection.count_documents({})
            unique_sources = self.collection.distinct("source")
            
            return {
                "status": "success",
                "document_count": doc_count,
                "unique_sources": unique_sources
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting collection stats: {str(e)}"
            }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
        logger.info("Closed MongoDB connection")

if __name__ == "__main__":
    # Example usage
    rag = RAGSystem()
    
    # Ingest a document
    response = rag.ingest_document("document.txt")
    print(response)
    
    # Query the system
    response = rag.query("What is the truthful qa?")
    print(response)
