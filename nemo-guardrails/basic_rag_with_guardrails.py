import os
from typing import List, Optional
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()

class RAGWithGuardrails:
    def __init__(self, config_path: str, openai_api_key: Optional[str] = None):
        """
        Initialize RAG with NeMo Guardrails.
        
        Args:
            config_path (str): Path to the guardrails configuration file
            openai_api_key (Optional[str]): OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable
        """
        # Get API key from parameter or environment variable
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either through the openai_api_key parameter "
                "or set as an environment variable OPENAI_API_KEY"
            )

        self.rails_config = RailsConfig.from_path(config_path)
        self.rails = LLMRails(self.rails_config)
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = OpenAI(openai_api_key=self.api_key)
        self.vector_store = None

    def load_documents(self, file_paths: List[str]):
        """
        Load and process documents into the vector store.
        
        Args:
            file_paths (List[str]): List of paths to text files to load
        """
        documents = []
        for file_path in file_paths:
            loader = TextLoader(file_path)
            documents.extend(loader.load())

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        split_docs = text_splitter.split_documents(documents)

        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings
        )

    def query(self, user_input: str) -> str:
        """
        Process user query through guardrails and RAG.
        
        Args:
            user_input (str): User's question or input
            
        Returns:
            str: Generated response
        """
        if not self.vector_store:
            return "Please load documents first using load_documents()"

        try:
            # Retrieve relevant documents, limiting to available documents
            total_docs = len(self.vector_store.get()['ids'])
            k = min(2, total_docs)  # Get at most 2 documents
            docs = self.vector_store.similarity_search(user_input, k=k)
            context = "\n".join([doc.page_content for doc in docs])

            # Add context to user message
            messages = [
                {"role": "system", "content": f"Context: {context}\n\nRespond based on the context provided. If the information isn't in the context, say so."},
                {"role": "user", "content": user_input}
            ]
            # Generate response using guardrails
            response = self.rails.generate(messages=messages)
            # Handle different response formats
            if isinstance(response, dict):
                return response.get('content', response.get('output', str(response)))
            elif hasattr(response, 'message'):
                return response.message.get('content', str(response.message))
            else:
                return str(response)

        except Exception as e:
            return f"Error processing query: {str(e)}"


def main():
    # Example usage
    config_path = "config"  # Path to your guardrails config directory
    
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    try:
        # Initialize RAG with API key
        rag = RAGWithGuardrails(config_path, openai_api_key=api_key)

        # Example: Load some documents
        documents = ["data/sample1.txt", "data/sample2.txt"]
        rag.load_documents(documents)
        
        # Example query
        question = "What are the key points in the documents?"
        response = rag.query(question)
        print(f"Question: {question}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
