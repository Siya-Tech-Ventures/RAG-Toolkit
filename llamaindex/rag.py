import os
from typing import List
from dotenv import load_dotenv
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    Prompt
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import HuggingFaceLLM
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import MetadataMode
import argparse
from rich.console import Console
from rich.markdown import Markdown
from rich import print as rprint
from rich.panel import Panel
from rich.prompt import Prompt as RichPrompt

# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()

class RAGSystem:
    def __init__(self, data_dir: str, persist_dir: str = "./storage"):
        """Initialize the RAG system with a directory containing documents."""
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        
        with console.status("[bold green]Setting up RAG system..."):
            self.setup_llm_and_embeddings()
            
            # Try to load existing index, otherwise create new one
            if os.path.exists(self.persist_dir):
                console.print("[yellow]Loading existing index...")
                self.load_from_disk()
            else:
                console.print("[yellow]Creating new index...")
                self.load_and_index_documents()

    def setup_llm_and_embeddings(self):
        """Set up the embedding model and LLM."""
        # Initialize embedding model
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Initialize LLM with simpler configuration
        llm = HuggingFaceLLM(
            model_name="distilgpt2",  # Using a smaller model
            tokenizer_name="distilgpt2",
            context_window=512,  # Smaller context window
            max_new_tokens=128,  # Reduced max tokens
            model_kwargs={"temperature": 0.7},
            tokenizer_kwargs={"padding": True}
        )

        # Create service context with simpler configuration
        self.service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            node_parser=SimpleNodeParser.from_defaults(
                chunk_size=256,  # Smaller chunk size
                chunk_overlap=10  # Reduced overlap
            )
        )

    def load_from_disk(self):
        """Load the index from disk if it exists."""
        storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
        self.index = load_index_from_storage(
            storage_context=storage_context,
            service_context=self.service_context
        )

    def load_and_index_documents(self):
        """Load documents and create a new index."""
        try:
            # Load documents
            documents = SimpleDirectoryReader(self.data_dir).load_data()
            console.print(f"[green]Loaded {len(documents)} documents from {self.data_dir}")
            
            # Create index
            self.index = VectorStoreIndex.from_documents(
                documents,
                service_context=self.service_context,
                show_progress=True
            )
            
            # Persist index to disk
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            console.print("[green]Successfully created and persisted the index!")
        except Exception as e:
            console.print(f"[red]Error loading documents: {str(e)}")
            raise

    def query(self, question: str, num_results: int = 3) -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question (str): The question to ask
            num_results (int): Number of relevant chunks to retrieve
            
        Returns:
            str: The generated answer
        """
        try:
            # Create a query engine with custom prompt
            query_engine = self.index.as_query_engine(
                similarity_top_k=num_results,
            )
            
            # Get response
            with console.status("[bold green]Generating response..."):
                response = query_engine.query(question)
            return str(response)
        except Exception as e:
            console.print(f"[red]Error during query: {str(e)}")
            return f"Error: {str(e)}"

    def get_relevant_chunks(self, question: str, num_results: int = 3) -> List[dict]:
        """
        Retrieve the most relevant document chunks for a given question.
        
        Args:
            question (str): The question to find relevant chunks for
            num_results (int): Number of chunks to retrieve
            
        Returns:
            List[dict]: List of relevant chunks with text and metadata
        """
        try:
            retriever = self.index.as_retriever(similarity_top_k=num_results)
            nodes = retriever.retrieve(question)
            
            chunks = []
            for node in nodes:
                chunks.append({
                    'text': node.node.get_content(metadata_mode=MetadataMode.NONE),
                    'metadata': node.node.metadata,
                    'score': node.score
                })
            return chunks
        except Exception as e:
            console.print(f"[red]Error retrieving chunks: {str(e)}")
            return []

def interactive_mode(rag: RAGSystem):
    """Run the RAG system in interactive mode."""
    console.print(Panel.fit(
        "[bold green]Welcome to the Interactive RAG System![/bold green]\n"
        "Type your questions and press Enter. Type 'exit' to quit.\n"
        "Commands:\n"
        "- [bold]!chunks[/bold]: Show relevant chunks for the last question\n"
        "- [bold]!help[/bold]: Show this help message\n"
        "- [bold]exit[/bold]: Exit the program"
    ))

    last_question = None
    while True:
        try:
            # Get user input
            question = RichPrompt.ask("\n[bold blue]Enter your question")
            
            # Handle commands
            if question.lower() == 'exit':
                console.print("[yellow]Goodbye!")
                break
            elif question.lower() == '!help':
                console.print(Panel.fit(
                    "Commands:\n"
                    "- [bold]!chunks[/bold]: Show relevant chunks for the last question\n"
                    "- [bold]!help[/bold]: Show this help message\n"
                    "- [bold]exit[/bold]: Exit the program"
                ))
                continue
            elif question.lower() == '!chunks' and last_question:
                chunks = rag.get_relevant_chunks(last_question)
                for i, chunk in enumerate(chunks, 1):
                    console.print(Panel(
                        f"[bold]Chunk {i}[/bold]\n"
                        f"Text: {chunk['text']}\n"
                        f"Score: {chunk['score']:.4f}",
                        title=f"Source: {chunk['metadata'].get('file_name', 'Unknown')}"
                    ))
                continue
            elif question.lower() == '!chunks':
                console.print("[yellow]No previous question to show chunks for!")
                continue
            
            # Process the question
            last_question = question
            answer = rag.query(question)
            
            # Display the answer
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Panel(Markdown(answer)))
            
            # Show chunk option
            if RichPrompt.ask("\nWould you like to see the relevant chunks?", choices=["y", "n"], default="n") == "y":
                chunks = rag.get_relevant_chunks(question)
                for i, chunk in enumerate(chunks, 1):
                    console.print(Panel(
                        f"[bold]Chunk {i}[/bold]\n"
                        f"Text: {chunk['text']}\n"
                        f"Score: {chunk['score']:.4f}",
                        title=f"Source: {chunk['metadata'].get('file_name', 'Unknown')}"
                    ))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive RAG System")
    parser.add_argument("--data_dir", type=str, default="data",
                      help="Directory containing documents to index")
    parser.add_argument("--persist_dir", type=str, default="./storage",
                      help="Directory to persist the index")
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG system
        rag = RAGSystem(args.data_dir, args.persist_dir)
        
        # Run interactive mode
        interactive_mode(rag)
    except Exception as e:
        console.print(f"[red bold]Fatal error: {str(e)}")
