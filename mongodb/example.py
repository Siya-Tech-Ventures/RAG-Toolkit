import os
import logging
from rag_mongodb import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize the RAG system with custom settings
    rag = RAGSystem(
        db_name="rag_demo",
        collection_name="vector_search",
        chunk_size=500,  # Smaller chunks for this example
        chunk_overlap=50,
        temperature=0.2  # Add some creativity to responses
    )
    
    try:
        # Create a directory for test documents
        os.makedirs("test_docs", exist_ok=True)
        
        # Create example documents of different types
        documents = {
            "ai_intro.txt": """
            Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, 
            especially computer systems. These processes include learning (the acquisition of information 
            and rules for using the information), reasoning (using rules to reach approximate or definite 
            conclusions) and self-correction.

            Particular applications of AI include expert systems, speech recognition and machine vision.
            """,
            
            "ai_types.txt": """
            There are several types of artificial intelligence:

            1. Narrow AI (Weak AI): Designed for a specific task
            2. General AI (Strong AI): Capable of performing any intellectual task
            3. Super AI: Hypothetical AI that surpasses human intelligence

            Each type has its own characteristics and applications.
            """,
            
            "ai_ethics.txt": """
            AI Ethics is a crucial consideration in development and deployment.
            Key principles include:
            - Transparency
            - Fairness
            - Privacy
            - Accountability
            - Safety

            These principles help ensure AI benefits society while minimizing risks.
            """
        }
        
        # Save example documents
        for filename, content in documents.items():
            filepath = os.path.join("test_docs", filename)
            with open(filepath, "w") as f:
                f.write(content)
        
        # Get initial collection stats
        logger.info("Initial collection stats:")
        stats = rag.get_collection_stats()
        logger.info(f"Documents in collection: {stats.get('document_count', 0)}")
        
        # Batch ingest documents
        logger.info("\nIngesting documents...")
        results = rag.batch_ingest_documents("test_docs")
        for result in results:
            if result["status"] == "success":
                logger.info(f"Successfully ingested {result['file']}")
            else:
                logger.error(f"Failed to ingest {result['file']}: {result['message']}")
        
        # Get updated collection stats
        logger.info("\nUpdated collection stats:")
        stats = rag.get_collection_stats()
        logger.info(f"Documents in collection: {stats.get('document_count', 0)}")
        logger.info(f"Unique sources: {stats.get('unique_sources', [])}")
        
        # Example queries with different search parameters
        questions = [
            ("What is Artificial Intelligence?", {"k": 3}),  # Use top 3 results
            ("What are the different types of AI?", {"k": 2}),  # Use top 2 results
            ("What are the ethical considerations in AI?", {"k": 4}),  # Use top 4 results
        ]
        
        logger.info("\nAsking questions...")
        for question, search_kwargs in questions:
            logger.info(f"\nQ: {question}")
            response = rag.query(question, search_kwargs=search_kwargs)
            
            if response["status"] == "success":
                logger.info(f"A: {response['answer']}")
                logger.info("\nSources used:")
                for doc in response["source_documents"]:
                    logger.info(f"- {doc['source']}")
            else:
                logger.error(f"Error: {response['message']}")
        
        # Clean up
        logger.info("\nCleaning up...")
        import shutil
        shutil.rmtree("test_docs")
        
        result = rag.clear_database()
        if result["status"] == "success":
            logger.info(result["message"])
        else:
            logger.error(f"Error clearing database: {result['message']}")
    
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
    
    finally:
        # Close MongoDB connection
        rag.client.close()
        logger.info("Finished example")

if __name__ == "__main__":
    main()
