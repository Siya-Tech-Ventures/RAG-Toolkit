# MongoDB RAG (Retrieval Augmented Generation) System

A powerful and flexible Retrieval Augmented Generation system built using MongoDB Atlas Vector Search and OpenAI's language models.

## Features

- **Multi-Document Support**: Handle various document types including:
  - Text files (.txt)
  - PDF documents (.pdf)
  - CSV files (.csv)
  - HTML documents (.html, .htm)

- **Advanced Document Processing**:
  - Automatic file type detection
  - Smart text chunking with configurable size and overlap
  - Batch document ingestion
  - Detailed ingestion reporting

- **Sophisticated Querying**:
  - Configurable search parameters
  - Source document tracking
  - Flexible retrieval options
  - Comprehensive response formatting

- **Robust Error Handling**:
  - Detailed error messages
  - Comprehensive logging
  - Graceful error recovery

- **Database Management**:
  - Collection statistics
  - Easy database cleanup
  - Connection management with context support

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RAG-Toolkit/mongodb
   ```

2. Create and activate a Python virtual environment:
   ```bash
   conda create -n mongodb-rag python=3.10
   conda activate mongodb-rag
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables in `.env`:
   ```
   MONGODB_ATLAS_CLUSTER_URI=your_mongodb_uri
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Basic Example

```python
from rag_mongodb import RAGSystem

# Initialize the system
rag = RAGSystem()

# Ingest a document
result = rag.ingest_document("path/to/document.txt")
print(result)

# Query the system
response = rag.query("What is artificial intelligence?")
print(response)
```

### Advanced Usage

```python
# Initialize with custom settings
rag = RAGSystem(
    db_name="custom_db",
    collection_name="custom_collection",
    chunk_size=500,
    chunk_overlap=50,
    temperature=0.2
)

# Batch ingest documents
results = rag.batch_ingest_documents("path/to/documents/")

# Query with custom search parameters
response = rag.query(
    "What is machine learning?",
    search_kwargs={"k": 5}  # Use top 5 results
)

# Get collection statistics
stats = rag.get_collection_stats()
print(stats)
```

## Configuration

### Environment Variables

- `MONGODB_ATLAS_CLUSTER_URI`: Your MongoDB Atlas connection string
- `OPENAI_API_KEY`: Your OpenAI API key

### Optional Parameters

- `db_name`: MongoDB database name (default: "rag_demo")
- `collection_name`: Collection name (default: "vector_search")
- `chunk_size`: Text chunk size (default: 1000)
- `chunk_overlap`: Chunk overlap size (default: 100)
- `model_name`: OpenAI model name (default: "gpt-3.5-turbo")
- `temperature`: Model temperature (default: 0)

## Dependencies

- `pymongo`: MongoDB Python driver
- `langchain`: LLM framework
- `openai`: OpenAI API client
- `python-magic`: File type detection
- Various document loaders for different file types

## Error Handling

The system provides detailed error messages and logging for:
- File type errors
- Connection issues
- Query processing problems
- Database operations

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
