# Verba RAG System

A Retrieval-Augmented Generation (RAG) system built using Verba, Weaviate, and OpenAI. This system allows you to ingest PDF documents and perform question-answering tasks with context from the documents.

## Features

- PDF document ingestion and processing
- Document chunking with overlap for better context
- Vector storage using Weaviate
- Conversational retrieval using OpenAI's GPT models
- Source attribution for answers
- Easy-to-use Python interface

## Prerequisites

- Python 3.10
- Conda (for environment management)
- OpenAI API key
- Weaviate instance (cloud or local)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG-Toolkit/verba
```

2. Create and activate the conda environment:
```bash
conda create -n verba-rag python=3.10 -y
conda activate verba-rag
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your credentials:
```env
OPENAI_API_KEY=your_openai_api_key
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API_KEY=your_weaviate_api_key
```

## Usage

### Basic Usage

```python
from verba_rag import VerbaRAG

# Initialize RAG system
rag = VerbaRAG()

# Ingest documents
pdf_paths = ["path/to/your/document.pdf"]
vectorstore = rag.ingest_documents(pdf_paths)

# Create QA chain
qa_chain = rag.create_qa_chain(vectorstore)

# Query the system
query = "What is this document about?"
result = rag.query(qa_chain, query)
print(f"Answer: {result['answer']}")
print("\nSources:")
for doc in result['source_documents']:
    print(f"- {doc.metadata['source']}")
```

### Running Tests

1. Place a test PDF file named `test.pdf` in the project directory
2. Run the test script:
```bash
python test_rag.py
```

## System Components

1. **Document Processing**
   - Uses PyPDFLoader for PDF ingestion
   - Implements RecursiveCharacterTextSplitter for document chunking
   - Configurable chunk size and overlap

2. **Vector Storage**
   - Uses Weaviate as the vector database
   - Stores document embeddings for efficient retrieval
   - Supports semantic search capabilities

3. **Language Model**
   - Utilizes OpenAI's GPT-3.5-turbo model
   - Supports conversational context
   - Provides source attribution

## Configuration

The system can be configured through the following environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `WEAVIATE_URL`: URL of your Weaviate instance
- `WEAVIATE_API_KEY`: API key for Weaviate authentication

## Project Structure

```
verba/
├── requirements.txt    # Project dependencies
├── verba_rag.py       # Main RAG implementation
├── test_rag.py        # Test script
└── .env               # Environment variables
```

## Dependencies

- weaviate-client==3.25.2
- openai==1.3.5
- python-dotenv==1.0.0
- langchain==0.1.0
- langchain-community==0.0.13
- langchain-core==0.1.10
- langchain-openai==0.0.8
- chromadb==0.4.18
- sentence-transformers==2.2.2
- pypdf==3.17.1
- tiktoken==0.5.1

## Error Handling

The system includes basic error handling for:
- Missing API credentials
- PDF processing errors
- Vector store operations
- Query processing

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[Your chosen license]
