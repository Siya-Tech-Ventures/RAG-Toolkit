# Phoenix RAG Implementation

This directory contains a Retrieval Augmented Generation (RAG) implementation using Phoenix for tracing and monitoring.

## Features

- Document loading from text files
- Vector storage with ChromaDB
- OpenAI integration for embeddings and text generation
- Phoenix tracing with OpenTelemetry
- Interactive query interface
- Persistent vector storage
- OpenInference instrumentation for OpenAI calls

## Setup Guide

### 1. Environment Setup

Choose one of the following methods:

#### Using Conda (Recommended)
```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate the environment
conda activate phoenix-rag
```

#### Using Pip
```bash
# Create conda environment
conda create -n phoenix-rag python=3.10 -y

# Activate the environment
conda activate phoenix-rag

# Install requirements
pip install -r requirements.txt
```

### 2. Environment Configuration

1. Create a copy of the example .env file:
```bash
cp .env.example .env
```

2. Update the `.env` file with your credentials:
```env
OPENAI_API_KEY=your_openai_api_key
PHOENIX_CLIENT_HEADERS='api_key=your_phoenix_api_key'
PHOENIX_COLLECTOR_ENDPOINT='https://app.phoenix.arize.com'
```

### 3. Prepare Documents

1. Create a data directory if it doesn't exist:
```bash
mkdir -p data
```

2. Add your text documents to the `data` directory:
- Files should be in .txt format
- Each file should contain relevant content for your RAG system
- Example structure:
  ```
  data/
  ├── document1.txt
  ├── document2.txt
  └── about_phoenix.txt
  ```

## Running the Project

### 1. Initial Document Loading

Before first use, load your documents into the vector store:
```bash
python rag_phoenix.py --load
```

This command will:
- Read all .txt files from the data directory
- Create embeddings using OpenAI
- Store vectors in the ChromaDB database
- Save the database in the `chroma_db` directory

### 2. Interactive Query Mode

Start the interactive query interface:
```bash
python rag_phoenix.py
```

### 3. Available Commands

In interactive mode:
- Type your question and press Enter to get an answer
- Special commands:
  * `exit`: Quit the program
  * `reload`: Reload documents from the data directory
  * Empty line: Skip and continue

### 4. Monitoring

1. View traces in Phoenix:
   - Visit https://app.phoenix.arize.com
   - Navigate to your project (phoenix-rag-demo)
   - View traces, embeddings, and LLM calls

2. Monitoring features:
   - Track OpenAI API calls
   - Monitor embedding generation
   - Analyze retrieval performance
   - View full conversation traces

## Project Structure

```
phoenix/
├── rag_phoenix.py      # Main implementation
├── requirements.txt    # Pip dependencies
├── environment.yml     # Conda environment
├── .env               # Environment variables
├── data/              # Text documents
│   └── *.txt
└── chroma_db/         # Vector database
```

## Troubleshooting

1. OpenAI API Issues:
   - Verify your API key in .env
   - Check API usage limits
   - Ensure internet connectivity

2. Document Loading Issues:
   - Verify text files are in UTF-8 format
   - Check file permissions
   - Ensure files are in .txt format

3. Vector Store Issues:
   - Delete chroma_db directory and reload if corrupted
   - Check disk space
   - Verify ChromaDB installation

4. Phoenix Tracing Issues:
   - Verify Phoenix API key
   - Check collector endpoint
   - Ensure proper environment variable setup

## Dependencies

Major dependencies and their versions:
- arize-phoenix-otel >= 0.6.1
- openai >= 1.0.0
- langchain >= 0.1.0
- chromadb >= 0.4.0
- openinference-semantic-conventions >= 0.1.12

For a complete list, see `requirements.txt`.
