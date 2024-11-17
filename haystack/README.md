# Haystack RAG Implementation

This directory contains implementation of a Retrieval-Augmented Generation (RAG) system using Haystack AI.

## Setup Instructions

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate haystack-rag
```

2. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env file and add your OpenAI API key
```

3. Run the example:
```bash
python rag_implementation.py
```

## Project Structure

- `rag_implementation.py`: Main implementation of the RAG system
- `environment.yml`: Conda environment specification
- `requirements.txt`: Python package dependencies
- `.env.example`: Template for environment variables

## Components

The implementation includes:
- Document Store: In-memory document storage
- Embedders: SentenceTransformers for document and query embedding
- Retriever: In-memory embedding retriever
- Generator: OpenAI-based text generation
- Document Preprocessor: For splitting documents into chunks

## Pipeline Architecture

The system uses two main pipelines:
1. Indexing Pipeline:
   - Document preprocessing
   - Document embedding
   - Storage in document store

2. Querying Pipeline:
   - Query embedding
   - Document retrieval
   - Response generation

## Usage

1. Add your documents to the system by modifying the documents list in `rag_implementation.py`
2. Run the script to see example usage
3. Modify the query in the main section to ask different questions

## Requirements

- Python 3.10
- OpenAI API key
- Dependencies listed in environment.yml
