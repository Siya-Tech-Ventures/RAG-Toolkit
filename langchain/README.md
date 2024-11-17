# PDF-based RAG System using LangChain

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, designed to process multiple PDF documents and answer questions based on their content. The system uses OpenAI's embeddings and the ChromaDB vector store for efficient retrieval.

## Features

- Multiple PDF document processing
- Automatic text chunking and embedding
- Semantic search using ChromaDB vector store
- Conversational question-answering interface
- Support for maintaining chat history

## Prerequisites

- Python 3.9 or higher
- Conda package manager
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langchain_rag
```

2. Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate langchain-rag
```

3. Set up your OpenAI API key:
   - Copy your API key to the `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Project Structure

```
langchain_rag/
├── data/               # Directory for PDF files
├── pdf_rag.py         # Main RAG implementation
├── environment.yml    # Conda environment file
├── requirements.txt   # Python dependencies
├── .env              # Environment variables
└── README.md         # This file
```

## Usage

1. Place your PDF files in the `data` directory.

2. Run the RAG system:
```bash
python pdf_rag.py
```

3. Start asking questions about your documents. Type 'quit' to exit.

## How it Works

1. **Document Loading**: The system loads all PDF files from the data directory using PyPDF Loader.

2. **Text Processing**: Documents are split into smaller chunks using RecursiveCharacterTextSplitter for better processing.

3. **Embedding Creation**: Text chunks are converted into embeddings using OpenAI's embedding model.

4. **Vector Storage**: Embeddings are stored in a ChromaDB vector store for efficient retrieval.

5. **Question Answering**: When a question is asked:
   - The system finds relevant text chunks using semantic search
   - These chunks are used as context for the LLM to generate accurate answers
   - The conversation history is maintained for context-aware responses

## Dependencies

- langchain==0.0.352
- python-dotenv==1.0.0
- openai==1.3.7
- chromadb==0.4.18
- pypdf==3.17.1
- tiktoken==0.5.1

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
