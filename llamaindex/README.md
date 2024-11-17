# LlamaIndex RAG Implementation

An interactive Retrieval-Augmented Generation (RAG) system built using LlamaIndex, offering an efficient way to query your document collection using natural language.

## Features

- 🚀 Interactive command-line interface with rich formatting
- 📚 Document indexing with automatic chunking and embedding
- 💾 Persistent storage of document indices
- 🔍 Semantic search capabilities
- 🤖 DistilGPT2-powered text generation
- 📊 Relevance scoring for retrieved chunks

## Prerequisites

- Python 3.10+
- Conda (recommended for environment management)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG-Toolkit/llamaindex
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate llamaindex-rag
```

Alternatively, you can install dependencies using pip:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your documents:
   - Create a `data` directory
   - Place your documents (PDF, TXT, etc.) in the data directory

2. Run the RAG system:
```bash
python rag.py --data_dir ./data --persist_dir ./storage
```

### Interactive Commands

- Type your questions and press Enter to get answers
- Special commands:
  - `!help`: Show available commands
  - `!chunks`: Show relevant chunks from the last question
  - `exit`: Exit the program

### Command Line Arguments

- `--data_dir`: Directory containing your documents (default: "data")
- `--persist_dir`: Directory to store the index (default: "./storage")

## System Components

### 1. Document Processing
- Automatic document loading and chunking
- Configurable chunk size (256 tokens) and overlap (10 tokens)
- Metadata preservation for source tracking

### 2. Embedding and Indexing
- Uses sentence-transformers/all-mpnet-base-v2 for embeddings
- Vector store indexing for efficient retrieval
- Persistent storage of computed indices

### 3. Language Model
- DistilGPT2 for text generation
- Configurable parameters:
  - Context window: 512 tokens
  - Max new tokens: 128
  - Temperature: 0.7

### 4. User Interface
- Rich terminal interface with colored output
- Progress indicators for long operations
- Markdown rendering for answers
- Source attribution for retrieved chunks

## Project Structure

```
llamaindex/
├── rag.py              # Main RAG implementation
├── requirements.txt    # Python dependencies
├── environment.yml     # Conda environment specification
├── README.md          # This documentation
├── data/              # Your documents go here
└── storage/           # Persisted index storage
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
