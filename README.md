# RAG-Toolkit: Comprehensive Guide to RAG Implementations

This repository provides a comprehensive collection of Retrieval-Augmented Generation (RAG) implementations using various modern frameworks and tools. Each implementation demonstrates different approaches and capabilities for building robust RAG solutions.

## 🚀 Implementations

### 1. [Verba Implementation](./verba/)
- **Tech Stack**: Verba, Weaviate, OpenAI
- **Key Features**:
  - PDF document ingestion and processing
  - Document chunking with overlap
  - Vector storage using Weaviate
  - Conversational retrieval using GPT models
  - Source attribution for answers
- **Best For**: Document-heavy applications requiring precise retrieval and attribution

### 2. [LangChain Implementation](./langchain/)
- **Tech Stack**: LangChain, ChromaDB, OpenAI
- **Key Features**:
  - Multiple PDF document processing
  - Automatic text chunking
  - Semantic search using ChromaDB
  - Conversational QA interface
  - Chat history support
- **Best For**: Complex RAG pipelines with multiple components

### 3. [LlamaIndex Implementation](./llamaindex/)
- **Tech Stack**: LlamaIndex, OpenAI, Vector Store
- **Key Features**:
  - Advanced data connectors
  - Structured data handling
  - Optimized query engines
  - Custom data indexing
- **Best For**: Data-intensive applications with diverse sources

### 4. [Phoenix Implementation](./phoenix/)
- **Tech Stack**: Phoenix, Vector DB
- **Key Features**:
  - Real-time vector indexing
  - High-performance retrieval
  - Scalable architecture
  - Modern deployment options
- **Best For**: High-performance, scalable RAG systems

### 5. [MongoDB Implementation](./mongodb/)
- **Tech Stack**: MongoDB Atlas, Vector Search
- **Key Features**:
  - Atlas Vector Search integration
  - Enterprise-grade security
  - Scalable document storage
  - Native vector indexing
- **Best For**: Enterprise applications requiring robust data management

### 6. [Haystack Implementation](./haystack/)
- **Tech Stack**: Haystack Framework
- **Key Features**:
  - Modular pipeline architecture
  - Multiple retriever options
  - Production-ready components
  - Flexible deployment
- **Best For**: Production-grade search and QA systems

### 7. [NeMo Guardrails Implementation](./nemo-guardrails/)
- **Tech Stack**: NeMo Guardrails, LLM
- **Key Features**:
  - Content filtering
  - Topic boundaries
  - Conversation flow control
  - Safety mechanisms
- **Best For**: Applications requiring controlled AI interactions

## 🛠 Getting Started

Each implementation directory contains:
- Detailed README with setup instructions
- Complete source code
- Configuration examples
- Usage demonstrations
- Testing scripts

### Prerequisites
- Python 3.9+
- Conda (for environment management)
- Relevant API keys (OpenAI, etc.)
- Vector store setup (varies by implementation)

### General Setup Steps
1. Clone the repository:
```bash
git clone <repository-url>
cd RAG-Toolkit
```

2. Choose an implementation directory
3. Follow the specific README instructions
4. Set up required API keys and services
5. Run the example code

## 📊 Comparison Matrix

| Implementation | Vector Store    | LLM Support | Document Types | Deployment Complexity |
|---------------|-----------------|-------------|----------------|---------------------|
| Verba         | Weaviate        | OpenAI      | PDF           | Medium              |
| LangChain     | ChromaDB        | Multiple    | Multiple      | Low                 |
| LlamaIndex    | Multiple        | Multiple    | Multiple      | Medium              |
| Phoenix       | Custom Vector DB| Multiple    | Multiple      | High                |
| MongoDB       | Atlas Search    | Multiple    | Multiple      | Medium              |
| Haystack      | Multiple        | Multiple    | Multiple      | Medium              |
| NeMo          | -               | Multiple    | Text          | Low                 |

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Additional Resources

- [Verba Documentation](https://github.com/weaviate/Verba)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/)
- [MongoDB Atlas Documentation](https://www.mongodb.com/docs/atlas/)
