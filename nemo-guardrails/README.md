# NeMo Guardrails Implementation

This directory contains implementation examples and guides for using NeMo Guardrails in RAG applications.

## Quick Start Guide

### 1. Environment Setup

Choose either Conda or virtual environment (venv) for your setup:

#### Option A: Using Conda (Recommended)
```bash
# Create new conda environment
conda create -n nemo-guardrails python=3.9

# Activate conda environment
conda activate nemo-guardrails

# Install requirements
pip install -r requirements.txt
```

#### Option B: Using Python venv
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

#### Setting up OpenAI API Key
Create a `.env` file in the root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

### 2. Project Structure

```
nemo-guardrails/
├── config/
│   └── config.yml         # Guardrails configuration
├── data/
│   ├── sample1.txt        # Sample document about AI Safety
│   └── sample2.txt        # Sample document about Guardrails
├── basic_rag_with_guardrails.py  # Main implementation
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

### 3. Running the Code

1. Make sure your environment (Conda or venv) is activated:
```bash
# If using Conda:
conda activate nemo-guardrails

# If using venv:
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate   # On Windows
```

2. Run the basic example:
```bash
python basic_rag_with_guardrails.py
```

3. For interactive use in Python:
```python
from basic_rag_with_guardrails import RAGWithGuardrails

# Initialize the RAG system
rag = RAGWithGuardrails("config")

# Load documents
documents = ["data/sample1.txt", "data/sample2.txt"]
rag.load_documents(documents)

# Make queries
response = rag.query("What are the key principles of AI safety?")
print(response)
```

### 4. Configuration

The `config.yml` file contains several guardrails:
- Content moderation
- Topic boundary checking
- Response quality control
- Factual accuracy verification

You can modify these guardrails in `config/config.yml` to suit your needs.

### 5. Sample Queries

Try these example queries:
```python
# Query about AI safety principles
response = rag.query("What are the main principles of responsible AI development?")

# Query about implementation guidelines
response = rag.query("How should AI guardrails be implemented in practice?")

# Off-topic query (will be handled by guardrails)
response = rag.query("What's the weather like today?")
```

### 6. Error Handling

The implementation includes error handling for:
- Missing documents
- Invalid queries
- API errors
- Configuration issues

If you encounter any errors, check:
1. Your OpenAI API key is correctly set in `.env`
2. All required files exist in their expected locations
3. The config directory is properly structured
4. Your environment (Conda or venv) is properly activated

### 7. Environment Management

#### Conda Commands
```bash
# List all conda environments
conda env list

# Update conda environment
conda update --all

# Remove conda environment
conda deactivate
conda env remove -n nemo-guardrails

# Export conda environment
conda env export > environment.yml

# Create environment from yml file
conda env create -f environment.yml
```

#### Venv Commands
```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv/
```

## Contents (Coming Soon)

1. Basic Setup
   - Installation guide
   - Configuration basics
   - First guardrail implementation

2. Core Features
   - Content filtering
   - Topic boundary control
   - Conversation flow management
   - Safety mechanisms

3. Integration Examples
   - RAG pipeline with guardrails
   - Custom guardrail rules
   - Multi-LLM setup

4. Best Practices
   - Security considerations
   - Performance optimization
   - Rule management

5. Advanced Use Cases
   - Enterprise compliance
   - Content moderation
   - Domain-specific implementations
