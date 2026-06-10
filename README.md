A basic Retrieval-Augmented Generation (RAG) system that lets users ask natural language questions about a collection of documents and get accurate, grounded answers with source citations.

## Tech Stack

- **Python 3.14**
- **LangChain** (core framework)
- **ChromaDB** (vector database)
- **HuggingFaceEmbeddings** (`sentence-transformers/all-MiniLM-L6-v2`)
- **Ollama** + `llama3.2` (LLM)
- **PyPDFLoader, Docx2txtLoader, TextLoader** (document loaders)
- **RecursiveCharacterTextSplitter**

## Architecture Overview

1. **Document Ingestion** → Load PDF, DOCX, TXT files from `/data`
2. **Chunking** → Split into 1000-character chunks with 200-character overlap
3. **Embedding** → Convert chunks to vectors using `all-MiniLM-L6-v2`
4. **Vector Store** → Store in persistent ChromaDB (`chroma_db/`)
5. **Retrieval** → Embed user query → similarity search → top 4 chunks
6. **Generation** → Send context + question to Ollama (`llama3.2`) → generate answer with citations

## Chunking Strategy

I used **`RecursiveCharacterTextSplitter`** with:
- `chunk_size = 1000`
- `chunk_overlap = 200`

**Why this strategy?**  
It intelligently splits on paragraphs, sentences, and punctuation while preserving context at chunk boundaries. This is a standard and effective approach for RAG systems.

## Embedding Model & Vector Database

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (fast, free, local, no API key needed)
- **Vector Database**: **ChromaDB** (lightweight, persists to disk, easy to use)

Both choices were made for simplicity, speed, and zero-cost deployment — ideal for a beginner-level internship project.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/eshwarsgithub/rag-document-qa-bot.git
   cd rag-document-qa-bot
2. Create and activate virtual environment:Bash
   python3 -m venv venv

Environment Variables
No API keys required (everything runs locally).
Example Queries

What are the main benefits of AI in education?
Why is education still important in the digital age?
What problem does RAG solve?
What are the core features of an Edtech Q&A bot?
What is the capital of France? (should say "I don't have enough information")

Known Limitations

Answers depend entirely on the quality of documents in /data
Local LLM (llama3.2) is slower than paid APIs
PDF page numbers may sometimes be off by 1
CLI only (no web UI yet)
   source venv/bin/activate    # On Mac/Linux
   # venv\Scripts\activate    # On Windows
3. Install dependencies:Bash
   pip install -r requirements.txt
   Install and run Ollama:
4. Download from https://ollama.com
   Run: ollama pull llama3.2
5. Add your documents to the data/ folder
6. Index the documents (run only once):Bash
   python index.py
7. Run the Q&A Bot:Bash
   python query.py
