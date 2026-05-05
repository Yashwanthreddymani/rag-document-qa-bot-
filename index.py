import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================
DATA_FOLDER = "data"
PERSIST_DIRECTORY = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# ======================================================

print("🚀 Starting document indexing...")

# Function to load documents with correct loader for each file type
def load_all_documents(data_folder: str):
    documents = []
    data_path = Path(data_folder)
    
    if not data_path.exists():
        print(f"❌ Folder '{data_folder}' not found!")
        return documents

    print(f"📂 Scanning folder: {data_folder}")
    
    for file_path in data_path.glob("**/*.*"):
        file_str = str(file_path)
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(file_str)
                print(f"   📄 Loading PDF: {file_path.name}")
            elif file_path.suffix.lower() in [".docx", ".doc"]:
                loader = Docx2txtLoader(file_str)
                print(f"   📝 Loading DOCX: {file_path.name}")
            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(file_str, encoding="utf-8")
                print(f"   📝 Loading TXT: {file_path.name}")
            else:
                print(f"   ⚠️  Skipping unsupported file: {file_path.name}")
                continue
                
            docs = loader.load()
            documents.extend(docs)
            
        except Exception as e:
            print(f"   ❌ Error loading {file_path.name}: {e}")
    
    return documents

# 1. Load all documents
documents = load_all_documents(DATA_FOLDER)

if len(documents) == 0:
    print("❌ No documents were loaded. Please check your 'data' folder.")
    exit()

print(f"✅ Successfully loaded {len(documents)} documents")

# 2. Split documents into chunks
print("✂️  Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)

print(f"✅ Created {len(chunks)} chunks")

# 3. Create embeddings and store in vector database
print("🔢 Creating embeddings and storing in vector database...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY
)

print("💾 Vector database saved successfully!")
print(f"   Location: {PERSIST_DIRECTORY}/")
print("\n🎉 Indexing completed! You can now run the query bot.")
