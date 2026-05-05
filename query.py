import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ==================== CONFIGURATION ====================
PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 4
# ======================================================

print("🚀 Loading RAG Q&A Bot...")

# 1. Load embeddings and vector database
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# 2. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# 3. Set up LLM (Ollama)
llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0.3,
)

# 4. Create prompt template (forces citations)
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the provided context.
If you don't know the answer, say "I don't have enough information in the documents to answer this."

Context (with sources):
{context}

Question: {question}

Answer:
""")

# 5. RAG Chain
def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        page_info = f" (page {int(page)+1})" if page != "" else ""
        formatted.append(f"Source: {source}{page_info}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

print("✅ Bot is ready!")
print("   Type your questions below.")
print("   Type 'exit' or 'quit' to stop.\n")

# 6. Interactive loop
while True:
    try:
        question = input("❓ Your question: ").strip()
        
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
            
        if not question:
            continue

        print("🤔 Thinking...")
        answer = rag_chain.invoke(question)
        
        print("\n📝 Answer:")
        print(answer)
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
