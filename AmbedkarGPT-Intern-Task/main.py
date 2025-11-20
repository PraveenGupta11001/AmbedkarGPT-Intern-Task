import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ----------------------------------------------
# 1. Load Document
# ----------------------------------------------
loader = TextLoader("speech.txt", encoding="utf-8")
docs = loader.load()

# Split
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------------------------
# 2. Chroma DB
# ----------------------------------------------
persist_dir = "chroma_db"

if os.path.exists(persist_dir) and os.listdir(persist_dir):
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    print("Loaded existing Chroma DB.")
else:
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    vectordb.persist()
    print("Created new Chroma DB.")

retriever = vectordb.as_retriever(search_kwargs={"k": 4})


# ----------------------------------------------
# 3. LLM (Ollama)
# ----------------------------------------------
llm = Ollama(model="mistral", temperature=0.0)


# ----------------------------------------------
# 4. LCEL RAG CHAIN (LangChain 1.0+)
# ----------------------------------------------
prompt = ChatPromptTemplate.from_template("""
    You are AmbedkarGPT. Answer the question using ONLY the context below.

    Context:
    {context}

    Question: {question}

    Answer:
""")

def combine_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | combine_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)


# ----------------------------------------------
# 5. Interactive loop
# ----------------------------------------------
print("AmbedkarGPT is ready! Type 'exit' to quit.\n")

while True:
    q = input("Your question: ").strip()

    if q.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    if not q:
        continue

    print("\nAnswer:")
    # -------------------------------
    # STREAMING RESPONSE (typing effect)
    # -------------------------------
    for chunk in rag_chain.stream(q):
        print(chunk, end="", flush=True)
    print("\n")

    # show retrieved docs
    docs = retriever.invoke(q)
    print("\nSources:")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d.page_content[:200]}...")

    print("-" * 80)
