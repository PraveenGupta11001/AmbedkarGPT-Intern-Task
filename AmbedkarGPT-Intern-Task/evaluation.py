import os
import json
import requests
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer
import nltk
nltk.download("punkt", quiet=True)

# LangChain Tools
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ============================================================================
# LLM SELECTION (TOGGLE)
# ============================================================================

USE_GEMINI = True   # True = Gemini API, False = local Ollama

# ============================================================================
# GEMINI DIRECT API CLIENT (your working version)
# ============================================================================

API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4")
]

# ENDPOINT = (
#     "https://generativelanguage.googleapis.com/v1beta/"
#     "models/gemini-1.5-flash:generateContent"
# )

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def generate_content_llm(prompt: str) -> str:
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    for key in API_KEYS:
        if not key:
            continue

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": key,
        }

        try:
            response = requests.post(
                ENDPOINT, headers=headers, json=payload, timeout=40
            )

            if response.status_code != 200:
                print(f"[ERROR] Key {key[:6]}... -> HTTP {response.status_code}")
                continue

            data = response.json()

            raw = (
                data["candidates"][0]["content"]["parts"][0]["text"]
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            return raw

        except Exception as e:
            print(f"[EXCEPTION] Key {key[:6]} -> {e}")
            continue

    print("All Gemini API keys failed")
    return ""

# ============================================================================
# Optional OLLAMA fallback
# ============================================================================
if not USE_GEMINI:
    from langchain_community.llms import Ollama
    llm_ollama = Ollama(model="mistral", temperature=0.0)
    print("Using Local Ollama Mistral\n")
else:
    print("⚡ Using Gemini Flash (Direct API) for answers\n")

# ============================================================================
# Load test dataset
# ============================================================================

with open("test_dataset.json") as f:
    raw = json.load(f)

# Allow both formats: list OR {"test_questions": [...]}
if isinstance(raw, dict) and "test_questions" in raw:
    test_data = raw["test_questions"]
else:
    test_data = raw

# ============================================================================
# Embeddings
# ============================================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================================
# Build RAG chain
# ============================================================================
def build_rag_chain(chunk_size):
    # load corpus
    docs = []
    for file in os.listdir("corpus"):
        loader = TextLoader(f"corpus/{file}", encoding="utf-8")
        docs.extend(loader.load())

    # chunk
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # vector DB
    vectordb = Chroma.from_documents(
        chunks, embeddings, persist_directory=f"chroma_eval_{chunk_size}"
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template("""
Use ONLY the following context to answer the question.

Context:
{context}

Question: {question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    } | prompt

    return chain, retriever, prompt

# ============================================================================
# Evaluation
# ============================================================================
results = {}
chunk_sizes = [300, 600, 900]

for size in chunk_sizes:
    print(f"\nEvaluating chunk_size = {size}")
    chain, retriever, prompt_template = build_rag_chain(size)
    size_results = []

    for item in tqdm(test_data):
        q = item["question"]
        gt = item["ground_truth"]
        true_docs = item["source_documents"]

        # Retrieve docs
        retrieved = retriever.invoke(q)
        retrieved_names = [
            d.metadata.get("source", "").split("/")[-1] for d in retrieved
        ]

        # Build full prompt text
        context_text = "\n\n".join([d.page_content for d in retrieved])
        final_prompt = f"""
            Use ONLY the following context to answer the question.

            Context:
            {context_text}

            Question: {q}

            Answer:
        """

        # GEMINI
        if USE_GEMINI:
            answer = generate_content_llm(final_prompt)

        # OLLAMA
        else:
            answer_obj = ({"context": context_text, "question": q} | prompt_template | llm_ollama)
            answer = answer_obj

        # Retrieval metrics
        hit = any(doc in retrieved_names for doc in true_docs)
        mrr = next(
            (1 / (i + 1) for i, d in enumerate(retrieved_names) if d in true_docs), 0
        )
        precision = (
            sum(1 for d in retrieved_names[:5] if d in true_docs) / 5
            if true_docs
            else 0
        )

        # ROUGE-L
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge = scorer.score(gt, answer)["rougeL"].fmeasure

        size_results.append(
            {
                "id": item["id"],
                "question": q,
                "answer": answer,
                "hit": hit,
                "mrr": mrr,
                "precision": precision,
                "rougeL": rouge,
                "retrieved": retrieved_names,
                "true_docs": true_docs,
            }
        )

    # Summary
    results[size] = {
        "hit_rate": float(np.mean([r["hit"] for r in size_results])),
        "mrr": float(np.mean([r["mrr"] for r in size_results])),
        "precision_at_5": float(np.mean([r["precision"] for r in size_results])),
        "rougeL": float(np.mean([r["rougeL"] for r in size_results])),
        "details": size_results,
    }

    print(
        f"✔ Chunk {size}: HR={results[size]['hit_rate']:.3f} | "
        f"MRR={results[size]['mrr']:.3f} | ROUGE-L={results[size]['rougeL']:.3f}"
    )

# ============================================================================
# Save results
# ============================================================================
with open("test_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ============================================================================
# Write Markdown analysis
# ============================================================================
best = max(results.items(), key=lambda x: x[1]["hit_rate"] + x[1]["mrr"])

with open("results_analysis.md", "w") as f:
    f.write("# Evaluation Results\n\n")
    f.write(f"**Best chunk size → {best[0]} characters**\n\n")

    for size, res in results.items():
        f.write(f"### Chunk Size: {size}\n")
        f.write(f"- Hit Rate: {res['hit_rate']:.3f}\n")
        f.write(f"- MRR: {res['mrr']:.3f}\n")
        f.write(f"- Precision@5: {res['precision_at_5']:.3f}\n")
        f.write(f"- ROUGE-L: {res['rougeL']:.3f}\n\n")

    f.write("Recommendation: Medium chunks (600) usually perform best.\n")

print("\nEvaluation complete! Generated: test_results.json + results_analysis.md")
