# AmbedkarGPT-Intern-Task

This repository contains the work submitted for the **Kalpit Pvt Ltd AI Intern Assignment**, consisting of:

* **Assignment 1:** A command-line RAG Q&A system built using LangChain, ChromaDB, HuggingFace Embeddings, and Ollama (Mistral 7B).
* **Assignment 2:** A complete evaluation pipeline for retrieval accuracy, answer quality, and chunk-size comparisons across a multi-document corpus.

The system is built using Python, runs fully locally, and includes optional support for Gemini Flash for faster evaluation.

---

## **1. Project Features**

### **Assignment 1 – RAG System**

* Loads a text document (`speech.txt`)
* Splits into chunks
* Embeds using `sentence-transformers/all-MiniLM-L6-v2`
* Stores vectors in ChromaDB
* Retrieves relevant context
* Generates answers using **Ollama Mistral 7B**
* Interactive question-answer CLI

### **Assignment 2 – Evaluation**

* Evaluates the RAG system over a 25-question test set
* Metrics implemented:

  * Hit Rate
  * MRR (Mean Reciprocal Rank)
  * Precision@5
  * ROUGE-L
* Tests multiple chunking strategies (300 / 600 / 900)
* Supports two LLMs:

  * **Ollama (local, slow)**
  * **Gemini Flash (API-based, fast)** via a toggle
* Generates:

  * `test_results.json`
  * `results_analysis.md`

---

## **2. Installation and Setup**

### **Clone the Repository**

```bash
git clone https://github.com/PraveenGupta11001/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### **Create Python Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## **3. Ollama Installation (Required for Assignment 1)**

Install Ollama:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Pull Mistral 7B:

```bash
ollama pull mistral
```

Make sure Ollama is running:

```bash
ollama serve
```

---

## **4. Gemini API Setup (Optional — For Fast Evaluation)**

Set your API key:

### Linux / macOS:

```bash
export GEMINI_API_KEY_1="YOUR_KEY_HERE"
```

### Windows PowerShell:

```powershell
setx GEMINI_API_KEY_1 "YOUR_KEY_HERE"
```

---

## **5. Running Assignment 1 (main RAG system)**

```bash
python main.py
```

This will:

1. Load `speech.txt`
2. Build ChromaDB in `chroma_db/`
3. Start an interactive CLI:

   ```
   Your question:
   ```

To exit:

```
exit
```

---

## **6. Running Assignment 2 (Full Evaluation)**

Before running, delete any previous evaluation vector stores (optional but recommended):

```bash
rm -rf chroma_eval_300 chroma_eval_600 chroma_eval_900
```

Start evaluation:

```bash
python evaluation.py
```

The script will:

* Build retrievers for chunk sizes 300 / 600 / 900
* Run all 25 questions
* Compute metrics
* Generate:

  * `test_results.json`
  * `results_analysis.md`

---

## **7. Switching LLM for Evaluation**

Inside `evaluation.py` there is a toggle:

```python
USE_GEMINI = True   # Use Gemini Flash
USE_GEMINI = False  # Use Ollama Mistral 7B
```

* **Gemini = Fast evaluation (recommended)**
* **Ollama = Required for assignment, slower**

---

## **8. Project Structure**

```
AmbedkarGPT-Intern-Task/
│
├── corpus/                     # 6 documents for assignment 2
├── chroma_db/                  # vector store for assignment 1
├── chroma_eval_300/            # generated during evaluation
├── chroma_eval_600/
├── chroma_eval_900/
│
├── speech.txt                  # assignment 1 source text
├── test_dataset.json           # 25 questions for assignment 2
│
├── main.py                     # assignment 1 RAG system
├── evaluation.py               # assignment 2 evaluation pipeline
├── test_results.json           # output after running evaluation
├── results_analysis.md         # summarized evaluation metrics
│
├── requirements.txt
└── README.md
```

---

## **9. Requirements**

The main packages include:

```
langchain
langchain-community
langchain-core
langchain-text-splitters
chromadb
sentence-transformers
ollama
ragas
rouge-score
nltk
datasets
pandas
tqdm
requests
```

Install using:

```bash
pip install -r requirements.txt
```

---

## **10. Notes**

* All embeddings and vector storage run locally.
* Evaluation supports both Gemini and Ollama.
* Gemini calls use direct HTTP requests for reliability.
* Ollama must be installed and running for Assignment 1.
* The repository is fully reproducible on any local machine.