# 🇲🇾 Malaysian Subsidy Eligibility Criteria Generator

A **Retrieval-Augmented Generation (RAG)** based intelligent document analysis system designed to generate subsidy eligibility scoring criteria from Malaysian poverty-related PDF documents. Built using **Streamlit**, **LangChain**, and **Ollama**, this tool simplifies large-scale policy document analysis through automation and AI-driven insights.

---

## 📘 Project Overview

This project is a document analysis tool powered by Retrieval-Augmented Generation (RAG), specifically created to analyze poverty-related PDFs in Malaysia and generate suggested subsidy eligibility criteria. It combines a user-friendly Streamlit interface with LangChain and Ollama for smart document processing and reasoning.

---

## 🚀 Key Features

- ✅ Parse and extract content from PDF documents  
- ✅ Store and retrieve document content using a vector database  
- ✅ Perform intelligent document analysis and Q&A  
- ✅ Automatically generate subsidy scoring criteria  
- ✅ Support for custom queries and deeper insights  
- ✅ Parallel processing for multiple documents  

---

## 🛠️ System Requirements

- Python 3.8 or higher  
- Ollama service running locally (default port: 11434)  
- Sufficient memory for document parsing and vector storage  

---

## 📦 Installation

1. **Clone the repository:**

```bash
git clone <repository_url>
cd <repository_name>
```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
# or
venv\\Scripts\\activate  # For Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Install and launch Ollama:**

Visit the [Ollama website](https://ollama.ai/) to download and install.

```bash
ollama serve
ollama pull llama3.2
```

---

## 📚 Dependencies

Main dependencies include:

- `streamlit` – UI framework  
- `langchain` – RAG implementation  
- `unstructured` – PDF parsing  
- `chromadb` – Vector database  
- `pandas` – Data handling  
- `requests` – API interaction  

See `requirements.txt` for the full list.

---

## 🧪 How to Use

1. **Launch the app:**

```bash
streamlit run reasoning/newapp.py
```

2. **Workflow:**

- Upload one or more PDF documents  
- Wait for the documents to be processed  
- View the generated subsidy scoring criteria  
- Use custom queries for further exploration  

3. **Cache Management:**

- Use the "Clear Cache and Restart" button in the sidebar  
- Caches include extracted text and vector data  

---

## 🧩 System Architecture

### 📄 Document Processing Flow

1. Upload PDF  
2. Extract text and tables using `unstructured`  
3. Chunk text and create embeddings  
4. Store in Chroma vector database  

### 🧠 RAG Workflow

- Uses Ollama's `llama3.2` model (fallback: HuggingFace `all-MiniLM-L6-v2`)  
- Vector-based document retrieval  
- Structured prompt engineering  
- LLM generates subsidy scoring criteria  

### ⚡ Performance Optimizations

- Multiprocessing for faster document handling  
- Persistent caching of results  
- Batched processing for large documents  

---

## ⚠️ Notes

1. Ensure Ollama service is running before launching the app  
2. Only text-extractable PDFs are supported (not scanned images)  
3. The system creates the following local directories:
   - `data_cache/` – Extracted document data  
   - `chroma_db/` – Vector database  
   - `logs/` – System logs  

---

## 🧯 Error Handling

- If Ollama is unavailable, the system falls back to HuggingFace embeddings  
- Document processing failures are logged  
- UI will display live status and error notifications  

---

## 🛠️ Development Roadmap

- [ ] Support additional document formats (e.g. DOCX, TXT)  
- [ ] Improve vector search relevance  
- [ ] Add document quality assessment  
- [ ] Integrate more LLM model options  
- [ ] Enhance UI interactivity  

---

## 🤝 Contribution Guidelines

Contributions are welcome via issues or pull requests. Please ensure:

1. Code follows PEP 8 standards  
2. Proper test coverage is added  
3. Related documentation is updated  


## 📬 Contact

szehaohar@1utar.my

