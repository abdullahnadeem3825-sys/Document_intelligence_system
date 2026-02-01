# 📄 Document Intelligence System

> AI-powered document classification, semantic search, and intelligent Q&A system
---

## 🎯 Overview

**Document Intelligence System** is a complete AI-powered solution for managing, classifying, and understanding documents. It automatically categorizes documents, extracts structured data, enables semantic search, and answers questions about your document collection.

### Key Features

- 📤 **Automatic Classification** - Upload documents and get instant categorization
- 🔍 **Semantic Search** - Find documents by meaning, not just keywords
- 💬 **Intelligent Q&A** - Ask questions and get answers with source citations
- 🎨 **Beautiful UI** - Modern, responsive web interface
- 🚀 **Fast & Accurate** - Powered by state-of-the-art AI models
- 🔒 **Privacy-First** - Runs completely offline, no cloud dependency

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Quick Start

Get up and running in 3 steps:

### 1. Install Dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

### 2. Update Configuration

Edit `api.py` line 17:
```python
os.environ["HF_HOME"] = r"YOUR_MODEL_PATH_HERE"
```

### 3. Start the System

```bash
# Terminal 1: Start API
python app.py

# Terminal 2: Start web server
python -m http.server 8080

# Browser: Open
http://localhost:8080/index.html
```

**First run?** Models will download (~4GB, takes 5-10 minutes, one-time only)

---

## ✨ Features

### 1. Upload & Classify 📤

Upload documents and get automatic classification with extracted structured data.

**Supported Formats:**
- PDF (`.pdf`)
- Microsoft Word (`.docx`, `.doc`)
- Plain Text (`.txt`)

**Document Types:**
- **Invoice** → Extract invoice number, date, company, amount
- **Resume** → Extract name, email, phone, experience
- **Utility Bill** → Extract account number, usage, billing date, amount
- **Other** → Generic classification

**Example:**
```python
# Upload invoice.pdf
→ Category: Invoice
→ Data: {
    invoice_number: "INV-2024-001",
    date: "2024-01-15",
    company: "Acme Corp",
    total_amount: 1250.50
  }
```

### 2. Semantic Search 🔍

Search documents by meaning using natural language queries.

**Example Queries:**
- "Find all invoices from January 2024"
- "Show me resumes with Python experience"
- "Documents about electricity bills over $100"
- "Payment receipts from Acme Corporation"

**Returns:**
- Ranked results with relevance scores
- Document previews
- Extracted data
- Sorted by similarity

### 3. Question & Answer 💬

Ask questions about your documents and get intelligent answers.

**Example Questions:**
- "What is the total amount due across all invoices?"
- "Who are the candidates with more than 5 years experience?"
- "What is my average monthly electricity usage?"

**Returns:**
- Natural language answer
- Source document citations
- Context-aware responses

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern, fast web framework for Python
- **Qwen2.5-3B-Instruct** - Large language model for classification & extraction
- **Sentence-BERT** - Text embeddings for semantic search
- **FAISS** - Efficient similarity search and clustering
- **PyPDF2** - PDF text extraction
- **python-docx** - Word document processing

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with gradients & animations
- **Vanilla JavaScript** - No frameworks, pure JS
- **Responsive Design** - Works on desktop and mobile

### AI/ML
- **Model:** Qwen/Qwen2.5-3B-Instruct
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Vector DB:** FAISS (Facebook AI Similarity Search)
- **Inference:** CPU (GPU optional)

---

## 📦 Installation

### Prerequisites

- **Python:** 3.8 or higher
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 5GB for models and data
- **OS:** Windows, Linux, or macOS

### Step 1: Clone/Download

```bash
# If using git
git clone https://github.com/yourusername/document-intelligence.git
cd document-intelligence

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

**Dependencies installed:**
- fastapi
- uvicorn
- sentence-transformers
- faiss-cpu
- transformers
- torch
- PyPDF2
- python-docx
- pydantic
- python-multipart

### Step 3: Configure Model Path

Edit `app.py` (line 17):

```python
os.environ["HF_HOME"] = r"C:\Users\YourName\Documents\ai_models"
```

This is where AI models will be cached.

---

## 🎮 Usage

### Starting the System


**Method 1: Manual Start**

```bash
# Terminal 1: API Server
python app.py

# Terminal 2: Web Server (for frontend)
python -m http.server 8080
```

Then open: `http://localhost:8080/index.html`

### Using the Web Interface

#### 1. Upload & Classify

1. Click **"Upload & Classify"**
2. Select or drag-drop a document
3. Click **"Upload & Process"**
4. View results:
   - Document category
   - Extracted structured data

#### 2. Semantic Search

1. Click **"Search Documents"**
2. Enter a natural language query
3. Click **"Search"**
4. View results:
   - Ranked by relevance
   - Preview text
   - Extracted data

#### 3. Ask Questions

1. Click **"Ask a Question"**
2. Type your question
3. Click **"Get Answer"**
4. View:
   - AI-generated answer
   - Source documents


## 🖼️ Screenshots

![Main Interface](images\main_page.PNG)
*Main dashboard with three powerful features*

![Upload Result](images\upload_result.PNG)
*Automatic classification and data extraction*

**Quick Example:**

```python
import requests

API_URL = "http://localhost:8000"

# Upload document
with open('invoice.pdf', 'rb') as f:
    response = requests.post(f"{API_URL}/upload", files={'file': f})
    print(response.json())

# Search documents
search = requests.post(f"{API_URL}/search", json={
    "query": "invoices from January",
    "top_k": 5
})
print(search.json())

# Ask question
qa = requests.post(f"{API_URL}/ask", json={
    "question": "What is the total amount due?"
})
print(qa.json()['answer'])
```

---
### Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API information |
| `/upload` | POST | Upload & classify document |
| `/search` | POST | Semantic search |
| `/ask` | POST | Question answering |
| `/stats` | GET | Database statistics |

---

## 📁 Project Structure

```
document-intelligence/
│
├── build_embeddings.py                      # create embeddings for documents
├── classify_and_extract.py                     # classify and extract info from all files 
├── app.py                      # FastAPI backend server
├── index.html               # Web interface
├── requirements.txt            # Python dependencies
│
│
├── embeddings/                       # Data directory (created on first run)
│   ├── faiss_index.bin               # FAISS vector index
│   ├── documents.json      # Document metadata
│   └── filenames.json       # file names
    └── metadata.json       # metadata
│
├── README.md                   # This file
└── LICENSE                     # MIT License
```

---

## ⚙️ Configuration

### Model Configuration

**Location:** `app.py`

```python
# Cache directory for models
os.environ["HF_HOME"] = r"YOUR_PATH_HERE"

# LLM parameters
pipeline_kwargs={
    "max_new_tokens": 512,      # Max output length
    "temperature": 0.1,         # Randomness (0.0-1.0)
    "top_p": 0.95,             # Nucleus sampling
    "do_sample": True,
    "repetition_penalty": 1.05,
}
```

### Server Configuration

**Location:** `app.py` (last line)

```python
uvicorn.run(
    app,
    host="0.0.0.0",  # Listen on all interfaces
    port=8000        # API port
)
```

### Search Configuration

```python
# Number of search results
top_k = 5  # Default in /search endpoint

# Number of documents for Q&A context
top_k = 3  # In answer_question method
```

### CORS Configuration

**Location:** `app.py`

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Production:** Change `allow_origins` to specific domains:
```python
allow_origins=[domain link]
```

---

## 💡 Examples

### Use Case 1: Invoice Management

```python
# Upload all invoices
import os
from pathlib import Path

invoices_dir = Path("./invoices")
for invoice in invoices_dir.glob("*.pdf"):
    with open(invoice, 'rb') as f:
        response = requests.post(
            "http://localhost:8000/upload",
            files={'file': f}
        )
        print(f"Uploaded: {invoice.name}")

# Search for unpaid invoices
results = requests.post(
    "http://localhost:8000/search",
    json={"query": "unpaid invoices", "top_k": 10}
).json()

# Calculate total
total = sum(r['extracted_data'].get('total_amount', 0) for r in results)
print(f"Total due: ${total:.2f}")
```

### Use Case 2: Resume Screening

```python
# Upload resumes
for resume in Path("./resumes").glob("*.pdf"):
    with open(resume, 'rb') as f:
        requests.post("http://localhost:8000/upload", files={'file': f})

# Find candidates with Python experience
candidates = requests.post(
    "http://localhost:8000/search",
    json={"query": "Python developer 5+ years experience", "top_k": 10}
).json()

# Filter by experience
qualified = [
    c for c in candidates 
    if c['extracted_data'].get('experience_years', 0) >= 5
]

print(f"Found {len(qualified)} qualified candidates")
```

### Use Case 3: Bill Tracking

```python
# Upload utility bills
for bill in Path("./bills").glob("*.pdf"):
    with open(bill, 'rb') as f:
        requests.post("http://localhost:8000/upload", files={'file': f})

# Ask about average usage
answer = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What is my average monthly electricity usage?"}
).json()

print(f"Answer: {answer['answer']}")
print(f"Sources: {', '.join(answer['sources'])}")
```

---

## 🔧 Troubleshooting


**Problem:** Port 8000 already in use

**Solution:**
```python
# In api.py, change port
uvicorn.run(app, host="0.0.0.0", port=8001)

# Update frontend.html
const API_URL = 'http://localhost:8001';
```

### Models Not Downloading

**Problem:** First run hangs or fails

**Solution:**
- Check internet connection
- Ensure sufficient disk space (~5GB)
- Wait patiently (first download takes 5-10 minutes)

### Empty Extracted Data

**Problem:** Upload succeeds but `extracted_data: {}`

**Causes:**
1. Document is unreadable (scanned image without OCR)
2. Model path incorrect
3. LLM failed to parse

**Solutions:**
1. Use text-based PDFs, not scanned images
2. Verify `HF_HOME` path exists
3. Check API terminal for errors

### CUDA Out of Memory

**Problem:** GPU memory error

**Solution:**
System uses CPU by default. If you enabled GPU, switch back:
```python
device=-1  # CPU mode
```

For more issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📊 Performance

### Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| First Upload | 5-10 sec | Model loading |
| Subsequent Uploads | 2-5 sec | Per document |
| Search Query | <1 sec | Up to 1000 docs |
| Q&A Query | 3-8 sec | Depends on context |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8GB | 16GB |
| Storage | 5GB | 10GB+ |
| Python | 3.8 | 3.10+ |

### Optimization Tips

1. **Use GPU** if available:
   ```python
   device=0  # Use first GPU
   ```

2. **Batch uploads** for multiple documents

3. **Limit search results:**
   ```python
   top_k=5  # Instead of 20
   ```

4. **Reduce context for Q&A:**
   ```python
   text[:2000]  # Instead of full text
   ```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

### Development Setup

```bash
# Fork and clone the repo
git clone https://github.com/yourusername/document-intelligence.git

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt


### Guidelines

1. **Code Style:** Follow PEP 8
2. **Documentation:** Update docs for new features
3. **Testing:** Add tests for new endpoints
4. **Commits:** Use clear, descriptive messages

---

## 🎯 Roadmap

### Current Version (1.0)
- ✅ Document classification
- ✅ Data extraction
- ✅ Semantic search
- ✅ Q&A system
- ✅ Web interface

### Planned Features

**v1.1**
- [ ] Batch upload UI
- [ ] Export results to CSV/Excel
- [ ] Advanced filtering (date range, amount)
- [ ] Document comparison

**v1.2**
- [ ] User authentication
- [ ] Multi-user support
- [ ] Cloud storage integration (S3, Google Drive)
- [ ] Email integration

**v2.0**
- [ ] OCR for scanned documents
- [ ] Multi-language support
- [ ] Custom document types
- [ ] Advanced analytics dashboard

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Document Intelligence Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 🙏 Acknowledgments

### Technologies
- **FastAPI** by Sebastián Ramírez
- **Qwen Models** by Alibaba Cloud
- **Sentence-BERT** by UKPLab
- **FAISS** by Facebook Research

### Inspiration
Built to solve real-world document management challenges in businesses and personal workflows.

---

## 📞 Support

### Getting Help


1. **Check Issues**
   - Search existing issues on GitHub
   - Create a new issue if needed

### Contact

- **Issues:** GitHub Issues
- **Email:** abdullahnadeem3825@gmail.com
- **Documentation:** See docs/ folder

---

## 📈 Stats

- **Lines of Code:** ~1,300
- **Documentation Pages:** 10+
- **Supported Formats:** 3 (PDF, DOCX, TXT)
- **Document Types:** 4 (Invoice, Resume, Utility Bill, Other)
- **API Endpoints:** 5
- **Languages:** Python, JavaScript, HTML, CSS

---

## 🌟 Star History

If you find this project helpful, please consider giving it a star! ⭐

---

**Built with ❤️ using FastAPI, Qwen2.5-3B, Sentence-BERT, and modern web technologies**

**Version:** 1.0  
**Status:** Production Ready ✅  
**Last Updated:** February 2026

---

[🔝 Back to Top](#-document-intelligence-system)