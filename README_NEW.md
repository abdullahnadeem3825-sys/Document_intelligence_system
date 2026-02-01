# Document AI System - Optimized Version

**Efficient document classification and search with persistent embeddings**

---

## 🎯 **Key Features**

✅ **Persistent Embeddings** - Build once, use many times (no rebuilding!)  
✅ **Separate Workflows** - Different scripts for embeddings vs classification  
✅ **FAISS on Disk** - Saved index for fast loading  
✅ **FastAPI** - Modern async API with automatic docs  
✅ **Client-Ready** - Put docs in `input/`, get results in `outputs/`

---

## 📁 **Folder Structure**

```
project/
├── input/                              ← Put client documents HERE
│   ├── invoice_1.pdf
│   ├── resume_1.pdf
│   └── bill_1.pdf
│
├── embeddings/                         ← Auto-created, saved embeddings
│   ├── faiss_index.bin                ← FAISS index (persistent)
│   ├── documents.json                 ← Document texts
│   ├── filenames.json                 ← Filename mapping
│   └── metadata.json                  ← Index metadata
│
├── outputs/                            ← Results go here
│   └── classification_results.json    ← Final output
│
├── build_embeddings.py                ← Run when NEW docs arrive
├── classify_and_extract.py            ← Run to classify & extract
├── api_server_fastapi.py              ← FastAPI server
├── requirements.txt
└── README.md
```

---

## 🚀 **Quick Start**

### **Step 1: Install (Once)**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: Add Client Documents**

```bash
# Create input folder and add documents
mkdir input
cp /path/to/client/documents/*.pdf input/
```

### **Step 3: Build Embeddings (Once per batch)**

```bash
python build_embeddings.py
```

**This creates:**
- `embeddings/faiss_index.bin` - FAISS index
- `embeddings/documents.json` - Document texts
- `embeddings/filenames.json` - Filename mapping

**⏱️ Time:** 5-10 minutes (first run downloads models)

### **Step 4: Classify & Extract**

```bash
python classify_and_extract.py
```

**This creates:**
- `outputs/classification_results.json` - All results

**⏱️ Time:** 5-10 seconds per document

---

## 📋 **Complete Workflow**

### **New Documents from Client**

```bash
# 1. Put documents in input/
cp /path/to/new/batch/*.pdf input/

# 2. Build embeddings (ONLY if new documents)
python build_embeddings.py

# 3. Classify and extract
python classify_and_extract.py

# 4. Get results
cat outputs/classification_results.json
```

### **Re-classify Same Documents (No New Docs)**

```bash
# Just run classification (embeddings already exist)
python classify_and_extract.py
```

---

## 🔄 **When to Run Each Script**

| Script | When to Run | What It Does | Time |
|--------|-------------|--------------|------|
| `build_embeddings.py` | **NEW documents** arrive | Creates embeddings & FAISS | 5-10 min (first run) |
| `classify_and_extract.py` | Want classification results | Classifies & extracts data | 5-10 sec/doc |
| `api_server_fastapi.py` | Want API access | Starts web server | Keeps running |

---

## 🌐 **Using the FastAPI Server**

### **Start Server**

```bash
python api_server_fastapi.py
```

Server runs on: `http://localhost:8000`

### **Interactive Docs**

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### **API Endpoints**

```bash
# Upload documents
curl -X POST http://localhost:8000/upload \
  -F "files=@invoice.pdf" \
  -F "files=@resume.pdf"

# Search documents
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "invoices from March", "top_k": 5}'

# Ask questions
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total amount?", "top_k": 3}'

# Get results
curl http://localhost:8000/results

# Download results
curl http://localhost:8000/results/download -o results.json

# Get statistics
curl http://localhost:8000/stats
```

---

## 📊 **Output Format**

### **classification_results.json**

```json
{
  "invoice_1.pdf": {
    "class": "Invoice",
    "invoice_number": "INV-1004",
    "date": "2025-03-26",
    "company": "Acme Corp",
    "total_amount": 1955.0
  },
  "resume_ali_khan.txt": {
    "class": "Resume",
    "name": "Ali Khan",
    "email": "ali.khan@example.com",
    "phone": "+1-555-980-6266",
    "experience_years": 4
  },
  "utility_bill.txt": {
    "class": "Utility Bill",
    "account_number": "ACC-49575",
    "billing_date": "2025-05-24",
    "usage": "406 kWh",
    "amount_due": 193.0
  }
}
```

---

## 🎯 **Key Improvements**

### **1. Persistent Embeddings**

**Before:**
```bash
# Every run rebuilds embeddings (slow!)
python document_ai_system.py folder/
# Takes 10-15 minutes every time
```

**Now:**
```bash
# Build embeddings once
python build_embeddings.py  # 10-15 min (once)

# Classify many times (fast!)
python classify_and_extract.py  # 1-2 min
python classify_and_extract.py  # 1-2 min
python classify_and_extract.py  # 1-2 min
```

### **2. Saved FAISS Index**

**Embeddings saved to disk:**
- `embeddings/faiss_index.bin` - Binary FAISS index
- Fast loading (<1 second)
- No rebuilding needed

### **3. Separate Concerns**

- **build_embeddings.py** - Embeddings only
- **classify_and_extract.py** - Classification only
- **api_server_fastapi.py** - API only

### **4. FastAPI Benefits**

- Async/await support
- Automatic OpenAPI docs
- Type validation with Pydantic
- Better performance than Flask

---

## 🔧 **Advanced Usage**

### **Process Multiple Batches**

```bash
# Batch 1
cp /path/batch1/*.pdf input/
python build_embeddings.py
python classify_and_extract.py
mv outputs/classification_results.json results_batch1.json

# Clear for batch 2
rm -rf input/* embeddings/*

# Batch 2
cp /path/batch2/*.pdf input/
python build_embeddings.py
python classify_and_extract.py
mv outputs/classification_results.json results_batch2.json
```

### **Custom Input/Output Folders**

```bash
# Custom folders
python build_embeddings.py --input my_docs --output my_embeddings
```

### **Check What's Indexed**

```bash
# View metadata
cat embeddings/metadata.json

# List indexed files
cat embeddings/filenames.json
```

---

## 🐛 **Troubleshooting**

### **"Embeddings not found"**

```bash
# You need to build embeddings first
python build_embeddings.py
```

### **"Input folder is empty"**

```bash
# Add documents to input/
mkdir input
cp /path/to/documents/*.pdf input/
```

### **"FAISS index not found"**

```bash
# Rebuild embeddings
python build_embeddings.py
```

### **Want to Start Fresh**

```bash
# Clear everything
rm -rf input/* embeddings/* outputs/*

# Or use API
curl -X DELETE http://localhost:8000/clear
```

---

## 📈 **Performance**

### **Build Embeddings (First Time)**
- Model download: 5-10 minutes (one-time)
- 10 documents: 30 seconds
- 100 documents: 3-5 minutes

### **Build Embeddings (Cached Models)**
- 10 documents: 10 seconds
- 100 documents: 1-2 minutes

### **Classification**
- 10 documents: 1-2 minutes
- 100 documents: 10-15 minutes

### **Search**
- Query response: <1 second

---

## 💡 **Tips**

1. **Build embeddings once** - Don't rebuild unless new docs
2. **Keep embeddings/** - It's your index cache
3. **API for integration** - Use FastAPI for other systems
4. **Batch processing** - Process 20-50 docs at a time for best performance

---

## 📝 **File Descriptions**

### **build_embeddings.py**
- Reads documents from `input/`
- Creates embeddings using sentence-transformers
- Builds FAISS index
- Saves everything to `embeddings/`

### **classify_and_extract.py**
- Loads documents from `embeddings/documents.json`
- Classifies using Phi-2
- Extracts structured data
- Saves results to `outputs/classification_results.json`

### **api_server_fastapi.py**
- FastAPI web server
- Endpoints for upload, search, Q&A
- Loads saved FAISS index
- Swagger UI at `/docs`

---

## 🎓 **Example Scenarios**

### **Scenario 1: Client Sends 15 PDFs**

```bash
# Day 1: First batch
cp ~/Downloads/client_docs/*.pdf input/
python build_embeddings.py           # 2-3 minutes
python classify_and_extract.py       # 2-3 minutes
cat outputs/classification_results.json
```

### **Scenario 2: Re-run Classification (Same Docs)**

```bash
# Day 2: Want to re-classify same docs
python classify_and_extract.py       # 2-3 minutes (fast!)
# No need to rebuild embeddings!
```

### **Scenario 3: New Batch Next Week**

```bash
# Week 2: New documents
rm -rf input/* embeddings/*          # Clear old
cp ~/Downloads/new_batch/*.pdf input/
python build_embeddings.py           # Rebuild for new docs
python classify_and_extract.py
```

### **Scenario 4: API Integration**

```bash
# Start API server
python api_server_fastapi.py &

# Your other system uploads docs
curl -X POST http://localhost:8000/upload -F "files=@doc.pdf"

# Your system gets results
curl http://localhost:8000/results
```

---

## ✅ **Quick Reference**

### **Setup (Once)**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **New Documents**
```bash
cp /path/to/docs/*.pdf input/
python build_embeddings.py
python classify_and_extract.py
```

### **Re-classify Existing**
```bash
python classify_and_extract.py
```

### **Start API**
```bash
python api_server_fastapi.py
```

---

## 🆘 **Support Checklist**

- [ ] Documents in `input/` folder
- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] `build_embeddings.py` run successfully
- [ ] `embeddings/` folder exists with files
- [ ] `classify_and_extract.py` run successfully
- [ ] `outputs/classification_results.json` created

---

## 🎉 **Summary**

**Old way:**
```bash
python document_ai_system.py folder/  # 15 min every time
```

**New way:**
```bash
python build_embeddings.py            # 3 min (once)
python classify_and_extract.py        # 2 min (many times)
```

**Result:** 5x faster for repeated classifications! 🚀
