# API Documentation

RESTful API for the Document AI System

## Base URL
```
http://localhost:5000
```

## Authentication
No authentication required (local deployment)

## Endpoints

### 1. Health Check
Check if the API is running

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "Document AI System",
  "version": "1.0.0"
}
```

**cURL Example:**
```bash
curl http://localhost:5000/health
```

---

### 2. Upload Documents
Upload one or more documents for processing

**Endpoint:** `POST /upload`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `files`: File(s) to upload (PDF, DOCX, or TXT)

**Response:**
```json
{
  "success": true,
  "uploaded_files": ["invoice.pdf", "resume.docx"],
  "message": "2 file(s) uploaded successfully"
}
```

**cURL Example:**
```bash
# Single file
curl -X POST http://localhost:5000/upload \
  -F "files=@invoice.pdf"

# Multiple files
curl -X POST http://localhost:5000/upload \
  -F "files=@invoice.pdf" \
  -F "files=@resume.docx" \
  -F "files=@bill.txt"
```

**Python Example:**
```python
import requests

url = "http://localhost:5000/upload"
files = [
    ('files', open('invoice.pdf', 'rb')),
    ('files', open('resume.docx', 'rb'))
]

response = requests.post(url, files=files)
print(response.json())
```

---

### 3. Process Documents
Process all uploaded documents (classify and extract data)

**Endpoint:** `POST /process`

**Response:**
```json
{
  "success": true,
  "results": {
    "invoice.pdf": {
      "class": "Invoice",
      "invoice_number": "INV-1234",
      "date": "2025-01-15",
      "company": "Acme Corp",
      "total_amount": 1250.50
    },
    "resume.docx": {
      "class": "Resume",
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "+1-555-123-4567",
      "experience_years": 5
    }
  },
  "total_documents": 2
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/process
```

**Python Example:**
```python
import requests

response = requests.post("http://localhost:5000/process")
results = response.json()

for filename, data in results['results'].items():
    print(f"{filename}: {data['class']}")
```

---

### 4. Semantic Search
Search documents by semantic meaning

**Endpoint:** `POST /search`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "query": "invoices from March 2025",
  "top_k": 5
}
```

**Response:**
```json
{
  "success": true,
  "query": "invoices from March 2025",
  "results": [
    {
      "filename": "invoice_1.pdf",
      "score": 0.234,
      "preview": "INVOICE\n\nInvoice #1001\nDate: 2025-03-15..."
    }
  ],
  "total_results": 1
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "resumes with 5+ years experience", "top_k": 3}'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:5000/search",
    json={"query": "utility bills", "top_k": 5}
)
print(response.json())
```

---

### 5. Ask Question
Ask a question and get an answer based on documents

**Endpoint:** `POST /ask`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "question": "What is the total amount on the Acme invoice?",
  "top_k": 3
}
```

**Response:**
```json
{
  "success": true,
  "question": "What is the total amount on the Acme invoice?",
  "answer": "The total amount on the Acme Corp invoice is $1,250.50, dated January 15, 2025.",
  "sources": [
    {
      "filename": "invoice.pdf",
      "score": 0.123,
      "preview": "..."
    }
  ]
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who has the most years of experience?"}'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:5000/ask",
    json={"question": "What is the highest utility bill amount?"}
)
print(f"Answer: {response.json()['answer']}")
```

---

### 6. Get Results
Retrieve all processing results

**Endpoint:** `GET /results`

**Response:**
```json
{
  "success": true,
  "results": {
    "invoice.pdf": {...},
    "resume.docx": {...}
  }
}
```

**cURL Example:**
```bash
curl http://localhost:5000/results
```

---

### 7. Download Results
Download results as a JSON file

**Endpoint:** `GET /results/download`

**Response:** JSON file download

**cURL Example:**
```bash
curl http://localhost:5000/results/download -o results.json
```

**Python Example:**
```python
import requests

response = requests.get("http://localhost:5000/results/download")
with open("results.json", "wb") as f:
    f.write(response.content)
```

---

### 8. Classify Single File
Classify a single document without full processing

**Endpoint:** `POST /classify`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file`: Single file to classify

**Response:**
```json
{
  "success": true,
  "filename": "document.pdf",
  "classification": "Invoice"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/classify \
  -F "file=@unknown_document.pdf"
```

**Python Example:**
```python
import requests

files = {'file': open('document.pdf', 'rb')}
response = requests.post("http://localhost:5000/classify", files=files)
print(f"Classification: {response.json()['classification']}")
```

---

### 9. Clear All Data
Clear all uploaded documents and results

**Endpoint:** `POST /clear`

**Response:**
```json
{
  "success": true,
  "message": "All documents cleared"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/clear
```

---

## Complete Workflow Example

### Python Script
```python
import requests
import time

BASE_URL = "http://localhost:5000"

# 1. Check health
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# 2. Upload documents
files = [
    ('files', open('invoice_1.pdf', 'rb')),
    ('files', open('invoice_2.pdf', 'rb')),
    ('files', open('resume.docx', 'rb'))
]
response = requests.post(f"{BASE_URL}/upload", files=files)
print("Upload:", response.json())

# 3. Process documents
print("\nProcessing documents...")
response = requests.post(f"{BASE_URL}/process")
results = response.json()
print(f"Processed {results['total_documents']} documents")

# 4. Search for invoices
response = requests.post(
    f"{BASE_URL}/search",
    json={"query": "invoices", "top_k": 5}
)
print("\nSearch results:", len(response.json()['results']))

# 5. Ask a question
response = requests.post(
    f"{BASE_URL}/ask",
    json={"question": "What is the total of all invoices?"}
)
print("\nAnswer:", response.json()['answer'])

# 6. Download results
response = requests.get(f"{BASE_URL}/results/download")
with open("api_results.json", "wb") as f:
    f.write(response.content)
print("\nResults saved to api_results.json")
```

### Bash Script
```bash
#!/bin/bash

BASE_URL="http://localhost:5000"

# Upload
curl -X POST $BASE_URL/upload \
  -F "files=@invoice.pdf" \
  -F "files=@resume.docx"

# Process
curl -X POST $BASE_URL/process

# Search
curl -X POST $BASE_URL/search \
  -H "Content-Type: application/json" \
  -d '{"query": "invoices", "top_k": 5}'

# Ask
curl -X POST $BASE_URL/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who has the most experience?"}'

# Download
curl $BASE_URL/results/download -o results.json
```

---

## Error Responses

All endpoints return error responses in this format:

```json
{
  "success": false,
  "error": "Error message here"
}
```

### Common HTTP Status Codes
- `200` - Success
- `400` - Bad request (missing parameters, invalid data)
- `413` - File too large (max 10MB)
- `500` - Internal server error

---

## Rate Limiting
No rate limiting (local deployment)

## File Size Limits
- Maximum file size: 10MB per file
- Supported formats: PDF, DOCX, TXT

## Notes
- The API must process documents (`POST /process`) before search and Q&A features are available
- Documents are stored in memory; restart the server or use `/clear` to reset
- First request to `/process` will be slow (model loading), subsequent requests are faster
