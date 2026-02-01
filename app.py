"""
app.py - FastAPI Backend for Document Intelligence System
----------------------------------------------------------
Features:
1. Upload & Classify: Upload a document → get classification + extracted data
2. Semantic Search: Search indexed documents by meaning
3. Q&A: Ask questions based on indexed documents

Important:
- Documents from /upload are ONLY classified & extracted
- They are NOT added to FAISS index or documents_db.json
"""

import os
os.environ["HF_HOME"] = r"D:\abdullah_work\atricent-scraper\langchain_models"

import json
import re
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

warnings.filterwarnings('ignore')

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss

try:
    import PyPDF2
except ImportError:
    print("Warning: PyPDF2 not installed. PDF support disabled.")

try:
    from docx import Document as DocxDocument
except ImportError:
    print("Warning: python-docx not installed. DOCX support disabled.")


# ─── Pydantic Models ────────────────────────────────────────────────────────

from langchain_core.pydantic_v1 import BaseModel as PydanticBaseModel, Field as PydanticField

class InvoiceData(PydanticBaseModel):
    invoice_number: Optional[str] = PydanticField(None)
    date: Optional[str] = PydanticField(None)
    company: Optional[str] = PydanticField(None)
    total_amount: Optional[float] = PydanticField(None)


class ResumeData(PydanticBaseModel):
    name: Optional[str] = PydanticField(None)
    email: Optional[str] = PydanticField(None)
    phone: Optional[str] = PydanticField(None)
    experience_years: Optional[int] = PydanticField(None)


class UtilityBillData(PydanticBaseModel):
    account_number: Optional[str] = PydanticField(None)
    billing_date: Optional[str] = PydanticField(None)
    usage: Optional[str] = PydanticField(None)
    amount_due: Optional[float] = PydanticField(None)


# ─── Request / Response Models ──────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class QuestionRequest(BaseModel):
    question: str


class SearchResult(BaseModel):
    filename: str
    relevance_score: float
    document_class: str
    preview: str
    extracted_data: Dict[str, Any]


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]


# ─── Document Extractor ─────────────────────────────────────────────────────

class DocumentExtractor:
    @staticmethod
    def extract_text(file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            return DocumentExtractor._extract_pdf(file_path)
        elif suffix in ['.docx', '.doc']:
            return DocumentExtractor._extract_docx(file_path)
        elif suffix == '.txt':
            return DocumentExtractor._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def _extract_pdf(file_path: Path) -> str:
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join(page.extract_text() or "" for page in reader.pages)
                return text.strip()
        except Exception as e:
            return f"UNREADABLE_DOCUMENT: {e}"

    @staticmethod
    def _extract_docx(file_path: Path) -> str:
        try:
            doc = DocxDocument(file_path)
            return "\n".join(p.text for p in doc.paragraphs).strip()
        except Exception as e:
            return f"UNREADABLE_DOCUMENT: {e}"

    @staticmethod
    def _extract_txt(file_path: Path) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read().strip()
        except Exception as e:
            return f"UNREADABLE_DOCUMENT: {e}"


# ─── Core Document Intelligence System ──────────────────────────────────────

class DocumentIntelligenceSystem:
    CATEGORIES = ["Invoice", "Resume", "Utility Bill", "Other", "Unclassifiable"]

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.uploads_dir = self.data_dir / "uploads"
        self.uploads_dir.mkdir(exist_ok=True)

        self.db_file = self.data_dir / "documents_db.json"
        self.index_file = self.data_dir / "embeddings.index"

        self.documents_db = self._load_db()

        print("Loading models...")
        self.pipe, self.tokenizer = self._load_llm()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = self._load_or_create_index()
        print(" System ready!")

    def _load_llm(self):
        model_id = "Qwen/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        return pipe, tokenizer

    def _load_db(self) -> Dict:
        if self.db_file.exists():
            with open(self.db_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_db(self):
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents_db, f, indent=2, ensure_ascii=False)

    def _load_or_create_index(self):
        if self.index_file.exists() and len(self.documents_db) > 0:
            return faiss.read_index(str(self.index_file))
        dimension = 384
        return faiss.IndexFlatIP(dimension)  # cosine similarity

    def _save_index(self):
        faiss.write_index(self.index, str(self.index_file))

    def _apply_chat_template(self, system_msg: str, user_msg: str) -> str:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def classify_document(self, text: str) -> str:
        if not text or text.startswith("UNREADABLE") or len(text.strip()) < 30:
            return "Unclassifiable"

        system_msg = """You are a strict document classifier.
You MUST choose EXACTLY ONE category.
Only use a positive category (Invoice, Resume, Utility Bill) if you see CLEAR evidence.

Rules:
- Invoice: ONLY if you see invoice number, INV-, total amount, amount due, subtotal, payment terms
- Resume: ONLY if you see name, email, phone, "experience", "skills", "education"
- Utility Bill: ONLY if you see account number, kWh/m³/units, billing period, utility company
- Other: anything else (letters, notes, forms, generic documents)
- Unclassifiable: too short, garbage, unreadable

Respond with ONLY the category name. Nothing else."""
        user_msg = f"Classify:\n\n{text[:2000]}"
        prompt = self._apply_chat_template(system_msg, user_msg)

        result = self.pipe(prompt, max_new_tokens=50, return_full_text=False)
        category = result[0]['generated_text'].strip().split('\n')[0].strip()

        if category in self.CATEGORIES:
            return category

        # fallback keyword check
        text_lower = text.lower()
        if sum(kw in text_lower for kw in ["invoice", "inv-", "total amount", "amount due", "subtotal"]) >= 2:
            return "Invoice"
        if sum(kw in text_lower for kw in ["experience", "skills", "education", "@", "curriculum"]) >= 2:
            return "Resume"
        if sum(kw in text_lower for kw in ["kwh", "account number", "billing period", "utility"]) >= 2:
            return "Utility Bill"
        return "Other"

    def extract_data(self, text: str, category: str) -> Dict[str, Any]:
        if category == "Invoice":
            return self._extract_invoice(text)
        elif category == "Resume":
            return self._extract_resume(text)
        elif category == "Utility Bill":
            return self._extract_utility(text)
        return {}

    def _extract_invoice(self, text: str) -> Dict[str, Any]:
        system_msg = """You are a data extraction assistant. Extract invoice information and return ONLY a valid JSON object.

Extract these fields if present:
- invoice_number: string (e.g. "INV-123")
- date: string (invoice date)
- company: string (company name)
- total_amount: number (final total as number, not string)

Use null for missing fields.

Example response:
{"invoice_number": "INV-001", "date": "2024-01-15", "company": "Acme Corp", "total_amount": 1250.50}

Return ONLY the JSON, no other text."""
        prompt = self._apply_chat_template(system_msg, f"Extract:\n\n{text[:3000]}")
        result = self.pipe(prompt, max_new_tokens=512, return_full_text=False)
        return self._parse_json(result[0]['generated_text'].strip(), InvoiceData)

    def _extract_resume(self, text: str) -> Dict[str, Any]:
        system_msg = """You are a data extraction assistant. Extract resume information and return ONLY a valid JSON object.

Extract these fields if present:
- name: string (full name)
- email: string (email address)
- phone: string (phone number)
- experience_years: integer (years of experience)

Use null for missing fields.

Example response:
{"name": "John Doe", "email": "john@example.com", "phone": "+1-555-0100", "experience_years": 5}

Return ONLY the JSON, no other text."""
        prompt = self._apply_chat_template(system_msg, f"Extract:\n\n{text[:3000]}")
        result = self.pipe(prompt, max_new_tokens=512, return_full_text=False)
        return self._parse_json(result[0]['generated_text'].strip(), ResumeData)

    def _extract_utility(self, text: str) -> Dict[str, Any]:
        system_msg = """You are a data extraction assistant. Extract utility bill information and return ONLY a valid JSON object.

Extract these fields if present:
- account_number: string (account number)
- billing_date: string (billing date)
- usage: string (usage with units, e.g. "450 kWh")
- amount_due: number (amount to pay as number)

Use null for missing fields.

Example response:
{"account_number": "ACC-12345", "billing_date": "2024-01-15", "usage": "450 kWh", "amount_due": 125.50}

Return ONLY the JSON, no other text."""
        prompt = self._apply_chat_template(system_msg, f"Extract:\n\n{text[:3000]}")
        result = self.pipe(prompt, max_new_tokens=512, return_full_text=False)
        return self._parse_json(result[0]['generated_text'].strip(), UtilityBillData)

    def _parse_json(self, response: str, schema_class) -> Dict[str, Any]:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                validated = schema_class(**data)
                return validated.dict(exclude_none=True)
            except:
                pass
        return {}

    def classify_and_extract_only(self, file_path: Path, filename: str) -> Dict[str, Any]:
        text = DocumentExtractor.extract_text(file_path)
        category = self.classify_document(text)
        extracted = self.extract_data(text, category)
        return {
            "filename": filename,
            "category": category,
            "extracted_data": extracted
        }

    def process_upload(self, file_path: Path, filename: str) -> Dict[str, Any]:
        """Full indexing version — use manually or via future /index endpoint"""
        text = DocumentExtractor.extract_text(file_path)
        category = self.classify_document(text)
        extracted = self.extract_data(text, category)
        print("Extracted data:", extracted)

        embedding = self.embedding_model.encode([text[:5000]])[0]
        embedding = embedding / np.linalg.norm(embedding)

        doc_id = len(self.documents_db)
        self.index.add(np.array([embedding], dtype=np.float32))

        self.documents_db[filename] = {
            "id": doc_id,
            "filename": filename,
            "category": category,
            "extracted_data": extracted,
            "text": text[:10000],
            "upload_date": datetime.now().isoformat(),
        }

        self._save_db()
        self._save_index()

        return {
            "filename": filename,
            "category": category,
            "extracted_data": extracted,
        }

    def semantic_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if len(self.documents_db) == 0:
            return []

        q_emb = self.embedding_model.encode([query])[0]
        q_emb = q_emb / np.linalg.norm(q_emb)

        distances, indices = self.index.search(
            np.array([q_emb], dtype=np.float32),
            min(top_k, len(self.documents_db))
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            for doc in self.documents_db.values():
                if doc["id"] == idx:
                    results.append(SearchResult(
                        filename=doc["filename"],
                        relevance_score=float(dist),
                        document_class=doc["category"],
                        preview=doc.get("text", "")[:200] + "...",
                        extracted_data=doc["extracted_data"]
                    ))
                    break
        return results

    def answer_question(self, question: str) -> AnswerResponse:
        results = self.semantic_search(question, top_k=3)
        if not results:
            return AnswerResponse(answer="No relevant documents found.", sources=[])

        context = ""
        sources = []
        for r in results:
            doc = self.documents_db[r.filename]
            context += f"\n--- {r.filename} ---\n{doc.get('text', '')[:2000]}\n"
            sources.append(r.filename)

        system_msg = """Answer based ONLY on the provided documents.
If the information is not there, say so. Be concise."""
        user_msg = f"Documents:\n{context}\n\nQuestion: {question}"
        prompt = self._apply_chat_template(system_msg, user_msg)

        result = self.pipe(prompt, max_new_tokens=512, return_full_text=False)
        answer = result[0]['generated_text'].strip()

        return AnswerResponse(answer=answer, sources=sources)


# ─── FastAPI App ────────────────────────────────────────────────────────────

app = FastAPI(title="Document Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system = DocumentIntelligenceSystem()


@app.get("/")
async def root():
    return {
        "message": "Document Intelligence API",
        "note": "Uploaded documents are classified only — not indexed. Use manual indexing for search/Q&A.",
        "endpoints": {
            "upload": "/upload → classify + extract (no indexing)",
            "search": "/search → semantic search (indexed docs only)",
            "ask": "/ask → question answering (indexed docs only)",
            "stats": "/stats → number of indexed documents"
        }
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        print(f"Upload received: {file.filename}")
        file_path = system.uploads_dir / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        result = system.classify_and_extract_only(file_path, file.filename)
        print(f"Processed (no index): {file.filename} → {result['category']}")
        print("Extracted result:", result)
        return result

    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(500, detail=str(e))


@app.post("/search")
async def search_documents(request: SearchRequest):
    try:
        results = system.semantic_search(request.query, request.top_k)
        print(f"Search '{request.query}' → {len(results)} results")
        return [r.dict() for r in results]
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        answer = system.answer_question(request.question)
        print(f"Q&A: '{request.question[:60]}...' → {len(answer.sources)} sources")
        return answer.dict()
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/stats")
async def get_stats():
    from collections import Counter
    categories = Counter(d["category"] for d in system.documents_db.values())
    return {
        "total_indexed_documents": len(system.documents_db),
        "categories": dict(categories)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)