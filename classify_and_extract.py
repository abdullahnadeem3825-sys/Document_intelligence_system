"""
Classifies documents and extracts structured data using LLM.
"""

import os
os.environ["HF_HOME"] = r"D:\abdullah_work\atricent-scraper\langchain_models"
import json
import re
import warnings
from pathlib import Path
from typing import Dict, Any, Optional


warnings.filterwarnings('ignore')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.pydantic_v1 import BaseModel, Field


# ─── Structured Output Schemas ──────────────────────────────────────────────

class InvoiceData(BaseModel):
    invoice_number: Optional[str] = Field(None, description="Invoice number e.g. INV-123, 2024-567")
    date: Optional[str] = Field(None, description="Invoice date (any common format)")
    company: Optional[str] = Field(None, description="Supplier or billed-to company name")
    total_amount: Optional[float] = Field(None, description="Final total amount (numeric)")


class ResumeData(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the person")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    experience_years: Optional[int] = Field(None, description="Years of experience (integer)")


class UtilityBillData(BaseModel):
    account_number: Optional[str] = Field(None, description="Account / customer number")
    billing_date: Optional[str] = Field(None, description="Billing or issue date")
    usage: Optional[str] = Field(None, description="Usage amount e.g. 450 kWh, 120 m³")
    amount_due: Optional[float] = Field(None, description="Amount to be paid (numeric)")


# ─── LLM Loader ─────────────────────────────────────────────────────────────

def load_llm_pipeline():
    """Load Qwen2.5-3B-Instruct using transformers pipeline directly"""
    print("Loading Qwen2.5-3B-Instruct...")
    
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    # Create pipeline
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
    
    print(" LLM loaded\n")
    return pipe, tokenizer


# ─── Document Processor ─────────────────────────────────────────────────────

class DocumentClassifierExtractorLLM:
    CATEGORIES = ["Invoice", "Resume", "Utility Bill", "Other", "Unclassifiable"]

    def __init__(
        self,
        embeddings_folder: str = "embeddings",
        output_folder: str = "outputs",
        debug: bool = True,
    ):
        self.embeddings_folder = Path(embeddings_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.debug = debug

        # Load documents
        docs_path = self.embeddings_folder / "documents.json"
        if not docs_path.exists():
            raise FileNotFoundError(
                f"Documents not found at {docs_path}. Run build_embeddings.py first."
            )

        with open(docs_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)

        print(f"Loaded {len(self.documents)} documents")

        # Load LLM
        self.pipe, self.tokenizer = load_llm_pipeline()

    def _apply_chat_template(self, system_msg: str, user_msg: str) -> str:
        """Apply Qwen chat template properly"""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        # Use tokenizer's chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def classify_document(self, text: str) -> str:
        """Classify document using LLM"""
        if not text or text == "UNREADABLE_DOCUMENT" or len(text.strip()) < 30:
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

        user_msg = f"Classify this document:\n\n{text[:2000]}"
        
        prompt = self._apply_chat_template(system_msg, user_msg)
        
        # Generate
        result = self.pipe(prompt, max_new_tokens=50, return_full_text=False)
        category = result[0]['generated_text'].strip()
        
        # Clean up response
        category = category.split('\n')[0].strip()
        
        if category in self.CATEGORIES:
            return category

        # Fallback logic
        text_lower = text.lower()
        has_invoice = sum(kw in text_lower for kw in ["invoice", "inv-", "total amount", "amount due", "subtotal"]) >= 2
        has_resume = sum(kw in text_lower for kw in ["experience", "skills", "education", "@", "curriculum"]) >= 2
        has_utility = sum(kw in text_lower for kw in ["kwh", "account number", "billing period", "utility"]) >= 2

        if has_invoice:
            return "Invoice"
        if has_resume:
            return "Resume"
        if has_utility:
            return "Utility Bill"

        return "Other"

    def extract_invoice(self, text: str) -> Dict[str, Any]:
        """Extract invoice data"""
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

        user_msg = f"Extract invoice data from:\n\n{text[:3000]}"
        
        prompt = self._apply_chat_template(system_msg, user_msg)
        
        # Generate
        result = self.pipe(prompt, max_new_tokens=512, return_full_text=False)
        print("result:", result)
        response = result[0]['generated_text'].strip()
        
        if self.debug:
            print(f"  → Raw output: {response[:150]}...")
        
        return self._parse_json_response(response, InvoiceData)

    def extract_resume(self, text: str) -> Dict[str, Any]:
        """Extract resume data"""
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

        user_msg = f"Extract resume data from:\n\n{text[:3000]}"
        
        prompt = self._apply_chat_template(system_msg, user_msg)
        
        result = self.pipe(prompt, max_new_tokens=512, return_full_text=False)
        response = result[0]['generated_text'].strip()
        
        if self.debug:
            print(f"  → Raw output: {response[:150]}...")
        
        return self._parse_json_response(response, ResumeData)

    def extract_utility(self, text: str) -> Dict[str, Any]:
        """Extract utility bill data"""
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

        user_msg = f"Extract utility bill data from:\n\n{text[:3000]}"
        
        prompt = self._apply_chat_template(system_msg, user_msg)
        
        result = self.pipe(prompt, max_new_tokens=512, return_full_text=False)
        response = result[0]['generated_text'].strip()
        
        if self.debug:
            print(f"  → Raw output: {response[:150]}...")
        
        return self._parse_json_response(response, UtilityBillData)

    def _parse_json_response(self, response: str, schema_class) -> Dict[str, Any]:
        """Parse JSON from LLM response with multiple fallback strategies"""
        
        # Strategy 1: Find JSON object in response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                data_dict = json.loads(json_str)
                validated = schema_class(**data_dict)
                return validated.dict(exclude_none=True)
            except (json.JSONDecodeError, Exception) as e:
                if self.debug:
                    print(f" Strategy 1 failed: {e}")
        
        # Strategy 2: Try to find nested JSON (sometimes model wraps it)
        nested_match = re.search(r'\{.*\{[^{}]*\}.*\}', response, re.DOTALL)
        if nested_match:
            json_str = nested_match.group(0)
            # Find innermost JSON
            inner_match = re.search(r'\{[^{}]*\}', json_str)
            if inner_match:
                try:
                    data_dict = json.loads(inner_match.group(0))
                    validated = schema_class(**data_dict)
                    return validated.dict(exclude_none=True)
                except (json.JSONDecodeError, Exception) as e:
                    if self.debug:
                        print(f"  Strategy 2 failed: {e}")
        
        # Strategy 3: Try entire response as JSON
        try:
            data_dict = json.loads(response)
            validated = schema_class(**data_dict)
            return validated.dict(exclude_none=True)
        except (json.JSONDecodeError, Exception) as e:
            if self.debug:
                print(f"  Strategy 3 failed: {e}")
        
        # All strategies failed
        if self.debug:
            print(f"   All JSON parsing strategies failed")
        
        return {}

    def extract_structured(self, text: str, category: str) -> Dict[str, Any]:
        """Route to appropriate extraction method"""
        if category == "Invoice":
            return self.extract_invoice(text)
        elif category == "Resume":
            return self.extract_resume(text)
        elif category == "Utility Bill":
            return self.extract_utility(text)
        else:
            return {}

    def process_all_documents(self) -> Dict[str, Dict[str, Any]]:
        print("\n" + "="*70)
        print("CLASSIFICATION + LLM EXTRACTION RESULTS")
        print("="*70 + "\n")

        results = {}

        for i, (filename, text) in enumerate(self.documents.items(), 1):
            print(f"[{i}/{len(self.documents)}] {filename}")

            category = self.classify_document(text)
            print(f"  → Class: {category}")

            extracted = self.extract_structured(text, category)
            print(f"  → Extracted: {extracted}")
            print()

            result = {"class": category}
            result.update(extracted)

            results[filename] = result

        print("\n" + "="*70)
        print("DONE")
        print("="*70)
        return results

    def save_results(self, results: Dict[str, Dict[str, Any]]):
        output_path = self.output_folder / "classification_results_llm.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved → {output_path}")

        # Summary
        from collections import Counter
        counts = Counter(d.get("class", "Unknown") for d in results.values())
        print("\nSummary:")
        for cat, cnt in sorted(counts.items()):
            print(f"  {cat:12} : {cnt}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="embeddings")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug output")
    args = parser.parse_args()

    extractor = DocumentClassifierExtractorLLM(
        embeddings_folder=args.embeddings,
        output_folder=args.output,
        debug=not args.no_debug,
    )

    results = extractor.process_all_documents()
    extractor.save_results(results)


if __name__ == "__main__":
    main()