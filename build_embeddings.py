"""
build_embeddings.py
-------------------
Run this ONLY when you get new documents from client.
Creates embeddings and FAISS index, saves them to disk.

Usage:
    python build_embeddings.py

This will:
1. Read all documents from input/ folder
2. Create embeddings for each document
3. Build FAISS index
4. Save everything to embeddings/ folder
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

import PyPDF2
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class EmbeddingBuilder:
    """Builds and saves embeddings + FAISS index"""
    
    def __init__(
        self, 
        input_folder: str = "input",
        embeddings_folder: str = "embeddings",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.input_folder = Path(input_folder)
        self.embeddings_folder = Path(embeddings_folder)
        self.embeddings_folder.mkdir(exist_ok=True)
        
        print(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        print("✓ Model loaded")
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
                return text.strip()
        except Exception as e:
            print(f"  Error reading PDF {pdf_path.name}: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_path: Path) -> str:
        """Extract text from DOCX"""
        try:
            doc = DocxDocument(docx_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text.strip()
        except Exception as e:
            print(f"  Error reading DOCX {docx_path.name}: {e}")
            return ""
    
    def extract_text_from_txt(self, txt_path: Path) -> str:
        """Extract text from TXT"""
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read().strip()
        except Exception as e:
            print(f"  Error reading TXT {txt_path.name}: {e}")
            return ""
    
    def read_all_documents(self) -> Dict[str, str]:
        """Read all documents from input folder"""
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {self.input_folder}")
        
        documents = {}
        supported_extensions = {'.pdf', '.docx', '.txt'}
        
        print(f"\nReading documents from: {self.input_folder}")
        print("=" * 60)
        
        for file_path in self.input_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                print(f"Reading: {file_path.name}")
                
                if file_path.suffix.lower() == '.pdf':
                    text = self.extract_text_from_pdf(file_path)
                elif file_path.suffix.lower() == '.docx':
                    text = self.extract_text_from_docx(file_path)
                elif file_path.suffix.lower() == '.txt':
                    text = self.extract_text_from_txt(file_path)
                else:
                    continue
                
                if text and len(text.strip()) > 20:
                    documents[file_path.name] = text
                else:
                    print(f"  Document is empty or too short, skipping")
        
        print("=" * 60)
        print(f"✓ Successfully read {len(documents)} documents\n")
        return documents
    
    def build_and_save_embeddings(self):
        """Main method: Build embeddings and save everything"""
        print("\n" + "=" * 60)
        print("BUILDING EMBEDDINGS & FAISS INDEX")
        print("=" * 60 + "\n")
        
        # Step 1: Read documents
        documents = self.read_all_documents()
        
        if not documents:
            print(" No valid documents found in input folder!")
            return
        
        # Step 2: Create embeddings
        print("Creating embeddings...")
        filenames = list(documents.keys())
        texts = list(documents.values())
        
        embeddings = self.encoder.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        embeddings = embeddings.astype('float32')
        
        print(f"✓ Created {len(embeddings)} embeddings (shape: {embeddings.shape})")
        
        # Step 3: Build FAISS index
        print("\nBuilding FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print(f" FAISS index built with dimension {dimension}")
        
        # Step 4: Save everything to disk
        print("\nSaving to disk...")
        
        # Save FAISS index
        faiss_path = self.embeddings_folder / "faiss_index.bin"
        faiss.write_index(index, str(faiss_path))
        print(f"  FAISS index saved: {faiss_path}")
        
        # Save documents (for retrieval later)
        docs_path = self.embeddings_folder / "documents.json"
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        print(f"  Documents saved: {docs_path}")
        
        # Save filenames (for mapping)
        filenames_path = self.embeddings_folder / "filenames.json"
        with open(filenames_path, 'w', encoding='utf-8') as f:
            json.dump(filenames, f, indent=2)
        print(f"  Filenames saved: {filenames_path}")
        
        # Save metadata
        metadata = {
            "num_documents": len(documents),
            "embedding_dimension": dimension,
            "embedding_model": "all-MiniLM-L6-v2"
        }
        metadata_path = self.embeddings_folder / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata saved: {metadata_path}")
        
        print("\n" + "=" * 60)
        print(" EMBEDDINGS BUILD COMPLETE")
        print("=" * 60)
        print(f"\nProcessed {len(documents)} documents")
        print(f"Embeddings saved in: {self.embeddings_folder}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build embeddings and FAISS index")
    parser.add_argument("--input", default="input", help="Input folder with documents")
    parser.add_argument("--output", default="embeddings", help="Output folder for embeddings")
    
    args = parser.parse_args()
    
    # Check if input folder exists
    if not Path(args.input).exists():
        print(f"\n Error: Input folder '{args.input}' does not exist!")
        print(f"\nPlease create it and add your documents:")
        print(f"  mkdir {args.input}")
        print(f"  cp /path/to/documents/*.pdf {args.input}/")
        return
    
    # Check if input folder has files
    input_path = Path(args.input)
    files = list(input_path.glob("*.*"))
    if not files:
        print(f"\n Error: Input folder '{args.input}' is empty!")
        print(f"\nPlease add your documents:")
        print(f"  path of your folder/*.pdf {args.input}/")
        return
    
    # Build embeddings
    builder = EmbeddingBuilder(args.input, args.output)
    builder.build_and_save_embeddings()


if __name__ == "__main__":
    main()
