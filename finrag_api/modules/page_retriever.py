import torch
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import fitz  # PyMuPDF
import os
import gc
from langchain.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PageRetriever:
    def __init__(self, args):
        self.args = args
        
        # GPU settings
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu_devices"]
            self.device = "cuda"  # Use all available GPUs
        else:
            self.device = "cpu"
            
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self.batch_size = args["batch_size"]
        self.pdf_path = args["pdf_path"]
        self.retrieve_strategy = 'page'
        
        # Text splitting settings
        self.chunk_size = 1024
        self.chunk_overlap = 30

        self.tokenizer_name = "naver/trecdl22-crossencoder-debertav3"
        
    def get_pdf_path(self, doc_name: str) -> str:
        """Generate PDF file path"""
        doc_name = os.path.basename(doc_name).replace(".pdf", "")
        ticker, period, report_type = doc_name.split("_")
        if report_type == "10K":
            doc_type = "10-K"
        elif report_type == "10Q":
            doc_type = "10-Q"
        elif report_type == "8K":
            doc_type = "8-K"
        else:
            doc_type = report_type

        pdf_path = os.path.join(self.pdf_path, ticker, doc_type, f"{doc_name}.pdf")
        return pdf_path
        
    async def initialize(self):
        """Initialize model"""
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Initialize Cross-encoder model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                trust_remote_code=True,
                use_fast=False
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.args["cross_encoder_model_name"],
                trust_remote_code=True
            ).to(self.device)  # Direct device assignment
            
            self.initialized = True
            print(f"Page Retriever initialization completed (Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']})")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise e
    
    def load_documents(self, doc_names: List[str]) -> List[Document]:
        """Load and split multiple documents"""
        all_documents = []
        for doc_name in doc_names:
            try:
                documents = self.load_document(doc_name)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading document {doc_name}: {str(e)}")
                continue
        return all_documents
    
    def load_document(self, doc_name: str) -> List[Document]:
        """Load and split a single document"""
        pdf_path = self.get_pdf_path(doc_name)
        if not os.path.exists(pdf_path):
            raise AssertionError(f"Document {pdf_path} not exists.")

        try:
            # Load PDF
            pdf_reader = PyMuPDFLoader(pdf_path)
            documents = pdf_reader.load()
            
            # Split text
            documents = self.split_text(documents)
            
            # Update source information for all documents
            for document in documents:
                document.metadata["source"] = doc_name

            return documents

        except Exception as e:
            print(f"Error loading document {doc_name}: {str(e)}")
            raise e

    def split_text(self, documents: List[Document]) -> List[Document]:
        """Split document text"""
        try:
            # Store full text for each page
            page_contents = {doc.metadata["page"]: doc.page_content for doc in documents}
            
            # Initialize text splitter and split
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                add_start_index=True,
            )
            split_documents = text_splitter.split_documents(documents)

            # Add full page text to split documents
            for document in split_documents:
                cur_page = document.metadata["page"]
                document.metadata["full_page_text"] = page_contents[cur_page]

            return split_documents

        except Exception as e:
            print(f"Error during text splitting: {str(e)}")
            return documents  # Return original documents if error occurs
    
    def document_to_dict(self, document: Document) -> Dict[str, Any]:
        """Convert Document object to dictionary"""
        page = document.metadata["page"]
        start_index = document.metadata.get("start_index")
        page_content = document.page_content
        source = os.path.basename(document.metadata["source"]).replace(".pdf", "")
        full_page_content = document.metadata["full_page_text"]
        return {
            "source": source,
            "page": page,
            "page_content": page_content,
            "full_page_content": full_page_content,
            "start_index": start_index,
        }

    async def retrieve_pages(self, query: str, doc_names: List[str], k: int = 3) -> Dict[str, Any]:
        """Retrieve pages"""
        if not self.initialized:
            raise Exception("Page Retriever is not initialized.")
        
        try:
            # Load all documents
            documents = self.load_documents(doc_names)
            
            # Calculate page scores and rank
            ranked_passages = await self._rank_passages(query, documents)
            
            # Filter by page
            if self.retrieve_strategy == "page":
                retrieved_pages = []
                retrieved_page_nums = set()
                for document in ranked_passages:
                    page = document.metadata["page"]
                    doc_name = os.path.basename(document.metadata["source"]).replace(".pdf", "")
                    page_key = f"{doc_name}_{page}"
                    
                    if page_key in retrieved_page_nums:
                        continue
                        
                    retrieved_pages.append(document)
                    retrieved_page_nums.add(page_key)
                    if len(retrieved_pages) == k:
                        break
            
            # Convert results
            results = [self.document_to_dict(doc) for doc in retrieved_pages]
            
            return {
                "params": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "retrieve_strategy": self.retrieve_strategy,
                    "k": k
                },
                "results": results
            }
            
        except Exception as e:
            print(f"Error during page retrieval: {str(e)}")
            return {
                "params": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "retrieve_strategy": self.retrieve_strategy,
                    "k": k
                },
                "results": []
            }
    
    async def _rank_passages(self, query: str, documents: List[Document]) -> List[Document]:
        """Calculate page scores and rank"""
        try:
            scores = []
            passages = ["Document title: "+' '.join(os.path.basename(doc.metadata['source']).replace('.pdf','').split("_")) + ". Context: " + doc.page_content for doc in documents]
            
            # Process in batches
            for i in range(0, len(passages), self.batch_size):
                batch = passages[i:i + self.batch_size]
                
                # Prepare batch data
                inputs = [f"{query} [SEP] {passage}" for passage in batch]
                batch_encoding = self.tokenizer(
                    inputs,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Calculate scores
                with torch.no_grad():
                    outputs = self.model(**batch_encoding)
                    logits = outputs.logits.squeeze(-1)
                    scores.extend(logits.cpu().tolist())
            
            # Rank documents
            ranks = [
                {"corpus_id": i, "score": score}
                for i, score in enumerate(scores)
            ]
            ranks = sorted(ranks, key=lambda x: x["score"], reverse=True)
            
            # Return documents in ranked order
            ranked_passages = [documents[rank["corpus_id"]] for rank in ranks]
            return ranked_passages
            
        except Exception as e:
            print(f"Error during page scoring: {str(e)}")
            return documents
    
    def cleanup(self):
        """Clean up resources"""
        if self.model:
            self.model = None
            del self.model
        if self.tokenizer:
            self.tokenizer = None
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Page Retriever resources cleaned up") 