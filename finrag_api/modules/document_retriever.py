import torch
from typing import Dict, Any, List
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import gc
from tqdm import tqdm
from langchain.document_loaders import PyMuPDFLoader
from pathlib import Path

class DocumentRetriever:
    def __init__(self, args):
        self.args = args
        
        # GPU settings
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu_devices"]
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.reranker = None
        self.tokenizer = None
        self.embeddings = None
        self.retriever = None
        self.initialized = False
        
        # PDF path settings
        self.pdf_path = args["pdf_path"]
        
    def get_pdf_path(self, doc_name: str) -> str:
        """Generate PDF file path"""
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
        """Initialize models"""
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Initialize embedding model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.args["model_name"],
                model_kwargs={"device": self.device}
            )
            
            # Include model name in path
            persist_dir = os.path.join(self.args["db_dir"], self.args["model_name"])
            
            # Initialize vector store
            self.retriever = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
            
            # Initialize Reranker (optional)
            if self.args["use_reranker"]:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.args["reranker_model_name"],
                )   
                self.reranker = AutoModelForSequenceClassification.from_pretrained(
                    self.args["reranker_model_name"],
                ).to(self.device)  # Direct device assignment
            
            self.initialized = True
            print(f"Document Retriever initialized (Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']})")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise e
    
    async def retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents"""
        if not self.initialized:
            raise Exception("Document Retriever is not initialized.")
        
        try:
            # Document retrieval
            retrieved = self.retriever.similarity_search(query, k=k)
            
            # Reranking (optional)
            if self.args["use_reranker"]:
                retrieved = await self._rerank_documents(query, retrieved)
            
            # Convert result format
            results = []
            for doc in retrieved:
                results.append({
                    "source": doc.metadata["source"],
                })
            
            return results
            
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            return []
    
    async def retrieve_pages(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve pages and extract PDF content"""
        try:
            # Document retrieval
            retrieved = self.retriever.similarity_search(query, k=k)
            
            # Reranking (optional)
            if self.args["use_reranker"]:
                retrieved = await self._rerank_documents(query, retrieved)
            
            # Convert result format and extract PDF content
            results = []
            for doc in retrieved:
                doc_name = doc.metadata["source"]
                page_num = doc.metadata.get("page", 1)
                
                # Generate PDF file path
                pdf_path = self.get_pdf_path(doc_name)
                
                # Load PDF and extract page content
                try:
                    loader = PyMuPDFLoader(pdf_path)
                    pages = loader.load()
                    if 0 <= page_num - 1 < len(pages):
                        page_content = pages[page_num - 1].page_content
                    else:
                        page_content = doc.page_content
                except Exception as e:
                    print(f"Error loading PDF ({pdf_path}): {str(e)}")
                    page_content = doc.page_content
                
                results.append({
                    "source": doc_name,
                    "page": page_num,
                    "text": doc.page_content,  # Retrieved text
                    "full_page_content": page_content  # Full page content
                })
            
            return results
            
        except Exception as e:
            print(f"Error during page retrieval: {str(e)}")
            return []
    
    async def _rerank_documents(self, query: str, documents: List[Any]) -> List[Any]:
        """Rerank documents"""
        try:
            passages = [doc.page_content for doc in documents]
            scores = []
            
            # Batch processing
            batch_size = self.args["batch_size"]
            batches = [
                passages[i : i + batch_size]
                for i in range(0, len(passages), batch_size)
            ]
            
            with torch.no_grad():
                for batch in batches:
                    inputs = [f"query: {query} [SEP] {passage}" for passage in batch]
                    
                    # Tokenization
                    batch_encoding = self.tokenizer(
                        inputs,
                        max_length=512,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Calculate scores
                    outputs = self.reranker(**batch_encoding)
                    logits = outputs.logits.squeeze(-1)
                    scores.extend(logits.cpu().tolist())
            
            # Sort by scores
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, _ in scored_docs]
            
        except Exception as e:
            print(f"Error during reranking: {str(e)}")
            return documents
    
    def cleanup(self):
        """Clean up resources"""
        if self.reranker:
            self.reranker = None
            del self.reranker
        if self.tokenizer:
            self.tokenizer = None
            del self.tokenizer
        if self.embeddings:
            self.embeddings = None
            del self.embeddings
        if self.retriever:
            self.retriever = None
            del self.retriever
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Document Retriever resources cleaned up") 