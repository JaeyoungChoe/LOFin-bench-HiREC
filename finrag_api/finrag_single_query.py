import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime
import random
import numpy as np
import torch

from modules.document_retriever import DocumentRetriever
from modules.page_retriever import PageRetriever
from modules.llm_server import LLMServer
from modules.query_transformer import QueryTransformer
from modules.evidence_curator import EvidenceCurator
from modules.generator import Generator
from modules.evaluator import FinRAGEvaluator

class FinRAGSingleQuery:
    def __init__(self, args):
        self.args = args
        self.dataset = None
        self.current_idx = 0
        self.document_retriever = None
        self.page_retriever = None
        self.llm_server = None
        self.query_transformer = None
        self.evidence_curator = None
        self.generator = None
        self.evaluator = None  # Set after generator initialization
        self.logger = logging.getLogger("FinRAG")
        
    async def initialize(self):
        """Initialize components"""
        try:
            self.logger.info("Starting component initialization")
            
            # Initialize LLM server (using multiple GPUs)
            self.llm_server = LLMServer(self.args)
            await self.llm_server.initialize()
            
            # Initialize Document Retriever (GPU 1)
            self.document_retriever = DocumentRetriever(self.args)
            await self.document_retriever.initialize()
            
            # Initialize Page Retriever (GPU 2)
            self.page_retriever = PageRetriever(self.args)
            await self.page_retriever.initialize()
            
            # Initialize Query Transformer
            self.query_transformer = QueryTransformer(self.llm_server)
            
            # Initialize Evidence Curator
            self.evidence_curator = EvidenceCurator(self.llm_server, self.args)
            
            # Initialize Generator
            self.generator = Generator(self.args)
            
            # Initialize Evaluator (with Generator)
            self.evaluator = FinRAGEvaluator(self.generator)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            await self.cleanup()
            raise e
    
    async def load_dataset(self, dataset_name: str):
        """Load dataset"""
        try:
            # Determine appropriate directory and filename based on dataset name
            if dataset_name in ["numeric_text", "numeric_table", "textual"]:
                dataset_path = os.path.join(
                    Path(__file__).parent.parent, "data", "by_answer_type", f"{dataset_name}_test.jsonl"
                )
            elif dataset_name in ["finqa", "financebench", "secqa"]:
                dataset_path = os.path.join(
                    Path(__file__).parent.parent, "data", "by_data_source", f"{dataset_name}_test.jsonl"
                )
            elif dataset_name == "all":
                dataset_path = os.path.join(
                    Path(__file__).parent.parent, "data", "all", "all_test.jsonl"
                )
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}")

            self.logger.info(f"Dataset file path: {dataset_path}")
            self.dataset = pd.read_json(dataset_path, lines=True)
            self.logger.info(f"Dataset loaded successfully: {len(self.dataset)} queries")
            self.logger.debug(f"Dataset columns: {self.dataset.columns.tolist()}")
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}", exc_info=True)
            raise e
    
    def get_next_query(self) -> Dict[str, Any]:
        """Get next query"""
        if self.current_idx >= len(self.dataset):
            return None
            
        query = self.dataset.iloc[self.current_idx].to_dict()
        self.current_idx += 1
        return query
    
    async def process_single_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Process single query"""
        try:
            original_question = query['question']
            question = query['question']
            relevant_pages = []
            for iteration in range(self.args.get("max_iteration", 4)):
               
                # 1. Query transformation
                self.logger.debug("Starting Query transformation")
                transformed_query = await self.query_transformer.transform_query(question)

                
                # 2. Document retrieval
                self.logger.debug("Starting Hierarchical retrieval")
                documents = await self.document_retriever.retrieve_documents(transformed_query, k=100)

                
                # 3. Page retrieval
                doc_names = [doc["source"] for doc in documents[:10]]
                page_result = await self.page_retriever.retrieve_pages(
                    question,
                    doc_names,
                    k=self.args.get("pages_per_doc", 10)
                )

                pages = page_result['results'][:5]
                
                # 4. Evidence collection
                self.logger.debug("Starting Evidence collection")
                pages = relevant_pages + pages
                evidence = await self.evidence_curator.curate_evidence(original_question, pages)
                
                relevant_pages = evidence['relevant_pages']

                # If answerable is 'answerable', break the loop
                if evidence['is_answerable'] == True:
                    break
                
                question = evidence['refined_query']
                iteration += 1
                  
            # Basic retrieval result
            retrieval_result = {
                "ground_truth": {
                    "answer": query['answer'],
                    "evidences": query['evidences']
                },
                "query": original_question,
                "transformed_query": transformed_query,
                "documents": doc_names,
                "retrieved_pages": [page["source"] +"_"+ str(page["page"]) for page in evidence["relevant_pages"]],
                "evidence": evidence
            }
            
            # Perform generation only if required
            if self.args.get("do_generate", False):
                answer_type = 'pot' if 'numeric' in self.args.get('dataset') else self.args.get("answer_type", "cot")  # cot, pot, direct
                generated_answer, generation_metadata = self.generator.generate_answer(
                    question=original_question,
                    retrieved_passages=evidence.get("relevant_pages", []),
                    answer_type=answer_type
                )
                    
                # Add generation results
                generation_result = {
                    "generated_answer": generated_answer,
                    "generation_metadata": generation_metadata,
                    "answer_type": answer_type
                }
                retrieval_result.update(generation_result)
            
            return retrieval_result
            
        except Exception as e:
            self.logger.error(f"Error during query processing: {str(e)}", exc_info=True)
            return {
                "query": query,
                "error": str(e)
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.document_retriever:
            self.document_retriever.cleanup()
        if self.page_retriever:
            self.page_retriever.cleanup()
        if self.llm_server:
            self.llm_server.cleanup()
        if self.generator:
            self.generator.cleanup() 