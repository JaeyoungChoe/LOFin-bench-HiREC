import logging
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import precision_score, recall_score
from .generator import Generator

class FinRAGEvaluator:
    def __init__(self, generator: Optional[Generator] = None):
        self.logger = logging.getLogger("FinRAG.Evaluator")
        self.generator = generator
        
    def evaluate_retrieval(self, 
                          retrieved_docs: List[str], 
                          ground_truth_docs: List[Dict[str, Any]],
                          k_values: List[int] = [5, 10]) -> Dict[str, float]:
        """
        Evaluate retrieval results
        
        Args:
            retrieved_docs: List of retrieved document IDs (e.g., ['AMT_2005_10K_93', 'AMT_2007_10K_115'])
            ground_truth_docs: List of ground truth documents (e.g., [{'page_num': 106, 'doc_name': 'AMT_2006_10K'}])
            k_values: List of k values to evaluate (e.g., [5, 10])
            
        Returns:
            Dict[str, float]: Precision and recall scores for each k
        """
        results = {}
        
        # Document ID normalization function
        def normalize_doc_id(doc_id: str) -> str:
            # Remove file extension
            doc_id = doc_id.replace('.pdf', '')
            # Normalize path separators
            doc_id = doc_id.replace('\\', '/')
            # Use only the last path component
            doc_id = doc_id.split('/')[-1]
            return doc_id.lower()
        
        # Normalize retrieved document IDs
        normalized_retrieved = [normalize_doc_id(doc) for doc in retrieved_docs]
        
        # Normalize ground truth document IDs (combine doc_name and page_num)
        normalized_ground_truth = []
        for gt_doc in ground_truth_docs:
            doc_name = normalize_doc_id(gt_doc['doc_name'])
            page_num = str(gt_doc['page_num'])
            normalized_gt = f"{doc_name}_{page_num}"
            normalized_ground_truth.append(normalized_gt)
        
        # Calculate precision and recall for each k
        for k in k_values:
            retrieved_at_k = normalized_retrieved[:k]
            
            # Precision@k
            relevant_at_k = len(set(retrieved_at_k) & set(normalized_ground_truth))
            precision = relevant_at_k / len(retrieved_at_k) if len(retrieved_at_k) > 0 else 0.0
            results[f'precision@{k}'] = precision
            
            # Recall@k
            recall = relevant_at_k / len(normalized_ground_truth) if normalized_ground_truth else 0.0
            results[f'recall@{k}'] = recall
            
        return results
    
    def evaluate_generation(self,
                          generated_answer: str,
                          ground_truth_answer: str,
                          question: str,
                          context: Optional[str] = None,
                          answer_type: str = "cot") -> Dict[str, Any]:
        """
        Evaluate generated answer (using Generator's evaluate_answer)
        
        Args:
            generated_answer: Generated answer
            ground_truth_answer: Ground truth answer
            question: Question
            context: Context (optional)
            answer_type: Answer generation method (cot, pot, direct)
            
        Returns:
            Dict[str, Any]: Evaluation results (numeric_accuracy, gpt_accuracy)
        """
        if not self.generator:
            self.logger.warning("Generator is not set. Cannot perform evaluation.")
            return {}
            
        return self.generator.evaluate_answer(
            question=question,
            answer=ground_truth_answer,
            generated=generated_answer,
            context=context or "",
            answer_type=answer_type
        )
    
    def evaluate_batch(self, 
                      results: List[Dict[str, Any]], 
                      ground_truth: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Batch evaluation
        
        Args:
            results: List of FinRAG results
            ground_truth: List of ground truth data (optional)
            
        Returns:
            Dict[str, Any]: Overall evaluation results
        """
        retrieval_metrics = []
        generation_metrics = []
        
        for idx, result in enumerate(results):
            # Retrieval evaluation
            if 'documents' in result and ground_truth:
                gt_docs = ground_truth[idx].get('evidences', [])
                retrieval_metrics.append(
                    self.evaluate_retrieval(
                        retrieved_docs=result['retrieved_pages'],
                        ground_truth_docs=gt_docs
                    )
                )
            
            # Generation evaluation
            if 'generated_answer' in result and ground_truth and self.generator:
                gt_answer = ground_truth[idx].get('answer', '')
                context = "\n".join([p['page_content'] for p in result.get('evidence', {}).get('relevant_pages', [])])
                answer_type = result.get('answer_type', 'cot')  # Get answer_type
                generation_metrics.append(
                    self.evaluate_generation(
                        generated_answer=result['generated_answer'],
                        ground_truth_answer=gt_answer,
                        question=result['query'],
                        context=context,
                        answer_type=answer_type
                    )
                )
        
        # Calculate overall averages
        final_metrics = {
            'retrieval': self._aggregate_metrics(retrieval_metrics) if retrieval_metrics else None,
            'generation': self._aggregate_metrics(generation_metrics) if generation_metrics else None
        }
        
        return final_metrics
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate metric averages"""
        if not metrics_list:
            return {}
            
        aggregated = {}
        for key in metrics_list[0].keys():
            if key == 'gpt_accuracy':  # For GPT accuracy, only average the score
                values = [m[key]['score'] for m in metrics_list if key in m]
            else:
                values = [m[key] for m in metrics_list if key in m]
            if values:  # Only calculate average if values exist
                aggregated[key] = float(np.mean(values))
            
        return aggregated 