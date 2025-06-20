import os
from dotenv import load_dotenv
load_dotenv(verbose=True)

import argparse
import json
import asyncio
import logging
from datetime import datetime
import random
import numpy as np
import torch
from tqdm import tqdm

from finrag_single_query import FinRAGSingleQuery

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set default seed
set_seed(42)

def setup_logging(debug_mode: bool, output_dir: str):
    """Setup logging configuration"""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"finrag_{timestamp}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("FinRAG")

async def main(args):
    # Convert args to dictionary
    args_dict = vars(args)

    for key in ['debug', 'use_full_page', 'do_generate', 'use_reranker', 'is_numeric_question', 'use_gpt_acc']:
        if key in args_dict:
            args_dict[key] = args_dict[key] == 'true'
    
    # Setup logging
    logger = setup_logging(args_dict["debug"], args_dict["output_dir"])
    logger.info("Starting FinRAG")
    
    # Set random seed if provided
    if args_dict["seed"] is not None:
        set_seed(args_dict["seed"])
        logger.info(f"Random seed set to: {args_dict['seed']}")
    
    # Initialize FinRAG
    finrag = FinRAGSingleQuery(args_dict)
    await finrag.initialize()
    await finrag.load_dataset(args_dict["dataset"])
    
    # Lists for storing results
    results = []
    ground_truth = []
    
    try:
        # Process queries
        if args_dict["debug"]:
            finrag.dataset = finrag.dataset.sample(n=3)
        
        total_queries = len(finrag.dataset)
        progress_bar = tqdm(total=total_queries, desc="Processing queries")
        
        while True:
            query = finrag.get_next_query()
            if query is None:
                break
                
            result = await finrag.process_single_query(query)
            results.append(result)
            
            # Store ground truth data for evaluation
            if "answer" in query or "evidences" in query:
                ground_truth.append({
                    "answer": query.get("answer"),
                    "evidences": query.get("evidences", [])
                })
            
            # Update progress
            progress_bar.update(1)
            logger.info(f"Processed queries: {len(results)}/{total_queries}")
        
        progress_bar.close()
    
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise e
    
    finally:
        # Cleanup resources
        await finrag.cleanup()
    
    # Save results
    output_path = os.path.join(args_dict["output_dir"], "retrieval_results.json")
    if args_dict["do_generate"] == 'True':
        output_path = os.path.join(args_dict["output_dir"], "retrieval_generation_results.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    
    # Perform evaluation
    if ground_truth:
        logger.info("Starting evaluation")
        evaluation_results = finrag.evaluator.evaluate_batch(results, ground_truth)
        
        # Save evaluation results
        eval_path = os.path.join(args_dict["output_dir"], "evaluation_results.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results saved to: {eval_path}")
        
        # Print evaluation results
        if evaluation_results['retrieval']:
            logger.info("Retrieval evaluation results:")
            for metric, value in evaluation_results['retrieval'].items():
                logger.info(f"  {metric}: {value:.4f}")
                
        if evaluation_results['generation']:
            logger.info("Generation evaluation results:")
            for metric, value in evaluation_results['generation'].items():
                logger.info(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Basic settings
    parser.add_argument("--dataset", type=str, required=True, help="Dataset file path (jsonl)")
    parser.add_argument("--pdf_path", type=str, required=True, help="PDF file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--db_dir", type=str, required=True, help="Vector DB directory")
    
    # Debug settings
    parser.add_argument("--debug", type=str.lower, choices=['true', 'false'], default='false', help="Enable debug mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    # Framework settings
    parser.add_argument("--max_iteration", type=int, default=4, help="Number of iterations")
    parser.add_argument("--max_contexts", type=int, default=10, help="Maximum number of contexts")
    parser.add_argument("--max_relevant_ids", type=int, default=10, help="Maximum number of relevant documents")
    parser.add_argument("--use_full_page", type=str.lower, choices=['true', 'false'], default='false', help="Use full page")
    parser.add_argument("--do_generate", type=str.lower, choices=['true', 'false'], default='false', help="Enable generation")
    
    # Retriever settings
    parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-large", help="Embedding model name")
    parser.add_argument("--cross_encoder_model_name", type=str, default="./models/DeBERTa_table_random-table_2e-7", help="Cross-encoder model name")
    parser.add_argument("--use_reranker", type=str.lower, choices=['true', 'false'], default='true', help="Use reranker")
    parser.add_argument("--reranker_model_name", type=str, default="naver/trecdl22-crossencoder-debertav3", help="Reranker model name")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens")
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="LLM model name")
    parser.add_argument("--openai_model_name", type=str, default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--gpu_devices", type=str, default="0,1,2", help="List of GPUs to use")
    parser.add_argument("--pages_per_doc", type=int, default=10, help="Number of pages per document")
    
    # Generator settings
    parser.add_argument("--temp", type=float, default=0.01, help="Generator temperature")
    parser.add_argument("--is_numeric_question", action="store_true", help="Is numeric question")
    parser.add_argument("--use_gpt_acc", action="store_true", help="Use GPT accuracy")
    parser.add_argument("--answer_type", type=str, default="cot", choices=["cot", "pot", "direct"], help="Answer generation type")
    
    args = parser.parse_args()
    asyncio.run(main(args))