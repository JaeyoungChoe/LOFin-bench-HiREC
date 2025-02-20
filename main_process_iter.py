import os
import argparse
from pathlib import Path
import random
from dotenv import load_dotenv

import numpy as np

import torch

from retriever.finrag_framework_iter import FinRAGFramework

load_dotenv(verbose=True)

root_path = Path(__file__).parent

def set_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    dataset_name = args.dataset
    pdf_path = "pdfs_v2"
    output_dir = f"HiREC/{args.output_dir}"
    seed = args.seed
    set_seed(seed)

    framework_name = args.framework_name
    numeric_dataset_list = ["numeric_table", "numeric_text"]
    if args.dataset in numeric_dataset_list:
        generate_method = 'pot'
    elif args.dataset in ["textual"]:
        generate_method = 'cot'
    else:
        generate_method = args.generate_method

    do_generate = True if args.do_generate in ['True', True] else False

    # Page Retrieval
    use_oracle_passage = False

    is_numeric_question = dataset_name in numeric_dataset_list

    device = args.device

    if framework_name in ["finrag", "finrag_dpr"]:
        framework = FinRAGFramework(
            dataset_name,
            pdf_path,
            output_dir,
            seed,
            do_generate=do_generate,
            use_oracle_passage=use_oracle_passage,
            is_numeric_question=is_numeric_question,
            generate_method=generate_method,
            device=device,
        )
        framework.use_gpt = True if args.use_gpt in ['True', True] else False
        framework.query_transformer_args['max_relevant_ids']=args.max_relevant_ids
        framework.max_iteration = args.iteration
        framework.generator_args["max_contexts"] = args.max_contexts
        framework.generator_args["use_full_page"] = True if args.use_full_page in ['True', True] else False
        framework.framework_name = framework_name
        if args.continue_iteration in ['True', True]:
            framework.continue_execute()
        else:
            framework.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--framework_name",
        type=str,
        # required=True,
        default="finrag",
        help="Framework name for evaluation",
    )
    parser.add_argument(
        "--generate_method",
        type=str,
        default="pot",
        help="generate methods. one of 'direct', 'cot', 'pot'",
    )

    # Evaluation Data
    parser.add_argument(
        "--dataset", type=str, default="numeric_text", help="Dataset for evaluation"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=int, default=0, help="Device number")
    parser.add_argument("--use_gpt", default=False, help="Use GPT4o")
    parser.add_argument("--max_relevant_ids", type=int, default=10, help="Max relevant ids")
    parser.add_argument("--iteration", type=int, default=2, help="Iteration count")
    parser.add_argument("--max_contexts", type=int, default=10, help="Max contexts")
    parser.add_argument("--use_full_page", default=False, help="Use full page")
    parser.add_argument("--do_generate", default=False, help="Do generate")
    parser.add_argument("--continue_iteration", default=False, help="Continue")
    args = parser.parse_args()
    
    main(args)
