import argparse
from pathlib import Path
import random
from dotenv import load_dotenv

import numpy as np

import torch

from retriever.gpt_direct import GPTDirect
from retriever.hhr import HHRFramework
from retriever.hybridsearch import HybridSearch
from retriever.dense import DenseFramework
from retriever.rq_rag import RQRag
from retriever.ircot import IRCoTFramework
from retriever.perplexity import Perplexity

load_dotenv(verbose=True)

root_path = Path(__file__).parent


def set_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    dataset_name = args.dataset
    pdf_path = "/data/jaeyoung/pdfs_v2"
    output_dir = args.output_dir
    seed = args.seed
    set_seed(seed)

    framework_name = args.framework_name

    do_generate = True

    is_numeric_question = "numeric" in dataset_name

    device = args.device

    if framework_name == "hhr":
        framework = HHRFramework(
            dataset_name,
            pdf_path,
            output_dir,
            seed,
            device,
            is_numeric_question=is_numeric_question,
        )
        framework.execute()
    elif framework_name == "hybridsearch":
        framework = HybridSearch(
            dataset_name,
            pdf_path,
            output_dir,
            seed,
            device,
            is_numeric_question=is_numeric_question,
        )
        framework.execute()
    elif framework_name == "dense":
        framework = DenseFramework(
            dataset_name,
            pdf_path,
            output_dir,
            seed,
            device,
            is_numeric_question=is_numeric_question,
        )
        framework.execute()
    elif framework_name == "rq-rag":
        framework = RQRag(
            dataset_name,
            pdf_path,
            output_dir,
            seed,
            is_numeric_question=is_numeric_question,
        )
        framework.execute()
    elif framework_name == "ircot":
        framework = IRCoTFramework(
            dataset_name,
            pdf_path,
            output_dir,
            seed,
            is_numeric_question=is_numeric_question,
        )
        framework.execute()
    elif framework_name == "gpt-direct":
        framework = GPTDirect(
            dataset_name,
            pdf_path,
            output_dir,
            seed,
            is_numeric_question=is_numeric_question,
        )
        framework.execute()
    elif framework_name == "perplexity":
        framework = Perplexity(
            dataset_name,
            pdf_path,
            output_dir,
            seed,
            is_numeric_question=is_numeric_question,
        )
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
        default="ircot",
        help="Framework name for inference",
    )

    # Evaluation Data
    parser.add_argument(
        "--dataset", type=str, default="numeric_table", help="Dataset for inference"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=int, default=4, help="Device number")
    args = parser.parse_args()
    main(args)
