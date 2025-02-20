import os

from lancedb.embeddings import get_registry
from lancedb.rerankers import ColbertReranker
import lancedb
import pandas as pd
from tqdm import tqdm
import numpy as np
from openai import OpenAI

from retriever.base_framework import BaseFramework
from retriever.generator import Generator
from retriever.utils import (
    calculate_numeric_accuracy,
    calculate_gpt_accuracy_text,
    get_original_dataset_name,
)

openai_embedder = get_registry().get("openai").create(name="text-embedding-3-small")

class HybridSearch(BaseFramework):
    framework_name = "hybridsearch"

    top_k_ = 50
    top_k = 10

    generator_args = {
        "openai_model": "gpt-4o",
        "temperature": 0.01,
        "max_tokens": 64536,
        "max_contexts": 10,
        "use_full_page": False,
    }

    def __init__(self, dataset_name, pdf_path, output_dir, seed, device, is_numeric_question=False):
        super().__init__(dataset_name, pdf_path, output_dir, seed, is_numeric_question=is_numeric_question)

        self.device = device

        self.retriever = self.get_retriever()
        self.reranker = ColbertReranker(device=f"cuda:{self.device}")

        self.openai_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=self.openai_key)
        
        generator_openai_model_name = self.generator_args["openai_model"]
        generator_temperature = self.generator_args["temperature"]
        generator_max_tokens = self.generator_args["max_tokens"]
        generator_max_context_count = self.generator_args["max_contexts"]
        generator_use_full_page = self.generator_args["use_full_page"]
        generate_method = "pot" if self.is_numeric_question else "cot"
        self.generator = Generator(
            pdf_path,
            generator_openai_model_name,
            generator_temperature,
            generator_max_tokens,
            generator_max_context_count,
            use_full_page=generator_use_full_page,
            is_numeric_question=is_numeric_question,
            generate_method=generate_method,
        )

        os.makedirs(self.output_dir, exist_ok=True)

    def get_retriever(self):
        lance_db_dir = "vectordb/lancedb/.lancedb"
        table_name = "hybrid_search"
        os.makedirs(lance_db_dir, exist_ok=True)
        
        db = lancedb.connect(lance_db_dir)
        hybrid_retriever = db.open_table(table_name)

        return hybrid_retriever

    def _rename_score_column(self, df: pd.DataFrame, new_name: str) -> bool:
        if  (new_name in df.columns) or (len(df) == 0): # column already exists
            return False
        
        if "_score" in df.columns:
            df.rename(columns={"_score": new_name}, inplace=True)
        elif "_relevance_score" in df.columns:
            df.rename(columns={"_relevance_score": new_name}, inplace=True)
        else:
            raise ValueError(f"No score column found in DataFrame columns: {df.columns}")
        return True
    
    def convert_to_dict(self, item):
        _, page, _ = item["passage_id"].split("__")
        return {
            "passage_id": item["passage_id"],
            "source": item["title"],
            "page": int(page),
            "page_content": item["text"],
        }
    
    def retrieve_passages(self, question):
        retriver = self.retriever
        reranker = self.reranker

        retrieved_docs = retriver \
            .search(query=question, query_type="hybrid") \
            .rerank(reranker=reranker) \
            .limit(self.top_k_) \
            .to_pandas()

        retrieved_docs.dropna(inplace=True)
        _ = self._rename_score_column(retrieved_docs, "score")
        retrieved_docs.rename(columns={"doc_id": "passage_id"}, inplace=True)

        retrieved_docs = retrieved_docs.sort_values(by='score', ascending=False).drop_duplicates(subset=['passage_id'], keep='first').reset_index(drop=True)
        retrieved_docs = [self.convert_to_dict(doc) for _, doc in retrieved_docs.iterrows()]
        retrieved_docs = retrieved_docs[:self.top_k]
        return retrieved_docs

    def execute(self):
        results = {}
        eval_results = {}
        scores = {
            metric: {
                "financebench": [],
                "open_secqa": [],
                "open_finqa": [],
                "total": [],
            }
            for metric in ["acc" if self.is_numeric_question else "gpt_acc", "recall", "precision", "k"]
        }

        for idx, (_, row) in enumerate(
            tqdm(self.dataset.iterrows(), total=len(self.dataset))
        ):
            qid = row["qid"]
            original_dataset = get_original_dataset_name(qid)
            question = row["question"]
            answer = str(row["answer"])
            evidences = row["evidences"]

            retrieved_passages = self.retrieve_passages(question)
            scores["k"][original_dataset].append(len(retrieved_passages))
            scores["k"]["total"].append(len(retrieved_passages))
            retrieval_scores = self.evaluate_pairs(evidences, retrieved_passages)
            recall = retrieval_scores["recall"]
            precision = retrieval_scores["precision"]
            scores["precision"][original_dataset].append(precision)
            scores["precision"]["total"].append(precision)
            scores["recall"][original_dataset].append(recall)
            scores["recall"]["total"].append(recall)

            generated_answer, generated_results = self.generator.generate_answer(question, retrieved_passages)
            contexts = self.load_contexts(evidences=evidences)
            contexts_ = [f"{passage.page_content}" for passage in contexts]
            contexts_ = "\n".join(contexts_)
            
            results[qid] = {
                "question": question,
                "answer": answer,
                "evidences": evidences,
                "generated": generated_answer,
                "original_dataset": original_dataset,
                "generated_results": generated_results,
                "retrieved_passages": retrieved_passages,
            }
            if self.is_numeric_question:
                acc = calculate_numeric_accuracy(answer, generated_answer)
                scores["acc"][original_dataset].append(acc)
                scores["acc"]["total"].append(acc)
                eval_results[qid] = {
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "extracted_answer": generated_answer,
                    "original_dataset": original_dataset,
                    "acc": acc,
                }
            else:
                contexts = self.load_contexts(evidences=evidences)
                contexts_ = [f"{passage.page_content}" for passage in contexts]
                contexts_ = "\n".join(contexts_)
                gpt_eval_results = calculate_gpt_accuracy_text(self.client, question, answer, generated_answer, contexts_)
                acc = gpt_eval_results["score"]
                scores["gpt_acc"][original_dataset].append(acc)
                scores["gpt_acc"]["total"].append(acc)
                results[qid]["gpt_eval_results"] = gpt_eval_results
                eval_results[qid] = {
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "extracted_answer": generated_answer,
                    "original_dataset": original_dataset,
                    "gpt_acc": acc,
                }
            
        results_path = os.path.join(self.output_dir, "results.json")
        self.save_json(results_path, results)

        eval_results_path = os.path.join(self.output_dir, "eval_results.json")
        self.save_json(eval_results_path, eval_results)

        final_scores = {"count": {}}
        for metric_key in scores.keys():
            final_scores[metric_key] = {}
            for dataset_name in scores[metric_key].keys():
                data_count = len(scores[metric_key][dataset_name])
                if dataset_name not in final_scores["count"]:
                    final_scores["count"][dataset_name] = data_count

                if data_count == 0:
                    final_scores[metric_key][dataset_name] = None
                else:
                    final_scores[metric_key][dataset_name] = np.mean(scores[metric_key][dataset_name])
        
        scores_path = os.path.join(self.output_dir, "scores.json")
        self.save_json(scores_path, final_scores)