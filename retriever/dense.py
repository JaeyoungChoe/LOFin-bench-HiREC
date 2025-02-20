import os

import numpy as np
import torch
from tqdm import tqdm
from langchain.vectorstores import Chroma
from langchain_community.embeddings.openai import OpenAIEmbeddings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

from retriever.base_framework import BaseFramework
from retriever.generator import Generator
from retriever.utils import (
    calculate_numeric_accuracy,
    calculate_gpt_accuracy_text,
    get_original_dataset_name,
)

class DenseFramework(BaseFramework):
    framework_name = "dense"

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
        self.reranker, self.tokenizer = self.get_reranker()
        self.rerank_batch_size = 64

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
        db_dir = "vectordb/dense"
        table_name = "dense"
        embedding_model_name = "text-embedding-3-small"

        # TODO - 삭제
        db_dir = "/data/jaeyoung/vectordb/gpt-embedding"
        table_name = "text_embedding_3_small-2"

        embedding_function_openai = OpenAIEmbeddings(model=embedding_model_name)
        retriever_openai = Chroma(
            persist_directory=db_dir,
            collection_name=table_name,
            embedding_function=embedding_function_openai,
        )

        return retriever_openai

    def get_reranker(self):
        reranker_model_name = "naver/trecdl22-crossencoder-debertav3"
        device = f"cuda:{self.device}"
        model = AutoModelForSequenceClassification.from_pretrained(
            reranker_model_name
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        return model, tokenizer
    
    def retrieve_passages(self, question):
        retrieved_passages = self.retriever.similarity_search(question, k=self.top_k_)

        reranked_passages = self._rank_passages(question, retrieved_passages)
        reranked_passages = reranked_passages[: self.top_k]
        reranked_passages = [self.convert_to_dict(passage) for passage in reranked_passages]
        return reranked_passages

    def _rank_passages(self, question, documents):
        scores = []
        passages = [f"Title: {document.metadata['source']}\nContent: {document.page_content}" for document in documents]

        # Split passages into batches
        batches = [
            passages[i : i + self.rerank_batch_size]
            for i in range(0, len(passages), self.rerank_batch_size)
        ]

        with torch.no_grad():
            # Loop over each batch of passages
            for batch in batches:
                inputs = [f"{question} [SEP] {passage}" for passage in batch]
                batch_encoding = self.tokenizer(
                    inputs,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.reranker.device)

                # Get logits from the model
                outputs = self.reranker(**batch_encoding)
                logits = outputs.logits.squeeze(-1)

                # Assume higher logits indicate higher relevance
                scores.extend(logits.cpu().tolist())

        # Rank documents by score
        ranks = [
            {"corpus_id": i, "score": score, "text": passages[i]}
            for i, score in enumerate(scores)
        ]
        ranks = sorted(ranks, key=lambda x: x["score"], reverse=True)

        # Retrieve documents in ranked order
        ranked_passages = [documents[rank["corpus_id"]] for rank in ranks]
        return ranked_passages

    def convert_to_dict(self, document):
        page = int(document.metadata["page"])
        page_content = document.page_content
        source = document.metadata["source"]
        return {
            "source": source,
            "page": page,
            "page_content": page_content,
            "full_page_content": page_content,
        }

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

        for idx, (_, row) in enumerate(tqdm(self.dataset.iterrows(), total=len(self.dataset))):
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