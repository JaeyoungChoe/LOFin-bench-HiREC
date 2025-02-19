from pathlib import Path
import pandas as pd
from datetime import datetime
import os
import json
from typing import List
from glob import glob
import numpy as np
import requests
import torch
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm.auto import tqdm
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification



class BaseFramework:
    framework_name = "base"

    def __init__(
        self,
        dataset_name: str,
        pdf_path: str,
        output_dir: str,
        seed: int,
        do_sample=False,
        is_numeric_question=False,
    ):
        self.dataset_name = dataset_name
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.seed = seed
        self.is_numeric_question = is_numeric_question

        dataset_path = os.path.join(
            Path(__file__).parent.parent, "data", dataset_name, "test.jsonl"
        )
        self.dataset = pd.read_json(dataset_path, lines=True)
        # TODO - 삭제
        if do_sample:
            if is_numeric_question:
                financebench_dataset = self.dataset[
                    self.dataset["qid"].str.startswith("financebench")
                ] 
                open_secqa_dataset = self.dataset[
                    self.dataset["qid"].str.startswith("openqa")
                ] 
                open_finqa_dataset = self.dataset[
                    ~(self.dataset["qid"].str.startswith("financebench"))
                    & ~(self.dataset["qid"].str.startswith("openqa"))
                ]
                sampled_open_finqa_dataset = open_finqa_dataset.sample(
                    n=100, random_state=self.seed
                ) 

                self.dataset = pd.concat(
                    [
                        financebench_dataset,
                        sampled_open_finqa_dataset,
                        open_secqa_dataset,
                    ]
                )

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(
            output_dir,
            "framework",
            self.framework_name,
            dataset_name,
            f"{self.framework_name}_{dataset_name}_{now_str}",
        )

    def save_json(self, filepath, data):
        try:
            with open(filepath, "w") as f:
                f.write(json.dumps(data, indent=4, ensure_ascii=False))
        except Exception as e:
            print(f"Error saving json file: {e}")
            with open(filepath, "w", encoding='utf-8') as f:
                f.write(json.dumps(data, indent=4, ensure_ascii=False, default=str))

    def execute(self):
        return None

class GPTEmbeddingPassageRetrieverModule:
    top_k_list = [1, 3, 5]

    def __init__(
        self,
        top_k: int = 50,
        device: int = 4,
        rerank = True,
        rerank_batch_size=64,
    ):
        super().__init__()
        
        self.top_k = top_k
        self.rerank = rerank 

        if rerank:
            if isinstance(device, str) and device.startswith("cuda"):
                self.device = device
            else:
                self.device = f"cuda:{device}"
            reranker_model_name = "naver/trecdl22-crossencoder-debertav3"
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    reranker_model_name
                ).to(self.device)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
            self.rerank_batch_size = rerank_batch_size

    def unload_model(self):
        if self.rerank:
            self.model = None
            del self.model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"DPR Reranker model unloaded")


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
                ).to(self.model.device)

                # Get logits from the model
                outputs = self.model(**batch_encoding)
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

    def query(self, query):
        openai_url = "http://127.0.0.1:8001/query/openai"
        query = {
            "query": query,
            "top_k": str(self.top_k),
        }
        response_openai = requests.post(openai_url, data=json.dumps(query), headers={"Content-Type": "application/json"})
        results = response_openai.json()["results"]
        retrieved_documents = []
        for result in results:
            document = Document(page_content=result["page_content"], metadata=result["metadata"])
            retrieved_documents.append(document)

        if self.rerank:
            retrieved_documents = self._rank_passages(query, retrieved_documents)

        return retrieved_documents

    def get_retrieved_documents(self, question) -> List[str]:
        results = self.query(question)
        return results

    def retrieve(self, query):
        def document_to_dict(document):
            page = document.metadata["page"]
            page_content = document.page_content
            source = document.metadata["source"]
            return {
                "source": source,
                "page": page,
                "page_content": page_content,
                "full_page_content": page_content,
            }
        retrieved_documents = self.get_retrieved_documents(question=query)
        retrieved_documents = retrieved_documents[: self.top_k]
        retrieved_documents = list(map(document_to_dict, retrieved_documents))

        return retrieved_documents

    def retrieve_documents(self, dataset: pd.DataFrame):
        results = {}

        for _, data in tqdm(dataset.iterrows(), total=len(dataset)):
            qid = data["qid"]
            question = data["question"]

            retrieved_documents = self.retrieve(question)

            results[qid] = retrieved_documents

        return results

    def evaluate_pairs(self, evidence_pages, retrieved_pages):
        correct = 0
        doc_correct = 0
        for evidence in evidence_pages:
            evidence_doc_name = evidence["doc_name"]
            evidence_page_num = evidence["page_num"]

            for retrieved_page in retrieved_pages:
                retrieved_page_num = retrieved_page["page"]
                retrieved_doc_name = retrieved_page["source"]
                if evidence_doc_name == retrieved_doc_name:
                    doc_correct = 1
                    if evidence_page_num == retrieved_page_num:
                        correct += 1
                        break

        recall = correct / len(evidence_pages)
        hit = 1 if correct > 0 else 0
        correct = 1 if correct == len(evidence_pages) else 0

        return {
            "correct": correct,
            "hit": hit,
            "recall": recall,
            "doc_accuracy": doc_correct,
        }

    def _evaluate(self, dataset: pd.DataFrame, results: dict):
        eval_results = {}

        for _, data in tqdm(dataset.iterrows(), total=len(dataset)):
            qid = data["qid"]
            evidences = data["evidences"]

            retrieved_passages = results[qid]
            for passage in retrieved_passages:
                passage["page"] = int(passage["page"])

            for k in self.top_k_list:
                top_k_passages = (
                    retrieved_passages[:k]
                    if len(retrieved_passages) > k
                    else retrieved_passages
                )
                score = self.evaluate_pairs(evidences, top_k_passages)
                if k not in eval_results:
                    eval_results[k] = {
                        "correct": [],
                        "recall": [],
                        "hit": [],
                        "doc_accuracy": [],
                    }
                eval_results[k]["correct"].append(score["correct"])
                eval_results[k]["recall"].append(score["recall"])
                eval_results[k]["hit"].append(score["hit"])
                eval_results[k]["doc_accuracy"].append(score["doc_accuracy"])

        scores = []
        for top_k in self.top_k_list:
            correct = np.mean(eval_results[top_k]["correct"])
            recall = np.mean(eval_results[top_k]["recall"])
            hit = np.mean(eval_results[top_k]["hit"])
            scores.append(
                {
                    "top_k": top_k,
                    "accuracy": correct,
                    "recall": recall,
                    "hit": hit,
                }
            )
        return scores

    def evaluate(self, dataset: pd.DataFrame):
        results = self.retrieve_documents(dataset)
        scores = self._evaluate(dataset, results)

        return {
            "params": {},
            "results": results,
            "scores": scores,
        }
