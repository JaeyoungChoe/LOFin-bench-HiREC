import os
import gc
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import openparse
from langchain.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import nltk
import spacy


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
random_seed(42)

class PageRetrieverModule:
    top_k_list = [1, 3, 5, 10]
    default_chunk_ver = "base"
    default_chunk_size = 1024
    default_chunk_overlap = 30

    def __init__(self, document_dir, use_oracle_passage=False, **kwargs):
        super().__init__()

        self.use_oracle_passage = use_oracle_passage

        self.kwargs = kwargs

        if self.kwargs.get("chunk_ver") is None:
            self.kwargs["chunk_ver"] = self.default_chunk_ver
        chunk_ver = self.kwargs["chunk_ver"]
        if chunk_ver == self.default_chunk_ver:
            if self.kwargs.get("chunk_size") is None:
                self.kwargs["chunk_size"] = self.default_chunk_size
            if self.kwargs.get("chunk_overlap") is None:
                self.kwargs["chunk_overlap"] = self.default_chunk_overlap

        if chunk_ver == "nltk":
            nltk.download("punkt")
        if chunk_ver == "spacy":
            self.nlp = spacy.load("en_core_web_sm")

        self.passage_type = self.kwargs.get("passage_type")

        self.document_dir = document_dir

        self.params = self.kwargs

    def unload_model(self):
        if self.model:
            self.model = None
            del self.model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Passage selection model unloaded")

    def get_pdf_path(self, doc_name):
        ticker, period, report_type = doc_name.split("_")
        if report_type == "10K":
            doc_type = "10-K"
        elif report_type == "10Q":
            doc_type = "10-Q"
        elif report_type == "8K":
            doc_type = "8-K"
        else:
            doc_type = report_type

        pdf_path = os.path.join(self.document_dir, ticker, doc_type, f"{doc_name}.pdf")
        return pdf_path

    def get_table_meta_path(self, doc_name):
        ticker, period, report_type = doc_name.split("_")
        if report_type == "10K":
            doc_type = "10-K"
        elif report_type == "10Q":
            doc_type = "10-Q"
        elif report_type == "8K":
            doc_type = "8-K"
        else:
            doc_type = report_type

        pdf_path = os.path.join(
            self.document_dir, ticker, doc_type, f"{doc_name}_table.jsonl"
        )
        return pdf_path

    def split_text(self, documents: List[Document]):
        page_contents = {doc.metadata["page"]: doc.page_content for doc in documents}
        splited_documents = None
        chunk_ver = self.kwargs.get("chunk_ver")
        if chunk_ver == "base":
            chunk_size = self.kwargs.get("chunk_size")
            chunk_overlap = self.kwargs.get("chunk_overlap")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            )
            splited_documents = text_splitter.split_documents(documents)
        elif chunk_ver == "nltk":
            splited_documents = []
            for doc in documents:
                metadata = doc.metadata
                content = doc.page_content

                sentences = nltk.sent_tokenize(content)
                for idx, sentence in enumerate(sentences):
                    new_metadata = {
                        **metadata,
                        "sentence_idx": idx,
                    }
                    new_doc = Document(sentence, metadata=new_metadata)
                    splited_documents.append(new_doc)
        elif chunk_ver == "spacy":
            splited_documents = []
            for doc in documents:
                metadata = doc.metadata
                content = doc.page_content

                doc = self.nlp(content)
                sentences = [sent.text for sent in doc.sents]
                for idx, sentence in enumerate(sentences):
                    new_metadata = {
                        **metadata,
                        "sentence_idx": idx,
                    }
                    new_doc = Document(sentence, metadata=new_metadata)
                    splited_documents.append(new_doc)

        if splited_documents is None:
            raise NotImplementedError("Please implement this method")

        for document in splited_documents:
            cur_page = document.metadata["page"]
            document.metadata["full_page_text"] = page_contents[cur_page]

        return splited_documents

    def load_documents(self, doc_names):
        documents = []
        for doc_name in doc_names:
            documents_ = self.load_document(doc_name)
            documents.extend(documents_)
        return documents

    def load_document(self, doc_name):
        pdf_path = self.get_pdf_path(doc_name)
        if not os.path.exists(pdf_path):
            raise AssertionError(f"Document {pdf_path} not exists.")

        chunk_ver = self.kwargs.get("chunk_ver")
        if chunk_ver in ["base", "nltk", "spacy"]:
            pdf_reader = PyMuPDFLoader(pdf_path)
            documents = pdf_reader.load()
            documents = self.split_text(documents)
        elif chunk_ver == "open_parse":
            parser = openparse.DocumentParser()
            docs = parser.parse(pdf_path)
            documents = []
            for doc in docs.nodes:
                content = doc.text
                if len(content.strip()) == 0:
                    continue
                page = doc.start_page
                doc_name = pdf_path.split("/")[-1]
                metadata = {
                    "page": page,
                    "source": doc_name,
                }
                document = Document(content, metadata=metadata)
                documents.append(document)
        else:
            raise NotImplementedError("Please implement this method")

        if self.passage_type == "base+table":
            table_meta_path = self.get_table_meta_path(doc_name)
            if os.path.exists(table_meta_path):
                tables_df = pd.read_json(table_meta_path, lines=True)
                for _, table_row in tables_df.iterrows():
                    table_text = table_row["text"]
                    table_page = table_row["page"]
                    if len(table_text) > 16:
                        metadata = {
                            "page": table_page,
                            "source": doc_name,
                        }
                        document = Document(table_text, metadata=metadata)
                        documents.append(document)

        return documents

    def _rank_passages(self, question, documents):
        raise NotImplementedError("Please implement this method")

    def rank_passages(self, question, documents, top_k):
        retrieve_strategy = self.kwargs["retrieve_strategy"]

        ranked_passages = self._rank_passages(question, documents)

        if retrieve_strategy == "page":
            retrieved_pages = []
            retrieved_page_nums = set()
            for document in ranked_passages:
                page = document.metadata["page"]
                if page in retrieved_page_nums:
                    continue

                retrieved_pages.append(document)
                retrieved_page_nums.add(page)
                if len(retrieved_pages) == top_k:
                    break
            return retrieved_pages
        elif retrieve_strategy == "passage":
            return ranked_passages[:top_k]
        raise AssertionError(f"Not supported retrieve_strategy {retrieve_strategy}")

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
        if len(retrieved_pages) == 0:
            precision = 0
        else:
            precision = correct / len(retrieved_pages)
        accuracy = 1 if correct == len(evidence_pages) else 0

        return {
            "correct": accuracy,
            "len_k": len(retrieved_pages),
            "hit": hit,
            "recall": recall,
            "doc_accuracy": doc_correct,
            "precision": precision,
        }

    def evaluate(self, dataset: pd.DataFrame, results: dict):
        eval_results = {}
        qid_scores = {}
        for _, data in tqdm(
            dataset.iterrows(), total=len(dataset), desc="Evaluation Passage retrieval"
        ):
            qid = data["qid"]
            evidences = data["evidences"]

            retrieved_passages = results[qid]

            for k in self.top_k_list:
                top_k_passages = (
                    retrieved_passages[:k]
                    if len(retrieved_passages) > k
                    else retrieved_passages
                )
                score = self.evaluate_pairs(evidences, top_k_passages)
                qid_scores[qid] = score
                if k not in eval_results:
                    eval_results[k] = {
                        "correct": [],
                        "recall": [],
                        "hit": [],
                        "doc_accuracy": [],
                        "precision": [],
                        "len_k": [],
                    }
                eval_results[k]["correct"].append(score["correct"])
                eval_results[k]["recall"].append(score["recall"])
                eval_results[k]["hit"].append(score["hit"])
                eval_results[k]["doc_accuracy"].append(score["doc_accuracy"])
                eval_results[k]["precision"].append(score["precision"])
                eval_results[k]["len_k"].append(score["len_k"])

        scores = []
        for top_k in self.top_k_list:
            correct = np.mean(eval_results[top_k]["correct"])
            recall = np.mean(eval_results[top_k]["recall"])
            hit = np.mean(eval_results[top_k]["hit"])
            precision = np.mean(eval_results[top_k]["precision"])
            len_k = np.sum(eval_results[top_k]["len_k"])
            scores.append(
                {
                    "top_k": top_k,
                    "accuracy": correct,
                    "recall": recall,
                    "hit": hit,
                    "precision": precision,
                    "len_k": len_k,
                }
            )
        return scores

    def retrieve_pages(
        self, dataset: pd.DataFrame, retrieved_documents: Optional[dict] = None
    ):
        def document_to_dict(document: Document):
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

        results = {}

        for _, data in tqdm(
            dataset.iterrows(), total=len(dataset), desc="Passage selection"
        ):
            qid = data["qid"]
            question = data["question"]

            if retrieved_documents is not None:
                doc_name = retrieved_documents[qid][:5]
            else:
                doc_name = list(set(map(lambda x: x["doc_name"], data["evidences"])))

            documents = self.load_documents(doc_name)

            if self.use_oracle_passage is False:
                top_k = max(self.top_k_list)
                top_passages = self.rank_passages(question, documents, top_k)
            else:
                evidence_infos = []
                for evidence in data["evidences"]:
                    evidence_infos.append((evidence["doc_name"], evidence["page_num"]))

                top_passages = []
                for passage in documents:
                    page_num = passage.metadata["page"]
                    doc_name = os.path.basename(passage.metadata["source"]).replace(
                        ".pdf", ""
                    )
                    if (doc_name, page_num) in evidence_infos:
                        top_passages.append(passage)

            top_passages = list(map(document_to_dict, top_passages))
            results[qid] = top_passages

        scores = self.evaluate(dataset, results)

        return {
            "params": self.params,
            "results": results,
            "scores": scores,
        }

class BaseCrossEncoderPageRetrieverModule(PageRetrieverModule):
    def __init__(self, model_name, document_dir, **kwargs):
        super().__init__(document_dir=document_dir, **kwargs)

        self.params["model_name"] = model_name

        self.model_name = model_name
        self.model = CrossEncoder(
            model_name,
            max_length=512,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def _rank_passages(self, question, documents):
        passages = [document.page_content for document in documents]
        ranks = self.model.rank(
            query=question, documents=passages, top_k=None, return_documents=True
        )
        ranked_passages = [documents[rank["corpus_id"]] for rank in ranks]
        return ranked_passages


class AutoModelForSequenceClassificationPageRetrieverModule(PageRetrieverModule):
    def __init__(self, model_name, document_dir, use_oracle_passage=False, **kwargs):
        super().__init__(
            document_dir=document_dir, use_oracle_passage=use_oracle_passage, **kwargs
        )
        self.model_name = model_name
        self.device = kwargs.get("device")

        if kwargs.get("only_eval") is False:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
                self.device
            )
        if self.kwargs["page_retriever"] == "deberta":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "naver/trecdl22-crossencoder-debertav3"
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = kwargs.get("batch_size")
        if self.batch_size is None:
            self.batch_size = 64

        self.params["model_name"] = model_name
        self.params["batch_size"] = self.batch_size

    def _rank_passages(self, question, documents):
        scores = []
        passages = ["Document title: "+' '.join(os.path.basename(document.metadata['source']).replace('.pdf','').split("_")) + ". Context: " + document.page_content for document in documents]
        
        # Split passages into batches
        batches = [
            passages[i : i + self.batch_size]
            for i in range(0, len(passages), self.batch_size)
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
