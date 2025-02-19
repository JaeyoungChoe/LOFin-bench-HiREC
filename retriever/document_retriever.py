from glob import glob
import os
import gc

import pandas as pd
from typing import List
import numpy as np
import torch
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DocumentRetrieverModule:
    top_k_list = [1, 3, 5, 10]

    def __init__(self):
        super().__init__()

    def get_retrieved_documents(self, question, new_question="") -> List[str]:
        raise NotImplementedError("Please implement this method.")

    def evaluate(self, dataset: pd.DataFrame, results: dict):
        eval_results = {}

        for _, data in tqdm(dataset.iterrows(), total=len(dataset), desc="Evaluation Document retrieval"):
            qid = data["qid"]
            evidences = data["evidences"]

            evidence_doc_names = set(map(lambda d: d["doc_name"], evidences))

            for top_k in self.top_k_list:
                retrieved_doc_names = set(results[qid][:top_k])

                correct_count = len(
                    retrieved_doc_names.intersection(evidence_doc_names)
                )

                correct = 1 if correct_count == len(evidence_doc_names) else 0
                recall = correct_count / len(evidence_doc_names)
                hit = 1 if correct_count > 0 else 0
                if len(retrieved_doc_names) == 0:
                    precision = 0
                else:
                    precision = correct_count / len(retrieved_doc_names)
                

                if top_k not in eval_results:
                    eval_results[top_k] = {
                        "correct": [],
                        "recall": [],
                        "hit": [],
                        "precision": [],
                    }

                eval_results[top_k]["correct"].append(correct)
                eval_results[top_k]["recall"].append(recall)
                eval_results[top_k]["hit"].append(hit)
                eval_results[top_k]["precision"].append(precision)

        scores = {}
        for top_k in self.top_k_list:
            correct = np.mean(eval_results[top_k]["correct"])
            recall = np.mean(eval_results[top_k]["recall"])
            hit = np.mean(eval_results[top_k]["hit"])
            precision = np.mean(eval_results[top_k]["precision"])
            scores[top_k] = {
                "accuracy": correct,
                "recall": recall,
                "hit": hit,
                "precision": precision
            }
        return scores

    def retrieve_documents(self, dataset: pd.DataFrame):
        results = {}
        top_k = max(self.top_k_list)
        for _, data in tqdm(dataset.iterrows(), total=len(dataset), desc="Document retrieval"):
            qid = data["qid"]
            question = data["question"]
            new_question = data["first_answer"]

            retrieved_documents = self.get_retrieved_documents(
                question=new_question, new_question=new_question
            )
            
            results[qid] = retrieved_documents[:top_k]

        scores = self.evaluate(dataset, results)
        return {
            "results": results,
            "scores": scores,
        }


class VectorstoreDocumentRerieverModule(DocumentRetrieverModule):
    def __init__(
        self,
        db_dir,
        document_dir,
        config,
        chunk_size=1024,
        chunk_overlap=30,
        rerank=False,
        rerank_batch_size=64,
    ):
        super().__init__()

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_dir = document_dir
        self.config = config


        self.db_dir = db_dir
        self.stopwords = set(stopwords.words("english"))
        self.rerank = rerank


        if config["only_eval"] == False:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config["doc_emb"], model_kwargs={"device": "cuda:0"}
            )
            self.retriever = self.get_vectorstore_retriever()
            if self.rerank is True:
                self.device = self.config["device"]
                reranker_model_name = "naver/trecdl22-crossencoder-debertav3"
                self.reranker = self.model = (
                    AutoModelForSequenceClassification.from_pretrained(
                        reranker_model_name
                    ).to(self.device)
                )
                self.tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
                self.rerank_batch_size = rerank_batch_size

    def unload_model(self):
        self.embeddings=None
        del self.embeddings
        
        self.reranker=None
        del self.reranker
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Document retrieval model unloaded")

    def process_doc(self, doc):
        pdf_loader = PyMuPDFLoader(doc)
        data = pdf_loader.load()[0]
        text = data.page_content
        text = text.lower().replace("\n", " ")
        words = word_tokenize(text)
        filtered_words = " ".join(
            [word for word in words if word.lower() not in self.stopwords]
        )
        data.page_content = filtered_words
        return data

    def make_documents_obj(self, chunk_ver):
        if chunk_ver == "document_selection":
            docs = glob(f"{self.document_dir}/*/*/*.pdf")
            documents = []

            for doc in tqdm(docs, desc="Making documents first pages"):
                try:
                    data = self.process_doc(doc, chunk_ver)
                    if data is not None:
                        documents.append(data)
                except Exception as e:
                    print(f"Error processing {doc}: {e}")

            return documents
        elif chunk_ver == "summary_document_selection":
            doc = pd.read_json("preprocessing/summarization_results.jsonl", lines=True)
            documents = []
            for idx, row in doc.iterrows():
                doc_name = row["doc_name"]
                content = row["summary"]
                document = Document(page_content=content, metadata={"source": doc_name})
                documents.append(document)
            return documents

    def build_vectorstore_retriever(self, documents):
        vectordb = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "ip"},
        )
        vectordb.persist()

        for doc in tqdm(documents, desc="Making documents first pages"):
            vectordb.add_documents(documents=[doc])
            vectordb.persist()

        return vectordb

    def get_vectorstore_retriever(self):
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir, exist_ok=True)
            documents = self.make_documents_obj(chunk_ver=self.config["chunk_ver"])
            retriever = self.build_vectorstore_retriever(documents)
        else:
            retriever = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "ip"},
            )

        return retriever

    def rerank_documents(
        self, question, retrieved_documents: List[Document]
    ) -> List[Document]:
        passages = [doc.page_content for doc in retrieved_documents]
        scores = []

        batches = [
            passages[i : i + self.rerank_batch_size]
            for i in range(0, len(passages), self.rerank_batch_size)
        ]

        with torch.no_grad():
            for batch in batches:
                inputs = [f"query: {question} [SEP] {passage}" for passage in batch]

                # Tokenize the batch
                batch_encoding = self.tokenizer(
                    inputs,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                # Get logits from the model
                outputs = self.reranker(**batch_encoding)
                logits = outputs.logits.squeeze(-1)

                # Append scores
                scores.extend(logits.cpu().tolist())

        # Create a list of (Document, score) tuples
        scored_documents = [
            (doc, score) for doc, score in zip(retrieved_documents, scores)
        ]

        # Sort documents by score in descending order
        scored_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)

        # Extract the ranked documents
        ranked_documents = [doc for doc, _ in scored_documents]

        return ranked_documents

    def get_retrieved_documents(self, question, new_question) -> List[str]:
        retrieved = self.retriever.similarity_search(question, k=100)
        if self.rerank:
            retrieved = self.rerank_documents(new_question, retrieved)
        doc_names = list(
            map(
                lambda x: x.metadata["source"].split("/")[-1].replace(".pdf", ""),
                retrieved,
            )
        )
        return doc_names
