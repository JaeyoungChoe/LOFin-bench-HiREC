import os

import lancedb
from lancedb.pydantic import LanceModel
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
import numpy as np

from retriever.base_framework import BaseFramework
from retriever.generator import Generator
from retriever.utils import (
    calculate_numeric_accuracy,
    calculate_gpt_accuracy_text,
    get_original_dataset_name,
)

def _define_schema():
    class TextSchema(LanceModel):
        passage_id: str
        title: str
        ticker: str
        report_type: str
        year: int
        page: int
        start_index: int
        content: str
    return TextSchema

class HHRFramework(BaseFramework):
    framework_name = "hhr"

    top_k = 10
    top_k_d = 5

    generator_args = {
        "openai_model": "gpt-4o",
        "temperature": 0.01,
        "max_tokens": 64536,
        "max_contexts": 10,
        "use_full_page": False,
    }

    def __init__(self, dataset_name, pdf_path, output_dir, seed, device, is_numeric_question=False, model_name="gpt-4o"):
        super().__init__(dataset_name, pdf_path, output_dir, seed, is_numeric_question=is_numeric_question)

        self.device = device

        self.openai_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=self.openai_key)
        self.model_name = model_name

        self.sparse_document_retriever = self.get_sparse_document_retriever()
        self.dense_document_retriever = self.get_dense_document_retriever()

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

    def get_sparse_document_retriever(self):
        lance_db_dir = "vectordb/lancedb/.lancedb"
        os.makedirs(lance_db_dir, exist_ok=True)
        table_name = "hhr_documents"

        db = lancedb.connect(lance_db_dir)
        retriever = db.open_table(table_name)

        return retriever
    
    def get_dense_document_retriever(self):
        db_dir = "vectordb/hhr"
        table_name = "hhr_documents"
        model_name = "intfloat/multilingual-e5-large"

        embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": f"cuda:{self.device}"}
        )

        retriever = Chroma(
            persist_directory=db_dir,
            collection_name=table_name,
            embedding_function=embedding_function,
        )

        return retriever

    def retrieve_documents(self, question):
        sparse_docs = self.retrieve_documents_sparse(question)
        dense_docs = self.retrieve_documents_dense(question)

        hybrid_retrieved = []
        selected_doc_ids = []

        turn = "dense"
        while len(hybrid_retrieved) < self.top_k_d:
            if turn == "dense":
                item = dense_docs.pop(0)
                doc_id = item["doc_id"]
                if doc_id in selected_doc_ids:
                    continue
                hybrid_retrieved.append(item)
                selected_doc_ids.append(doc_id)
                turn = "sparse"
            elif turn == "sparse":
                item = sparse_docs.pop(0)
                doc_id = item["doc_id"]
                if doc_id in selected_doc_ids:
                    continue
                hybrid_retrieved.append(item)
                selected_doc_ids.append(doc_id)
                turn = "dense"

        return hybrid_retrieved
    
    def _rename_score_column(self, df: pd.DataFrame, new_name: str) -> bool:
        if  (new_name in df.columns) or (len(df) == 0): # column already exists
            return False
        
        if "_score" in df.columns:
            df.rename(columns={"_score": new_name}, inplace=True)
        elif "_relevance_score" in df.columns:
            df.rename(columns={"_relevance_score": new_name}, inplace=True)
        elif "_distance" in df.columns:
            df.rename(columns={"_distance": new_name}, inplace=True)
        else:
            raise ValueError(f"No score column found in DataFrame columns: {df.columns}")
        return True

    def convert_lance_document_to_dict(self, x):
        doc_id = x["doc_id"]
        metadata = {
            "title": x["title"],
            "file_path": x["file_path"],
            "ticker": x["ticker"],
            "report_type": x["report_type"],
            "year": x["year"],
            "score": x["score"],
        }
        summary = x["summary"]
        result = {
            "doc_id": doc_id,
            "metadata": metadata,
            "content": summary,
        }
        return result

    def retrieve_documents_sparse(self, question):
        retriever = self.sparse_document_retriever
        retrieved_docs = retriever \
            .search(query=question, query_type="fts") \
            .limit(self.top_k_d * 4) \
            .to_pandas()
        
        retrieved_docs.dropna(inplace=True)
        _ = self._rename_score_column(retrieved_docs, "score")
        retrieved_docs = retrieved_docs.sort_values(by='score', ascending=False).drop_duplicates(subset=['doc_id'], keep='first').reset_index(drop=True)
        retrieved_docs_ = [self.convert_lance_document_to_dict(x) for (_, x) in retrieved_docs.iterrows()]

        return retrieved_docs_
    
    def convert_document_to_dict(self, x):
        metadata = x.metadata
        summary = x.page_content[len("passage: "):]
        doc_id = metadata["title"]
        result = {
            "doc_id": doc_id,
            "metadata": metadata,
            "content": summary,
        }
        return result
    
    def retrieve_documents_dense(self, question):
        retriever = self.dense_document_retriever

        retrieved_docs = retriever.similarity_search(query=question, k=self.top_k_d * 4)
        retrieved_docs = [self.convert_document_to_dict(doc) for doc in retrieved_docs]
        
        return retrieved_docs

    def retrieve_passages(self, question):
        documents = self.retrieve_documents(question)

        retrieved_passages_sparse = self.retrieve_passages_sparse(question, documents)
        retrieved_passages_dense = self.retrieve_passages_dense(question, documents)

        hybrid_retrieved = []
        selected_passage_ids = []

        turn = "dense"
        while len(hybrid_retrieved) < self.top_k:
            if turn == "dense":
                item = retrieved_passages_dense.pop(0)
                passage_id = item["passage_id"]
                if passage_id in selected_passage_ids:
                    continue
                hybrid_retrieved.append(item)
                selected_passage_ids.append(passage_id)
                turn = "sparse"
            elif turn == "sparse":
                item = retrieved_passages_sparse.pop(0)
                passage_id = item["passage_id"]
                if passage_id in selected_passage_ids:
                    continue
                hybrid_retrieved.append(item)
                selected_passage_ids.append(passage_id)
                turn = "dense"
        
        return hybrid_retrieved

    def convert_lance_passage_to_dict(self, x):
        passage_id = x["passage_id"]
        metadata = {
            "passage_id": x["passage_id"],
            "title": x["title"],
            "ticker": x["ticker"],
            "report_type": x["report_type"],
            "year": x["year"],
            "page": x["page"],
            "start_index": int(x["start_index"]),
            "score": x["score"],
        }
        content = x["content"]
        content = content[content.find("Content : ") + len("Content : "):]
        source = metadata["title"]
        page = metadata["page"]
        start_index = metadata["start_index"]

        result = {
            "passage_id": passage_id,
            "source": source,
            "page": page,
            "start_index": start_index,
            "page_content": content,
            "metadata": metadata,
        }   

        return result
    
    def retrieve_passages_sparse(self, question, documents):
        lance_db_dir = "vectordb/lancedb/.lancedb"
        table_name = "hhr_passages"

        db = lancedb.connect(lance_db_dir)
        table_names = db.table_names()
        if table_name in table_names:
            db.drop_table(table_name)
        
        schema = _define_schema()
        retriever = db.create_table(table_name, schema=schema, on_bad_vectors="drop")

        passages = []

        for doc in documents:
            title = doc["metadata"]["title"]
            metadata = doc["metadata"]
            documents_ = self.load_document(title)
            
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1024,
                    chunk_overlap=30,
                    add_start_index=True,
                )
                splitted_documents = text_splitter.split_documents(documents_)
                for splitted_document in splitted_documents:
                    page = splitted_document.metadata["page"]
                    chunk_start_idx = splitted_document.metadata["start_index"]
                    passage_id = f"{title}__{page}__{chunk_start_idx}"
                    document = {
                        "passage_id": passage_id,
                        "title": title,
                        "ticker": metadata["ticker"],
                        "report_type": metadata["report_type"],
                        "year": metadata["year"],
                        "page": page,
                        "start_index": chunk_start_idx,
                        "content": f'Document Title : {title}\nContent : {splitted_document.page_content}',
                    }
                    passages.append(document)
            except Exception as e:
                print('Exception : ' + str(e))
            
        retriever.add(passages)
        retriever.create_fts_index('content', replace=True, use_tantivy=False)

        retrieved_passages = retriever \
            .search(query=question, query_type="fts") \
            .limit(self.top_k * 2) \
            .to_pandas()

        retrieved_passages.dropna(inplace=True)
        _ = self._rename_score_column(retrieved_passages, "score")
        retrieved_passages = retrieved_passages.sort_values(by='score', ascending=False).drop_duplicates(subset=['passage_id'], keep='first').reset_index(drop=True)
        retrieved_passages = [self.convert_lance_passage_to_dict(passage) for _, passage in retrieved_passages.iterrows()]

        db.drop_table(table_name)
        return retrieved_passages

    def convert_passage_to_dict(self, x):
        metadata = x.metadata
        content = x.page_content
        content = content[content.find("Content : ") + len("Content : "):]
        passage_id = metadata["passage_id"]
        source = metadata["title"]
        page = metadata["page"]
        start_index = metadata["start_index"]

        result = {
            "passage_id": passage_id,
            "source": source,
            "page": page,
            "start_index": start_index,
            "page_content": content,
            "metadata": metadata,
        }   
        return result

    def retrieve_passages_dense(self, question, retrieved_documents): 
        def convert_to_document(passages):
            ids = []
            documents = []

            for passage in passages:
                ids.append(passage["passage_id"])
                page_content = passage["content"]
                metadata = passage.copy()
                del metadata["content"]
                document = Document(page_content=page_content, metadata=metadata)
                documents.append(document)
            
            return ids, documents

        model_name = "intfloat/multilingual-e5-large"
        embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": f"cuda:{self.device}"}
        )
        retriever = Chroma(
            embedding_function=embedding_function
        )

        passages = []

        for doc in retrieved_documents:
            title = doc["metadata"]["title"]
            metadata = doc["metadata"].copy()
            if "score" in metadata:
                del metadata["score"]
            
            documents = self.load_document(title)
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1024,
                    chunk_overlap=30,
                    add_start_index=True,
                )
                splitted_documents = text_splitter.split_documents(documents)
                for splitted_document in splitted_documents:
                    page = splitted_document.metadata["page"]
                    chunk_start_idx = int(splitted_document.metadata["start_index"])
                    passage_id = f"{title}__{page}__{chunk_start_idx}"
                    document = {
                        "passage_id": passage_id,
                        "title": title,
                        "ticker": metadata["ticker"],
                        "report_type": metadata["report_type"],
                        "year": metadata["year"],
                        "page": page,
                        "start_index": chunk_start_idx,
                        "content": f'Document Title : {title}\nContent : {splitted_document.page_content}',
                    }
                    passages.append(document)
            except Exception as e:
                print('Exception : ' + str(e))

        passage_ids, passages_ = convert_to_document(passages)
        retriever.add_documents(documents=passages_, ids=passage_ids)

        retrieved_passages = retriever.similarity_search(query=question, k=self.top_k * 2)
        retrieved_passages = [self.convert_passage_to_dict(x) for x in retrieved_passages]

        return retrieved_passages

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


