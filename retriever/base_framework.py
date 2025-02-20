from datetime import datetime
import json
import os


from langchain.document_loaders import PyMuPDFLoader
import pandas as pd
from pathlib import Path


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
        self.dataset = self.dataset.sample(n=1, random_state=self.seed)
        
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

        pdf_path = os.path.join(self.pdf_path, ticker, doc_type, f"{doc_name}.pdf")
        return pdf_path

    def load_document(self, doc_name):
        pdf_path = self.get_pdf_path(doc_name)
        if not os.path.exists(pdf_path):
            raise AssertionError(f"Document {pdf_path} not exists.")

        pdf_reader = PyMuPDFLoader(pdf_path)
        documents = pdf_reader.load()
        return documents

    def load_documents(self, doc_names):
        documents = []
        for doc_name in doc_names:
            documents_ = self.load_document(doc_name)
            documents.extend(documents_)
        return documents

    def load_contexts(self, evidences):
        doc_names = list(set(map(lambda x: x["doc_name"], evidences)))
        documents = self.load_documents(doc_names)

        evidence_infos = []
        for evidence in evidences:
            evidence_infos.append((evidence["doc_name"], evidence["page_num"]))

        contexts = []
        for passage in documents:
            page_num = passage.metadata["page"]
            doc_name = os.path.basename(passage.metadata["source"]).replace(".pdf", "")
            if (doc_name, page_num) in evidence_infos:
                contexts.append(passage)

        return contexts

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

    def execute(self):
        return None
