import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.nn import DataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.document_loaders import PyMuPDFLoader

device = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"

class SummarizationModel:
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.tokenizer = self.load_tokenizer(model_name)

    def load_model(self, model_name):
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def generate_summary(self, text):
        messages = self.create_messages(text)
        model_inputs = self.prepare_inputs(messages)
        generated_ids = self.generate_text(model_inputs)
        return self.decode_response(generated_ids, model_inputs)

    def create_messages(self, text):
        prompt = "Summarize the following text:"
        answer_template = (
            "answer: "
        )
        return [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}\n\n{text}\n\n{answer_template}"},
        ]
    def prepare_inputs(self, messages):
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer([chat_text], return_tensors="pt").to(self.model.device)

    def generate_text(self, model_inputs):
        return self.model.generate(
            **model_inputs,
            max_new_tokens=256,
        )

    def decode_response(self, generated_ids, model_inputs):
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def extract_text_from_pdf(pdf_path):
    reader = PyMuPDFLoader(pdf_path)
    documents = reader.load()    
    return documents

def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    summarization_model = SummarizationModel(model_name)
    
    pdfs_path = "pdfs"
    pdfs = glob(f"{pdfs_path}/*/*/*.pdf")
    pdfs = np.array_split(pdfs, 2)[1]

    summarization_results = []
    blank_pdfs = []
    for pdf_path in tqdm(pdfs, desc="Summarizing PDFs"):
        try:
            documents = extract_text_from_pdf(pdf_path)
            first_page = documents[0].page_content
            if len(first_page) < 100:
                first_page = documents[1].page_content

            summary = summarization_model.generate_summary(first_page)
            doc_name = os.path.basename(pdf_path)
            summarization_results.append({
                "doc_name": doc_name,
                "summary": summary
            })
        except Exception as e:
            print(f"Error occurred while summarizing {pdf_path}: {e}")
    
    pd.DataFrame(summarization_results).to_json(f"preprocessing/summarization_results.jsonl", orient="records", lines=True)

if __name__ == "__main__":
    main()
        