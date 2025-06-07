import json
import os
from typing import Optional
import re

from langchain.document_loaders import PyMuPDFLoader
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import numpy as np

from retriever.prompts import *

from retriever.utils import (
    calculate_numeric_accuracy,
    calculate_gpt_accuracy_text,
    run_program,
    get_original_dataset_name,
)

def process_single_pot_output(output):
    if not output or "argparse" in output:
        return "", ""

    function_name = ""
    processed_output = ""

    tmp = re.findall(r"```python(.*?)```", output, re.DOTALL)
    if len(tmp) > 0:
        processed_output = tmp[0].strip("\n")
    else:
        tmp = re.findall(r"```(.*?)```", output, re.DOTALL)
        if len(tmp) > 0:
            processed_output = tmp[0].strip("\n")
        else:
            tmp = re.findall(r"```", output, re.DOTALL)
            if len(tmp) == 1:
                if len(output) > 4 and output[:4] == "    ":
                    processed_output = "def solution():\n" + output.split("```")[0]
                else:
                    processed_output = "def solution():\n    " + output.split("```")[0]
            else:
                if len(output) > 4 and output[:4] == "    ":
                    processed_output = "def solution():\n" + output
                else:
                    processed_output = "def solution():\n    " + output

    processed_output = re.sub(r"(?<=\d),(?=\d)", "", processed_output)

    match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", processed_output)
    if match:
        function_name = match.group(1)
    else:
        function_name = "solution"

    processed_output = processed_output.strip("```").strip()

    return function_name, processed_output


class Generator:
    pot_system_prompt = """
You are a financial expert, you are supposed to generate a Python program to answer the given question based on the provided financial document context. The returned value of the program is supposed to be the answer. 
```python
def solution():
    # Define variables name and value based on the given context
    guarantees = 210
    total_exposure = 716

    #Do math calculation to get the answer
    answer = (guarantees / total_exposure) * 100

    # return answer
    return answer
```
"""
    cot_system_prompt = (
        "You are a financial expert, you are supposed to answer the given question based on the provided financial document context. "
        "You need to first think through the problem step by step, documenting each necessary step. "
        "Then you are required to conclude your response with the final answer in your last sentence as 'Therefore, the answer is {final answer}'. "
        "The final answer should be a numeric value and No text should be added after the final answer."
    )
    cot_user_prompt_postfix = "Let's think step by step to answer the given question."

    text_cot_system_prompt = (
        "You are a financial expert, you are supposed to answer the given question based on the provided financial document context. "
        "You need to first think through the problem step by step, documenting each necessary step. "
        "Then you are required to conclude your response with the final answer in your last sentence as 'Therefore, the answer is {final answer}'. "
    )

    text_direct_system_prompt = (
        "You are a financial expert, you are supposed to answer the given question directly. "
        "Write the answer concisely in 1~2 sentences."
    )
    numeracy_direct_system_prompt = (
        "You are a financial expert, you are supposed to answer the given question directly. "
        "Do not include units such as thousands, millions, or billions, or mention currency if the question specifically requests them to be excluded. "
        "Do not use commas for thousand separators in the answer. "
        "You must just give a answer without any other reasons and respond with 'The answer is {final answer}.'. "
    )

    pot_user_prompt_postfix = """Please generate a Python program to answer the given question. The format of the program should be the following:
```python
def solution():
    # Define variables name and value based on the given context
    ...
    # Do math calculation to get the answer
    ...
    # return answer
    return answer
```

Continue the program to answer the question. The returned value of the program is supposed to be the answer:
```python
def solution():
    # Define variables name and value based on the given context
"""

    user_prompt = "{}\n\nQuestion: {}"

    def __init__(
        self,
        pdf_path,
        openai_model_name="gpt-4o-mini",
        temp=0.01,
        max_tokens=2048,
        max_context_count=10,
        use_full_page=True,
        is_numeric_question=True,
        generate_method="direct",
        enable_extra_step=False,
        args=None,
    ):
        self.pdf_path = pdf_path
        self.openai_model_name = openai_model_name
        self.temp = temp
        self.max_tokens = max_tokens
        self.max_context_count = max_context_count
        self.use_full_page = use_full_page
        self.is_numeric_question = is_numeric_question
        self.generate_method = generate_method
        self.enable_extra_step = enable_extra_step
        self.use_gpt_acc = False
        self.args = args

        self.openai_key = os.environ["OPENAI_API_KEY"]

        self.client = OpenAI(api_key=self.openai_key)

        self.context_prefix = "[START OF FILING] "
        self.context_postfix = " [END OF FILING]\n"

        prompts = self.get_prompts()
        self.params = {
            "openai_model_name": openai_model_name,
            "temp": temp,
            "max_tokens": max_context_count,
            "max_context_count": max_context_count,
            "prompt": prompts,
        }

    def get_prompts(self):
        if not self.is_numeric_question:
            return {
                "system_prompt": self.text_direct_system_prompt,
                "user_prompt": self.user_prompt,
                "context_prefix": self.context_postfix,
                "context_post_fix": self.context_postfix,
            }
        else:
            system_prompt = ""
            user_prompt = ""
            if self.generate_method == "direct":
                system_prompt = self.numeracy_direct_system_prompt
                user_prompt = self.user_prompt
            elif self.generate_method == "cot":
                system_prompt = self.cot_system_prompt
                user_prompt = self.user_prompt + "\n\n" + self.cot_user_prompt_postfix
            elif self.generate_method == "pot":
                system_prompt = self.pot_system_prompt
                user_prompt = self.user_prompt + "\n\n" + self.pot_user_prompt_postfix
            return {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "context_prefix": self.context_postfix,
                "context_post_fix": self.context_postfix,
            }

    def prepare_text_prompt(self, question, context):
        text_system_prompt = self.text_direct_system_prompt
        user_prompt = self.user_prompt.format(context, question)
        return text_system_prompt, user_prompt

    def prepare_direct_prompt(self, question, context):
        if self.is_numeric_question:
            numeric_system_prompt = self.numeracy_direct_system_prompt
            user_prompt = self.user_prompt.format(context, question)
            return numeric_system_prompt, user_prompt
        else:
            return self.prepare_text_prompt(question, context)

    def prepare_cot_prompt(self, question, context):
        if self.is_numeric_question:
            numeric_system_prompt = self.cot_system_prompt
            user_prompt = self.user_prompt.format(context, question)
            user_prompt += "\n\n" + self.cot_user_prompt_postfix
            return numeric_system_prompt, user_prompt
        else:
            text_system_prompt = self.text_cot_system_prompt
            user_prompt = self.user_prompt.format(context, question)
            user_prompt += "\n\n" + self.cot_user_prompt_postfix
            return text_system_prompt, user_prompt

    def prepare_pot_prompt(self, question, context):
        if self.is_numeric_question:
            numeric_system_prompt = self.pot_system_prompt
            user_prompt = self.user_prompt.format(context, question)
            user_prompt += "\n\n" + self.pot_user_prompt_postfix
            return numeric_system_prompt, user_prompt
        else:
            return self.prepare_text_prompt(question, context)

    def extract_direct_answer(self, response):
        if self.is_numeric_question:
            splited = response.split("The answer is")
            if len(splited) == 1:
                return ""
            return splited[-1].lstrip(" ").rstrip('. "')
        else:
            return response

    def extract_cot_answer(self, response):
        splited = response.split("Therefore, the answer is")
        if len(splited) == 1:
            return ""
        return splited[-1].lstrip(" ").rstrip('. "')

    def extract_pot_answer(self, response):
        if self.is_numeric_question:
            def_name, pot_code = process_single_pot_output(response)
            extracted = run_program(pot_code, def_name)
            return extracted, pot_code
        else:
            return response, None

    def generate_answer(self, question, retrieved_contexts):
        # finish_retrieved_contexts = retrieved_contexts
        context_prompt = ""
        used_context_count = 0
        
        final_retrieved_contexts = {}
        document_page = []
        for idx, retrieved in enumerate(retrieved_contexts):
            source = retrieved["source"]
            page = retrieved["page"]

            key = f"{source}_{page}"
            if key not in document_page:
                document_page.append(key)
            
            if key not in final_retrieved_contexts:
                final_retrieved_contexts[key] = [retrieved]
            else:
                final_retrieved_contexts[key].append(retrieved)
        
        finish_retrieved_contexts = []
        for key in document_page:
            retrieved = final_retrieved_contexts[key]
            if "start_index" in retrieved[0]:
                retrieved = sorted(retrieved, key=lambda x: x["start_index"])
            source = retrieved[0]["source"]
            page = retrieved[0]["page"]
            page_content = ' '.join([retrieved["page_content"] for retrieved in retrieved])
            if not self.use_full_page:
                full_page_content = ' '.join([retrieved["page_content"] for retrieved in retrieved])
            else:
                full_page_content = retrieved[0]["full_page_content"]

            final_retrieved = {"source": source, "page_content": page_content, "full_page_content": full_page_content}
            finish_retrieved_contexts.append(final_retrieved)
            
        for idx, retrieved in enumerate(finish_retrieved_contexts):
            source = retrieved["source"]
            page_content = retrieved["page_content"]
            full_page_content = retrieved["full_page_content"]
            content = full_page_content if self.use_full_page else page_content
            # context = f"Context{idx+1}: Title is {source}. Content is {content}\n\n"
            context = self.context_prefix + f"{source} {content}" + self.context_postfix
            context_prompt += context
            used_context_count += 1
            if used_context_count == self.max_context_count:
                break

        if self.generate_method == "direct":
            system_prompt, user_prompt = self.prepare_direct_prompt(
                question, context_prompt
            )
        elif self.generate_method == "cot":
            system_prompt, user_prompt = self.prepare_cot_prompt(
                question, context_prompt
            )
        elif self.generate_method == "pot":
            system_prompt, user_prompt = self.prepare_pot_prompt(
                question, context_prompt
            )
        else:
            raise AssertionError()

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        response = self.client.chat.completions.create(
            model=self.openai_model_name,
            messages=messages,
            temperature=self.temp,
        )

        generated = response.choices[0].message.content

        results = {
            "prompt": messages,
            "used_context_count": used_context_count,
            "response": response.to_dict(),
            "generated": generated,
        }

        if self.generate_method == "direct":
            extracted_answer = self.extract_direct_answer(generated)
            results["extracted"] = extracted_answer
        elif self.generate_method == "cot":
            extracted_answer = self.extract_cot_answer(generated)
            results["extracted"] = extracted_answer
        elif self.generate_method == "pot":
            extracted_answer, pot_code = self.extract_pot_answer(generated)
            results["extracted"] = extracted_answer
            if pot_code:
                results["pot_code"] = pot_code

        return extracted_answer, results

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

    def evaluate(self, dataset: pd.DataFrame, results: dict):
        if self.is_numeric_question:
            eval_results = {
                metric: {
                    "financebench": [],
                    "open_secqa": [],
                    "open_finqa": [],
                    "total": [],
                }
                for metric in ["acc", #"acc2", 
                               "gpt_acc"]
            }
        else:
            eval_results = {
                metric: {
                    "financebench": [],
                    "open_secqa": [],
                    "open_finqa": [],
                    "total": [],
                }
                for metric in ["gpt_acc"]
            }
        eval_results2 = {}

        for _, data in tqdm(dataset.iterrows(), total=len(dataset), desc="Answer Evaluation"):
            qid = data["qid"]
            original_dataset = get_original_dataset_name(qid)

            question = data["question"]
            answer = str(data["answer"])
            generated = results[qid]
            evidences = data["evidences"]
            contexts = self.load_contexts(evidences=evidences)
            contexts_ = [f"{passage.page_content}" for passage in contexts]
            contexts_ = "\n".join(contexts_)

            eval_results2[qid] = {
                "qid": qid,
                "question": question,
                "answer": answer,
                "extracted_answer": generated,
                "original_dataset": original_dataset,
            }

            if self.is_numeric_question:
                # Accuracy
                acc = calculate_numeric_accuracy(answer, generated)
                eval_results2[qid]["accuracy"] = acc
                eval_results["acc"][original_dataset].append(acc)
                eval_results["acc"]["total"].append(acc)

                if self.use_gpt_acc:
                    # GPT Accuracy
                    gpt_accuracy_result = calculate_gpt_accuracy_text(
                        self.client,
                        question,
                        answer,
                        generated,
                        contexts_,
                    )
                    gpt_acc_prompt = gpt_accuracy_result["prompt"]
                    gpt_accuracy = gpt_accuracy_result["score"]
                    gpt_accuracy_generated = gpt_accuracy_result["generated"]

                    eval_results2[qid]["gpt_acc"] = gpt_accuracy
                    eval_results2[qid]["gpt_acc_generated"] = gpt_accuracy_generated
                    eval_results2[qid]["gpt_acc_prompt"] = gpt_acc_prompt
                    eval_results["gpt_acc"][original_dataset].append(gpt_accuracy)
                    eval_results["gpt_acc"]["total"].append(gpt_accuracy)
            else:
                # GPT Accuracy
                gpt_accuracy_result = calculate_gpt_accuracy_text(
                    self.client, question, answer, generated, contexts_
                )
                gpt_acc_prompt = gpt_accuracy_result["prompt"]
                gpt_accuracy = gpt_accuracy_result["score"]
                gpt_accuracy_generated = gpt_accuracy_result["generated"]

                eval_results2[qid]["gpt_acc"] = gpt_accuracy
                eval_results2[qid]["gpt_acc_generated"] = gpt_accuracy_generated
                eval_results2[qid]["gpt_acc_prompt"] = gpt_acc_prompt
                eval_results["gpt_acc"][original_dataset].append(gpt_accuracy)
                eval_results["gpt_acc"]["total"].append(gpt_accuracy)

        scores = {"count": {}}
        for metric_key in eval_results.keys():
            scores[metric_key] = {}
            for dataset_name in eval_results[metric_key].keys():
                data_count = len(eval_results[metric_key][dataset_name])
                if dataset_name not in scores["count"]:
                    scores["count"][dataset_name] = data_count

                if data_count == 0:
                    scores[metric_key][dataset_name] = None
                else:
                    scores[metric_key][dataset_name] = float(np.mean(
                        eval_results[metric_key][dataset_name]
                    ))

        return {
            "scores": scores,
            "results": eval_results2,
        }

    def generate_answers(
        self, dataset: pd.DataFrame, retrieved_results: Optional[dict] = None,
    ):
        results = {}
        answers = {}

        for _, data in tqdm(dataset.iterrows(), total=len(dataset), desc="Answer generation"):
            qid = data["qid"]
            original_dataset = get_original_dataset_name(qid)
            retrieved_contexts = retrieved_results[qid]
            question = "## Question: " + data["question"]
            if self.enable_extra_step:
                new_question = data['gen_answer']
                question = question + '\n' + "Message: " + new_question

            generated_answer, generated_result = self.generate_answer(
                question, retrieved_contexts
            )

            answers[qid] = generated_answer
            results[qid] = {
                "question": question,
                "original_dataset": original_dataset,
                "answer": generated_answer,
                "result": generated_result,
            }

        evaluation_result = self.evaluate(dataset, answers)

        return {
            "params": self.params,
            "results": results,
            "scores": evaluation_result["scores"],
            "eval_results": evaluation_result["results"],
        }
    
    def generate_text(self, system_prompt, full_prompt):
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": full_prompt,
            },
        ]
        response = self.client.chat.completions.create(
            model=self.openai_model_name,
            messages=messages,
            temperature=self.temp,
        )

        generated = response.choices[0].message.content
        return generated, response.usage.to_dict()