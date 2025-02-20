import os

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from retriever.base_framework import BaseFramework
from retriever.utils import (
    calculate_numeric_accuracy,
    calculate_gpt_accuracy_text,
    get_original_dataset_name,
)


class GPTDirect(BaseFramework):
    numeric_system_prompt = (
        "You are a financial expert, you are supposed to answer the given question directly. "
        "Do not include units such as thousands, millions, or billions, or mention currency if the question specifically requests them to be excluded. "
        "Do not use commas for thousand separators in the answer. "
        "You must just give a answer without any other reasons and respond with 'The answer is {final answer}.'. "
    )

    text_system_prompt = (
        "You are a financial expert, you are supposed to answer the given question directly. "
        "Write the answer concisely in a sentence or a phrase."
    )

    framework_name = "gpt-direct"

    def __init__(
        self,
        dataset_name,
        pdf_path,
        output_dir,
        seed,
        is_numeric_question=False,
    ):
        super().__init__(
            dataset_name,
            pdf_path,
            output_dir,
            seed,
            is_numeric_question=is_numeric_question,
        )

        self.openai_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=self.openai_key)
        self.openai_model_name = "gpt-4o"
        self.temperature = 0.01
        os.makedirs(self.output_dir, exist_ok=True)
    

    def generate_answer(self, question):
        def extract_answer(response):
            if self.is_numeric_question:
                splited = response.split("The answer is")
                if len(splited) == 1:
                    return ""
                return splited[-1].lstrip(" ").rstrip('. "')
            return response

        user_prompt = "Question: {}".format(question)


        messages = [
            {
                "role": "system",
                "content": (
                    self.numeric_system_prompt
                    if self.is_numeric_question
                    else self.text_system_prompt
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        response = self.client.chat.completions.create(
            model=self.openai_model_name,
            messages=messages,
            temperature=self.temperature,
        )
        generated = response.choices[0].message.content

        extracted_answer = extract_answer(generated)
        if extracted_answer.startswith("{") and extracted_answer.endswith("}"):
            extracted_answer = extracted_answer[1:-1]

        return {
            "prompt": messages,
            "generated": generated,
            "extracted": extracted_answer,
            "response": response,
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
            for metric in ["acc" if self.is_numeric_question else "gpt_acc"]
        }

        for idx, (_, row) in enumerate(
            tqdm(self.dataset.iterrows(), total=len(self.dataset))
        ):
            qid = row["qid"]
            original_dataset = get_original_dataset_name(qid)
            question = row["question"]
            generated_result = self.generate_answer(question)
            prompt = generated_result["prompt"]
            generated = generated_result["generated"]
            extracted_answer = generated_result["extracted"]
            response = generated_result["response"]
            evidences = row["evidences"]

            answer = str(row["answer"])

            results[qid] = {
                "question": question,
                "answer": answer,
                "extracted_answer": extracted_answer,
                "prompt": prompt,
                "gpt_generated": generated,
                "response": response.to_dict() if response is not None else None,
            }

            if self.is_numeric_question:
                acc = calculate_numeric_accuracy(answer, extracted_answer)
                scores["acc"][original_dataset].append(acc)
                scores["acc"]["total"].append(acc)
                eval_results[qid] = {
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "extracted_answer": extracted_answer,
                    "original_dataset": original_dataset,
                    "acc": acc,
                }
            else:
                contexts = self.load_contexts(evidences=evidences)
                contexts_ = [f"{passage.page_content}" for passage in contexts]
                contexts_ = "\n".join(contexts_)
                gpt_eval_results = calculate_gpt_accuracy_text(self.client, question, answer, extracted_answer, contexts_)
                acc = gpt_eval_results["score"]
                scores["gpt_acc"][original_dataset].append(acc)
                scores["gpt_acc"]["total"].append(acc)
                results[qid]["gpt_eval_results"] = gpt_eval_results
                eval_results[qid] = {
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "extracted_answer": extracted_answer,
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
        