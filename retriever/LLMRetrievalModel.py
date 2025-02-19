import re
import gc
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import json
from retriever.prompts import *
from retriever.utils import (
    get_original_dataset_name,
)

class LLMRetrievalModel:
    def __init__(self, args):
        self.args = args
        self.device = args['device']
        if args['use_transform']:
            model_name = args['model_name']
            self.model = self.load_model(model_name) if args['model_name'] else None
            self.tokenizer = self.load_tokenizer(model_name) if args['model_name'] != "" else None
        else:
            self.model = None
            self.tokenizer = None            
        
        self.max_new_tokens = 1024
        self.logit_threshold = 0.01
        self.dataset_name = args['dataset_name']

    def load_model(self, model_name):
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            # device_map=self.device
            device_map='auto'
        )

    def unload_model(self):
        self.model=None
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"{self.args['model_name']} model unloaded")

    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def create_messages(self, instruction, text):

        return [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": f"{instruction}\n\n{text}"},
        ]

    def prepare_inputs(self, messages):
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer([chat_text], return_tensors="pt").to(self.model.device)

    def _generate(self, model_inputs):
        return self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
        )

    def decode_response(self, generated_ids, model_inputs):
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def generate_text(self, instruction, text):
        messages = self.create_messages(instruction, text)
        model_inputs = self.prepare_inputs(messages)
        generated_ids = self._generate(model_inputs)
        return self.decode_response(generated_ids, model_inputs)

    def rewrite_query(self, df_questions):
        rewrite_results = {}
        reranking_bar = tqdm(total=len(df_questions), desc="Query rewriting process")
        for idx, row in df_questions.iterrows():
            instruction = QUERY_REWRITING_PROMPT_3
            qid = row["qid"]
            question = row["question"]
            instruction = instruction.replace("{Question}", question)
            answer = self.generate_text(instruction, "")
            
            rewrite_results[qid] = {}
            rewrite_results[qid]["input_prompt"] = instruction
            rewrite_results[qid]['rewritten_query'] = answer
            
            answer = answer.replace("Query: ", "").replace("## ","")
            df_questions.loc[idx, "first_answer"] = answer.replace("\n", " ")
            
            reranking_bar.update(1)
        reranking_bar.close()

        return df_questions, rewrite_results

    def processing_answer(self, answer):
        doc_ret_answer = answer.split("[Document retrieval]:")[1].split("[Passage selection]:")[0]
        pas_sel_answer = answer.split("[Passage selection]:")[1].split("[Answer generation]:")[0]
        gen_answer = answer.split("[Answer generation]:")[1]
        return doc_ret_answer, pas_sel_answer, gen_answer
    
    def unanswerable_check(self, df_questions, page_retrieval_results):
        PROMPT = ENHANCED_PROMPT_3

        suggested_queries = {}
        correct_yes = 0
        correct_no = 0
        incorrect_yes = 0
        incorrect_no = 0
        
        pbar = tqdm(total=len(df_questions), desc="Unanswerable check")
        for idx, row in df_questions.iterrows():
            qid = row["qid"]
            label = row['answer']
            question = row["question"]
            evidences = row["evidences"]
            evidences_len = len(evidences)
            correct = 0
            evidence_key = []
            retrieved_passages = page_retrieval_results[qid]
            for evidence in evidences:
                doc_name = evidence["doc_name"]
                page_num = evidence["page_num"]
                key = f"{doc_name}_{page_num}"
                evidence_key.append(key)
            
            # top_5 passages
            count = 0
            for passage in retrieved_passages:
                key = f"{passage['source']}_{passage['page']}"
                if key in evidence_key:
                    count += 1
            
            is_correct = (count == evidences_len) 

            passages = '\n'.join([
            f"Context{idx} (ID: {idx}): " 
            "Title is "+ value["source"] +". Content is " + value['page_content'] 
            for idx, value in enumerate(retrieved_passages)
            ])
            inputs = ("Context: ", passages, "Question: ", question)

            full_prompt = FULL_PROMPT.format(task=PROMPT["task_description"], instructions='\n '.join(PROMPT["instructions"]), inputs=inputs, output_format=PROMPT["output_format"])
            answer = self.generate_text(full_prompt,"")
            
            try:
                parse_output = self.parse_output(answer)
                answerable, generated_answer, answerable_doc_ids, suggested_query = parse_output["is_answerable"], parse_output["answer"], parse_output["answerable_doc_ids"], parse_output["refined_query"]
            except:
                print(f"Error in parsing output for question {qid}")
                answerable, generated_answer, answerable_doc_ids, suggested_query = "unanswerable", None, [], question
            answerable = True if answerable in ["true", "yes", "answerable"] else False
            
            if is_correct and answerable:
                correct_yes += 1
            elif is_correct and not answerable:
                correct_no += 1
            elif not is_correct and answerable:
                incorrect_yes += 1
            else:
                incorrect_no += 1

            if answerable is False and len(answerable_doc_ids) >= self.args["max_relevant_ids"]:
                answerable = True

            suggested_queries[qid] = {}
            suggested_queries[qid]["suggested_query"] = None if suggested_query in ["None","", "none"] else suggested_query
            suggested_queries[qid]["generated_answer"] = None if answerable in ["None", "", "none"] else generated_answer
            suggested_queries[qid]["is_answerable"] = answerable
            suggested_queries[qid]["explanation"] = parse_output.get("missing_information", "")
            suggested_queries[qid]['input_prompt'] = full_prompt
            suggested_queries[qid]['output_answer'] = answer
            
            try:
                relevant_passages = [retrieved_passages[idx] for idx in answerable_doc_ids]
                suggested_queries[qid]["relevant_passages"] = relevant_passages
            except:
                suggested_queries[qid]["relevant_passages"] = []
            
            pbar.update(1)

        analysis_counts = {
            "correct_yes": correct_yes,
            "correct_no": correct_no,
            "incorrect_yes": incorrect_yes,
            "incorrect_no": incorrect_no,
            }

        self.print_out_analysis(analysis_counts)
        
        return suggested_queries
    
    def print_out_analysis(self, analysis_counts):
        print("===== Result of Answerability checker =====")
        print(f"True & Positive: {analysis_counts['correct_yes']}")
        print(f"True & Negative: {analysis_counts['correct_no']}")
        print(f"False & Positive: {analysis_counts['incorrect_yes']}")
        print(f"False & Negative: {analysis_counts['incorrect_no']}")
        print("================================")

        correct_yes = analysis_counts["correct_yes"]
        correct_no = analysis_counts["correct_no"]
        incorrect_yes = analysis_counts["incorrect_yes"]
        incorrect_no = analysis_counts["incorrect_no"]

        total_count = correct_yes + correct_no + incorrect_yes + incorrect_no
        correct_yes_ratio = correct_yes / total_count
        correct_no_ratio = correct_no / total_count
        incorrect_yes_ratio = incorrect_yes / total_count
        incorrect_no_ratio = incorrect_no / total_count

        print(f"True & Positive Ratio: {correct_yes_ratio}")
        print(f"True & Negative Ratio: {correct_no_ratio}")
        print(f"False & Positive Ratio: {incorrect_yes_ratio}")
        print(f"False & Negative Ratio: {incorrect_no_ratio}")
                
        sufficiency = (correct_yes + incorrect_yes) / total_count
        print(f"Sufficiency: {sufficiency}")
            
    def remove_special_characters(self, text):
        text = text.replace("#", "").replace("##", "").replace(":","").replace("-","").replace("_","")
        text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ").replace("  ", " ")
        return text.lower().strip()

    def parse_output(self, output_str):
        # Split into lines
        output_str = output_str.replace("\n",'')
        lines = output_str.strip().split("##")
        
        # Dictionary to store parsed values
        parsed_data = {}
        
        answerable_marker = "is_answerable:"
        missing_information_marker = "missing_information:"
        answer_marker = "answer:"
        answerable_doc_ids_marker = "answerable_doc_ids:"
        suggested_query_marker = "refined_query:"
        
        for line in lines:
            line = line.strip()
            if answerable_marker in line:
                parsed_data["is_answerable"] = line.split(answerable_marker)[1].strip()
            elif missing_information_marker in line:
                parsed_data["missing_information"] = line.split(missing_information_marker)[1].strip()
            elif answer_marker in line:
                parsed_data["answer"] = line.split(answer_marker)[1].strip()
            elif answerable_doc_ids_marker in line:
                parsed_data["answerable_doc_ids"] = line.split(answerable_doc_ids_marker)[1].strip()
            elif suggested_query_marker in line:
                parsed_data["refined_query"] = line.split(suggested_query_marker)[1].strip()

        relevant_ids = []
        answerable_doc_ids = parsed_data.get("answerable_doc_ids", "")
        if answerable_doc_ids:
            #"[1, 3]" -> "1, 3" -> ["1", "3"] -> [1,3]
            trimmed_ids = answerable_doc_ids.strip("[] \n\t")
            if trimmed_ids:
                relevant_ids = [int(x.strip()) for x in trimmed_ids.split(",") if x.strip().isdigit()]
        parsed_data["answerable_doc_ids"] = relevant_ids[:self.args["max_relevant_ids"]]
        return parsed_data