import os
import re

from tqdm import tqdm
import torch
from langchain.vectorstores import Chroma
from langchain_community.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from .base_framework import BaseFramework
from .utils import (
    calculate_numeric_accuracy,
    calculate_gpt_accuracy_text,
    get_original_dataset_name,
)


MAX_QUERY_LENGTH = 300


class RQRag(BaseFramework):
    framework_name = "rq-rag"
    expand_on_tokens = [
        "[S_Rewritten_Query]",
        "[S_Decomposed_Query]",
        "[S_Disambiguated_Query]",
        "[A_Response]",
    ]

    model_name = "zorowin123/rq_rag_llama2_7B"

    numeric_system_prompt = "Given a question that requires multi-hop reasoning, you need to decompose the question and answer based on the given context. You must just give a answer without any other reasons and respond with 'The answer is {final answer}.'."
    text_system_prompt = "Given a question that requires multi-hop reasoning, you need to decompose the question and answer based on the given context. Write the answer concisely in a sentence or a phrase."

    top_k = 3

    def __init__(self, dataset_name, pdf_path, output_dir, seed, is_numeric_question):
        super().__init__(
            dataset_name,
            pdf_path,
            output_dir,
            seed,
            is_numeric_question=is_numeric_question,
        )

        self.retriever = self.get_retriever()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        self.special_tokens_dict = self.load_sag_special_tokens(self.tokenizer)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            from_tf=bool(".ckpt" in self.model_name),
            device_map="auto",
            torch_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
        )
        self.model.generation_config.eos_token_id = [
            self.tokenizer.convert_tokens_to_ids("</s>"),
            self.tokenizer.convert_tokens_to_ids("[EOS]"),
        ]

        self.max_depth = 2

        self.openai_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=self.openai_key)
        self.openai_model_name = "gpt-4o"
        self.temp = 0.01

        os.makedirs(self.output_dir, exist_ok=True)

    def get_retriever(self):
        db_dir = "vectordb/dense"
        table_name = "dense"
        embedding_model_name = "text-embedding-3-small"

        embedding_function_openai = OpenAIEmbeddings(model=embedding_model_name)
        retriever_openai = Chroma(
            persist_directory=db_dir,
            collection_name=table_name,
            embedding_function=embedding_function_openai,
        )

        return retriever_openai

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
    
    def retrieve_passages(self, question):
        retrieved_passages = self.retriever.similarity_search(question, k=self.top_k)
        retrieved_passages = retrieved_passages[: self.top_k]
        retrieved_passages = [self.convert_to_dict(passage) for passage in retrieved_passages]
        return retrieved_passages

    def load_sag_special_tokens(self, tokenizer):
        special_tokens_dict = {}

        for token in tokenizer.additional_special_tokens:
            special_tokens_dict[token] = tokenizer.convert_tokens_to_ids(token)

        return special_tokens_dict

    def generate_tree_of_thoughts(self, initial_prompts):
        def format_evidences(evidences: list, max_length_per_evidence=350):
            tmp = ""

            for cur_evidence in evidences:
                page_content = re.sub(
                    r"[\s\n\t]+", " ", cur_evidence["page_content"]
                ).strip()
                if len(page_content) > max_length_per_evidence:
                    page_content = page_content[:max_length_per_evidence]
                tmp += "Title: " + cur_evidence["source"] + "\n"
                tmp += "Text: " + page_content + "\n"

            return tmp

        temp_query = []
        retrieval_results = []

        paths = [
            {
                "prompt": prompt,
                "depth": 0,
                "text": "",
                "done": False,
                "retrieved_index": [],
            }
            for prompt in initial_prompts
        ]
        final_outputs = []
        
        while paths:
            current_path = paths.pop(0)

            if current_path["done"]:
                final_outputs.append(current_path)
                continue

            for special_token in self.expand_on_tokens:
                if (
                    current_path["depth"] >= self.max_depth
                    and special_token != "[A_Response]"
                ):
                    continue

                input_text = current_path["prompt"] + special_token
                # print(input_text)
                # print("=" * 20)
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                ).to(self.model.device)

                if special_token != "[A_Response]":
                    # when not generating the final answer, we adjust the temp to increase diversity
                    outputs = self.model.generate(
                        **inputs,
                        return_dict_in_generate=True,
                        temperature=1.0,
                        max_length=8192,
                    )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        return_dict_in_generate=True,
                        do_sample=False,
                        max_length=8192,
                    )

                decoded_output = self.tokenizer.batch_decode(
                    outputs.sequences, skip_special_tokens=False
                )[0].replace("<s> ", "<s>")

                input_tokens_count = inputs['input_ids'].size(1)  # Number of tokens in the input
                output_tokens_count = outputs.sequences.size(1)  # Number of tokens in the output

                temp_query.append({
                    "prompt": input_text,
                    "token": special_token,
                    "response": decoded_output,
                    "input_tokens_count": input_tokens_count,
                    "output_tokens_count": output_tokens_count,
                })

                if special_token == "[A_Response]":
                    # done
                    pattern = r"\[A_Response\](.*?)\[EOS\]"
                    matches = re.findall(pattern, decoded_output, re.DOTALL)
                    if len(matches) > 0:
                        result = matches[-1].strip()
                    else:
                        result = "dummy results, unable to detect valid answer"

                    new_path = {
                        "prompt": decoded_output,
                        "depth": current_path["depth"] + 1,
                        "done": True,
                        "final_answer": result,
                        "retrieved_index": current_path["retrieved_index"],
                    }
                else:
                    # get the query and ask search_engine
                    pattern = r"\[(S_Rewritten_Query|S_Decomposed_Query|S_Disambiguated_Query)\](.*?)\[EOS\]"
                    matches = re.findall(pattern, decoded_output, re.DOTALL)
                    if len(matches) > 0:
                        query_for_search = matches[-1][1].strip()
                    else:
                        query_for_search = "dummy"

                    top_indices = [None]
                    evidences = self.retrieve_passages(query_for_search)
                    retrieval_results.append({
                        "query": query_for_search,
                        "retrieved": evidences,
                    })
                    evidences_list = format_evidences(evidences)

                    new_path = {
                        "prompt": decoded_output
                        + "[R_Evidences]"
                        + evidences_list
                        + "[/R_Evidences]",
                        "depth": current_path["depth"] + 1,
                        "done": False,
                        "cur_special_token": special_token,
                        "cur_query": query_for_search,
                        "cur_evidence": evidences_list,
                        "retrieved_index": current_path["retrieved_index"]
                        + top_indices,
                    }

                paths.append(new_path)

        return {
            "final_outputs": final_outputs,
            "responses": temp_query,
            "retrieval_results": retrieval_results,
        }

    def get_ppl_and_answer_confidence_and_option(self, ins):
        inputs = self.tokenizer(
            [ins], return_tensors="pt", add_special_tokens=False, padding=False
        ).to(self.model.device)

        answer_pattern = r"(?:is|was)\s+(.*?)(?:[.,]\s|\[EOS\])"

        matches = re.findall(answer_pattern, ins.split("[A_Response]")[-1], re.DOTALL)
        if matches:
            extracted_result = matches[-1]
        else:
            extracted_result = None

        with torch.no_grad():
            output = self.model(**inputs, labels=inputs["input_ids"])

        # get confidence
        start_token = "[A_Response]"
        end_token = "[EOS]"
        inputs = self.tokenizer.encode(
            ins, return_tensors="pt", add_special_tokens=False
        )

        start_pos = (
            inputs == self.tokenizer.encode(start_token, add_special_tokens=False)[0]
        ).nonzero(as_tuple=True)[1][
            -1:
        ]  # although not preferable, sometimes it does emerge multiple time
        end_pos = (
            inputs == self.tokenizer.encode(end_token, add_special_tokens=False)[0]
        ).nonzero(as_tuple=True)[1][
            -1:
        ]  # there might be multiple [EOS]
        logits = output.logits
        # the next token id is 13 which is "\n", and do not know why sometimes have two \n, and it's different from final answer
        while start_pos < len(inputs[0]) - 1 and inputs[0][start_pos].item() in [
            32000,
            29871,
            13,
        ]:
            start_pos += 1

        if len(start_pos) == 0:
            start_pos = 0
        if len(end_pos) == 0:
            end_pos = 0

        selected_logits = logits[
            0, start_pos - 1 : start_pos, :
        ]  # [A_Response] ** only this one ** [EOS], should be aware, the start_pos in logits means the first output

        # get answers after [A_Response] configence
        probabilities = torch.softmax(selected_logits, dim=-1)
        max_confidence, _ = torch.max(probabilities, dim=-1)
        average_confidence = torch.mean(max_confidence).item()

        # get ppl
        ppl = torch.exp(output.loss).item()

        selected_option = extracted_result  # might be none
        selected_option_score = average_confidence

        return (
            ppl,
            average_confidence,
            selected_option,
            selected_option_score,
            extracted_result,
        )

    def extract_direct_answer(self, response):
        if self.is_numeric_question:
            splited = response.split("The answer is")
            if len(splited) == 1:
                return ""
            return splited[-1].lstrip(" ").rstrip('. "')
        else:
            return response

    def generate_answer(self, row):
        system_prompt = (
            self.numeric_system_prompt
            if self.is_numeric_question
            else self.text_system_prompt
        )
        prompt = (
            f"<s><|system|>\n{system_prompt}"
            + self.tokenizer.eos_token
            + "\n<|user|>\n"
            + row["question"]
            + self.tokenizer.eos_token
            + "\n"
        )
        prompt += "<|assistant|>\n"
        prompts = [prompt]

        results = self.generate_tree_of_thoughts(prompts)

        cur_score_list = []
        answer2score = {}

        tot_results = {
            "tot": [],
        }

        for result in results["final_outputs"]:
            final_answer = result["final_answer"]
            final_answer = self.extract_direct_answer(final_answer)
            if final_answer == "":
                continue

            cur_scores = {}
            (
                ppl,
                confidence,
                selected_option,
                selected_option_score,
                extracted_result,
            ) = self.get_ppl_and_answer_confidence_and_option(result["prompt"])
            if extracted_result is None:
                extracted_result = final_answer
            if selected_option is None:
                selected_option = final_answer

            cur_scores["ppl"] = ppl
            cur_scores["confidence"] = confidence
            cur_scores["extracted"] = extracted_result
            cur_scores["final_answer"] = final_answer
            cur_score_list.append(cur_scores)

            tot_result = {
                "prompt": result["prompt"],
                "final_answer": final_answer,
                "ppl": ppl,
                "confidence": confidence,
                "extracted": extracted_result,
            }
            tot_results["tot"].append(tot_result)

            if final_answer not in answer2score:
                answer2score[final_answer] = 0
            answer2score[final_answer] += confidence

        max_conf_item = max(cur_score_list, key=lambda x: x["confidence"])
        generated_answer = max_conf_item["final_answer"]

        tot_results["answer2score"] = answer2score
        
        return generated_answer, tot_results, results["responses"], results["retrieval_results"]
    
    def extract_retrieved_passages(self, retrieved_results):
        retrieved_passages = []
        for retrieved_result in retrieved_results:
            for retrieved in retrieved_result["retrieved"]:
                retrieved = retrieved.copy()
                retrieved["page"] = int(retrieved["page"])
                retrieved_passages.append(retrieved)
        
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

        for _, row in tqdm(self.dataset.iterrows(), total=len(self.dataset)):
            qid = row["qid"]
            original_dataset = get_original_dataset_name(qid)
            question = row["question"]
            extracted_answer, tot_results, responses, retrieval_results = self.generate_answer(row)
            answer = str(row["answer"])
            evidences = row["evidences"]

            retrieved_passages = self.extract_retrieved_passages(retrieval_results)
            scores["k"][original_dataset].append(len(retrieved_passages))
            scores["k"]["total"].append(len(retrieved_passages))
            retrieval_scores = self.evaluate_pairs(evidences, retrieved_passages)
            recall = retrieval_scores["recall"]
            precision = retrieval_scores["precision"]
            scores["precision"][original_dataset].append(precision)
            scores["precision"]["total"].append(precision)
            scores["recall"][original_dataset].append(recall)
            scores["recall"]["total"].append(recall)

            contexts = self.load_contexts(evidences=evidences)
            contexts_ = [f"{passage.page_content}" for passage in contexts]
            contexts_ = "\n".join(contexts_)

            results[qid] = {
                "question": question,
                "answer": answer,
                "evidences": evidences,
                "generated": extracted_answer,
                "tot": tot_results,
                "original_dataset": original_dataset,
                "responses": responses,
                "retrieval_results": retrieval_results
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