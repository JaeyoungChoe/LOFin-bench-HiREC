import os
import time
import re

import requests
import spacy
from openai import OpenAI
from tqdm import tqdm
import numpy as np

from retriever.base_framework import BaseFramework
from langchain.text_splitter import RecursiveCharacterTextSplitter
from retriever.generator import Generator
from retriever.utils import (
    calculate_numeric_accuracy,
    calculate_gpt_accuracy_text,
    get_original_dataset_name,
)

def remove_wh_words(text: str) -> str:
    wh_words = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}
    words = [word for word in text.split(" ") if word.strip().lower() not in wh_words]
    text = " ".join(words)
    return text

class IRCoTFramework(BaseFramework):
    framework_name = "ircot"

    # Retrieval
    source_corpus_name = "fin_documents"
    retrieval_count = 5
    max_iter = 4
    retrieval_method = "retrieve_from_elasticsearch"

    cot_system_prompt = (
        "You are a financial expert, you are supposed to answer the given question based on the provided financial document context. "
        "You need to first think through the problem step by step, documenting each necessary step. "
        "Then you are required to conclude your response with the final answer in your last sentence as 'Therefore, the answer is {final answer}'. "
    )

    temperature = 0.1

    # Answer Extractor
    answer_extractor_regex = re.compile(".* answer is (.*)")

    generator_args = {
        "openai_model": "gpt-4o",
        "temperature": 0.01,
        "max_tokens": 64536,
        "max_contexts": 10,
        "use_full_page": False,
    }

    def __init__(self, dataset_name, pdf_path, output_dir, seed, is_numeric_question=True, model_name="gpt-4o"):
        super().__init__(
            dataset_name,
            pdf_path,
            output_dir,
            seed,
            is_numeric_question=is_numeric_question,
        )

        self.model_name = model_name

        self.retriever_url = "http://localhost:8020/retrieve"

        self.openai_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=self.openai_key)

        self.spacy_object = spacy.load("en_core_web_sm")

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

    
    def retrieve(self, query):
        params = {
            "retrieval_method": self.retrieval_method,
            "query_text": query,
            "max_hits_count": self.retrieval_count,
            "corpus_name": self.source_corpus_name,
            "document_type": "title_paragraph_text"
        }
        
        for _ in range(10):
            try:
                response =  requests.post(self.retriever_url, json=params)
                result = response.json()
                retrieval = result["retrieval"]

                retrieved_documents = []

                for retrieval_item in retrieval:
                    pid = retrieval_item["id"]
                    title = retrieval_item["title"]
                    content = retrieval_item["paragraph_text"]
                    score = retrieval_item["score"]

                    document = {
                        "pid": pid,
                        "title": title,
                        "content": content,
                        "score": score,
                    }
                    retrieved_documents.append(document)
                
                return retrieved_documents
            except:
                print("Post request didn't succeed. Will wait 20s and retry.")
                time.sleep(20)
        raise Exception("Post request couldn't succeed after several attempts.")
    
    def prepare_prompts(self, retrieved_passages, question, generated=[]):
        prompts = ""
        for retrieved_passage in retrieved_passages:
            prompts += f"Document Title: {retrieved_passage['title']}\n"
            prompts += f"{retrieved_passage['content']}\n\n"
        
        prompts += (
            "Q: Answer the following question by reasoning step-by-step.\n"
            f"{question}\n"
            f"A: {' '.join(generated)}"
        )

        return prompts
    
    def format_context(self, contexts):
        context_prompt = ""
        for retrieved in contexts:
            title = retrieved["title"]
            content = retrieved["content"]
            context_prompt +=  f"Sources: {title} - {content}\n\n"
        return context_prompt
    
    def generate(self, system_prompt, user_prompt):
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
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )

        generated = response.choices[0].message.content

        results = {
            "prompt": messages,
            "response": response.to_dict(),
            "generated": generated,
        }

        return results
    
    def extract_first_sentence(self, output, generated):
        new_sents = list(self.spacy_object(output).sents)
        if new_sents:
            for new_sent in new_sents:
                sentence = new_sent.text.strip()
                if sentence not in generated:
                    return sentence

            first_sentence = new_sents[0].text
            return first_sentence
        return output


    def process_iter_item(self, row):
        qid = row["qid"]
        question = row["question"]
        exit_flag = False
        cur_iter = 0

        cur_query = question
        last_generated = question
        generated = []
        retrieved = []

        results = {
            "qid": qid,
            "question": question,
        }

        while cur_iter < self.max_iter and not exit_flag:
            cur_iter += 1

            cur_query = last_generated
            results[f"iter{cur_iter}"] = {
                "query": cur_query
            }
            
            # Retrieve Relevant Documents
            preprocessed_cur_query = remove_wh_words(cur_query)
            cur_retrieved = self.retrieve(cur_query)
            results[f"iter{cur_iter}"]["retrieval"] = {
                "query": preprocessed_cur_query,
                "retrieved": cur_retrieved,
            }
            for retreived_passage in cur_retrieved:
                retrieved.append(retreived_passage)

            # Gen LLM CoT
            user_prompt = self.prepare_prompts(retrieved, question, generated)
            generated_results = self.generate(self.cot_system_prompt, user_prompt)
            generated_output = generated_results["generated"]
            first_sentence = self.extract_first_sentence(generated_output, generated)
            if first_sentence:
                generated.append(first_sentence)
            results[f"iter{cur_iter}"]["cot"] = {
                "generated_results": generated_results,
                "first_sentence": first_sentence,
            }
            last_generated = first_sentence
            
            # Control Exit
            exit_reason = None
            return_answer = None

            if cur_iter == self.max_iter:
                exit_flag = True
                exit_reason = "reach_max_iter"
            if generated and self.answer_extractor_regex.match(generated[-1]):
                return_answer = self.answer_extractor_regex.match(generated[-1]).group(1)
                exit_flag = True
                exit_reason = "return_answer"
            results[f"iter{cur_iter}"]["exit"] = {
                "exit_flag": exit_flag,
            }
            if exit_reason:
                results[f"iter{cur_iter}"]["exit"]["exit_reason"] = exit_reason
            if return_answer:
                results[f"iter{cur_iter}"]["exit"]["return_answer"] = return_answer

        return results, retrieved
    
    def add_page(self, x):
        source = x["title"]
        content = x["content"]

        documents = self.load_document(source)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=30,
            add_start_index=True,
        )
        page = -1
        content_ = re.sub(r"[\n\t\s]+", " ", content).strip()
        splitted_documents = text_splitter.split_documents(documents)
        for splitted_document in splitted_documents:
            page_ = splitted_document.metadata["page"]
            page_content_ = re.sub(r"[\n\t\s]+", " ", splitted_document.page_content).strip()

            if content_ == page_content_:
                page = page_
                break
        
        if page == -1:
            for idx, sd in enumerate(splitted_documents):
                page_ = sd.metadata["page"]
                page_content_ = re.sub(r"[\n\t\s]+", " ", sd.page_content).strip()
                if page_content_.startswith(content_):
                    page = page_
                    break

        return {
            "source": source,
            "page_content": content,
            "page": page,
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
            
            reasoning_results, retrieved_passages = self.process_iter_item(row)

            retrieved_passages = [self.add_page(passage) for passage in retrieved_passages]
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
                "reasoning_results": reasoning_results,
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
