import os
import sys
import json
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from retriever.base_framework import BaseFramework
from retriever.document_retriever import VectorstoreDocumentRerieverModule
from retriever.generator import Generator
from retriever.page_retriever import (
    AutoModelForSequenceClassificationPageRetrieverModule,
)
from retriever.LLMRetrievalModel import LLMRetrievalModel

import torch
from retriever.base_framework import GPTEmbeddingPassageRetrieverModule

class FinRAGFramework(BaseFramework):
    framework_name = "finrag"
    
    query_transformer_args = {"model_name": "Qwen/Qwen2.5-7B-Instruct", "use_transform": True, "max_relevant_ids": 10}

    document_retriver_args = {
        "config": {
            "doc_emb": "intfloat/multilingual-e5-large",
            "chunk_ver": "summary_document_selection",
            "only_eval": False,
        },
        "db_dir": f"vectordb/dense_retriever/summary_document_selection/intfloat/multilingual-e5-large",
        "rerank": True,
    }

    page_retriever_args = {
        "page_retriever": "deberta",
        "batch_size": 64,
        "retrieve_strategy": "passage",
        "passage_type": "base",
        "only_eval": False,
    }
    # page_retriever_model_name = "finqa_models/DeBERTa_table_random-table_2e-7"
    page_retriever_model_name = "naver/trecdl22-crossencoder-debertav3"

    generator_args = {
        "openai_model": "gpt-4o",
        "temperature": 0.01,
        "max_tokens": 64536,
        "max_contexts": 10,
        "use_full_page": False,
        "enable_extra_step":False,
    }

    use_gpt = False

    def __init__(
        self,
        dataset_name: str,
        pdf_path: str,
        output_dir: str,
        seed: int,
        do_generate=False,
        use_oracle_passage=False,
        is_numeric_question=True,
        generate_method="pot",
        device="0",
    ):
        super().__init__(
            dataset_name,
            pdf_path,
            output_dir,
            seed,
            is_numeric_question=is_numeric_question,
        )

        self.pdf_path = pdf_path
        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

        print("use_oracle_passage", use_oracle_passage)

        self.dataset_name = dataset_name
        self.use_oracle_passage = use_oracle_passage

        self.do_generate = do_generate

        self.generate_method = generate_method
        if self.pdf_path is not None:
            os.makedirs(self.output_dir, exist_ok=True)

        self.max_iteration = 1
        self.current_count = 0
    
    def init_rewriter(self):
        # Query Transformer
        self.query_transformer_args["device"] = self.device
        self.query_transformer_args["dataset_name"] = self.dataset_name
        query_transformer = LLMRetrievalModel(
            args=self.query_transformer_args,
        )
        return query_transformer

    def init_document_retriever(self):
        # Document Retriever
        self.document_retriver_args["document_dir"] = self.pdf_path
        self.document_retriver_args["config"]["device"] = self.device
        document_retriever = VectorstoreDocumentRerieverModule(
            **self.document_retriver_args
        )
        return document_retriever

    def init_page_retriever(self):
        # Page Retriever
        self.page_retriever_args["device"] = self.device
        page_retriever = AutoModelForSequenceClassificationPageRetrieverModule(
            self.page_retriever_model_name,
            self.pdf_path,
            use_oracle_passage=self.use_oracle_passage,
            **self.page_retriever_args,
        )
        return page_retriever

    def init_generator(self):
        # Generator
        generator_openai_model_name = self.generator_args["openai_model"]
        generator_temperature = self.generator_args["temperature"]
        generator_max_tokens = self.generator_args["max_tokens"]
        generator_max_context_count = self.generator_args["max_contexts"]
        generator_use_full_page = self.generator_args["use_full_page"]
        enable_extra_step = self.generator_args["enable_extra_step"]
        self.generator_args["max_relevant_ids"] = self.query_transformer_args["max_relevant_ids"]
        generator = Generator(
            self.pdf_path,
            generator_openai_model_name,
            generator_temperature,
            generator_max_tokens,
            generator_max_context_count,
            use_full_page=generator_use_full_page,
            is_numeric_question=self.is_numeric_question,
            generate_method=self.generate_method,
            enable_extra_step=enable_extra_step,
            args=self.generator_args,
        )
        return generator

    def query_rewriting(self, iteration_dataset):
        query_transformer = self.init_rewriter()
        dataset, rewrite_results = query_transformer.rewrite_query(iteration_dataset)
        query_transformer.unload_model()

        save_path = os.path.join(self.output_dir, "query_rewriting_results.json")
        self.save_json(save_path, rewrite_results)
        return dataset

    def document_retrieval(self, iteration_dataset):
        document_retriever = self.init_document_retriever()
        document_retriever.top_k_list = [1,3,5,10]
        document_retrieval_results = document_retriever.retrieve_documents(iteration_dataset)
        document_retriever.unload_model()
        return document_retrieval_results

    def page_retrieval(self, iteration_dataset, document_retrieval_results):
        page_retriever = self.init_page_retriever()
        page_retriever.top_k_list = [1,3,5,10,15]
        page_retrieval_results = page_retriever.retrieve_pages(
            iteration_dataset, retrieved_documents=document_retrieval_results
        )
        page_retriever.unload_model()
        return page_retrieval_results

    def hierarchical_retrieval(self, iteration_dataset):
        if self.use_oracle_passage:
            document_retrieval_results = {"scores": {}, "results": {}}
            page_retrieval_results = self.page_retrieval(iteration_dataset, None)
        else:
            iteration_dataset = self.query_rewriting(iteration_dataset)
            document_retrieval_results = self.document_retrieval(iteration_dataset)
            
            page_retrieval_results = self.page_retrieval(iteration_dataset, document_retrieval_results["results"])
        return document_retrieval_results, page_retrieval_results, iteration_dataset
    
    def init_dpr(self):
        device = self.device
        dpr_retriever = GPTEmbeddingPassageRetrieverModule(device=device)
        return dpr_retriever
    
    def dpr_retrieval(self, iteration_dataset):
        dpr_retriever = self.init_dpr()
        dpr_retriever.top_k_list = [1,3,5,10,15]
        dpr_retrieval_results = dpr_retriever.evaluate(iteration_dataset)
        dpr_retriever.unload_model()
        return {}, dpr_retrieval_results, iteration_dataset
    
    def generate_answers(self, page_retrieval_results, iteration_dataset):
        generator = self.init_generator()
        generator_results = generator.generate_answers(
            iteration_dataset, retrieved_results=page_retrieval_results
        )
        return generator_results
    
    def iteration_evaluation(self, document_retrieval_results, page_retrieval_results, iteration_dataset):
        doc_retriever = self.init_document_retriever()
        doc_retriever.top_k_list = [1,3,5,10,15]
        scores = doc_retriever.evaluate(iteration_dataset, document_retrieval_results['results'])
        document_retrieval_results["scores"] = scores
        
        page_retriever = self.init_page_retriever()
        page_retriever.top_k_list = [1,3,5,10,15]
        scores = page_retriever.evaluate(iteration_dataset, page_retrieval_results['results'])
        page_retrieval_results["scores"] = scores
        return document_retrieval_results, page_retrieval_results
    
    def iterative_check(self, iteration_dataset, iter_passage_input):
        llm = self.init_rewriter()
        suggested_queries = llm.unanswerable_check(iteration_dataset, iter_passage_input)
        llm.unload_model()
        return suggested_queries       
    
    def save_generated_results(self, generator_results, document_retrieval_results, page_retrieval_results, tag):
        final_score = generator_results["scores"]
        final_score_path = os.path.join(self.output_dir, f"{tag}_final_score.json")
        self.save_json(final_score_path, final_score)

        evaluation_results = generator_results["eval_results"]
        evaluation_results_path = os.path.join(self.output_dir, f"{tag}_eval_results.json")
        self.save_json(evaluation_results_path, evaluation_results)
        
        params = {
            "document_retriever": self.document_retriver_args,
            "page_retriever": self.page_retriever_args,
            "generator": self.generator_args,
            "version": "iteration" + str(self.max_iteration),
            "use_gpt": self.use_gpt,
            "query_transformer": self.query_transformer_args,
            "max_relevant_ids": self.query_transformer_args["max_relevant_ids"],
            "generate_method": self.generate_method,
            
        }
        params_path = os.path.join(self.output_dir, f"{tag}_params.json")
        self.save_json(params_path, params)

        scores = {
            "document_retrieval": document_retrieval_results["scores"],
            "page_retrieval": page_retrieval_results["scores"],
            "generator": generator_results["scores"],
        }
        scores_path = os.path.join(self.output_dir, f"{tag}_scores.json")
        self.save_json(scores_path, scores)

    def save_results(self, generator_results, document_retrieval_results, page_retrieval_results, suggested_queries_results, iteration_dataset, tag):
        results = {}
        for _, row in iteration_dataset.iterrows():
            qid = row["qid"]
            question = row["question"]
            answer = row["answer"]
            evidences = row["evidences"]
            retrieved_docs = document_retrieval_results["results"].get(qid)[:self.query_transformer_args["max_relevant_ids"]]
            retrieved_passages = page_retrieval_results["results"][qid][:self.query_transformer_args["max_relevant_ids"]]
            generated = generator_results["results"][qid]["answer"]
            generated_result = generator_results["results"][qid]["result"]
                
            results[qid] = {
                "question": question,
                "answer": answer,
                "evidences": evidences,
                "retrieved_docs": retrieved_docs,
                "retrieved_passages": retrieved_passages,
                "generated": generated,
                "generated_result": generated_result,
            }

        results_path = os.path.join(self.output_dir, f"{tag}_results.json")
        self.save_json(results_path, results)

        suggested_queries_results_path = os.path.join(self.output_dir, f"{tag}_suggested_queries_results.json")
        self.save_json(suggested_queries_results_path, suggested_queries_results)

    def change_retrieval_result(self, page_retrieval_results):
        # duplicated passage remove by (source, page, content) key
        doc_results = {}
        for qid, value in page_retrieval_results['results'].items():
            unique_results = {}
            for item in value:
                key = (item['source'], item['page'], item['page_content'])
                if key not in unique_results:
                    unique_results[key] = item
            
            doc_results[qid] = list({v['source'] for v in value})
            
            page_retrieval_results['results'][qid] = list(unique_results.values())

        return doc_results, page_retrieval_results
    
    def save_retrieval_result_not_generate(self, document_retrieval_results, page_retrieval_results, dataset,tag):
        params = {
            "document_retriever": self.document_retriver_args,
            "page_retriever": self.page_retriever_args,
        }
        params_path = os.path.join(self.output_dir, f"{tag}_retrieval_params.json")
        self.save_json(params_path, params)

        scores = {
            "document_retrieval": document_retrieval_results["scores"],
            "page_retrieval": page_retrieval_results["scores"],
            "count": len(dataset),
        }
        scores_path = os.path.join(self.output_dir, f"{tag}_scores.json")
        self.save_json(scores_path, scores)

        results = {}
        for _, row in dataset.iterrows():
            qid = row["qid"]
            question = row["question"]
            answer = row["answer"]
            evidences = row["evidences"]
            retrieved_docs = document_retrieval_results["results"].get(qid)
            retrieved_passages = page_retrieval_results["results"][qid]
            results[qid] = {
                "question": question,
                "answer": answer,
                "evidences": evidences,
                "retrieved_docs": retrieved_docs,
                "retrieved_passages": retrieved_passages,
            }
        results_path = os.path.join(self.output_dir, f"{tag}_retrieval_results.json")
        self.save_json(results_path, results)
    
    def save_initial_process(self, document_retrieval_results, page_retrieval_results, iteration_dataset):
        self.output_dir = os.path.join(self.output_dir, "initial")
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_retrieval_result_not_generate(document_retrieval_results, page_retrieval_results, iteration_dataset, 'initial')

    def execute(self):
        iteration_dataset = self.dataset.copy()[:10]
        origin_output_dir = self.output_dir
        
        document_retrieval_results, retrieval_results = {}, []
        iteration_document_results, iteration_passage_results, iteration_generate_results = {}, {}, {}

        if self.framework_name == "finrag_dpr":
            document_retrieval_results, page_retrieval_results, iteration_dataset = self.dpr_retrieval(iteration_dataset)
            document_retrieval_results['scores'], document_retrieval_results['results'] = {}, {}
        else:
            document_retrieval_results, page_retrieval_results, iteration_dataset = self.hierarchical_retrieval(iteration_dataset)
        self.dataset = iteration_dataset.copy()
        retrieval_results.append(page_retrieval_results['results'].copy())
        
        self.save_initial_process(document_retrieval_results, page_retrieval_results, iteration_dataset)
        # Initial retrieval passages limit to 5
        page_retrieval_results['results'] = {qid:passages[:5] for qid,passages in page_retrieval_results['results'].items()}

        suggested_queries_results = {}
        while self.current_count < self.max_iteration:
            self.output_dir = os.path.join(origin_output_dir, "iteration" + str(self.current_count))
            os.makedirs(self.output_dir, exist_ok=True)

            print(f"Current Iteration: {self.current_count}")
            print(f"Lenght of iteration_dataset: {len(iteration_dataset)}")
            
            suggested_queries = self.iterative_check(iteration_dataset, page_retrieval_results['results'])
                
            for qid, data in suggested_queries.items():
                new_query = data['suggested_query']
                print(f"QID: {qid}, New Query: {new_query}")
                is_answerable = data['is_answerable']
                iteration_dataset.loc[iteration_dataset['qid'] == qid, 'question'] = new_query
                iteration_dataset.loc[iteration_dataset['qid'] == qid, 'is_answerable'] = is_answerable

                if qid in suggested_queries_results:
                    suggested_queries_results[qid].append(data)
                else:
                    suggested_queries_results[qid] = [data]

            page_retrieval_results['results'][qid] = data['relevant_passages']

            true_iteration_dataset = iteration_dataset[iteration_dataset['is_answerable'] == True]
            iteration_dataset = iteration_dataset[iteration_dataset['is_answerable'] == False]

            # save true results
            if len(true_iteration_dataset) > 0:
                document_retrieval_results['results'], page_retrieval_results = self.change_retrieval_result(page_retrieval_results)
                true_document_results, true_passage_results = self.iteration_evaluation(document_retrieval_results, page_retrieval_results, true_iteration_dataset)
                if self.do_generate is False:
                    self.save_retrieval_result_not_generate(document_retrieval_results, page_retrieval_results, true_iteration_dataset, "true")
                else:
                    true_iteration_dataset['question'] = self.dataset['question']
                    true_iteration_generate_results = self.generate_answers(true_passage_results['results'], true_iteration_dataset)
                    self.save_generated_results(true_iteration_generate_results, true_document_results, true_passage_results, "true")
                    self.save_results(true_iteration_generate_results, true_document_results, true_passage_results, suggested_queries_results, true_iteration_dataset, "true")
                

            # Re-process the search process for the false case
            if self.framework_name == "finrag_dpr":
                iteration_document_results, iteration_passage_results, _ = self.dpr_retrieval(iteration_dataset)
            else:
                iteration_document_results, iteration_passage_results, _ = self.hierarchical_retrieval(iteration_dataset)
    
            # sum-set operation for the passage retrieval results
            for qid in iteration_passage_results['results']:                
                page_retrieval_results['results'][qid] += iteration_passage_results['results'][qid][:5]

            document_retrieval_results['results'], page_retrieval_results = self.change_retrieval_result(page_retrieval_results)

            false_document_results, false_passage_results = self.iteration_evaluation(document_retrieval_results, page_retrieval_results, iteration_dataset)
            self.save_retrieval_result_not_generate(false_document_results, false_passage_results, iteration_dataset, "false")
            all_document_results, all_passage_results = self.iteration_evaluation(document_retrieval_results, page_retrieval_results, self.dataset)
            self.save_retrieval_result_not_generate(all_document_results, all_passage_results, self.dataset, "all")

            iteration_dataset['question'] = self.dataset['question']
            self.current_count += 1
            print(f"Number of current iteration: {self.current_count}")

        
        if self.current_count == self.max_iteration and self.max_iteration != 0:
            print("Reaching the maximum number of iterations.")
            print("Save the final results.")
        
    def only_generate_answer(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        document_retrieval_results = {}
        qids = list(data.keys())
        dataset = self.dataset[self.dataset['qid'].isin(qids)]
        
        page_retrieval_results = {}
        document_retrieval_results['results'], page_retrieval_results["results"] = self.change_last_retrieval_result(data)
        document_retrieval_results, page_retrieval_results = self.iteration_evaluation(document_retrieval_results, page_retrieval_results, dataset)

        iteration_generate_results = self.generate_answers(page_retrieval_results["results"], dataset)

        self.save_generated_results(iteration_generate_results, document_retrieval_results, page_retrieval_results, "only_generate")
        self.save_results(iteration_generate_results, document_retrieval_results, page_retrieval_results, {}, dataset, "only_generate")
    
    def change_last_retrieval_result(self, results_data):
        doc_results = {}
        page_result_data = {}
        for qid, value in results_data.items():
            unique_results = {}
            for item in value['retrieved_passages']:
                
                key = (item['source'], item['page'], item['page_content'])
                
                if key not in unique_results:
                    unique_results[key] = item
                

            doc_results[qid] = list({v['source'] for v in value['retrieved_passages']})
            
            page_result_data[qid] = list(unique_results.values())
        return doc_results, page_result_data
    
    def continue_execute(self):
        iteration_dataset = self.dataset.copy()
        origin_output_dir = self.output_dir

        all_retrieval_path = ""
        with open(all_retrieval_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        false_path = ""
        with open(false_path, 'r', encoding='utf-8') as f:
            false_result_data = json.load(f)

        # last iter + 1
        self.current_count = 0
        false_qids = list(false_result_data.keys())
        iteration_dataset = iteration_dataset[iteration_dataset['qid'].isin(false_qids)]
        document_retrieval_results, retrieval_results = {}, []
        iteration_document_results, page_retrieval_results, iteration_generate_results = {}, {}, {}
        
        document_retrieval_results['results'], page_retrieval_results["results"] = self.change_last_retrieval_result(data)
        document_retrieval_results, page_retrieval_results = self.iteration_evaluation(document_retrieval_results, page_retrieval_results, iteration_dataset)

        page_retrieval_results['results'] = {qid:passages[:5] for qid,passages in page_retrieval_results['results'].items()}

        suggested_queries_results = {}
        while self.current_count < self.max_iteration:
            self.output_dir = os.path.join(origin_output_dir, "iteration" + str(self.current_count))
            os.makedirs(self.output_dir, exist_ok=True)

            print(f"Current Iteration: {self.current_count}")
            print(f"Lenght of iteration_dataset: {len(iteration_dataset)}")
            
            suggested_queries = self.iterative_check(iteration_dataset, page_retrieval_results['results'])
                
            for qid, data in suggested_queries.items():
                new_query = data['suggested_query']
                print(f"QID: {qid}, New Query: {new_query}")
                is_answerable = data['is_answerable']
                iteration_dataset.loc[iteration_dataset['qid'] == qid, 'question'] = new_query
                iteration_dataset.loc[iteration_dataset['qid'] == qid, 'is_answerable'] = is_answerable

                if qid in suggested_queries_results:
                    suggested_queries_results[qid].append(data)
                else:
                    suggested_queries_results[qid] = [data]

                page_retrieval_results['results'][qid] = data['relevant_passages']

            true_iteration_dataset = iteration_dataset[iteration_dataset['is_answerable'] == True]
            iteration_dataset = iteration_dataset[iteration_dataset['is_answerable'] == False]

            if len(true_iteration_dataset) > 0:
                document_retrieval_results['results'], page_retrieval_results = self.change_retrieval_result(page_retrieval_results)
                true_document_results, true_passage_results = self.iteration_evaluation(document_retrieval_results, page_retrieval_results, true_iteration_dataset)
                if self.do_generate is False:
                    self.save_retrieval_result_not_generate(document_retrieval_results, page_retrieval_results, true_iteration_dataset, "true")
                else:
                    true_iteration_dataset['question'] = self.dataset['question']
                    true_iteration_generate_results = self.generate_answers(true_passage_results['results'], true_iteration_dataset)
                    self.save_generated_results(true_iteration_generate_results, true_document_results, true_passage_results, "true")
                    self.save_results(true_iteration_generate_results, true_document_results, true_passage_results, suggested_queries_results, true_iteration_dataset, "true")

            iteration_document_results, iteration_passage_results, _ = self.hierarchical_retrieval(iteration_dataset)

            for qid in iteration_passage_results['results']:                
                page_retrieval_results['results'][qid] += iteration_passage_results['results'][qid][:5]

            document_retrieval_results['results'], page_retrieval_results = self.change_retrieval_result(page_retrieval_results)

            false_document_results, false_passage_results = self.iteration_evaluation(document_retrieval_results, page_retrieval_results, iteration_dataset)
            self.save_retrieval_result_not_generate(false_document_results, false_passage_results, iteration_dataset, "false")
            all_document_results, all_passage_results = self.iteration_evaluation(document_retrieval_results, page_retrieval_results, self.dataset)
            self.save_retrieval_result_not_generate(all_document_results, all_passage_results, self.dataset, "all")

            iteration_dataset['question'] = self.dataset['question']
            self.current_count += 1

        # False case는 iteration이 끝났을 경우에만 추론
        if self.current_count == self.max_iteration and self.max_iteration != 0:
            print("Reaching the maximum number of iterations.")
            print("Saving the final results.")