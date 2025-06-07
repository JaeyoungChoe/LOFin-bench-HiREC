from typing import Dict, Any
from .llm_server import LLMServer
from batch_evaluation.retriever.prompts import QUERY_REWRITING_PROMPT_3

class QueryTransformer:
    def __init__(self, llm_server: LLMServer):
        self.llm = llm_server
        
    async def transform_query(self, query: str) -> str:
        """Transform query"""
        try:
            instruction = QUERY_REWRITING_PROMPT_3.replace("{Question}", query)
            response = await self.llm.generate(instruction, "")
            
            # Extract only the Query: part from response
            if "Query:" in response:
                response = response.split("Query:")[1].strip()
            response = response.replace("## ", "").replace("\n", " ").strip()
            
            return response
            
        except Exception as e:
            print(f"Error during query transformation: {str(e)}")
            return query 