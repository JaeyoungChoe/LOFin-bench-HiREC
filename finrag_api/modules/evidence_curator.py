from typing import Dict, Any, List
from .llm_server import LLMServer
from utils.prompts import ENHANCED_PROMPT_3, FULL_PROMPT

class EvidenceCurator:
    def __init__(self, llm_server: LLMServer, args):
        self.args = args
        self.llm = llm_server
        self.max_contexts = args["max_contexts"]
        self.max_relevant_ids = args["max_relevant_ids"]
        
    def parse_output(self, output_str: str) -> Dict[str, Any]:
        """Parse LLM output"""
        # Remove special characters and handle newlines
        output_str = output_str.replace("\n", '')
        lines = output_str.strip().split("##")
        
        # Define parsing markers
        markers = {
            "is_answerable": "is_answerable:",
            "missing_information": "missing_information:",
            "answer": "answer:",
            "answerable_doc_ids": "answerable_doc_ids:",
            "refined_query": "refined_query:"
        }
        
        # Result dictionary
        parsed_data = {}
        
        # Parse each line
        for line in lines:
            line = line.strip()
            for key, marker in markers.items():
                if marker in line:
                    parsed_data[key] = line.split(marker)[1].strip()
        
        # Process answerable_doc_ids
        relevant_ids = []
        answerable_doc_ids = parsed_data.get("answerable_doc_ids", "")
        if answerable_doc_ids:
            trimmed_ids = answerable_doc_ids.strip("[] \n\t")
            if trimmed_ids:
                relevant_ids = [int(x.strip()) for x in trimmed_ids.split(",") if x.strip().isdigit()]
        parsed_data["answerable_doc_ids"] = relevant_ids[:self.max_relevant_ids]
        
        return parsed_data
        
    async def curate_evidence(self, query: str, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Curate evidence from retrieved pages"""
        try:
            # Use only up to max_contexts pages
            pages = pages[:self.max_contexts]
            
            # Construct page information
            passages = '\n'.join([
                f"Context{idx} (ID: {idx}): " 
                "Title is "+ page["source"] +". Content is following.\n" + page['page_content'] 
                for idx, page in enumerate(pages)  # Use pages list
            ])
            
            # Construct prompt
            inputs = ("Context: ", passages, "Question: ", query)
            full_prompt = FULL_PROMPT.format(
                task=ENHANCED_PROMPT_3["task_description"],
                instructions='\n '.join(ENHANCED_PROMPT_3["instructions"]),
                inputs=inputs,
                output_format=ENHANCED_PROMPT_3["output_format"]
            )
            
            # Generate LLM response
            response = await self.llm.generate(full_prompt, "")
            
            # Parse response
            try:
                parsed_output = self.parse_output(response)
                
                # Check if answerable
                is_answerable = parsed_output.get("is_answerable", "").lower()
                is_answerable = True if is_answerable in ["true", "yes", "answerable"] else False
                
                # Extract relevant pages
                relevant_pages = []
                answerable_doc_ids = parsed_output.get("answerable_doc_ids", [])
                if answerable_doc_ids:
                    try:
                        relevant_pages = [pages[idx] for idx in answerable_doc_ids]
                    except:
                        relevant_pages = []
                
                # Construct result
                result = {
                    "is_answerable": is_answerable,
                    "answer": parsed_output.get("answer", ""),
                    "relevant_pages": relevant_pages,
                    "missing_information": parsed_output.get("missing_information", ""),
                    "refined_query": parsed_output.get("refined_query", ""),
                }
                
                # Additional processing for unanswerable cases
                if not is_answerable and len(answerable_doc_ids) >= self.max_relevant_ids:
                    result["is_answerable"] = True
                
                return result
                
            except Exception as e:
                print(f"Error parsing response: {str(e)}")
                return {
                    "is_answerable": False,
                    "answer": "",
                    "relevant_pages": [],
                    "missing_information": f"Parsing error: {str(e)}",
                    "refined_query": "",
                    "input_prompt": full_prompt,
                    "output_answer": response
                }
            
        except Exception as e:
            print(f"Error during evidence curation: {str(e)}")
            return {
                "is_answerable": False,
                "relevant_pages": [],
                "missing_information": str(e),
                "refined_query": "",
                "input_prompt": "",
                "output_answer": ""
            }

    def cleanup(self):
        """Clean up resources"""
        if self.llm:
            self.llm.cleanup() 