from typing import Dict, Any, List, Tuple
import re
from openai import OpenAI
import os
import json
from utils.utils import run_program
import numpy as np
from sigfig import round as sigfig_round

class Generator:
    def __init__(self, args):
        self.args = args
        self.client = OpenAI()
        self.openai_model_name = args["openai_model_name"]
        self.temp = args["temp"]
        self.use_gpt_acc = args["use_gpt_acc"]
        
        self.openai_key = os.environ["OPENAI_API_KEY"]
        
        # Prompt templates
        self.cot_system_prompt = (
            "You are a financial expert, you are supposed to answer the given question based on the provided financial document context. "
            "You need to first think through the problem step by step, documenting each necessary step. "
            "Then you are required to conclude your response with the final answer in your last sentence as 'Therefore, the answer is {final answer}'. "
            "The final answer should be a numeric value and No text should be added after the final answer."
        )
        
        self.text_cot_system_prompt = (
            "You are a financial expert, you are supposed to answer the given question based on the provided financial document context. "
            "You need to first think through the problem step by step, documenting each necessary step. "
            "Then you are required to conclude your response with the final answer in your last sentence as 'Therefore, the answer is {final answer}'. "
        )
        
        self.text_direct_system_prompt = (
            "You are a financial expert, you are supposed to answer the given question directly. "
            "Write the answer concisely in 1~2 sentences."
        )
        
        self.numeracy_direct_system_prompt = (
            "You are a financial expert, you are supposed to answer the given question directly. "
            "Do not include units such as thousands, millions, or billions, or mention currency if the question specifically requests them to be excluded. "
            "Do not use commas for thousand separators in the answer. "
            "You must just give a answer without any other reasons and respond with 'The answer is {final answer}.'. "
        )

        self.pot_system_prompt = """
You are a financial expert, you are supposed to generate a Python program to answer the given question based on the provided financial document context. The returned value of the program is supposed to be the answer. 
```python
def solution():
    # Define variables name and value based on the given context
    guarantees = 210
    total_exposure = 716

    # Do math calculation to get the answer
    answer = (guarantees / total_exposure) * 100

    # return answer
    return answer
```
"""

        self.pot_user_prompt_postfix = """Please generate a Python program to answer the given question. The format of the program should be the following:
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

    def process_single_pot_output(self, output: str) -> Tuple[str, str]:
        """Process POT output"""
        # Initial input validation
        if not output or "argparse" in output:
            return "", ""

        # Initial function name and body
        function_name = "solution"
        processed_output = ""

        # Extract code block and remove markers
        tmp = re.findall(r"```python(.*?)```", output, re.DOTALL)
        if len(tmp) > 0:
            processed_output = tmp[0].strip("\n")
        else:
            tmp = re.findall(r"```(.*?)```", output, re.DOTALL)
            if len(tmp) > 0:
                processed_output = tmp[0].strip("\n")
            else:
                # If no code block, use entire output and remove markers
                processed_output = re.sub(r"```", "", output).strip()

        # Handle numbers with commas
        processed_output = re.sub(r"(?<=\d),(?=\d)", "", processed_output)

        # Check if function definition exists
        if not re.search(r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(", processed_output):
            # If no function definition, move entire code inside function
            lines = processed_output.split("\n")
            indented_lines = ["    " + line for line in lines]
            processed_output = "def solution():\n" + "\n".join(indented_lines)
        else:
            # If function definition exists, extract function name
            match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", processed_output)
            if match:
                function_name = match.group(1)

        # Clean up function body (including marker removal)
        processed_output = re.sub(r"```", "", processed_output).strip()

        return function_name, processed_output

    def extract_cot_answer(self, response: str) -> str:
        """Extract final answer from CoT response"""
        splited = response.split("Therefore, the answer is")
        if len(splited) == 1:
            return ""
        return splited[-1].lstrip(" ").rstrip('. "')

    def extract_direct_answer(self, response: str, is_numeric_question: bool) -> str:
        """Extract final answer from direct response"""
        if is_numeric_question:
            splited = response.split("The answer is")
            if len(splited) == 1:
                return ""
            return splited[-1].lstrip(" ").rstrip('. "')
        else:
            return response

    def extract_pot_answer(self, response: str, answer_type: str = "cot") -> Tuple[str, str]:
        """Extract final answer from POT response"""
        is_numeric_question = answer_type == "pot"
        
        if not is_numeric_question:
            return response, None
            
        def_name, pot_code = self.process_single_pot_output(response)
        extracted = run_program(pot_code, def_name)
        return extracted, pot_code

    def calculate_numeric_accuracy(self, answer: str, generated: str) -> float:
        """Calculate numeric accuracy
        
        Args:
            answer (str): Ground truth answer string (e.g., '70.2%', '6.4%')
            generated (str): Generated answer string (e.g., '70.20335985853228', '6.41025641025641')
            
        Returns:
            float: Accuracy score (1.0 or 0.0)
        """
        try:
            # Remove percentage signs and commas
            answer = answer.replace("%", "").replace(",", "")
            generated = generated.replace("%", "").replace(",", "")
            
            # Convert strings to float
            answer_float = float(answer)
            generated_float = float(generated)
            
            # Prevent division by zero
            if answer_float == 0:
                return 1.0 if generated_float == 0 else 0.0
            
            # Check number of significant figures in answer
            if "." in answer:
                # For decimal numbers
                sig_figs = len(answer.replace(".", "").lstrip("0"))
            else:
                # For whole numbers
                sig_figs = len(answer.lstrip("0"))
            
            # Round generated answer to same number of significant figures
            rounded_generated = sigfig_round(generated_float, sigfigs=sig_figs)
            
            # Compare using numpy.isclose (considering both relative and absolute error)
            return 1.0 if np.isclose(answer_float, rounded_generated, rtol=1e-10, atol=1e-10) else 0.0
            
        except Exception as e:
            print(f"Error in calculate_numeric_accuracy: {e}")
            return 0.0

    def calculate_gpt_accuracy(self, question: str, answer: str, generated: str, context: str) -> Dict[str, Any]:
        """Calculate accuracy using GPT"""
        system_prompt = """You are a financial expert, you are supposed to evaluate if the generated answer is correct based on the given context and question.
You need to respond with a JSON object with the following format:
{
    "score": 1.0 or 0.0,
    "reason": "reason for the score"
}"""

        user_prompt = f"""Context: {context}
Question: {question}
Answer: {answer}
Generated Answer: {generated}

Please evaluate if the generated answer is correct based on the given context and question."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.openai_model_name,
            messages=messages,
            temperature=0.0,
        )

        try:
            result = json.loads(response.choices[0].message.content)
            return {
                "score": result.get("score", 0.0),
                "reason": result.get("reason", ""),
                "prompt": user_prompt,
                "generated": response.choices[0].message.content
            }
        except:
            return {
                "score": 0.0,
                "reason": "Failed to parse GPT response",
                "prompt": user_prompt,
                "generated": response.choices[0].message.content
            }

    def generate_answer(self, question: str, retrieved_passages: List[Dict], answer_type: str = "cot") -> Tuple[str, Dict]:
        """Generate answer for a single question"""
        # Determine if numeric question based on answer_type
        is_numeric_question = answer_type == "pot"
        
        # Construct context
        contexts = []
        for passage in retrieved_passages:
            contexts.append(f"Page {passage['page']} from {passage['source']}:\n{passage['page_content']}")
        context = "\n\n".join(contexts)

        # Construct prompt
        if answer_type == "pot":
            system_prompt = self.pot_system_prompt
            full_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\n{self.pot_user_prompt_postfix}"
        elif answer_type == "direct":
            system_prompt = self.numeracy_direct_system_prompt if is_numeric_question else self.text_direct_system_prompt
            full_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        else:  # cot
            system_prompt = self.cot_system_prompt if is_numeric_question else self.text_cot_system_prompt
            full_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        
        # Generate answer
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
        
        # Extract answer
        if answer_type == "pot":
            extracted_answer, pot_code = self.extract_pot_answer(generated, answer_type)
            metadata = {
                "prompt": full_prompt,
                "generated": generated,
                "pot_code": pot_code,
                "usage": response.usage.to_dict()
            }
        elif answer_type == "direct":
            extracted_answer = self.extract_direct_answer(generated, is_numeric_question)
            metadata = {
                "prompt": full_prompt,
                "generated": generated,
                "usage": response.usage.to_dict()
            }
        else:  # cot
            extracted_answer = self.extract_cot_answer(generated)
            metadata = {
                "prompt": full_prompt,
                "generated": generated,
                "usage": response.usage.to_dict()
            }

        return extracted_answer, metadata

    def evaluate_answer(self, question: str, answer: str, generated: str, context: str, answer_type: str = "cot") -> Dict[str, Any]:
        """Evaluate answer"""
        # Determine if numeric question based on answer_type
        is_numeric_question = answer_type == "pot"
        
        result = {}
        
        if is_numeric_question:
            # Calculate numeric accuracy
            numeric_acc = self.calculate_numeric_accuracy(answer, generated)
            result["numeric_accuracy"] = numeric_acc
            
            # Calculate GPT accuracy
            if self.use_gpt_acc:
                gpt_acc = self.calculate_gpt_accuracy(question, answer, generated, context)
                result["gpt_accuracy"] = gpt_acc
        else:
            # For text questions, only calculate GPT accuracy
            gpt_acc = self.calculate_gpt_accuracy(question, answer, generated, context)
            result["gpt_accuracy"] = gpt_acc
            
        return result

    def cleanup(self):
        """Clean up resources"""
        pass 