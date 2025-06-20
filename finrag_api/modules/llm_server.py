import torch
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import gc

class LLMServer:
    def __init__(self, args):
        self.args = args
        
        # GPU settings
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu_devices"]
            self.device = "cuda"  # Use all available GPUs
        else:
            self.device = "cpu"
            
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self.max_new_tokens = args["max_new_tokens"]
        
    async def initialize(self):
        """Initialize model"""
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Initialize LLM model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args["llm_model_name"],
            )
            
            # Load model distributed across multiple GPUs
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args["llm_model_name"],
                device_map="auto",
                torch_dtype="auto"
            )
            
            self.initialized = True
            print(f"LLM Server initialized (Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}, device_map: {self.model.hf_device_map})")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise e
    
    def create_messages(self, instruction: str, text: str = "") -> List[Dict[str, str]]:
        """Create chat messages"""
        return [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": f"{instruction}\n\n{text}"},
        ]
    
    def prepare_inputs(self, messages: List[Dict[str, str]]):
        """Prepare inputs for model"""
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer([chat_text], return_tensors="pt").to(self.device)
    
    async def generate(self, instruction: str, text: str = "", max_length: int = None) -> str:
        """Generate text"""
        if not self.initialized:
            raise Exception("LLM is not initialized.")
        
        try:
            # Create messages and prepare inputs
            messages = self.create_messages(instruction, text)
            model_inputs = self.prepare_inputs(messages)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens
                )
            
            # Decode results
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
            ]
            generated_text = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            return generated_text
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return ""
    
    def cleanup(self):
        """Clean up resources"""
        if self.model:
            self.model = None
            del self.model
        if self.tokenizer:
            self.tokenizer = None
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"LLM Server resources cleaned up") 