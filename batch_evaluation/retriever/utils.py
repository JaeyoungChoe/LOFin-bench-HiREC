import signal
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP, InvalidOperation
import json
from typing import Dict, Any

def get_original_dataset_name(qid):
    if qid.startswith("financebench"):
        return "financebench"
    elif qid.startswith("openqa"):
        return "open_secqa"
    return "open_finqa"

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Code execution took too long!")

signal.signal(signal.SIGALRM, timeout_handler)

def calculate_gpt_accuracy(
    client,
    question: str,
    answer: str,
    generated: str,
    context: str,
) -> Dict[str, Any]:
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

    response = client.chat.completions.create(
        model="gpt-4",
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

def calculate_gpt_accuracy_text(
    client,
    question: str,
    reference: str,
    candidate: str,
    contexts: str,
):
    system_prompt = """You are a financial expert, you are supposed to evaluate if the generated answer is correct based on the given context and question.
You need to respond with a JSON object with the following format:
{
    "score": 1.0 or 0.0,
    "reason": "reason for the score"
}"""

    user_prompt = f"""Context: {contexts}
Question: {question}
Answer: {reference}
Generated Answer: {candidate}

Please evaluate if the generated answer is correct based on the given context and question."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.0,
    )

    generated = response.choices[0].message.content
    try:
        result = json.loads(generated)
        score = result.get("score", 0.0)
    except:
        score = 0 if "incorrect" in generated.strip().lower() else 1

    return {"prompt": user_prompt, "generated": generated, "score": score}
