import signal
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP, InvalidOperation

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

def dummy_print(*args, **kwargs):
    pass

def preprocess_answer(answer):
    answer = answer.strip()

    if answer.startswith("$"):
        answer = answer.lstrip("$ ")

    if answer.endswith("%"):
        answer = answer.rstrip("% ")

    return answer

def run_program(program_code, def_name="solution"):
    try:
        namespace = {"print": dummy_print}
        exec(program_code, namespace)
        executed = namespace[def_name]()
        return str(executed)
    except (TimeoutException, Exception, OverflowError):
        return ""

def calculate_numeric_accuracy(num1, num2):
    # function to count decimal places
    def count_decimal_places(num):
        try:
            num_str = str(Decimal(num)).rstrip("0").split(".")
            return len(num_str[1]) if len(num_str) > 1 else 0
        except InvalidOperation:
            raise ValueError(f"Invalid input for decimal conversion: {num}")

    # function to count significant digits
    def significant_digits(num):
        try:
            num_str = str(Decimal(num)).replace(".", "").lstrip("0").rstrip("0")
            return len(num_str)
        except InvalidOperation:
            raise ValueError(f"Invalid input for decimal conversion: {num}")

    if num1 == num2:
        return 1.0

    num1 = preprocess_answer(num1)
    num2 = preprocess_answer(num2)

    try:
        dec_places1 = count_decimal_places(num1)
        dec_places2 = count_decimal_places(num2)
        sig_digits1 = significant_digits(num1)
        sig_digits2 = significant_digits(num2)
    except ValueError as e:
        print(e)
        return 0.0 

    if Decimal(num1) == Decimal(num2):
        return 1.0 

    if Decimal(num1) == 0 or Decimal(num2) == 0:
        return 0.0

    # check to signicant digits
    if max(sig_digits1, sig_digits2) < 2:
        return 0.0

    if dec_places1 == dec_places2:
        return 0.0

    # separate larger and smaller numbers
    larger, smaller = (num1, num2) if dec_places1 > dec_places2 else (num2, num1)
    target_places = min(dec_places1, dec_places2)

    try:
        rounded = Decimal(larger).quantize(
            Decimal("1e-{0}".format(target_places)), rounding=ROUND_HALF_UP
        )
        truncated = Decimal(larger).quantize(
            Decimal("1e-{0}".format(target_places)), rounding=ROUND_FLOOR
        )

        smaller_decimal = Decimal(smaller).quantize(
            Decimal("1e-{0}".format(target_places))
        )
        return (
            1.0 if rounded == smaller_decimal or truncated == smaller_decimal else 0.0
        )
    except InvalidOperation as e:
        return 0.0

gpt_acc_prompt_template = """You are a highly knowledgeable expert and teacher in the finance domain.

You are reviewing a student’s answers to financial questions.

You are given the context, the question, the student’s answer and the student’s explanation and the ground−truth answer.

Please use the given information and refer to the ground−truth answer to determine if the student’s answer is correct.

The input information is as follows:

context: `{}`
question: `{}`
ground−truth answer: `{}`
student’s answer: `{}`

Please respond directly as either ’correct’ or ’incorrect’."""


def calculate_gpt_accuracy_text(
    client,
    question: str,
    reference: str,
    candidate: str,
    contexts: str,
):
    prompt = gpt_acc_prompt_template.format(contexts, question, reference, candidate)
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
    )
    generated = response.choices[0].message.content
    score = 0 if "incorrect" in generated.strip().lower() else 1
    return {"prompt": prompt, "generated": generated, "score": score}
