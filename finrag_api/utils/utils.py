import torch
import torch.nn as nn
from typing import Set, Tuple, Union, List
from transformers import BartTokenizer, BartForConditionalGeneration
import traceback
import numpy as np
import re
import string
import bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from scipy.optimize import linear_sum_assignment
import signal
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP, InvalidOperation
import math
from pathlib import Path


nltk.download("wordnet")


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
    """Preprocess answer by removing unnecessary characters"""
    answer = answer.strip()

    if answer.startswith("$"):
        answer = answer.lstrip("$ ")

    if answer.endswith("%"):
        answer = answer.rstrip("% ")

    return answer


def run_program(program_code, def_name="solution"):
    """Execute program code and return result"""
    try:
        namespace = {"print": dummy_print}
        exec(program_code, namespace)
        executed = namespace[def_name]()
        return str(executed)
    except (TimeoutException, Exception, OverflowError):
        return ""


def calculate_numeric_accuracy(num1, num2):
    """Calculate numeric accuracy between two numbers"""
    # Function to count decimal places
    def count_decimal_places(num):
        try:
            num_str = str(Decimal(num)).rstrip("0").split(".")
            return len(num_str[1]) if len(num_str) > 1 else 0
        except InvalidOperation:
            raise ValueError(f"Invalid input for decimal conversion: {num}")

    # Function to count significant digits
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
        # Get decimal places and significant digits
        dec_places1 = count_decimal_places(num1)
        dec_places2 = count_decimal_places(num2)
        sig_digits1 = significant_digits(num1)
        sig_digits2 = significant_digits(num2)
    except ValueError as e:
        print(e)
        return 0.0  # Return False for invalid input

    # Check if values are identical
    if Decimal(num1) == Decimal(num2):
        return 1.0  # Return True if numeric values match

    if Decimal(num1) == 0 or Decimal(num2) == 0:
        return 0.0

    # Check if significant digits are at least 2
    if max(sig_digits1, sig_digits2) < 2:
        return 0.0  # Skip if significant digits are less than 2

    # Compare decimal places
    if dec_places1 == dec_places2:
        return 0.0  # No need to process if decimal places are the same

    # Identify larger and smaller decimal places
    larger, smaller = (num1, num2) if dec_places1 > dec_places2 else (num2, num1)
    target_places = min(dec_places1, dec_places2)

    try:
        # Handle rounding and truncation
        rounded = Decimal(larger).quantize(
            Decimal("1e-{0}".format(target_places)), rounding=ROUND_HALF_UP
        )
        truncated = Decimal(larger).quantize(
            Decimal("1e-{0}".format(target_places)), rounding=ROUND_FLOOR
        )

        # Check if processed result matches the number with fewer decimal places
        smaller_decimal = Decimal(smaller).quantize(
            Decimal("1e-{0}".format(target_places))
        )
        return (
            1.0 if rounded == smaller_decimal or truncated == smaller_decimal else 0.0
        )
    except InvalidOperation as e:
        return 0.0


class BARTScorer:
    def __init__(
        self, device="cuda", max_length=1024, checkpoint="facebook/bart-large-cnn"
    ):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(
            reduction="none", ignore_index=self.model.config.pad_token_id
        )
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """Load model from paraphrase finetuning"""
        if path is None:
            path = Path(__file__).parent.parent / "models" / "bart.pth"
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """Score a batch of examples"""
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i : i + batch_size]
            tgt_list = tgts[i : i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    src_tokens = encoded_src["input_ids"].to(self.device)
                    src_mask = encoded_src["attention_mask"].to(self.device)

                    tgt_tokens = encoded_tgt["input_ids"].to(self.device)
                    tgt_mask = encoded_tgt["attention_mask"]
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f"source: {src_list}")
                print(f"target: {tgt_list}")
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        """Score with multiple references"""
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """Test the scorer"""
        src_list = [
            "This is a very good idea. Although simple, but very insightful.",
            "Can I take a look?",
            "Do not trust him, he is a liar.",
        ]

        tgt_list = ["That's stupid.", "What's the problem?", "He is trustworthy."]

        print(self.score(src_list, tgt_list, batch_size))


def extract_numeric_value(text):
    """Extract numeric values from text"""
    try:
        cleaned_text = text.replace(",", "").strip("% ")
        numeric_string = re.findall(r"[-+]?\d*\.\d+|\d+", cleaned_text)
        if numeric_string:
            # Return first number if multiple numbers are found
            return numeric_string
        return None
    except Exception as e:
        print(f"Error in extract_numeric_value: {e}")
        return None


def scale_to_num(scale):
    """Convert scale words to numeric values"""
    scale = scale.lower()
    num = 1
    if "hundred" in scale:  # hundred
        num = 100
    elif "thousand" in scale:  # thousand
        num = 1000
    elif "million" in scale:  # million
        num = 1000000
    elif "billion" in scale:  # billion
        num = 1000000000
    elif "percent" in scale:  # percent
        num = 0.01
    return num


def extract_one_num_from_str(s):
    """Extract a single number from string"""
    s = _clean_num(s)
    r_num = r"([+-]?\d+(\.\d+)?)|([+-]?\.\d+)"
    groups = re.findall(r_num, s)
    if len(groups) == 0:
        return None
    num = groups[0][0]
    if num == "":
        return None
    if "." in num:
        return float(num)
    return int(num)


EXCLUDE_IN_NUM = "'\"\\$€£¥%(),[]"


def _clean_num(text: str):
    """Clean text by removing excluded characters"""
    return "".join([ch for ch in str(text) if ch not in EXCLUDE_IN_NUM])


def is_number(text: str) -> bool:
    """Check if text is a valid number"""
    try:
        words = " ".join([_clean_num(w) for w in text.split()]).split()
        if len(words) == 0:
            return False
        num = float(words[0])
        if np.isnan(num):
            return False
        if len(words) >= 2:
            if scale_to_num(words[1]) == 1:
                return False
        return True
    except ValueError:
        return False


def negative_num_handle(x):
    """Handle negative numbers in parentheses"""
    all = re.findall("(\([\d.\s]+\))", x.strip())
    if len(all) > 0:
        return -1
    return 1


def percent_num_handle(x):
    """Handle percentage numbers"""
    all = re.findall("([\d.\s]+%)", x.strip())
    if len(all) > 0:
        return 0.01
    return 1


def word_scale_handle(x):
    """Handle word scales (e.g., million, billion)"""
    iter = re.finditer("([\d.]+\s?[a-zA-Z]+)", x)
    for one in iter:
        text = one.group(0).lower()
        scale_val = scale_to_num(text)
        return scale_val
    return 1


def to_number(text: str) -> float:
    """Convert text to number"""
    num = extract_one_num_from_str(text)
    scale_val = word_scale_handle(text)
    negative_flag = negative_num_handle(text)
    percent_flag = percent_num_handle(text)
    if num is not None:
        return round(num * scale_val * negative_flag * percent_flag, 4)
    return None


def remove_articles(text: str) -> str:
    """Remove articles from text"""
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def white_space_fix(text: str) -> str:
    """Fix whitespace in text"""
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def remove_punc(text: str) -> str:
    """Remove punctuation from text"""
    if not is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def lower(text: str) -> str:
    """Convert text to lowercase"""
    return text.lower()


def tokenize(text: str) -> List[str]:
    """Tokenize text"""
    return re.split(" ", text)


def normalize_number(text: str) -> str:
    """Normalize number in text"""
    if is_number(text):
        return str(to_number(text))
    else:
        return text


def normalize_answer(text: str) -> str:
    """Normalize answer text by removing punctuation, articles and extra whitespace"""
    parts = [
        white_space_fix(remove_articles(normalize_number(remove_punc(lower(token)))))
        for token in tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


STRIPPED_CHARACTERS = string.punctuation + "".join(["'", "'", "´", "`", "_"])


def ws_tokenize(text):
    """Tokenize text with basic whitespace cleaning"""
    text = text.strip().lower()
    if not text:
        return []
    text = white_space_fix(text)
    tokens = text.split()
    tokens = [token.strip(STRIPPED_CHARACTERS) for token in tokens]
    return tokens


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    """Convert answer to bags of tokens"""
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    """Compute F1 score between predicted and gold bags"""
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """Align predicted and gold bags and compute maximum metric values"""
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def calculate_em_and_f1(reference, candidate):
    """Calculate exact match and F1 score"""
    reference = reference.rstrip("% ")
    candidate = candidate.rstrip("% ")

    predicted_bags = _answer_to_bags(candidate)
    gold_bags = _answer_to_bags(reference)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(
        gold_bags[0]
    ):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def calculate_bert_score(reference, candidate):
    """Calculate BERT score"""
    reference_ = [reference]
    candidate_ = [candidate]
    _, _, f1 = bert_score.score(candidate_, reference_, lang="en")
    return f1.item()


def calculate_bleu(reference, candidate):
    """Calculate BLEU score"""
    reference_ = [reference.split()]
    candidate_ = candidate.split()
    return sentence_bleu(reference_, candidate_)


def calculate_rouge(reference, candidate):
    """Calculate ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {key: value.fmeasure for key, value in scores.items()}


def calculate_meteor(reference, candidate):
    """Calculate METEOR score"""
    reference_ = reference.split()
    candidate_ = candidate.split()
    return meteor_score([reference_], candidate_)


def calculate_bart_score(bart_scorer, reference, candidate):
    """Calculate BART score"""
    reference_ = [reference]
    candidate_ = [candidate]
    score = bart_scorer.score(candidate_, reference_)
    return score[0]


def calculate_gpt_accuracy(
    client,
    question: str,
    reference: str,
    candidate: str,
    openai_model_name,
    temperature,
):
    """Calculate GPT accuracy with detailed evaluation criteria"""
    prompt = """The following is a question in the financial domain. Compare the Gold Answer and the Model Answer to evaluate if the two answers match.

Output 'True' if:
The Model Answer conveys the core meaning or conclusion of the Gold Answer, even if certain specific details (e.g., numerical values, additional context, or supporting explanations) are omitted.
For classification-type questions (e.g., Yes/No, or categorical answers), the Model Answer must align with the overall conclusion provided in the Gold Answer. If the conclusion matches, minor omissions of context or details do not affect the judgment.
For numerical answers:
If both the Gold Answer and Model Answer can be represented as decimals:
- If the Gold Answer and Model Answer have the same number of decimal places, the values must match exactly to be considered equivalent.
- If the number of decimal places differs, the numerical values are considered equivalent if rounding or truncation of the number with more decimal places matches the one with fewer decimal places.
If this adjustment is not possible and the values differ, output 'False.'

Output 'False' if:
The Model Answer provides a conclusion that contradicts the core meaning of the Gold Answer.
The omission of critical information substantially changes the interpretation of the question or answer, leading to a different understanding.
For numerical answers:
If the Gold Answer and Model Answer have the same number of decimal places and the values do not match exactly, the answers are considered unequal.
If the values cannot be made equivalent through rounding or truncation to the precision of the shorter decimal, the answers are also considered unequal."""
    question_prompt = f"Question: {question}"
    reference = reference.rstrip("% ")
    gold_answer_prompt = f"Gold Answer: {reference}"
    candidate = candidate.rstrip("% ")
    model_answer_prompt = f"Model Answer: {candidate}"
    prompt += f"\n\n{question_prompt}\n{gold_answer_prompt}\n{model_answer_prompt}"

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    response = client.chat.completions.create(
        model=openai_model_name,
        messages=messages,
        temperature=temperature,
    )
    generated = response.choices[0].message.content
    score = 1 if "True" in generated else 0
    return {"generated": generated, "score": score}


def calculate_gpt_accuracy_text(
    client,
    question: str,
    reference: str,
    candidate: str,
    openai_model_name,
    temperature=0.1,
):
    """Calculate GPT accuracy for text answers"""
    prompt = "The following is a question in the financial domain. Compare the Gold Answer and the Model Answer to evaluate if the two answers match. Output 'True' if the answers are the same and 'False' if they are different."
    question_prompt = f"Question: {question}"
    gold_answer_prompt = f"Gold Answer: {reference}"
    model_answer_prompt = f"Model Answer: {candidate}"
    prompt += f"\n\n{question_prompt}\n{gold_answer_prompt}\n{model_answer_prompt}"

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    response = client.chat.completions.create(
        model=openai_model_name,
        messages=messages,
        temperature=temperature,
    )
    generated = response.choices[0].message.content
    score = 1 if "True" in generated else 0
    return {"generated": generated, "score": score}


gpt_acc_prompt_template = """You are a highly knowledgeable expert and teacher in the finance domain.

You are reviewing a student's answers to financial questions.

You are given the context, the question, the student's answer and the student's explanation and the ground−truth answer.

Please use the given information and refer to the ground−truth answer to determine if the student's answer is correct.

The input information is as follows:

context: `{}`
question: `{}`
ground−truth answer: `{}`
student's answer: `{}`

Please respond directly as either 'correct' or 'incorrect'."""


def calculate_gpt_accuracy_text_1(
    client,
    question: str,
    reference: str,
    candidate: str,
    contexts: str,
):
    """Calculate GPT accuracy with context"""
    prompt = gpt_acc_prompt_template.format(contexts, question, reference, candidate)
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.0,
    )
    generated = response.choices[0].message.content
    score = 0 if "incorrect" in generated.strip().lower() else 1
    return {"prompt": prompt, "generated": generated, "score": score}


def round_up_to_decimal(number, decimals):
    """Round up number to specified decimal places"""
    factor = 10**decimals
    return math.ceil(number * factor) / factor


def within_eps(pred: float, gt: float):
    """Check if prediction is within epsilon range of ground truth"""
    eps = abs(gt) * 0.0015
    if pred >= gt - eps and pred <= gt + eps:
        return 1
    else:
        return 0


def compare_two_numbers(p, gt):
    """Compare two numbers with various tolerance levels"""
    p = preprocess_answer(p)
    gt = preprocess_answer(gt)
    try:
        p = float(p)
        gt = float(gt)
    except Exception:
        return 0

    v1, v2 = max(abs(gt), abs(p)), min(abs(gt), abs(p))
    if (v1 != 0 and v2 != 0) and int(math.log10(v1 / v2)) == math.log10(v1 / v2):
        return 1

    if v2 <= v1 / 50 and within_eps(pred=v2 * 100, gt=v1):
        return 1
    elif v2 <= v1 / 500 and within_eps(pred=v2 * 1000, gt=v1):
        return 1
    elif v2 <= v1 / 50000 and within_eps(pred=v2 * 100000, gt=v1):
        return 1

    if round_up_to_decimal(v1, 3) == round_up_to_decimal(v2, 3):
        return 1

    return within_eps(pred=p, gt=gt)