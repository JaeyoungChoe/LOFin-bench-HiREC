QUERY_REWRITING_PROMPT = """The following question can be solved in three steps. Answer what information is needed to perform each of the three steps below.
## Process steps
Document retrieval: The step of retrieval for the titles of documents related to the question.
Passage selection: The step of identifying the most relevant passage within the content of the retrieved documents.
Answer generation: The step of generating the required answer based on the given contexts.

## Question: {Question}

## Output format 
[Document retrieval]:
[Passage selection]:
[Answer generation]: 

"""
example = """Make sure the rewritten question explicitly states the search intent and clarifies which types of documents (or formats) are being sought.
Indicate the information desired from these documents (titles, key findings, major issues, etc.).
Do not alter the main topic of the original question, but rephrase it with greater specificity and clarity to facilitate document retrieval."""

QUERY_REWRITING_PROMPT_2 = """As a financial expert, you need to revise the given question to find document titles related to it. Analyze the question, determine the necessary information, and rewrite the query accordingly.
Make sure the rewritten question explicitly states the search intent and clarifies which types of documents (or formats) are being sought.
Indicate the information desired from these documents (titles, key findings, major issues, etc.).
Do not alter the main topic of the original question, but rephrase it with greater specificity and clarity to facilitate document retrieval.

## Question: {Question}

### Output format
## Query: {Rewritten query}"""

QUERY_REWRITING_PROMPT_3 = """You are an AI that rewrites user questions about financial topics into concise meta-focused queries. 
1) Identify the key financial terms or metrics in the question. 
2) Determine which type of documents typically contain those terms. 
3) Transform the user’s question into a short query referencing the financial terms and the relevant documents. 
4) Do not reveal the transformation process or provide examples. 
5) Output only the final rewritten query.

## Question: {Question}

### Output format
## Query: {Rewritten query}"""

ENHANCED_PROMPT_3 = {
  "task_description": "You are a financial expert. Evaluate the provided context to determine if it contains enough information to answer the given question.",
  "instructions": [
    "1. Read the context carefully and decide if it contains enough information to answer the question.",
    "2. If it is answerable, set 'is_answerable: answerable' and provide the answer in 'answer'.",
    "3. If it is not answerable, set 'is_answerable: unanswerable'.",
    " - List the relevant document IDs in 'answerable_doc_ids' in order of relevance (from most to least relevant)."
    " - Explain what specific information is missing in 'missing_information.'",
    " - Provide a concise question in 'refined_query' to search for exactly that missing information.",
    "4. Output your result strictly in the specified format below using '##' headers."
  ],
  "input_format": [
    "Context: {Insert the context here}",
    "Question: {Insert the question here}"
  ],
  "output_format": (
    "## is_answerable: answerable or unanswerable",
    "## missing_information: If 'unanswerable', specify the details or data needed; if 'answerable', None",
    "## answer: If 'answerable', provide the answer; if 'unanswerable', then None",
    "## answerable_doc_ids: Provide a list of document IDs that contain relevant information (e.g., [1, 2]). If none, use []",
    "## refined_query: If 'unanswerable', provide a refined question to obtain the missing information;"
  )
}

FULL_PROMPT = """
### Instruction
{task}
{instructions}

### Inputs
{inputs}

### Output format
{output_format}
"""