from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage

GENERATION_EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. Do not reduce the score for differences in phrasing, word order, or synonyms, as long as the response conveys the same meaning as the reference answer.
3. Focus on correctness, accuracy, and factual alignment rather than stylistic differences.
4. After writing a feedback, write a score that is an integer between 1 and 5. Ensure that the score follows the rubric and is not lowered due to wording variations.
5. The output format should be a JSON as follows: 
{{
    "feedback": "write a feedback for criteria", 
    "score": "an integer number between 1 and 5"
    }}
6. Please do not generate any other opening, closing, and explanations. Be sure to output a valid JSON.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual, but lacks a really crucial detail.
Score 5: The response is completely correct, accurate, and factual. Minor rewordings, synonyms, or phrasing differences that do not change the meaning should not lower the score.

###Output:"""
