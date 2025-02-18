from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage

OLD_GENERATION_EVALUATION_PROMPT = """###Task Description:
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


GENERATION_EVALUATION_PROMPT = """
###Task Description:
You are an expert evaluator assessing the quality of a generated answer based on its similarity in meaning to a reference answer. The answers do not need to be identical word-for-word, but they must convey the same essential information.

###Input Information:
	•	Document Page Content (Domain Knowledge):
The following document page provides relevant background information related to the topic. However, it may contain multiple facts or numbers, and not all of them are correct for this specific evaluation.
Do not assume any information from the document page is the correct answer unless it aligns with the reference answer.
{document_page}
	•	Reference Answer (Correct Answer):
This is the correct answer that the generated answer should align with in meaning.
{reference_answer}
	•	Generated Answer:
{generated_answer}

###Evaluation Criteria:
Assess whether the generated answer conveys the same meaning as the reference answer. Consider:
	1.	Key Information: Does it contain the same essential details as the reference answer?
	2.	Paraphrasing Quality: Even if phrased differently, does it preserve the intended meaning?
	3.	Factual Consistency: Does it avoid introducing incorrect details from the document page?
	4.	Completeness: Does it omit or alter key information that changes the meaning?

###Scoring Guidelines (1-5):
	•	5 – Perfect match: The generated answer conveys the same meaning as the reference answer.
	•	4 – Very close: Minor differences in phrasing, but the meaning is intact.
	•	3 – Somewhat similar: Includes partial information, but some key details are missing or slightly altered.
	•	2 – Weak similarity: The answer is loosely related but misses or changes important details.
	•	1 – Poor match: The answer does not convey the intended meaning or contains major inaccuracies.


###Output Format (JSON):
Provide your evaluation as a structured JSON object:
{{
    "feedback": "Provide a concise explanation of how well the generated answer aligns with the reference answer, and where it could be improved.",
    "score": "an integer number between 1 and 5"
    }}
    
Now, evaluate the provided answers and generate the JSON output.
"""
