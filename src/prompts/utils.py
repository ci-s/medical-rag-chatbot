from question import QUESTION_PROMPT


def create_question_prompt(retrieved_documents, background, query):
    documents_str = "".join([f"{document}\n" for document in retrieved_documents])
    return QUESTION_PROMPT.format(retrieved_documents=documents_str, background=background, query=query)
