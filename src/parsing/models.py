from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: str = Field(description="the answer to the question")

    def to_dict(self):
        return {
            "answer": self.answer,
        }


class ReasoningAnswer(BaseModel):
    answer: str = Field(description="the answer to the question")
    reasoning: str = Field(description="the reasoning behind the answer")

    def to_dict(self):
        return {
            "answer": self.answer,
            "reasoning": self.reasoning,
        }


class ThinkingAnswer(BaseModel):
    """
    Represents an AI-generated response that includes both a reasoning process
    and the final answer. This structure encourages the model to articulate
    its thought process before providing a direct response.

    Attributes:
        thinking (str): A step-by-step reasoning or thought process leading
                       to the final answer. This field should capture
                       intermediate deductions, considerations, and logical
                       steps taken to arrive at the conclusion.

        answer (str): The final response to the given question, derived
                      from the preceding thinking process.
    """

    thinking: str = Field(description="Step-by-step reasoning process leading to the final answer.")
    answer: str = Field(description="The final answer derived from the thinking process.")

    def to_dict(self):
        """
        Converts the instance to a dictionary format.

        Returns:
            dict: A dictionary containing the 'thinking' and 'answer' fields.
        """
        return {
            "thinking": self.thinking,
            "answer": self.answer,
        }


class Feedback(BaseModel):
    feedback: str = Field(description="the feedback for the answer")
    score: int = Field(description="the score for the answer")


class Summary(BaseModel):
    summary: str = Field(description="the summary of the provided texts")


class Statements(BaseModel):
    statements: list[str] = Field(description="list of statements")


class StatementResult(BaseModel):
    statement: str = Field(description="The statement being evaluated")
    verdict: str = Field(description="The verdict on the statement (yes/no)")
    explanation: str = Field(description="Explanation for the verdict")


class ResultsResponse(BaseModel):
    results: list[StatementResult] = Field(description="List of statement evaluations")


class AnswerRelevanceResultResponse(BaseModel):
    questions: list[str] = Field(description="List of questions")
    noncommittal: bool = Field(description="Whether the answer is noncommittal")


class ContextRelevanceResultResponse(BaseModel):
    relevant_sentences: list[str] = Field(description="List of relevant sentences")
    irrelevant_sentences: list[str] = Field(description="List of irrelevant sentences")


class ParaphrasedGroundTruth(BaseModel):
    paraphrased: list[str] = Field(description="List of paraphrased versions of the ground truth")


class WhitespaceInjectionResponse(BaseModel):
    processed_text: str = Field(description="The text with injected whitespace")


class TableText(BaseModel):
    table_texts: list[str] = Field(description="The parts of the text that contains the tables unchanged")


class TableDescription(BaseModel):
    description: str = Field(description="the description of the provided table")


class TableMarkdown(BaseModel):
    markdown: str = Field(description="the markdown representation of the table")


class FlowchartDescription(BaseModel):
    description: str = Field(description="the description of the provided flowchart")


class TextInFlowchartPage(BaseModel):
    text: str = Field(description="the text present outside of flowchart")
