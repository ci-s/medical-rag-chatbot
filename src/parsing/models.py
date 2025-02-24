from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: str = Field(description="the answer to the question")

    def to_dict(self):
        return {
            "answer": self.answer,
        }


class ExtendedAnswer(BaseModel):
    answer: str = Field(description="the answer to the question")
    reasoning: str = Field(description="the reasoning behind the answer")

    def to_dict(self):
        return {
            "answer": self.answer,
            "reasoning": self.reasoning,
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
