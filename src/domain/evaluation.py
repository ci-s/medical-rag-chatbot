class Feedback:  # TODO: Add question id and/or
    def __init__(self, question_id: int, feedback: str, score: int, generated_answer: str):
        self.question_id = question_id
        self.text = feedback
        self.score = score
        self.generated_answer = generated_answer

    def __repr__(self):
        return f"Feedback(question_id={self.question_id}, feedback={self.text}, score={self.score}, generated_answer={self.generated_answer})"

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "feedback": self.text,
            "score": self.score,
            "generated_answer": self.generated_answer,
        }


class StatementResult(Feedback):
    def __init__(
        self,
        statements: list[str],
        explanations: list[str],
        verdicts: list[str],
        question_id: int,
        score: float,
        feedback: str = "",
        generated_answer: str = "",
    ):
        super().__init__(question_id=question_id, feedback=feedback, score=score, generated_answer=generated_answer)
        self.statements = statements
        self.explanations = explanations
        self.verdicts = verdicts

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "statements": self.statements,
            "explanations": self.explanations,
            "verdicts": self.verdicts,
            "score": self.score,
            "generated_answer": self.generated_answer,
        }


class AnswerRelevanceResult(Feedback):
    def __init__(
        self,
        answer: str,
        generated_questions: list[str],
        noncommittal: int,
        question_id: int,
        score: float,
        feedback: str = "",
        generated_answer: str = "",
    ):
        super().__init__(question_id=question_id, feedback=feedback, score=score, generated_answer=generated_answer)
        self.generated_questions = generated_questions
        self.answer = answer
        self.noncommittal = noncommittal

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "answer": self.answer,
            "generated_questions": self.generated_questions,
            "noncommittal": self.noncommittal,
            "score": self.score,
            "generated_answer": self.generated_answer,
        }


class ContextRelevanceResult(Feedback):
    def __init__(
        self,
        relevant_sentences: str,
        irrelevant_sentences: str,
        question_id: int,
        score: float,
        feedback: str = "",
        generated_answer: str = "",
    ):
        super().__init__(question_id=question_id, feedback=feedback, score=score, generated_answer=generated_answer)
        self.relevant_sentences = relevant_sentences
        self.irrelevant_sentences = irrelevant_sentences

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "relevant_sentences": self.relevant_sentences,
            "irrelevant_sentences": self.irrelevant_sentences,
            "score": self.score,
            "generated_answer": self.generated_answer,
        }


class Stats:
    def __init__(self, recall: float, precision: float):
        self.recall = recall
        self.precision = precision

    def to_dict(self):
        return {"recall": self.recall, "precision": self.precision}
