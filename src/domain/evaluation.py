class Feedback:  # TODO: Add question id and/or
    def __init__(self, feedback: str, score: int):
        self.text = feedback
        self.score = score

    def __repr__(self):
        return f"Feedback(feedback={self.text}, score={self.score})"

    def to_dict(self):
        return {"feedback": self.text, "score": self.score}


class StatementResult(Feedback):
    def __init__(
        self, statements: list[str], explanations: list[str], verdicts: list[str], score: float, feedback: str = ""
    ):
        super().__init__(feedback=feedback, score=score)
        self.statements = statements
        self.explanations = explanations
        self.verdicts = verdicts

    def to_dict(self):
        return {
            "statements": self.statements,
            "explanations": self.explanations,
            "verdicts": self.verdicts,
            "score": self.score,
        }


class AnswerRelevanceResult(Feedback):
    def __init__(
        self, answer: str, generated_questions: list[str], noncommittals: list[int], score: float, feedback: str = ""
    ):
        super().__init__(feedback=feedback, score=score)
        self.generated_questions = generated_questions
        self.answer = answer
        self.noncommittals = noncommittals

    def to_dict(self):
        return {
            "answer": self.answer,
            "generated_questions": self.generated_questions,
            "noncommittals": self.noncommittals,
            "score": self.score,
        }


class ContextRelevanceResult(Feedback):
    def __init__(self, relevant_sentences: str, irrelevant_sentences: str, score: float, feedback: str = ""):
        super().__init__(feedback=feedback, score=score)
        self.relevant_sentences = relevant_sentences
        self.irrelevant_sentences = irrelevant_sentences

    def to_dict(self):
        return {
            "relevant_sentences": self.relevant_sentences,
            "irrelevant_sentences": self.irrelevant_sentences,
            "score": self.score,
        }


class Stats:
    def __init__(self, pct: float, total: int):
        self.pct = pct
        self.total = total
