class Feedback:
    def __init__(self, feedback: str, score: int):
        self.text = feedback
        self.score = score

    def __repr__(self):
        return f"Feedback(feedback={self.text}, score={self.score})"

    def to_dict(self):
        return {"feedback": self.text, "score": self.score}


class Stats:
    def __init__(self, pct: float, total: int):
        self.pct = pct
        self.total = total
