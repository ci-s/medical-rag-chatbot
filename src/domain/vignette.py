import yaml
from typing import Literal


class Question:
    def __init__(
        self,
        id: int,
        question: str,
        answer: str,
        reference: list[dict],
        source: Literal["Handbuch", "Antibiotika"],
    ) -> None:
        self.id = id
        self.question = question
        self.answer = answer
        self.references = self._process_references(reference)
        self.source = source

    def get_id(self) -> int:
        return self.id

    def get_question(self) -> str:
        return self.question

    def get_answer(self) -> str:
        return self.answer

    def get_references(self) -> dict[int, str]:
        return self.references

    def get_reference_pages(self) -> list[int]:
        return self.references.keys()

    def _process_references(self, reference_list: list[dict]):
        """Convert reference list into a dictionary with page numbers as keys."""
        return {ref["page"]: ref["type"] for ref in reference_list}

    def get_source(self) -> Literal["Handbuch", "Antibiotika"]:
        return self.source

    def __str__(self) -> str:
        return f"Question {self.id}: Question: {self.question[:100]}... \nAnswer: {self.answer} \nReference: {self.references} \nSource: {self.source}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "reference": [{"page": k, "type": v} for k, v in self.references.items()],
            "source": self.source,
        }


class Vignette:
    def __init__(self, id: int, background: str, context: int, questions: list[Question] | None) -> None:
        self.id = id
        self.background = background
        self.context = context
        self.questions = questions

    def get_id(self) -> int:
        return self.id

    def get_questions(self) -> list[Question] | None:
        return self.questions

    def get_question(self, id: int) -> Question | None:
        for question in self.questions:
            if question.get_id() == id:
                return question
        print(f"Question with id {id} not found in vignette {self.id}")
        return None

    def get_preceding_questions(self, id: int) -> list[Question]:
        id_found = False
        preceding_questions = []
        for question in self.questions:
            if question.get_id() < id:
                preceding_questions.append(question)
            elif question.get_id() == id:
                id_found = True
                break

        if not id_found:
            print(f"Question with id {id} not found in vignette {self.id}")
            return []

        if len(preceding_questions) == 0:
            print(f"No preceding questions found for question with id {id} in vignette {self.id}")

        return preceding_questions

    def get_background(self) -> str:
        return self.background

    def get_context(self) -> str:
        return self.context

    def __str__(self) -> str:
        return f"Vignette {self.id}: \nBackground: {self.background[:100]}...\nContext: {self.context}\nQuestions: {len(self.questions)}"

    def filter_vignette(self, categories: list[Literal["Text", "Table", "Flowchart"]] | None, pages: list[int] | None):
        filtered_questions = []

        for question in self.get_questions():
            references = question.get_references()

            if categories:
                if any(source_type not in categories for _, source_type in references.items()):
                    continue
            if pages:
                if any(page_number not in pages for page_number, _ in references.items()):
                    continue

            filtered_questions.append(question)

        self.questions = filtered_questions

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "background": self.background,
            "context": self.context,
            "questions": [q.to_dict() for q in self.questions] if self.questions else [],
        }


class VignetteCollection:
    def __init__(self):
        self.vignettes: list[Vignette] = []

    def load_from_yaml(self, file_path, filter_categories: list[str] | None, filter_pages: list[int] | None):
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            for vignette_data in data["vignettes"]:
                vignette = Vignette(
                    background=vignette_data["background"],
                    context=vignette_data["context"],
                    id=vignette_data["id"],
                    questions=[Question(**q) for q in vignette_data["questions"]],
                )

                if filter_categories:
                    vignette.filter_vignette(filter_categories, filter_pages)
                self.vignettes.append(vignette)

    def get_vignettes(self) -> list[Vignette]:
        return self.vignettes

    def get_vignette_by_id(self, id) -> Vignette | None:
        for vignette in self.vignettes:
            if vignette.id == id:
                return vignette
        return None

    def save_to_yaml(self, file_path: str) -> None:
        data = {"vignettes": [v.to_dict() for v in self.vignettes]}
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
