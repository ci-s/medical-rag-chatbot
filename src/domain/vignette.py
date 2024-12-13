import yaml
from typing import Literal


class Question:
    def __init__(
        self,
        id: int,
        question: str,
        answer: str,
        reference: list[int],
        source: Literal["Handbuch", "Antibiotika"],
        text_only: bool = False,
    ) -> None:
        self.id = id
        self.question = question
        self.answer = answer
        self.reference = reference
        self.source = source
        self.text_only = text_only

    def get_id(self) -> int:
        return self.id

    def get_question(self) -> str:
        return self.question

    def get_answer(self) -> str:
        return self.answer

    def get_reference(self) -> list[int]:
        return self.reference

    def get_source(self) -> Literal["Handbuch", "Antibiotika"]:
        return self.source

    def __str__(self) -> str:
        return f"Question {self.id}: Question: {self.question[:100]}... \nAnswer: {self.answer} \nReference: {self.reference} \nSource: {self.source}"


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


class VignetteCollection:
    def __init__(self):
        self.vignettes: list[Vignette] = []

    def load_from_yaml(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            for vignette_data in data["vignettes"]:
                vignette = Vignette(
                    background=vignette_data["background"],
                    context=vignette_data["context"],
                    id=vignette_data["id"],
                    questions=[Question(**q) for q in vignette_data["questions"]],
                )

                self.vignettes.append(vignette)

    def get_vignettes(self) -> list[Vignette]:
        return self.vignettes

    def get_vignette_by_id(self, id) -> Vignette | None:
        for vignette in self.vignettes:
            if vignette.id == id:
                return vignette
        return None

    def label_text_only_questions(self, text_pages) -> None:
        id_list = []  # TODO: decide should it stay here or not
        for vignette in self.vignettes:
            for question in vignette.get_questions():
                if all(p in text_pages for p in question.reference):
                    question.text_only = True
                    id_list.append(question.id)
        print("Text only questions labeled, ids are as follows: ", id_list)
