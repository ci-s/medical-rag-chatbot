from enum import Enum
import pickle


class Page:
    def __init__(self, page_number: int, token_count: int, raw_content: str, processed_content: str | None = None):
        self.page_number = page_number
        self.token_count = token_count
        self.raw_content = raw_content
        if processed_content is None:
            self.processed_content = raw_content
        else:
            self.processed_content = processed_content


class Document:
    def __init__(self, path: str = ""):
        self.path = path
        self.pages = []

    def add_page(self, page):
        self.pages.append(page)

    def get_page(self, page_number):
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def get_raw_content(self, page_number):
        page = self.get_page(page_number)
        return page.raw_content if page else None

    def get_processed_content(self, page_number):
        page = self.get_page(page_number)
        return page.processed_content if page else None

    def save(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
        print(f"Document saved to {filepath}")

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as file:
            document = pickle.load(file)
        print(f"Document loaded from {filepath}")
        return document

    def __str__(self) -> str:
        print(f"Document with {len(self.pages)} pages from {self.path}")


class ChunkType(Enum):
    TEXT = "Text"
    TABLE = "Table"
    FLOWCHART = "Flowchart"


class Chunk:
    def __init__(
        self,
        text,
        start_page,
        end_page,
        section_heading=None,
        index=None,
        type: ChunkType | None = None,
    ):
        self.text = text
        self.index = index
        self.start_page = start_page
        self.end_page = end_page
        self.section_heading = section_heading
        self.type = type

    def __str__(self) -> str:
        if self.section_heading:
            return f"Chunk {self.index} ({self.type}) in section {self.section_heading}, from page {self.start_page} to {self.end_page}: {self.text[:150]}..."
        else:
            return (
                f"Chunk {self.index} ({self.type}) from page {self.start_page} to {self.end_page}: {self.text[:150]}..."
            )

    def to_dict(self):
        return {
            "text": self.text,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "section_heading": self.section_heading,
            "index": self.index,
            "type": self.type.value if self.type else None,
        }

    def copy(self):
        return Chunk(
            self.text,
            self.start_page,
            self.end_page,
            self.section_heading,
            self.index,
            self.type,
        )

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            text=data.get("text"),
            start_page=data.get("start_page"),
            end_page=data.get("end_page"),
            section_heading=data.get("section_heading"),
            index=data.get("index"),
            type=ChunkType(data["type"]) if data.get("type") else None,
        )
