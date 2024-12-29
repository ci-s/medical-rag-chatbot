import pickle


class Page:
    def __init__(self, page_number: int, token_count: int, raw_content: str, processed_content: str):
        self.page_number = page_number
        self.token_count = token_count
        self.raw_content = raw_content
        self.processed_content = processed_content


class Document:
    def __init__(self, path: str):
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


class Chunk:
    def __init__(self, text, start_page, end_page):
        self.text = text
        self.start_page = start_page
        self.end_page = end_page

    def __str__(self) -> str:
        return f"Chunk from page {self.start_page} to {self.end_page}: {self.text[:150]}..."
