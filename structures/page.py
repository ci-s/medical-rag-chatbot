class Page:
    def __init__(self, page_number: int, token_count: int, raw_content: str, processed_content: str):
        self.page_number = page_number
        self.token_count = token_count
        self.raw_content = raw_content
        self.processed_content = processed_content


class Document:
    def __init__(self, path: str, pages: list[Page] = []):
        self.path = path
        self.pages = pages

    def add_page(self, page):
        self.pages.append(page)

    def get_page(self, page_number):
        return next(
            (page for page in self.pages if page.page_number == page_number), None
        )  # TODO: does it work as intended?


class Chunk:
    def __init__(self, text, start_page, end_page):
        self.text = text
        self.start_page = start_page
        self.end_page = end_page

    def __str__(self) -> str:
        return f"Chunk from page {self.start_page} to {self.end_page}: {self.text[:150]}..."
