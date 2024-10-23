class PageInfo:
    page_number: int
    token_count: int | None = None
    raw_content: str
    processed_content: str | None = None

    def __init__(self, page_number, token_count, raw_content, processed_content):
        self.page_number = page_number
        self.token_count = token_count
        self.raw_content = raw_content
        self.processed_content = processed_content


class Chunk:
    def __init__(self, text, start_page, end_page):
        self.text = text
        self.start_page = start_page
        self.end_page = end_page
