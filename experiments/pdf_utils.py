from pypdf import PdfReader


def read_pdf(file_path):
    reader = PdfReader(file_path)
    doc = "\nnewpage"
    for page in reader.pages:
        doc = doc + page.extract_text()
    return doc
