from io import BytesIO
from pathlib import Path
import pymupdf
from docx import Document
from bs4 import BeautifulSoup


def make_document(text, source, file_format, page_number=None, section="Document content"):

    metadata = {
        "source": source,
        "source_format": file_format,
        "section": section
    }

    if page_number is not None:
        metadata["page_number"] = page_number

    return {
        "text": text,
        "metadata": metadata
    }


def extract_pdf(uploaded_file):

    pdf = pymupdf.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    documents = []

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text("text").strip()

        if text:
            documents.append(
                make_document(
                    text,
                    uploaded_file.name,
                    "PDF",
                    page_number
                )
            )

    pdf.close()
    return documents


def extract_docx(uploaded_file):

    doc = Document(BytesIO(uploaded_file.getvalue()))

    content = []

    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            content.append(paragraph.text.strip())

    for table in doc.tables:
        for row in table.rows:
            content.append(
                " | ".join(cell.text.strip() for cell in row.cells)
            )

    text = "\n".join(content).strip()

    return [
        make_document(text, uploaded_file.name, "DOCX")
    ] if text else []


def extract_txt(uploaded_file):

    raw = uploaded_file.getvalue()

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")

    text = text.strip()

    return [
        make_document(text, uploaded_file.name, "TXT")
    ] if text else []


def extract_html(uploaded_file):

    soup = BeautifulSoup(
        uploaded_file.getvalue(),
        "html.parser"
    )

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text = soup.get_text(separator="\n", strip=True)

    title = (
        soup.title.string.strip()
        if soup.title and soup.title.string
        else "Document content"
    )

    return [
        make_document(
            text,
            uploaded_file.name,
            "HTML",
            section=title
        )
    ] if text else []


def process_uploaded_file(uploaded_file):

    extension = Path(uploaded_file.name).suffix.lower()

    extractors = {
        ".pdf": extract_pdf,
        ".docx": extract_docx,
        ".txt": extract_txt,
        ".html": extract_html,
        ".htm": extract_html
    }

    if extension not in extractors:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    documents = extractors[extension](uploaded_file)

    if not documents:
        raise ValueError(
            "No readable text found in this document."
        )

    return documents
