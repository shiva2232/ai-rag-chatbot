import PyPDF2

def read_pdf(file_path):
    pages = []

    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            
            if text:  # avoid None pages
                pages.append(text)

    return pages

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks