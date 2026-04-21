"""
cv_chunker.py
 
Extracts hierarchical chunks from a CV PDF for use with SentenceTransformers.
 
Each chunk is a string like:
  "Work Experience | Acme Corp | Senior Engineer (2019-2023) | Reduced AWS costs by 30%"
 
Usage:
    from cv_chunker import chunk_cv
    chunks = chunk_cv("my_cv.pdf")
    for chunk in chunks:
        print(chunk)
"""
 
import re
import pdfplumber
 
 
# ── Section heading detection ─────────────────────────────────────────────────
# Adjust these keywords to match the section headings in your CV
SECTION_KEYWORDS = [
    "experience", "employment", "work history",
    "education", "qualifications",
    "skills", "technologies", "competencies",
    "summary", "profile", "objective",
    "projects", "achievements", "certifications", "awards",
    "publications", "volunteer", "interests",
]
 
def _is_section_heading(line: str) -> bool:
    """Return True if the line looks like a CV section heading."""
    clean = line.strip().lower()
    # Headings are usually short and match a known keyword
    return any(clean.startswith(kw) for kw in SECTION_KEYWORDS) and len(clean) < 60
 
 
# ── Sub-heading detection (company / institution / role lines) ────────────────
def _is_sub_heading(line: str) -> bool:
    """
    Return True if the line looks like a role/company/date line.
    These typically contain a year (4 digits) or a dash between two years.
    Examples:
        "Senior Software Engineer | Acme Corp | 2019 - 2023"
        "Microsoft  2020–Present"
        "BSc Computer Science, MIT (2012–2016)"
    """
    return bool(re.search(r'\b(19|20)\d{2}\b', line))
 
 
# ── Bullet detection ──────────────────────────────────────────────────────────
def _is_bullet(line: str) -> bool:
    """Return True if the line starts with a bullet character."""
    return line.strip().startswith("•")
 
 
def _clean_bullet(line: str) -> str:
    """Strip the bullet character and surrounding whitespace."""
    return line.strip().lstrip("•").strip()
 
 
# ── Main chunker ──────────────────────────────────────────────────────────────
def chunk_cv(pdf_path: str) -> list[str]:
    """
    Parse a CV PDF and return a list of hierarchical chunk strings.
 
    Each chunk has the form:
        "<Section> | <Sub-heading> | <Bullet text>"
    or for section-level non-bullet lines:
        "<Section> | <line text>"
 
    Args:
        pdf_path: Path to the CV PDF file.
 
    Returns:
        List of chunk strings ready for embedding.
    """
    # ── Extract raw text from PDF ─────────────────────────────────────────────
    raw_lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                raw_lines.extend(text.splitlines())
 
    # ── Parse lines into hierarchical chunks ─────────────────────────────────
    chunks = []
    current_section = "General"
    current_sub = ""          # company / role / institution line
 
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue  # skip blank lines
 
        if _is_section_heading(stripped):
            current_section = stripped.title()
            current_sub = ""  # reset sub-heading when section changes
 
        elif _is_sub_heading(stripped):
            current_sub = stripped
 
        elif _is_bullet(stripped):
            bullet_text = _clean_bullet(stripped)
            if not bullet_text:
                continue
 
            # Build the hierarchical string
            if current_sub:
                chunk = f"{current_section} | {current_sub} | {bullet_text}"
            else:
                chunk = f"{current_section} | {bullet_text}"
 
            chunks.append(chunk)
 
        else:
            # Non-bullet, non-heading line (e.g. a summary paragraph)
            # Include it as a chunk if it's substantial enough
            if len(stripped) > 30:
                if current_sub:
                    chunk = f"{current_section} | {current_sub} | {stripped}"
                else:
                    chunk = f"{current_section} | {stripped}"
                chunks.append(chunk)
 
    return chunks
 
 
# ── Optional: embed chunks with SentenceTransformers ─────────────────────────
def embed_chunks(chunks: list[str], model_name: str = "all-MiniLM-L6-v2"):
    """
    Embed a list of chunk strings using SentenceTransformers.
 
    Args:
        chunks:     List of chunk strings from chunk_cv()
        model_name: SentenceTransformers model to use
 
    Returns:
        numpy array of shape (len(chunks), embedding_dim)
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(chunks, show_progress_bar=True)
 
 
# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
 
    if len(sys.argv) < 2:
        print("Usage: python cv_chunker.py path/to/cv.pdf")
        sys.exit(1)
 
    pdf_path = sys.argv[1]
    print(f"Chunking: {pdf_path}\n")
 
    chunks = chunk_cv(pdf_path)
 
    print(f"Found {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i:02d}] {chunk}")
 
