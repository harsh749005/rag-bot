"""
Simple multimodal document Q&A backend (Gemini + RAG).

Upload:  file -> get text out of it -> split into chunks -> embed chunks -> save
Ask:     question -> find the best chunks -> Gemini answers using them
"""
import os
import json
import time

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "100"))  # biggest upload we accept
SEND_ALL_BELOW = 80000    # if all your files together are smaller than this many characters, send everything to Gemini
TOP_K = 8                 # otherwise send only the 8 best matching chunks
SOURCE_MIN_SCORE = 0.5    # only show a source chip if the chunk is this similar to the question

UPLOAD_DIR = "uploads"
DB_FILE = "data/chunks.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

app = FastAPI()
# On Render set ALLOWED_ORIGIN to your Vercel URL, e.g. https://my-app.vercel.app
allowed_origins = os.getenv("ALLOWED_ORIGIN", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["*"], allow_headers=["*"])


# ---------- tiny "database" (a JSON file) ----------
def load_db():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE) as f:
        return json.load(f)


def save_db(chunks):
    with open(DB_FILE, "w") as f:
        json.dump(chunks, f)


# ---------- helper: try again if Gemini says "too many requests" ----------
def retry(func, tries=3):
    for i in range(tries):
        try:
            return func()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


# ---------- helper: let Gemini read a file (image, scanned PDF, audio, video) ----------
def ask_gemini_about_file(path, prompt):
    uploaded = client.files.upload(file=path)
    while uploaded.state.name == "PROCESSING":  # videos need a moment to be processed
        time.sleep(3)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name == "FAILED":
        raise ValueError("Gemini could not process this file")
    try:
        result = retry(lambda: client.models.generate_content(
            model=CHAT_MODEL,
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(max_output_tokens=60000),  # room for long transcripts
        ))
        return result.text or ""
    finally:
        client.files.delete(name=uploaded.name)  # Gemini also deletes it after 48 hours anyway


IMAGE_PROMPT = (
    "First write out ALL text visible in this image exactly as written (OCR). "
    "Then describe what the image shows in detail: objects, people, charts, numbers, colours."
)
MEDIA_PROMPT = (
    "Transcribe all speech word for word (speech-to-text). Start every sentence with a [mm:ss] timestamp. "
    "After the transcript add a section called ON SCREEN describing anything shown visually "
    "(slides, text, objects) with timestamps. If there is no speech, say so and only describe what is shown."
)
SCANNED_PDF_PROMPT = "This is a scanned document. Write out all of its text exactly as written, page by page."


# ---------- 1. get text out of each file type ----------
# Every reader returns a list of (where, text) pairs, e.g. ("page 3", "some text...")

def read_pdf(path):
    pages = PdfReader(path).pages
    result = [(f"page {i + 1}", page.extract_text() or "") for i, page in enumerate(pages)]
    if sum(len(text.strip()) for _, text in result) < 50:  # no real text -> it is a scan, let Gemini read it
        return [("scanned pdf", ask_gemini_about_file(path, SCANNED_PDF_PROMPT))]
    return result


def read_word(path):
    doc = Document(path)
    lines = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:  # also read tables
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    return [("document", "\n".join(lines))]


def read_text(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [("text", f.read())]


def extract_text(path):
    ext = path.lower().split(".")[-1]
    if ext == "pdf":
        return read_pdf(path)
    if ext == "docx":
        return read_word(path)
    if ext in ("txt", "md", "csv"):
        return read_text(path)
    if ext in ("png", "jpg", "jpeg", "webp"):
        return [("image", ask_gemini_about_file(path, IMAGE_PROMPT))]
    if ext in ("mp4", "mov", "webm", "avi", "mpeg", "mpg", "wmv", "mp3", "wav", "aac", "flac", "ogg"):
        return [("transcript", ask_gemini_about_file(path, MEDIA_PROMPT))]
    raise ValueError("Unsupported file type: ." + ext)


# ---------- 2. split text into small chunks ----------
def split_text(text, size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks


# ---------- 3. turn text into numbers (embeddings) ----------
# task is "RETRIEVAL_DOCUMENT" for file chunks and "RETRIEVAL_QUERY" for questions (gives better matches)
def embed(texts, task):
    config = types.EmbedContentConfig(task_type=task)
    vectors = []
    for i in range(0, len(texts), 50):  # send 50 at a time
        batch = texts[i:i + 50]
        result = retry(lambda: client.models.embed_content(model=EMBED_MODEL, contents=batch, config=config))
        vectors += [e.values for e in result.embeddings]
    return vectors


# ---------- API endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}  # used by the keep-alive ping, does no real work


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    # (a normal "def" runs in a background thread, so long videos don't freeze the server)
    name = os.path.basename(file.filename)
    path = os.path.join(UPLOAD_DIR, name)

    # save to disk 1 MB at a time so big files don't fill the memory
    size = 0
    too_big = False
    with open(path, "wb") as f:
        while True:
            piece = file.file.read(1024 * 1024)
            if not piece:
                break
            size += len(piece)
            if size > MAX_FILE_MB * 1024 * 1024:
                too_big = True
                break
            f.write(piece)
    if too_big:
        os.remove(path)
        return {"ok": False, "error": f"File too large (max {MAX_FILE_MB} MB)"}

    try:
        parts = extract_text(path)
    except Exception as error:
        return {"ok": False, "error": str(error)[:200]}
    finally:
        os.remove(path)  # we only need the extracted text, not the original file

    new_chunks = []
    for where, text in parts:
        for piece in split_text(text):
            new_chunks.append({"file": name, "where": where, "text": piece})
    if not new_chunks:
        return {"ok": False, "error": "No text found in this file"}

    try:
        vectors = embed([f"{c['file']} - {c['where']}\n{c['text']}" for c in new_chunks], "RETRIEVAL_DOCUMENT")
    except Exception as error:
        return {"ok": False, "error": "Embedding failed: " + str(error)[:150]}
    for chunk, vector in zip(new_chunks, vectors):
        chunk["embedding"] = vector

    db = [c for c in load_db() if c["file"] != name]  # replace if uploaded before
    save_db(db + new_chunks)
    return {"ok": True, "file": name, "chunks": len(new_chunks)}


@app.get("/files")
def list_files():
    names = sorted({c["file"] for c in load_db()})
    return {"files": names}


@app.delete("/files/{name}")
def delete_file(name: str):
    save_db([c for c in load_db() if c["file"] != name])
    return {"ok": True}


class Question(BaseModel):
    question: str
    history: list = []  # earlier messages: [{"role": "user" or "assistant", "text": "..."}]


@app.post("/ask")
def ask(body: Question):
    db = load_db()
    history = body.history[-6:]

    # 1. find the chunks that match the question (include the last question so follow-ups work)
    chosen = []
    sources = []
    if db:
        search_text = body.question
        earlier_questions = [m for m in history if m.get("role") == "user"]
        if earlier_questions:
            search_text = earlier_questions[-1].get("text", "") + " " + body.question

        try:
            question_vector = np.array(embed([search_text], "RETRIEVAL_QUERY")[0])
        except Exception:
            return {"answer": "Sorry, I couldn't reach Gemini right now. Please try again in a minute.", "sources": []}
        all_vectors = np.array([c["embedding"] for c in db])
        scores = all_vectors @ question_vector / (
            np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(question_vector))

        total_chars = sum(len(c["text"]) for c in db)
        if total_chars <= SEND_ALL_BELOW:
            chosen_ids = list(range(len(db)))                    # small knowledge base: send everything
        else:
            chosen_ids = list(np.argsort(scores)[::-1][:TOP_K])  # big one: send the best matches

        chosen = [db[i] for i in chosen_ids]
        best_first = sorted(chosen_ids, key=lambda i: -scores[i])
        sources = [{"file": db[i]["file"], "where": db[i]["where"]}
                   for i in best_first if scores[i] >= SOURCE_MIN_SCORE][:5]

    # 2. build the prompt
    context = "\n\n".join(f"[{c['file']} - {c['where']}]\n{c['text']}" for c in chosen) or "(no files uploaded yet)"
    chat_so_far = "\n".join(f"{m.get('role', 'user').capitalize()}: {m.get('text', '')}" for m in history)
    prompt = (
        "You are a helpful assistant. The user has uploaded files (documents, images, videos).\n"
        "Rules:\n"
        "1. Use the CONTEXT first, and mention the file name and page or timestamp you used.\n"
        "2. If the answer is not in the context, still help using your own knowledge, "
        "but start with: \"This isn't in your files, but...\"\n"
        "3. Be clear and concise.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"CONVERSATION SO FAR:\n{chat_so_far or '(none)'}\n\n"
        f"User: {body.question}"
    )

    # 3. ask Gemini
    try:
        result = retry(lambda: client.models.generate_content(model=CHAT_MODEL, contents=prompt))
        return {"answer": result.text or "I couldn't generate an answer. Please rephrase.", "sources": sources}
    except Exception:
        return {"answer": "Sorry, Gemini is busy or the free limit was reached. Please try again in a minute.", "sources": []}


# ---------- serve the web page locally (must stay at the bottom) ----------
from fastapi.staticfiles import StaticFiles
if os.path.isdir("frontend"):  # skipped on Render, where the page is hosted on Vercel
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
