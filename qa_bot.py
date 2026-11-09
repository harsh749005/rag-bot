"""
QA Bot: ask questions about a PDF, Word (.docx) file, image, or video.

How it works (5 simple steps):
1. Load the file and turn it into text.
2. Split the text into small chunks.
3. Store the chunks in a search index (using embeddings).
4. For each question, find the chunks that match best.
5. Send those chunks and the question to the LLM to get an answer.

For a video, step 1 does two things:
- It writes down what people say (speech to text).
- It looks at a few frames and asks the LLM to describe them.

Run it:
    python qa_bot.py my_file.pdf
    python qa_bot.py my_video.mp4
"""

import base64
import mimetypes
import os
import sys

from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()  # read ANTHROPIC_API_KEY from the .env file
# Settings you may want to change
MODEL_NAME = "claude-sonnet-5-5"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000      # characters in each chunk
CHUNK_OVERLAP = 150    # characters shared between neighbour chunks
CHUNKS_TO_USE = 4      # how many chunks we give the LLM per question

# Video settings
WHISPER_MODEL = "base"     # speech-to-text model: tiny, base, small, medium
SPEECH_BLOCK_SECONDS = 60  # group the transcript into blocks of about 1 minute
FRAME_EVERY_SECONDS = 30   # look at one frame every 30 seconds...
MAX_FRAMES = 10            # ...but never more than this many (each one costs an LLM call)

IMAGE_TYPES = [".png", ".jpg", ".jpeg", ".webp", ".gif"]
VIDEO_TYPES = [".mp4", ".mov", ".mkv", ".avi", ".webm"]

# The instructions we give the LLM for every question
PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the context below.\n"
    "If the answer is not in the context, say \"I could not find that in the document.\"\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


# ---------- Small helpers ----------

def get_text(message):
    # The reply can be a plain string or a list of blocks.
    # We keep only the blocks that hold the answer text.
    if isinstance(message.content, str):
        return message.content

    parts = []
    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
    return "".join(parts)


def format_time(seconds):
    # 125 seconds becomes "02:05"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


# ---------- Step 1: load the file ----------

def load_pdf(path):
    # Read the PDF page by page
    return PyPDFLoader(path).load()


def load_docx(path):
    # Read the text inside the Word file
    return Docx2txtLoader(path).load()


def describe_image(image_bytes, mime_type, llm):
    # An image has no text yet, so we ask the LLM to read it for us.
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    message = HumanMessage(content=[
        {
            "type": "text",
            "text": "Extract all the text in this image. "
                    "Then describe what the image shows in detail.",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
        },
    ])
    return get_text(llm.invoke([message]))


def load_image(path, llm):
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as file:
        image_bytes = file.read()

    text = describe_image(image_bytes, mime_type, llm)
    return [Document(page_content=text, metadata={"source": path})]


def make_speech_document(lines, start_time, path):
    text = f"[Video speech from {format_time(start_time)}] " + " ".join(lines)
    return Document(page_content=text, metadata={"source": path, "start": start_time})


def load_video_speech(path):
    # Turn the speech in the video into text.
    # We import here so the other file types still work without this library.
    from faster_whisper import WhisperModel

    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(path)

    documents = []
    lines = []
    block_start = 0

    # Whisper gives short segments. We join them into blocks of about 1 minute.
    for segment in segments:
        lines.append(segment.text.strip())
        if segment.end - block_start >= SPEECH_BLOCK_SECONDS:
            documents.append(make_speech_document(lines, block_start, path))
            lines = []
            block_start = segment.end

    # Keep the last few lines too
    if lines:
        documents.append(make_speech_document(lines, block_start, path))

    return documents


def load_video_frames(path, llm):
    # Take a few pictures from the video and ask the LLM to describe each one.
    import cv2

    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS) or 25
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    # Spread the frames across the whole video, but not closer than FRAME_EVERY_SECONDS
    step = max(FRAME_EVERY_SECONDS, duration / MAX_FRAMES)

    documents = []
    seconds = 0
    while seconds < duration and len(documents) < MAX_FRAMES:
        video.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        found, frame = video.read()

        if found:
            ok, jpg = cv2.imencode(".jpg", frame)
            text = describe_image(jpg.tobytes(), "image/jpeg", llm)
            text = f"[Video frame at {format_time(seconds)}] " + text
            documents.append(Document(page_content=text, metadata={"source": path, "start": seconds}))

        seconds += step

    video.release()
    return documents


def load_video(path, llm):
    print("  Reading the speech...")
    speech = load_video_speech(path)
    print("  Looking at video frames...")
    frames = load_video_frames(path, llm)
    return speech + frames


def load_file(path, llm):
    # Pick the right loader using the file extension
    extension = os.path.splitext(path)[1].lower()

    if extension == ".pdf":
        return load_pdf(path)
    if extension == ".docx":
        return load_docx(path)
    if extension in IMAGE_TYPES:
        return load_image(path, llm)
    if extension in VIDEO_TYPES:
        return load_video(path, llm)

    raise ValueError(f"Sorry, {extension} files are not supported yet.")


# ---------- Steps 2 and 3: split the text and build the index ----------

def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_index(chunks):
    # Embeddings turn text into numbers so we can search by meaning
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    index = InMemoryVectorStore(embeddings)
    index.add_documents(chunks)
    return index


# ---------- Steps 4 and 5: find chunks and answer ----------

def find_chunks(index, question):
    return index.similarity_search(question, k=CHUNKS_TO_USE)


def ask_question(llm, index, question):
    chunks = find_chunks(index, question)
    context = "\n\n".join(chunk.page_content for chunk in chunks)

    messages = PROMPT.invoke({"context": context, "question": question})
    answer = get_text(llm.invoke(messages))
    return answer, chunks


def show_sources(chunks):
    # PDF chunks know their page number (it starts at 0, so we add 1)
    pages = sorted({chunk.metadata["page"] + 1 for chunk in chunks if "page" in chunk.metadata})
    if pages:
        print("Pages used:", ", ".join(str(page) for page in pages))

    # Video chunks know the time in the video where they came from
    times = sorted({chunk.metadata["start"] for chunk in chunks if "start" in chunk.metadata})
    if times:
        print("Video times used:", ", ".join(format_time(t) for t in times))


# ---------- Main program ----------

def main():
    if len(sys.argv) != 2:
        print("Usage: python qa_bot.py <file.pdf | file.docx | image | video>")
        return

    path = sys.argv[1]
    if not os.path.exists(path):
        print("File not found:", path)
        return

    llm = ChatAnthropic(model=MODEL_NAME)

    print("Loading file...")
    documents = load_file(path, llm)

    chunks = split_into_chunks(documents)
    if not chunks:
        print("No text found. If this is a scanned PDF, save the pages as images and try again.")
        return

    print("Building search index...")
    index = build_index(chunks)

    print("Ready! Ask a question (type 'quit' to stop).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", ""):
            break

        answer, used_chunks = ask_question(llm, index, question)
        print("\nBot:", answer)
        show_sources(used_chunks)
        print()


if __name__ == "__main__":
    main()