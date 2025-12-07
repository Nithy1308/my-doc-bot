import os
import re
from pathlib import Path
import streamlit as st

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# BLOCK ANY OPENAI USAGE
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("OPENAI_API_BASE", None)
os.environ.pop("OPENAI_BASE_URL", None)

DOCS_DIR = Path("docs")
STORAGE_DIR = Path("storage")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# -------------- SIMPLE GREETING MAP --------------
GREETINGS = {
    "hi": "Hello there, how can I help you today?",
    "hello": "Hi, how can I help you today?",
    "hey": "Hey there! How can I help you today?",
    "good morning": "Good morning! How may I assist you today?",
    "good afternoon": "Good afternoon! How may I assist you today?",
    "good evening": "Good evening! How may I assist you today?",
}


# ----------- LOAD / BUILD INDEX -----------
def build_or_load_index(rebuild=False):

    if not rebuild and STORAGE_DIR.exists():
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=str(STORAGE_DIR)
            )
            return load_index_from_storage(storage_context)
        except Exception:
            pass

    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True, exist_ok=True)

    docs = SimpleDirectoryReader(
        input_dir=str(DOCS_DIR),
        exclude=["*.json"],
    ).load_data()

    if not docs:
        raise ValueError(
            f"No documents found in {DOCS_DIR.resolve()} (excluding *.json). "
            "Please add at least one .txt/.pdf/.docx file."
        )

    # split into small chunks so one country / paragraph tends to be separate
    parser = SentenceSplitter(chunk_size=120, chunk_overlap=0)
    nodes = parser.get_nodes_from_documents(docs)

    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    return index


# -------------- MAIN ANSWER LOGIC (NO LLM) -------------
def answer_from_docs(index, question: str) -> str:
    """
    Retrieve relevant text from docs and then keep only the lines
    that contain at least one non-stopword keyword from the question.
    If nothing matches, say we don't know.
    """
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    nodes = retriever.retrieve(question)

    if not nodes:
        return "I don't know based on the documents."

    # combine a few top nodes (more chance to hit correct line)
    combined_text = "\n".join(
        [(n.text or "").strip() for n in nodes if (n.text or "").strip()]
    )

    if not combined_text:
        return "I don't know based on the documents."

    print("\n=== Retrieved context for question ===")
    print("Q:", question)
    print(combined_text[:500])
    print("=== END CONTEXT ===\n")

    # 1) extract keywords from question
    words = re.findall(r"\w+", question.lower())
    stop = {
        "what", "who", "why", "where", "when", "how",
        "is", "are", "the", "a", "an", "in", "of", "to",
        "please", "tell", "me", "about",
        "capital", "country", "city"   # generic words – ignore them
    }
    keywords = [w for w in words if w not in stop]

    print(">>> keywords:", keywords)

    if not keywords:
        return "I don't know based on the documents."

    # 2) keep only lines that contain at least one keyword
    lines = [ln.strip() for ln in combined_text.splitlines() if ln.strip()]
    matched_lines = [
        ln for ln in lines
        if any(k in ln.lower() for k in keywords)
    ]

    print(">>> matched_lines:", matched_lines)

    if not matched_lines:
        return "I don't know based on the documents."

    # Usually this will be exactly one line, e.g.
    # "India is a country in South Asia. Capital: New Delhi."
    return "\n".join(matched_lines)


# -------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Docs Bot", layout="wide")
st.title("📚 Private Document Q&A Bot")

with st.sidebar:
    st.header("Admin Area")
    admin_key = st.text_input("Admin key", type="password")
    rebuild = False

    if admin_key == "mysecret":
        st.success("Admin mode active.")
        rebuild = st.button("Rebuild index from docs/")
    else:
        st.info("Enter admin key to rebuild.")

try:
    index = build_or_load_index(rebuild)
except Exception as e:
    st.error(str(e))
    st.stop()


# ---------- CHAT ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    norm = user_input.lower().strip()

    # GREETING
    if norm in GREETINGS:
        reply = GREETINGS[norm]
    else:
        # DOC ANSWER (NO LLM)
        reply = answer_from_docs(index, user_input)

    with st.chat_message("assistant"):
        st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

