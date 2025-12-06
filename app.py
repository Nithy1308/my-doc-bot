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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ---------- BLOCK ANY OPENAI USAGE ----------
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("OPENAI_API_BASE", None)
os.environ.pop("OPENAI_BASE_URL", None)
os.environ["LLAMA_INDEX_USE_OPENAI_EMBEDDINGS"] = "false"


# ---------- PATH CONFIG ----------
DOCS_DIR = Path("docs")
STORAGE_DIR = Path("storage")


# ---------- EMBEDDING MODEL (LOCAL) ----------
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
print(">>> Using embedding model: BAAI/bge-small-en-v1.5")


# ---------- SIMPLE GREETING MAP ----------
GREETINGS_RESPONSES = {
    "hi": "Hello there, how can I help you today?",
    "hi!": "Hello there, how can I help you today?",
    "hello": "Hi, how can I help you today?",
    "hello!": "Hi, how can I help you today?",
    "hey": "Hey there! How can I help you today?",
    "helo": "Hi there, how can I help you today?",
    "good morning": "Good morning! How may I assist you today?",
    "good afternoon": "Good afternoon! How may I assist you today?",
    "good evening": "Good evening! How may I assist you today?",
    "what's up?": "Not much, how about you?",
    "how are you?": "I'm doing well, thank you for asking. How may I assist you today?",
}
print(">>> Greeting keys:", list(GREETINGS_RESPONSES.keys()))


# ---------- INDEX BUILD / LOAD ----------
def build_or_load_index(rebuild: bool = False):
    """
    Build the index from docs/ if it doesn't exist,
    or load it from storage/ if already built.

    EXCLUDES any *.json files so config/greeting JSON is never treated as docs.
    """
    if not rebuild and STORAGE_DIR.exists():
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
            index = load_index_from_storage(storage_context)
            print(">>> Loaded index from storage/")
            return index
        except Exception as e:
            print(">>> Failed to load index, rebuilding. Error:", e)

    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> Building index from docs/ (excluding *.json)")
    docs = SimpleDirectoryReader(
        input_dir=str(DOCS_DIR),
        recursive=False,
        exclude=["*.json"],   # do not index greetings.json etc.
    ).load_data()

    print(">>> Number of docs indexed:", len(docs))

    if len(docs) == 0:
        raise ValueError(
            f"No documents found in {DOCS_DIR.resolve()} (excluding *.json). "
            "Please add at least one .txt/.pdf/.docx file."
        )

    index = VectorStoreIndex.from_documents(
    docs,
    embed_model=embed_model,
    chunk_size=200,   # force small chunks
    chunk_overlap=0
)


    if not STORAGE_DIR.exists():
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    print(">>> Index built and persisted to storage/")
    return index


# ---------- ANSWER FROM DOCS WITH GUARDS ----------
def answer_from_docs(index, question: str) -> str:
    """
    Use vector search to find the single most relevant chunk.

    Only answer if:
      - similarity score is high enough, AND
      - at least one meaningful keyword from the question appears
        in the candidate text.

    Otherwise: "I don't know based on the documents."
    """
    retriever = VectorIndexRetriever(index=index, similarity_top_k=1)
    nodes = retriever.retrieve(question)

    if not nodes:
        return "I don't know based on the documents."

    best = nodes[0]
    score = getattr(best, "score", None)
    text = best.text if hasattr(best, "text") else best.node.get_content()
    text = text or ""
    text_lower = text.lower()

    print("\n=== Retrieved context for question ===")
    print("Q:", question)
    print("Score:", score)
    print(text[:500])
    print("=== END CONTEXT ===\n")

    # 1) Similarity threshold (quite strict)
    if score is not None and score < 0.7:
        return "I don't know based on the documents."

    # 2) Keyword overlap guard
    words = re.findall(r"\w+", question.lower())
    stopwords = {
        "who", "what", "where", "when", "why", "how",
        "is", "are", "am", "was", "were",
        "a", "an", "the", "of", "in", "on", "at", "for",
        "tell", "me", "about", "please", "do", "does", "did",
        "you", "your"
    }
    keywords = [w for w in words if w not in stopwords]

    keyword_match = any(k in text_lower for k in keywords) if keywords else False

    print(">>> keywords:", keywords)
    print(">>> keyword_match:", keyword_match)

    if not keyword_match:
        return "I don't know based on the documents."

    # Passed both checks → return full text (no artificial cutoff)
    text = text.strip()
    if not text:
        return "I don't know based on the documents."

    return text


# ---------- STREAMLIT UI SETUP ----------
st.set_page_config(page_title="Q&A Bot", page_icon="📚", layout="wide")
st.title("📚 Q&A Bot")

st.markdown(
    "Ask questions related to capital of countries"
    "If something is not in the docs, the bot will say it doesn't know."
)


# ---------- SIDEBAR (ADMIN AREA) ----------
with st.sidebar:
    st.header("Admin Area")

    admin_key = st.text_input("Admin key (optional)", type="password")
    rebuild_pressed = False

    if admin_key == "mysecret":
        st.success("Admin mode active.")
        st.markdown(
            f"Docs folder: `{DOCS_DIR.resolve()}`\n\n"
            "Add or update files there, then click the button below."
        )
        rebuild_pressed = st.button("🔁 Rebuild index from docs/")
    else:
        if admin_key:
            st.error("Incorrect admin key.")
        st.info("Enter admin key to rebuild the index after changing docs.")

    st.markdown("---")
    st.caption("Users will only see the chat on the main page.")


# ---------- BUILD / LOAD INDEX ----------
try:
    index = build_or_load_index(rebuild=rebuild_pressed)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Error while building/loading index: {e}")
    st.stop()


# ---------- CHAT STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- USER INPUT ----------
user_input = st.chat_input("Ask a question about the documents...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    normalized = user_input.strip().lower()
    print("\n>>> USER INPUT RAW:", repr(user_input))
    print(">>> normalized:", repr(normalized))

    # 1) Exact greeting match
    if normalized in GREETINGS_RESPONSES:
        answer = GREETINGS_RESPONSES[normalized]
        print(">>> Matched greeting, using canned reply.")
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    else:
        # 2) Otherwise, answer from docs with similarity + keyword guards
        print(">>> Not a greeting, answering from docs.")
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                try:
                    answer = answer_from_docs(index, user_input)
                except Exception as e:
                    answer = f"Error while generating answer: {e}"
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

