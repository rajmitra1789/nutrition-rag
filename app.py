"""Streamlit chat UI for the dietary guidance RAG assistant."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.answer.generate import answer_question
from src.retrieve.index import document_catalog

st.set_page_config(
    page_title="Dietary Guidance Assistant",
    page_icon="🥗",
    layout="wide",
)

CATALOG = document_catalog()
DOC_LABELS = {
    "all": "All documents",
    **{doc["doc_id"]: f"{doc['document_name']} ({doc['year']})" for doc in CATALOG},
}


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _render_hits(hits) -> None:
    if not hits:
        return
    with st.expander("Retrieved passages", expanded=False):
        for hit in hits:
            st.markdown(
                f"**{hit.document_name}** · {hit.section_heading} · similarity {hit.similarity:.2f}"
            )
            st.caption(f"{hit.publisher}, {hit.year} — {hit.source_url}")
            st.code(hit.text, language=None)


def main() -> None:
    _init_state()

    st.title("Dietary Guidance Assistant")
    st.caption(
        "Answers come only from official public dietary guidance. Every claim is cited. "
        "This is not medical advice and it will not set calorie or weight targets."
    )

    with st.sidebar:
        st.header("Search scope")
        selected = st.selectbox(
            "Retrieve from",
            options=list(DOC_LABELS.keys()),
            format_func=lambda key: DOC_LABELS[key],
        )
        st.markdown("### Corpus")
        for doc in CATALOG:
            st.markdown(
                f"- [{doc['document_name']}]({doc['source_url']})  \n"
                f"  {doc['publisher']}, {doc['year']}"
            )
        st.markdown("---")
        st.markdown(
            "Out of scope: medical advice, personal calorie targets, "
            "and what anyone should weigh."
        )
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("hits"):
                _render_hits(item["hits"])

    prompt = st.chat_input("Ask about food, nutrition, or food safety guidance")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt, "hits": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    doc_id = None if selected == "all" else selected
    with st.chat_message("assistant"):
        with st.spinner("Searching official guidance…"):
            try:
                message, result, refusal = answer_question(prompt, doc_id=doc_id)
            except Exception as exc:
                message = f"I could not complete that request: {exc}"
                result, refusal = None, None
        st.markdown(message)
        hits = result.hits if result and not refusal else []
        if hits:
            _render_hits(hits)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": message,
            "hits": hits,
        }
    )


if __name__ == "__main__":
    main()
