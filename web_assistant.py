from typing import Any

import faiss
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.vectorstores import FAISS
from streamlit_feedback import streamlit_feedback

from retrievers.web_research import WebRetriever as WebResearchRetriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container=None, initial_text=""):
        self.text = initial_text
        self.container = container

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        # TODO: Some tokens may not appear in the final response
        with self.container:
            message = st.chat_message("assistant")
            message.write(self.text)

    def on_chain_start(
        self,
        *args,
        **kwargs,
    ) -> Any:
        with self.container:
            st.status("Running...")

    def on_retriever_start(self, serialized, query: str, **kwargs):
        with self.container:
            st.status(f"Searching query: {query}")


def initialize_page():
    title = "Web Assistant"
    st.set_page_config(page_title=title, page_icon="🌐")
    st.session_state.setdefault("user", [])
    st.session_state.setdefault("assistant", [])
    st.sidebar.title(title)


def setup():
    # Vectorstore
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore_public = FAISS(
        embeddings_model.embed_query, index, InMemoryDocstore({}), {}
    )

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True)

    # Search
    search = DuckDuckGoSearchAPIWrapper()

    # Initialize
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public, llm=llm, search=search, num_search_results=2
    )

    return web_retriever, llm


def on_btn_click():
    del st.session_state.user[:]
    del st.session_state.assistant[:]


def record_feedback(*args, **kwargs):
    # TODO: Implement feedback logging
    vote = args[0]
    print(f"Response {kwargs['ind']} has the vote of {vote['score']}")


# Main app page
initialize_page()
web_retriever, llm = setup()  # Init retriever and llm
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm, retriever=web_retriever, return_source_documents=True
)

# st.chat_input(on_submit=refresh_page, key="user_input")
user_input = st.chat_input(key="user_input")
st.sidebar.button("Clear message", on_click=on_btn_click)
if user_input:
    st.session_state.user.append(user_input)
    st.session_state.assistant.append(
        {"type": "normal", "data": ""}
    )  # Create a placeholder response in the session_state

for i in range(len(st.session_state.assistant)):
    st.chat_message("user").write(st.session_state.user[i])
    assistant_msg = st.session_state.assistant[i]["data"]
    if assistant_msg:
        st.chat_message("assistant").write(st.session_state.assistant[i]["data"])
    else:  # Calls to the chain is only triggered when a placeholder is detected
        container = st.empty()
        with container:
            st.chat_message("assistant").write("")
            stream_handler = StreamHandler(container)
            response = qa_chain(
                {"question": user_input},
                callbacks=[stream_handler],
            )  # Update the empty container in the callback
            st.session_state.assistant[-1] = {  # Update the placeholder response
                "type": "normal",
                "data": response["answer"],
                "sources": response["sources"],
                "docs": response["source_documents"],
            }
            st.chat_message("assistant").write(st.session_state.assistant[-1]["data"])
        container = st.empty()
        with container:
            expander = container.expander("Retrieved results")
            documents = response["source_documents"]
            for idx, doc in enumerate(documents):
                source = doc.metadata["source"]
                expander.write(f"**Results from {source}**")
                expander.text(doc.page_content)

    feedback = streamlit_feedback(
        feedback_type="thumbs",
        key=f"feedback_{i}",
        on_submit=record_feedback,
        kwargs={"ind": i},
    )  # triggers re-rendering after clicking

# TODO: Log conversations
