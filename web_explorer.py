import streamlit as st
import logging
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from streamlit_chat import message

from retrievers.web_research import WebRetriever as WebResearchRetriever

title = "Web Assistant"

st.set_page_config(page_title=title, page_icon="🌐")
# st.sidebar.image("img/ai.png")
st.header(f"`{title}`")
st.info(
    "`I am an AI that can answer questions by exploring, reading, and summarizing web pages."  # noqa: E501
    "I can be configured to use different modes: public API or private (no data sharing).`"  # noqa: E501
)


def settings():
    # Vectorstore
    import faiss
    from langchain.docstore import InMemoryDocstore
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS

    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore_public = FAISS(
        embeddings_model.embed_query, index, InMemoryDocstore({}), {}
    )

    # LLM
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True)

    # Search
    search = DuckDuckGoSearchAPIWrapper()

    # Initialize
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public, llm=llm, search=search, num_search_results=2
    )

    return web_retriever, llm


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)

# Make retriever and llm
if "retriever" not in st.session_state:
    st.session_state["retriever"], st.session_state["llm"] = settings()
web_retriever = st.session_state.retriever
llm = st.session_state.llm

# User input
question = st.text_input("`Ask a question:`")

with st.container():
    if question:
        # Generate answer (w/ citations)
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)

        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)

        # Write answer and sources
        retrieval_streamer_cb = PrintRetrievalHandler(st.container())
        answer = st.empty()
        stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
        result = qa_chain(
            {"question": question}, callbacks=[retrieval_streamer_cb, stream_handler]
        )
        answer.info("`Answer:`\n\n" + result["answer"])
        st.info("`Sources:`\n\n" + result["sources"])
