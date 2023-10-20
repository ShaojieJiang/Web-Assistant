import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from streamlit_chat import message
from streamlit_feedback import streamlit_feedback

from retrievers.web_research import WebRetriever as WebResearchRetriever


def initialize_page():
    title = "Web Assistant"
    st.set_page_config(page_title=title, page_icon="🌐")
    st.session_state.setdefault("past", [])
    st.session_state.setdefault("generated", [])
    st.sidebar.title(title)


def setup():
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


def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)
    # TODO: Uncomment the lines below when finished debugging
    # response = qa_chain({"question": user_input})
    # st.session_state.generated.append({'type': 'normal', 'data': response['answer']})
    st.session_state.generated.append({"type": "normal", "data": "blah"})


def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]


initialize_page()
# Init retriever and llm
web_retriever, llm = setup()
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)

for i in range(len(st.session_state["generated"])):
    message(st.session_state["past"][i], is_user=True, key=f"{i}_user")
    message(
        st.session_state["generated"][i]["data"],
        key=f"{i}",
        allow_html=True,
        is_table=True if st.session_state["generated"][i]["type"] == "table" else False,
    )
    feedback = streamlit_feedback(
        feedback_type="thumbs", align="flex-start", key=f"{i}_feedback"
    )

st.chat_input(on_submit=on_input_change, key="user_input")
st.sidebar.button("Clear message", on_click=on_btn_click)
