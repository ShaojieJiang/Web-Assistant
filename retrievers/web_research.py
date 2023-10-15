from typing import List, Union

from langchain.chains import LLMChain
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities import DuckDuckGoSearchAPIWrapper, GoogleSearchAPIWrapper
from langchain.vectorstores.base import VectorStore
from pydantic import Field


class WebRetriever(WebResearchRetriever):
    # Inputs
    vectorstore: VectorStore = Field(
        ..., description="Vector store for storing web pages"
    )
    llm_chain: LLMChain
    search: Union[DuckDuckGoSearchAPIWrapper, GoogleSearchAPIWrapper] = Field(
        ..., description="Google Search API Wrapper"
    )
    num_search_results: int = Field(1, description="Number of pages per Google search")
    text_splitter: RecursiveCharacterTextSplitter = Field(
        RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50),
        description="Text splitter for splitting web pages into chunks",
    )
    url_database: List[str] = Field(
        default_factory=list, description="List of processed URLs"
    )
