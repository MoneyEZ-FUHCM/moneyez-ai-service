from typing import Annotated, List, Optional

from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # The add_messages annotation ensures proper handling of message history for conversation memory
    messages: Annotated[list, add_messages]
    # RAG context storage for retrieved knowledge
    rag_context: Optional[List[str]]
    # Store the user's query for RAG processing
    queries: Optional[List[str]]
    # Store retrieved documents for more advanced processing
    retrieved_docs: Optional[List[Document]]


class InputState(TypedDict):
    """The expected input to the graph from users."""
    messages: list


class IndexState(TypedDict):
    """State for the document indexing process."""
    docs: List[Document]
