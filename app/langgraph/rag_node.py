from typing import Dict, List, Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from .state import AgentState
from ..knowledge.vectordb import make_retriever


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""
    query: str


def get_message_text(msg):
    """Extract text content from various message formats."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        texts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(texts).strip()


def format_docs(docs: List[Document]) -> str:
    """Format documents as XML for inclusion in prompts."""
    if not docs:
        return "<documents></documents>"

    formatted_docs = []
    for doc in docs:
        metadata = doc.metadata or {}
        meta_str = "".join(f" {k}={v!r}" for k, v in metadata.items())
        if meta_str:
            meta_str = f" {meta_str}"
        formatted_docs.append(f"<document{meta_str}>\n{doc.page_content}\n</document>")

    return f"<documents>\n{''.join(formatted_docs)}\n</documents>"


async def should_use_rag(state, config):
    """Determine if RAG should be used for the current query."""
    use_rag = config.get("configurable", {}).get("use_rag", True)

    # Return a state update with an empty list for queries to ensure we update a valid state key
    return {
        "need_rag": False,
        "queries": []  # Include an empty list for queries to satisfy the state update requirement
    }


async def generate_query(state: AgentState, config: Dict[str, Any]) -> Dict:
    """Generate a search query based on user input."""
    messages = state.get("messages", [])

    # Only process if there's at least one message and it's from the user
    if not messages or not isinstance(messages[-1], HumanMessage):
        # Return empty queries list instead of empty dict
        return {"queries": []}

    # Extract the user query
    human_input = get_message_text(messages[-1])

    # For now, use the raw query directly
    queries = state.get("queries", [])
    queries.append(human_input)

    return {"queries": queries}


async def retrieve_knowledge(state: AgentState, config: Dict[str, Any]) -> Dict:
    """Retrieve relevant documents based on the query."""
    # Check if we have queries
    queries = state.get("queries", [])
    if not queries:
        # Return empty lists instead of empty dict
        return {
            "rag_context": [],
            "retrieved_docs": []
        }

    # Use the most recent query
    user_query = queries[-1]

    try:
        # Use the context manager to get a retriever
        with make_retriever(config) as retriever:
            # Retrieve documents
            doc_objects = await retriever(user_query)

            # Always return the keys even if empty
            return {
                "rag_context": [doc.page_content for doc in doc_objects] if doc_objects else [],
                "retrieved_docs": doc_objects if doc_objects else []
            }
    except Exception as e:
        print(f"[RAG NODE] Error in knowledge retrieval: {str(e)}")
        # Return empty lists instead of empty dict
        return {
            "rag_context": [],
            "retrieved_docs": []
        }
