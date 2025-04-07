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
    # Get configuration setting for RAG
    use_rag = config.get("configurable", {}).get("use_rag", True)
    
    # If RAG is disabled in config, don't use it
    if not use_rag:
        return {
            "need_rag": False,
            "queries": [] 
        }
    
    messages = state.get("messages", [])
    
    # Only process if there's at least one message and it's from the user
    if not messages or not isinstance(messages[-1], HumanMessage):
        return {
            "need_rag": False,
            "queries": []
        }
    
    # Extract the user query
    human_input = get_message_text(messages[-1])
    
    # Determine if the query is informational and would benefit from RAG
    # Keywords that suggest knowledge retrieval would be helpful
    knowledge_keywords = [
        "tài chính", "thông tin", "giải thích", "là gì", "định nghĩa", 
        "khái niệm", "cách", "làm sao", "tư vấn", "nên", "hướng dẫn",
        "quy định", "luật", "chính sách", "so sánh", "khác nhau"
    ]
    
    # Check if any knowledge keywords are in the query
    need_rag = any(keyword in human_input.lower() for keyword in knowledge_keywords)
    
    print(f"[RAG DECISION] Query: {human_input}")
    print(f"[RAG DECISION] Using RAG: {need_rag}")
    
    return {
        "need_rag": need_rag,
        "queries": []  # We'll generate the actual search query in the next step if needed
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
    
    # Optimize query for search
    # Remove filler words and phrases to focus on key terms
    filler_words = [
        "cho tôi", "giúp tôi", "làm ơn", "xin vui lòng", "bạn có thể", 
        "tôi muốn", "tôi cần", "xin", "hãy", "thông tin về"
    ]
    
    optimized_query = human_input
    for word in filler_words:
        optimized_query = optimized_query.replace(word, "").strip()
    
    # Extract key financial terms from the query
    financial_terms = [
        "đầu tư", "tiết kiệm", "chi tiêu", "thu nhập", "lãi suất", 
        "chứng khoán", "cổ phiếu", "trái phiếu", "ngân sách", "vay", "nợ",
        "thuế", "bảo hiểm", "quỹ", "tài chính cá nhân"
    ]
    
    # If query is very short after optimization, use the original
    if len(optimized_query) < len(human_input) / 2:
        optimized_query = human_input
    
    print(f"[QUERY GENERATION] Original: {human_input}")
    print(f"[QUERY GENERATION] Optimized: {optimized_query}")
    
    # Store both original and optimized queries
    queries = state.get("queries", [])
    queries.append(optimized_query)
    
    # If we have financial keywords in the query, add them as a focused query as well
    keyword_query = " ".join([term for term in financial_terms if term in human_input.lower()])
    if keyword_query:
        queries.append(keyword_query)
        print(f"[QUERY GENERATION] Keyword query: {keyword_query}")

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

    # Prepare for document collection from all queries
    all_docs = []
    all_doc_content = []
    
    try:
        # Use the context manager to get a retriever
        with make_retriever(config) as retriever:
            # Process each query
            for query in queries:
                print(f"[RAG RETRIEVAL] Processing query: {query}")
                
                # Retrieve documents for this query
                doc_objects = await retriever(query)
                
                if doc_objects:
                    print(f"[RAG RETRIEVAL] Found {len(doc_objects)} documents for query: {query}")
                    
                    # Add unique documents to our collection
                    for doc in doc_objects:
                        # Check if this document is unique (by content)
                        if doc.page_content not in all_doc_content:
                            all_docs.append(doc)
                            all_doc_content.append(doc.page_content)
            
            # Sort documents by relevance if we have multiple
            # This is a simple implementation - in a real system you might want more sophisticated ranking
            if len(all_docs) > 1:
                # Use the first query (usually the most comprehensive) for ranking
                main_query = queries[0]
                
                # Simple ranking function - could be replaced with a more sophisticated one
                def rank_doc(doc):
                    # Count term overlap between query and document
                    query_terms = set(main_query.lower().split())
                    doc_terms = set(doc.page_content.lower().split())
                    overlap = len(query_terms.intersection(doc_terms))
                    return overlap
                
                # Sort documents by our ranking function
                all_docs.sort(key=rank_doc, reverse=True)
                
                # Update the content list to match the new order
                all_doc_content = [doc.page_content for doc in all_docs]
            
            print(f"[RAG RETRIEVAL] Total unique documents retrieved: {len(all_docs)}")
            
            # Always return the keys even if empty
            return {
                "rag_context": all_doc_content,
                "retrieved_docs": all_docs
            }
    except Exception as e:
        print(f"[RAG NODE] Error in knowledge retrieval: {str(e)}")
        # Return empty lists instead of empty dict
        return {
            "rag_context": [],
            "retrieved_docs": []
        }
