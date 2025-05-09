import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Any, Generator, Sequence

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

# Load environment variables from .env file
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    print("GOOGLE_API_KEY not found in environment variables. Please set it.")

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Qdrant configuration
COLLECTION_NAME = "knowledge_base"
VECTOR_SIZE = 768  # Gemini embedding size
QDRANT_URL = os.environ.get("QDRANT_URL", "http://178.128.118.171:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# Document metadata store - in-memory for simplicity
# In production, use a database
document_metadata = {}

# Create Qdrant client - connect to external server
print(f"[VECTORDB] Connecting to Qdrant at {QDRANT_URL}")
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60  # Increase timeout for operations
)

# Try to create collection if it doesn't exist
try:
    qdrant_client.get_collection(COLLECTION_NAME)
    print(f"[VECTORDB] Using existing Qdrant collection: {COLLECTION_NAME}")
except Exception:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"[VECTORDB] Created new Qdrant collection: {COLLECTION_NAME}")

# Initialize vector store - Fix the parameter name from embeddings to embedding
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,  # Changed from embeddings to embedding
)


def make_text_encoder(model_name: str) -> Embeddings:
    """Create an embedding model based on configuration."""
    # We already have Gemini embeddings initialized, so just return them
    # This could be expanded to support other models as needed
    return embeddings


# Helper functions for document processing
def get_document_loader(file_path, content_type):
    """Returns the appropriate document loader based on file type."""
    if content_type == "application/pdf":
        return PyPDFLoader(file_path)
    elif content_type == "text/plain":
        return TextLoader(file_path)
    elif content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                          "application/msword"]:
        return Docx2txtLoader(file_path)
    elif content_type == "text/html":
        return UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {content_type}")


def process_and_store_document(file, filename, content_type):
    """Process document and store it in the vector database."""
    print(f"\n[VECTORDB] Processing document: {filename} ({content_type})")
    print(f"[VECTORDB] File size: {len(file)} bytes")

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file)
        temp_file_path = temp_file.name

    try:
        # Load document
        loader = get_document_loader(temp_file_path, content_type)
        documents = loader.load()
        print(f"[VECTORDB] Loaded {len(documents)} document segments")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[VECTORDB] Created {len(chunks)} chunks for embedding")

        # Generate document ID
        document_id = str(uuid.uuid4())

        # Add document ID and user_id to metadata for each chunk
        for chunk in chunks:
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["document_id"] = document_id
            chunk.metadata["document_name"] = filename

        # Store document chunks in vector database
        vector_store.add_documents(chunks)
        print(f"[VECTORDB] Successfully stored chunks in vector database")

        # Store document metadata
        document_metadata[document_id] = {
            "document_id": document_id,
            "name": filename,
            "size": len(file),
            "created_at": datetime.now().isoformat(),
            "content_type": content_type,
            "chunk_count": len(chunks),
        }

        return document_id

    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


def delete_document(document_id):
    """Delete document from vector database."""
    print(f"\n[VECTORDB] Attempting to delete document: {document_id}")
    
    try:
        # Create a proper filter using Qdrant's models
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )

        # Get point IDs with the specified document_id
        search_result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filter_condition,
            limit=10000,  # Adjust based on expected max chunks per document
            with_payload=True,  # Get payload to extract document name for logging
            with_vectors=False
        )

        doc_name = document_id
        if document_id in document_metadata:
            doc_name = document_metadata[document_id].get('name', document_id)
            
        # Try to delete points even if not in memory metadata
        if search_result and search_result[0]:
            point_ids = [point.id for point in search_result[0]]
            
            if point_ids:
                # Extract document name from the first point if available
                if not doc_name and search_result[0][0].payload and "metadata" in search_result[0][0].payload:
                    doc_name = search_result[0][0].payload["metadata"].get("document_name", document_id)
                    
                # Delete from Qdrant - Updated to use the new API format with points_selector
                qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )

                print(f"[VECTORDB] Successfully deleted document: {doc_name} with {len(point_ids)} chunks")
                
                # Also delete from in-memory metadata if it exists
                if document_id in document_metadata:
                    del document_metadata[document_id]
                    
                return True
            else:
                print(f"[VECTORDB] Found document but no points to delete: {doc_name}")
                
        # Clean up metadata even if no chunks were found
        if document_id in document_metadata:
            del document_metadata[document_id]
            print(f"[VECTORDB] Removed document metadata for: {doc_name}")
            return True
                
        print(f"[VECTORDB] No document found with ID: {document_id}")
        return False

    except Exception as e:
        print(f"[VECTORDB] Error deleting document: {str(e)}")
        return False


def _get_documents_from_qdrant():
    """Get unique documents from Qdrant collection."""
    try:
        # Get all points from collection with metadata
        result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            with_vectors=False,
            limit=10000
        )
        
        # Extract unique document IDs and metadata
        unique_docs = {}
        if result and result[0]:
            # Calculate total content size for each document
            content_sizes = {}
            
            for point in result[0]:
                if point.payload and "metadata" in point.payload:
                    doc_id = point.payload["metadata"].get("document_id")
                    doc_name = point.payload["metadata"].get("document_name")
                    
                    if doc_id and doc_name:
                        # Add content length to document's total size
                        if "page_content" in point.payload and isinstance(point.payload["page_content"], str):
                            content_sizes[doc_id] = content_sizes.get(doc_id, 0) + len(point.payload["page_content"])
                        
                        if doc_id not in unique_docs:
                            unique_docs[doc_id] = {
                                "document_id": doc_id,
                                "name": doc_name,
                                "size": 0,  # Will update with calculated size
                                "created_at": datetime.now().isoformat(),
                                "content_type": point.payload["metadata"].get("content_type", "unknown")
                            }
            
            # Update document sizes
            for doc_id, size in content_sizes.items():
                if doc_id in unique_docs:
                    unique_docs[doc_id]["size"] = max(size * 2, 1024)  # Estimate original file size, min 1KB
                    
        return unique_docs
    except Exception as e:
        print(f"[VECTORDB] Error getting documents from Qdrant: {str(e)}")
        return {}


def get_document_list():
    """Get list of documents from both metadata and vector store."""
    try:
        print(f"[VECTORDB] Getting document list")
        
        # Get documents from both sources
        metadata_docs = document_metadata
        vector_docs = _get_documents_from_qdrant()
        
        # Merge documents (vector store docs take precedence)
        all_docs = {**metadata_docs, **vector_docs}
        
        print(f"[VECTORDB] Found {len(metadata_docs)} docs in metadata, {len(vector_docs)} in vector store")
        
        result = [
            {
                "id": doc["document_id"],
                "name": doc["name"],
                "size": doc["size"],
                "createdDate": doc["created_at"],
                "contentType": doc["content_type"]
            }
            for doc in all_docs.values()
        ]
        
        print(f"[VECTORDB] Returning {len(result)} total documents")
        return result
    except Exception as e:
        print(f"[VECTORDB] Error getting document list: {str(e)}")
        return []


def query_knowledge_base(query: str, top_k: int = 3) -> List[Document]:
    """Query the knowledge base for relevant document chunks."""
    print(f"\n[VECTORDB] Querying knowledge base with: {query[:50]}...")
    print(f"[VECTORDB] Retrieving top {top_k} results")

    try:

        # Perform similarity search
        results = vector_store.similarity_search(query, k=top_k)
        print(f"[VECTORDB] Found {len(results)} results")

        for i, doc in enumerate(results):
            metadata_str = ", ".join([f"{k}={v}" for k, v in doc.metadata.items()])
            print(f"[VECTORDB] Result {i + 1} metadata: {metadata_str}")
            print(f"[VECTORDB] Result {i + 1} snippet: {doc.page_content[:50]}...")

        return results
    except Exception as e:
        print(f"[VECTORDB] Error querying knowledge base: {str(e)}")
        return []


@contextmanager
def make_retriever(config: Dict[str, Any]) -> Generator[Any, None, None]:
    """Create a retriever for the agent based on the configuration."""
    try:

        # Create a retriever function that wraps our query_knowledge_base
        async def retriever(query: str) -> List[Document]:
            print(f"[VECTORDB] Retrieving documents for query: {query[:50]}...")
            return query_knowledge_base(query)

        yield retriever
    except Exception as e:
        print(f"[VECTORDB] Error creating retriever: {str(e)}")

        # Provide a fallback retriever that returns no results
        async def fallback_retriever(query: str) -> List[Document]:
            print(f"[VECTORDB] Using fallback retriever (returns no results)")
            return []

        yield fallback_retriever


async def index_documents(docs: List[Document]) -> bool:
    """Index documents directly into the vector store."""
    try:
        # Add documents to vector store
        vector_store.add_documents(docs)

        # Store basic metadata about each document
        for doc in docs:
            document_id = doc.metadata.get("document_id") or str(uuid.uuid4())
            document_metadata[document_id] = {
                "document_id": document_id,
                "name": doc.metadata.get("source", "Unnamed document"),
                "size": len(doc.page_content),
                "created_at": datetime.now().isoformat(),
                "content_type": "text/plain"
            }

        print(f"[VECTORDB] Successfully indexed {len(docs)} documents")
        return True
    except Exception as e:
        print(f"[VECTORDB] Error indexing documents: {str(e)}")
        return False
