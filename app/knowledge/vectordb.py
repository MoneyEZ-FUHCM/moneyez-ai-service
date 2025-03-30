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

# Initialize Qdrant client - use local file storage for development
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_data")
COLLECTION_NAME = "knowledge_base"
VECTOR_SIZE = 768  # Gemini embedding size

# Document metadata store - in-memory for simplicity
# In production, use a database
document_metadata = {}

# Create Qdrant client
qdrant_client = QdrantClient(path=QDRANT_PATH)

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


def process_and_store_document(file, filename, content_type, user_id="default_user"):
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
            chunk.metadata["user_id"] = user_id

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
            "user_id": user_id
        }

        return document_id

    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


def delete_document(document_id):
    """Delete document from vector database."""
    print(f"\n[VECTORDB] Attempting to delete document: {document_id}")

    if document_id in document_metadata:
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
                with_payload=False,
                with_vectors=False
            )

            if search_result and search_result[0]:
                point_ids = [point.id for point in search_result[0]]
                if point_ids:
                    qdrant_client.delete(
                        collection_name=COLLECTION_NAME,
                        points_ids=point_ids
                    )

                    # Delete metadata
                    doc_info = document_metadata[document_id]
                    del document_metadata[document_id]
                    print(
                        f"[VECTORDB] Successfully deleted document: {doc_info.get('name', document_id)} with {len(point_ids)} chunks")
                    return True

            print(f"[VECTORDB] No chunks found for document: {document_id}")
            # Clean up metadata even if no chunks were found
            if document_id in document_metadata:
                del document_metadata[document_id]
            return True

        except Exception as e:
            print(f"[VECTORDB] Error deleting document: {str(e)}")
            return False

    print(f"[VECTORDB] Document not found: {document_id}")
    return False


def get_document_list(user_id=None):
    """Get list of documents in the knowledge base."""
    docs = document_metadata.values()

    if user_id:
        docs = [doc for doc in docs if doc.get("user_id") == user_id]

    return [
        {
            "document_id": doc["document_id"],
            "name": doc["name"],
            "size": doc["size"],
            "created_at": doc["created_at"],
            "content_type": doc["content_type"]
        }
        for doc in docs
    ]


def query_knowledge_base(query: str, user_id=None, top_k: int = 3) -> List[Document]:
    """Query the knowledge base for relevant document chunks."""
    print(f"\n[VECTORDB] Querying knowledge base with: {query[:50]}...")
    print(f"[VECTORDB] Retrieving top {top_k} results")

    try:
        # Filter by user_id if provided
        search_kwargs = {}
        if user_id:
            search_kwargs = {
                "filter": Filter(
                    must=[
                        FieldCondition(
                            key="metadata.user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                )
            }

        # Perform similarity search
        results = vector_store.similarity_search(query, k=top_k, **search_kwargs)
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
        # Extract user_id from config if available
        configurable = config.get("configurable", {})
        user_id = configurable.get("user_id", "default_user")

        # Create a retriever function that wraps our query_knowledge_base
        async def retriever(query: str) -> List[Document]:
            print(f"[VECTORDB] Retrieving documents for query: {query[:50]}...")
            return query_knowledge_base(query, user_id=user_id)

        yield retriever
    except Exception as e:
        print(f"[VECTORDB] Error creating retriever: {str(e)}")

        # Provide a fallback retriever that returns no results
        async def fallback_retriever(query: str) -> List[Document]:
            print(f"[VECTORDB] Using fallback retriever (returns no results)")
            return []

        yield fallback_retriever


def ensure_docs_have_user_id(docs: Sequence[Document], user_id: str) -> List[Document]:
    """Ensure all documents have a user_id in their metadata."""
    return [
        Document(
            page_content=doc.page_content,
            metadata={**(doc.metadata or {}), "user_id": user_id}
        )
        for doc in docs
    ]


async def index_documents(docs: List[Document], user_id: str = "default_user") -> bool:
    """Index documents for a specific user."""
    try:
        # Ensure all documents have the user_id in metadata
        stamped_docs = ensure_docs_have_user_id(docs, user_id)

        # Add documents to vector store
        vector_store.add_documents(stamped_docs)

        # Store basic metadata about each document
        for doc in stamped_docs:
            document_id = doc.metadata.get("document_id") or str(uuid.uuid4())
            document_metadata[document_id] = {
                "document_id": document_id,
                "name": doc.metadata.get("source", "Unnamed document"),
                "size": len(doc.page_content),
                "created_at": datetime.now().isoformat(),
                "content_type": "text/plain",
                "user_id": user_id
            }

        print(f"[VECTORDB] Successfully indexed {len(docs)} documents for user {user_id}")
        return True
    except Exception as e:
        print(f"[VECTORDB] Error indexing documents: {str(e)}")
        return False
