from typing import List, Optional, Dict, Any

from pydantic import BaseModel


class Document(BaseModel):
    """Model for document metadata stored in the knowledge base."""
    document_id: str
    name: str
    size: int
    created_at: str
    content_type: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    """Response model for document information."""
    document_id: str
    name: str
    size: int
    created_at: str
    content_type: str


class DocumentListResponse(BaseModel):
    """Response model for list of documents."""
    documents: List[DocumentResponse]


class DocumentDeleteResponse(BaseModel):
    """Response model for document deletion."""
    status: str
    message: str
