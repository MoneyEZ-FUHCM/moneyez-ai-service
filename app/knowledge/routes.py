from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Query

from app.exception.exception import CustomHTTPException

from .models import DocumentResponse, DocumentListResponse, DocumentDeleteResponse
from .vectordb import process_and_store_document, delete_document, get_document_list, document_metadata

router = APIRouter()


@router.post("/knowledge/upload", response_model=DocumentResponse)
async def upload_document(
        file: UploadFile = File(...),
        user_id: str = Form("default_user")
):
    """Upload a document to the knowledge base."""
    print(f"\n[KNOWLEDGE API] Document upload requested: {file.filename}")
    print(f"[KNOWLEDGE API] Content type: {file.content_type}")
    print(f"[KNOWLEDGE API] User ID: {user_id}")

    try:
        # Read file content
        file_content = await file.read()
        print(f"[KNOWLEDGE API] Read file content, size: {len(file_content)} bytes")

        # Process and store document
        print(f"[KNOWLEDGE API] Processing and storing document...")
        document_id = process_and_store_document(
            file=file_content,
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            user_id=user_id
        )

        # Return document info
        doc_info = document_metadata[document_id]
        print(f"[KNOWLEDGE API] Document processed successfully, ID: {document_id}")
        return DocumentResponse(
            document_id=doc_info["document_id"],
            name=doc_info["name"],
            size=doc_info["size"],
            created_at=doc_info["created_at"],
            content_type=doc_info["content_type"]
        )
    except Exception as e:
        print(f"[KNOWLEDGE API] Error processing document: {str(e)}")
        raise CustomHTTPException(
            status_code=500,
            error_code="DOCUMENT_PROCESSING_ERROR",
            message=f"Error processing document: {str(e)}"
        )


@router.delete("/knowledge/delete/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document_endpoint(document_id: str):
    """Delete a document from the knowledge base."""
    print(f"\n[KNOWLEDGE API] Document deletion requested: {document_id}")

    success = delete_document(document_id)
    if success:
        print(f"[KNOWLEDGE API] Document deleted successfully: {document_id}")
        return DocumentDeleteResponse(
            status="success",
            message=f"Document {document_id} deleted successfully"
        )
    else:
        print(f"[KNOWLEDGE API] Document not found: {document_id}")
        raise CustomHTTPException(
            status_code=404,
            error_code="DOCUMENT_NOT_FOUND",
            message=f"Document {document_id} not found"
        )


@router.get("/knowledge/documents", response_model=DocumentListResponse)
async def get_documents(user_id: Optional[str] = Query(None)):
    """Get list of documents in the knowledge base."""
    print(f"\n[KNOWLEDGE API] Document list requested")
    if user_id:
        print(f"[KNOWLEDGE API] Filtering for user: {user_id}")

    documents = get_document_list(user_id=user_id)
    print(f"[KNOWLEDGE API] Returning {len(documents)} documents")
    return DocumentListResponse(documents=documents)
