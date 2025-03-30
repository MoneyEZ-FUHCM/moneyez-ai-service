from typing import Generic, Optional, TypeVar

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.models import BaseResponse

T = TypeVar("T")


class CustomException(Exception, Generic[T]):
    def __init__(
            self,
            status_code: int,
            error_code: Optional[str] = None,
            message: Optional[str] = None,
            data: Optional[T] = None,
    ):
        self.status_code = status_code
        self.error_code = error_code or str(status_code)
        self.message = message or "An error occurred"
        self.data = data
        self.response = BaseResponse[T](
            status=status_code,
            error_code=self.error_code,
            message=self.message,
            data=data
        )


class CustomHTTPException(HTTPException):
    def __init__(
            self,
            status_code: int,
            error_code: Optional[str] = None,
            message: Optional[str] = None,
            data: Optional[T] = None,
    ):
        self.status_code = status_code
        self.error_code = error_code or str(status_code)
        self.message = message or "An error occurred"
        self.data = data
        self.response = BaseResponse(
            status=status_code,
            error_code=self.error_code,
            message=self.message,
            data=data,
        )
        self.detail = self.message
        super().__init__(status_code=status_code, detail=self.detail)


# Exception handlers to register with FastAPI app
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=BaseResponse(
            status=exc.status_code,
            error_code=str(exc.status_code),
            message=str(exc.detail),
            data=None
        ).dict()
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=BaseResponse(
            status=422,
            error_code="422",
            message="Validation error",
            data=[{"loc": err["loc"], "msg": err["msg"], "type": err["type"]} for err in exc.errors()]
        ).dict()
    )


async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=BaseResponse(
            status=500,
            error_code="500",
            message=str(exc),
            data=None
        ).dict()
    )


def register_exception_handlers(app):
    """Register all exception handlers with the FastAPI app"""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)


