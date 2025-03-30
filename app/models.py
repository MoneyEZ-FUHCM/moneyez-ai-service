from typing import Generic, List, Literal, Union, Optional, Dict, Any

from pydantic import BaseModel
from typing import TypeVar, Optional, Dict, Any


class Conversation(BaseModel):
    conversation_id: str
    title: Optional[str] = "New Conversation"
    created_at: str
    updated_at: str


class ConversationCreate(BaseModel):
    conversation_id: str
    title: Optional[str] = "New Conversation"


class ConversationUpdate(BaseModel):
    title: Optional[str] = None


class MessageBase(BaseModel):
    conversation_id: str
    role: str
    content: str
    timestamp: Optional[str] = None


class MessageCreate(MessageBase):
    pass


class MessageResponse(MessageBase):
    id: Optional[str] = None
    tool_info: Optional[Dict[str, Any]] = None


class LanguageModelTextPart(BaseModel):
    type: Literal["text"]
    text: str
    providerMetadata: Optional[Any] = None


class LanguageModelImagePart(BaseModel):
    type: Literal["image"]
    mimeType: Optional[str] = None
    providerMetadata: Optional[Any] = None


class LanguageModelFilePart(BaseModel):
    type: Literal["file"]
    mimeType: str
    providerMetadata: Optional[Any] = None


class LanguageModelToolCallPart(BaseModel):
    type: Literal["tool-call"]
    toolCallId: str
    toolName: str
    args: Any
    providerMetadata: Optional[Any] = None


class LanguageModelToolResultContentPart(BaseModel):
    type: Literal["text", "image"]
    text: Optional[str] = None
    data: Optional[str] = None
    mimeType: Optional[str] = None


class LanguageModelToolResultPart(BaseModel):
    type: Literal["tool-result"]
    toolCallId: str
    toolName: str
    result: Any
    isError: Optional[bool] = None
    content: Optional[List[LanguageModelToolResultContentPart]] = None
    providerMetadata: Optional[Any] = None


class LanguageModelSystemMessage(BaseModel):
    role: Literal["system"]
    content: str


class LanguageModelUserMessage(BaseModel):
    role: Literal["user"]
    content: List[
        Union[LanguageModelTextPart, LanguageModelImagePart, LanguageModelFilePart]
    ]


class LanguageModelAssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: List[Union[LanguageModelTextPart, LanguageModelToolCallPart]]


class LanguageModelToolMessage(BaseModel):
    role: Literal["tool"]
    content: List[LanguageModelToolResultPart]


LanguageModelV1Message = Union[
    LanguageModelSystemMessage,
    LanguageModelUserMessage,
    LanguageModelAssistantMessage,
    LanguageModelToolMessage,
]


class FrontendToolCall(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: dict[str, Any]


class ChatRequest(BaseModel):
    system: Optional[str] = ""
    tools: Optional[List[FrontendToolCall]] = []
    messages: List[LanguageModelV1Message]


class AnyArgsSchema(BaseModel):
    class Config:
        extra = "allow"


class StatusResponse(BaseModel):
    status: str
    message: str


class HealthCheckResponse(BaseModel):
    status: str
    mongodb: bool


class ConversationHistoryResponse(BaseModel):
    messages: List[Dict[str, Any]]
    error: Optional[str] = None

class MessageBaseCShap(BaseModel):
    ConversationId: str
    Content: str
    Role: str
    Timestamp: Optional[str] = None

# class RevicedMessage(BaseModel):
#     userId: str
#     message: str
#     conversationId: str
#     PreviousMessages: Optional[List[MessageBaseCShap]] = None
class RevicedMessage(BaseModel):
    data: str

class RevicedMessageParse(BaseModel):
    userId: str
    message: str
    conversationId: str
    PreviousMessages: Optional[List[MessageBaseCShap]] = None

T = TypeVar('T')

class BaseResponse(BaseModel, Generic[T]):
    status: Optional[int] = 200
    message: Optional[str] = None
    error_code: Optional[str] = None
    data: Optional[T] = None

class MessageResponse(BaseModel):
                    status: Literal["success", "error"]
                    conversation_id: str
                    message: Dict[str, Any]
