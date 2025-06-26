from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Session ID (tạo mới nếu không có)")
    model_id: str = Field(..., description="ID của model để sử dụng")
    message: str = Field(..., description="Nội dung tin nhắn")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    context_files: Optional[List[str]] = Field(default=None, description="Danh sách file context")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Cấu hình chat")

class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    response: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    timestamp: datetime

class ChatSessionCreate(BaseModel):
    model_id: str = Field(..., description="ID của model")
    session_name: Optional[str] = Field(default=None, description="Tên session")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    context_files: Optional[List[str]] = Field(default=None, description="Danh sách file context")

class ChatSessionResponse(BaseModel):
    session_id: str
    model_id: str
    session_name: Optional[str]
    system_prompt: Optional[str]
    context_files: Optional[List[str]]
    message_count: int
    created_at: datetime
    last_activity: datetime

class ChatMessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    content_type: str
    attached_files: Optional[List[str]]
    timestamp: datetime
    tokens_used: Optional[int]
    response_time: Optional[float]

class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessageResponse]
    total: int 