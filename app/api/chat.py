from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.schemas.chat import (
    ChatRequest, ChatResponse, ChatSessionCreate, ChatSessionResponse,
    ChatHistoryResponse
)
from app.services.chat_service import ChatService
from app.config import settings

router = APIRouter(prefix="/api/chat", tags=["Chat"])

# Service instance
chat_service = ChatService()

@router.post("/send", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """Gửi tin nhắn và nhận response"""
    try:
        result = chat_service.send_message(db, request)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return ChatResponse(
            session_id=result["session_id"],
            message_id=result["message_id"],
            response=result["response"],
            tokens_used=result["tokens_used"],
            response_time=result["response_time"],
            timestamp=result["timestamp"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session", response_model=ChatSessionResponse)
async def create_chat_session(
    request: ChatSessionCreate,
    db: Session = Depends(get_db)
):
    """Tạo chat session mới"""
    try:
        session = chat_service.create_chat_session(db, request)
        
        return ChatSessionResponse(
            session_id=session.session_id,
            model_id=session.model_id,
            session_name=session.session_name,
            system_prompt=session.system_prompt,
            context_files=session.context_files,
            message_count=session.message_count,
            created_at=session.created_at,
            last_activity=session.last_activity
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Lấy thông tin chat session"""
    try:
        session = chat_service.get_chat_session(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return ChatSessionResponse(
            session_id=session.session_id,
            model_id=session.model_id,
            session_name=session.session_name,
            system_prompt=session.system_prompt,
            context_files=session.context_files,
            message_count=session.message_count,
            created_at=session.created_at,
            last_activity=session.last_activity
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    page: int = 1,
    size: int = 20,
    db: Session = Depends(get_db)
):
    """Lấy lịch sử chat"""
    try:
        result = chat_service.get_chat_history(db, session_id, page, size)
        
        return ChatHistoryResponse(
            session_id=result["session_id"],
            messages=result["messages"],
            total=result["total"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/{session_id}/upload")
async def upload_context_files(
    session_id: str,
    files: List[str],
    db: Session = Depends(get_db)
):
    """Upload context files cho session"""
    try:
        success = chat_service.upload_context_files(db, session_id, files)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Context files uploaded successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions", response_model=dict)
async def get_active_sessions(
    user_id: Optional[str] = None,
    page: int = 1,
    size: int = 10,
    db: Session = Depends(get_db)
):
    """Lấy danh sách active sessions"""
    try:
        result = chat_service.get_active_sessions(db, user_id, page, size)
        
        return {
            "sessions": result["sessions"],
            "total": result["total"],
            "page": result["page"],
            "size": result["size"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/session/{session_id}")
async def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Xóa chat session"""
    try:
        success = chat_service.delete_session(db, session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Session deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/{session_id}/update")
async def update_session(
    session_id: str,
    session_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Cập nhật thông tin session"""
    try:
        session = chat_service.get_chat_session(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Cập nhật các trường được phép
        if session_name is not None:
            session.session_name = session_name
        if system_prompt is not None:
            session.system_prompt = system_prompt
        
        db.commit()
        
        return {"message": "Session updated successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 