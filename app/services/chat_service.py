import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import logging

from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage
from app.models.model_info import ModelInfo
from app.schemas.chat import ChatRequest, ChatSessionCreate
from app.utils.chat_utils import ChatUtils
from app.utils.model_utils import ModelUtils
from app.utils.file_processor import FileProcessor
from app.config import settings

logger = logging.getLogger(__name__)

class ChatService:
    """Service để xử lý chat functionality"""
    
    def __init__(self):
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
    
    def create_chat_session(
        self,
        db: Session,
        request: ChatSessionCreate
    ) -> ChatSession:
        """Tạo chat session mới"""
        try:
            # Kiểm tra model có tồn tại không
            model = db.query(ModelInfo).filter(ModelInfo.model_id == request.model_id).first()
            if not model:
                raise ValueError(f"Model not found: {request.model_id}")
            
            # Tạo session data
            session_data = ChatUtils.create_chat_session_data(
                model_id=request.model_id,
                session_name=request.session_name,
                system_prompt=request.system_prompt,
                context_files=request.context_files
            )
            
            # Tạo session record
            session = ChatSession(**session_data)
            db.add(session)
            db.commit()
            db.refresh(session)
            
            logger.info(f"Created chat session: {session.session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            db.rollback()
            raise
    
    def get_chat_session(
        self,
        db: Session,
        session_id: str
    ) -> Optional[ChatSession]:
        """Lấy chat session"""
        return db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    
    def send_message(
        self,
        db: Session,
        request: ChatRequest
    ) -> Dict[str, Any]:
        """Gửi tin nhắn và nhận response"""
        try:
            start_time = time.time()
            
            # Validate request
            validation = ChatUtils.validate_chat_request(request.message, request.model_id)
            if not validation["valid"]:
                return {
                    "error": "Validation failed",
                    "details": validation["errors"]
                }
            
            # Lấy hoặc tạo session
            session = None
            if request.session_id:
                session = self.get_chat_session(db, request.session_id)
            
            if not session:
                # Tạo session mới
                session_create = ChatSessionCreate(
                    model_id=request.model_id,
                    session_name=f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    system_prompt=request.system_prompt,
                    context_files=request.context_files
                )
                session = self.create_chat_session(db, session_create)
            
            # Lấy model
            model_info = db.query(ModelInfo).filter(ModelInfo.model_id == request.model_id).first()
            if not model_info:
                return {"error": "Model not found"}
            
            # Load model nếu chưa load
            model, tokenizer = self._get_or_load_model(model_info.model_path)
            
            # Lấy chat history
            chat_history = self._get_chat_history(db, session.session_id)
            
            # Xây dựng prompt
            prompt = ChatUtils.build_chat_prompt(
                message=request.message,
                system_prompt=session.system_prompt or "",
                context_files=session.context_files,
                chat_history=chat_history
            )
            
            # Sanitize message
            sanitized_message = ChatUtils.sanitize_message(request.message)
            
            # Lưu user message
            user_message_data = ChatUtils.create_chat_message_data(
                session_id=session.session_id,
                role="user",
                content=sanitized_message
            )
            user_message = ChatMessage(**user_message_data)
            db.add(user_message)
            
            # Generate response
            response = ModelUtils.generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt
            )
            
            # Tính thời gian response
            response_time = time.time() - start_time
            
            # Lưu assistant message
            assistant_message_data = ChatUtils.create_chat_message_data(
                session_id=session.session_id,
                role="assistant",
                content=response,
                tokens_used=len(tokenizer.encode(response)),
                response_time=response_time
            )
            assistant_message = ChatMessage(**assistant_message_data)
            db.add(assistant_message)
            
            # Cập nhật session
            session.message_count += 2
            session.last_activity = datetime.now()
            
            db.commit()
            
            return {
                "session_id": session.session_id,
                "message_id": assistant_message.message_id,
                "response": response,
                "tokens_used": assistant_message.tokens_used,
                "response_time": response_time,
                "timestamp": assistant_message.timestamp
            }
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            db.rollback()
            return {"error": str(e)}
    
    def _get_or_load_model(self, model_path: str) -> tuple:
        """Lấy hoặc load model"""
        if model_path in self.loaded_models:
            return self.loaded_models[model_path]["model"], self.loaded_models[model_path]["tokenizer"]
        
        try:
            # Load model
            model, tokenizer = ModelUtils.load_finetuned_model(model_path)
            
            # Cache model
            self.loaded_models[model_path] = {
                "model": model,
                "tokenizer": tokenizer,
                "loaded_at": datetime.now()
            }
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            raise
    
    def _get_chat_history(
        self,
        db: Session,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """Lấy chat history"""
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.timestamp.desc()).limit(limit).all()
        
        # Đảo ngược để có thứ tự đúng
        messages.reverse()
        
        history = []
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return history
    
    def get_chat_history(
        self,
        db: Session,
        session_id: str,
        page: int = 1,
        size: int = 20
    ) -> Dict[str, Any]:
        """Lấy chat history với pagination"""
        offset = (page - 1) * size
        
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.timestamp.asc()).offset(offset).limit(size).all()
        
        total = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).count()
        
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "message_id": msg.message_id,
                "role": msg.role,
                "content": msg.content,
                "content_type": msg.content_type,
                "timestamp": msg.timestamp,
                "tokens_used": msg.tokens_used,
                "response_time": msg.response_time
            })
        
        return {
            "session_id": session_id,
            "messages": formatted_messages,
            "total": total,
            "page": page,
            "size": size
        }
    
    def upload_context_files(
        self,
        db: Session,
        session_id: str,
        files: List[str]
    ) -> bool:
        """Upload context files cho session"""
        try:
            session = self.get_chat_session(db, session_id)
            if not session:
                return False
            
            # Validate files
            for file_path in files:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                if not FileProcessor.is_supported_file(
                    file_path, 
                    settings.allowed_extensions
                ):
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
            
            # Cập nhật context files
            current_files = session.context_files or []
            session.context_files = current_files + files
            session.last_activity = datetime.now()
            
            db.commit()
            
            logger.info(f"Updated context files for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading context files: {e}")
            db.rollback()
            return False
    
    def get_active_sessions(
        self,
        db: Session,
        user_id: Optional[str] = None,
        page: int = 1,
        size: int = 10
    ) -> Dict[str, Any]:
        """Lấy danh sách active sessions"""
        offset = (page - 1) * size
        
        query = db.query(ChatSession)
        if user_id:
            query = query.filter(ChatSession.user_id == user_id)
        
        sessions = query.order_by(
            ChatSession.last_activity.desc()
        ).offset(offset).limit(size).all()
        
        total = query.count()
        
        formatted_sessions = []
        for session in sessions:
            formatted_sessions.append({
                "session_id": session.session_id,
                "model_id": session.model_id,
                "session_name": session.session_name,
                "message_count": session.message_count,
                "created_at": session.created_at,
                "last_activity": session.last_activity
            })
        
        return {
            "sessions": formatted_sessions,
            "total": total,
            "page": page,
            "size": size
        }
    
    def delete_session(
        self,
        db: Session,
        session_id: str
    ) -> bool:
        """Xóa chat session"""
        try:
            session = self.get_chat_session(db, session_id)
            if not session:
                return False
            
            # Xóa tất cả messages
            db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).delete()
            
            # Xóa session
            db.delete(session)
            db.commit()
            
            logger.info(f"Deleted chat session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            db.rollback()
            return False 