import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ChatUtils:
    """Utility class cho chat functionality"""
    
    @staticmethod
    def generate_session_id() -> str:
        """Tạo session ID duy nhất"""
        return f"session_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def generate_message_id() -> str:
        """Tạo message ID duy nhất"""
        return f"msg_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def create_system_prompt(context_files: Optional[List[str]] = None) -> str:
        """Tạo system prompt với context từ files"""
        base_prompt = """Bạn là một AI assistant hữu ích. Hãy trả lời câu hỏi một cách chính xác và hữu ích."""
        
        if context_files:
            context_prompt = "\n\nBạn có thể sử dụng thông tin từ các file sau để trả lời:\n"
            for i, file_path in enumerate(context_files, 1):
                filename = file_path.split('/')[-1]
                context_prompt += f"{i}. {filename}\n"
            base_prompt += context_prompt
        
        return base_prompt
    
    @staticmethod
    def build_chat_prompt(
        message: str,
        system_prompt: str = "",
        context_files: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Xây dựng prompt cho chat"""
        prompt_parts = []
        
        # System prompt
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")
        
        # Context từ files
        if context_files:
            from .file_processor import FileProcessor
            context_text = FileProcessor.extract_text_from_files(context_files)
            if context_text:
                prompt_parts.append(f"Context:\n{context_text}")
        
        # Chat history
        if chat_history:
            for msg in chat_history[-5:]:  # Chỉ lấy 5 tin nhắn gần nhất
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")
        
        # Current message
        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    @staticmethod
    def format_chat_response(
        response: str,
        tokens_used: Optional[int] = None,
        response_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Format response cho chat"""
        return {
            "response": response,
            "tokens_used": tokens_used,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def validate_chat_request(
        message: str,
        model_id: str,
        max_length: int = 1000
    ) -> Dict[str, Any]:
        """Validate chat request"""
        errors = []
        
        if not message or not message.strip():
            errors.append("Message cannot be empty")
        
        if len(message) > max_length:
            errors.append(f"Message too long (max {max_length} characters)")
        
        if not model_id:
            errors.append("Model ID is required")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    @staticmethod
    def extract_context_from_files(
        file_paths: List[str],
        max_context_length: int = 2000
    ) -> str:
        """Trích xuất context từ files"""
        try:
            from .file_processor import FileProcessor
            context_text = FileProcessor.extract_text_from_files(file_paths)
            
            # Cắt ngắn nếu quá dài
            if len(context_text) > max_context_length:
                context_text = context_text[:max_context_length] + "..."
            
            return context_text
            
        except Exception as e:
            logger.error(f"Error extracting context from files: {e}")
            return ""
    
    @staticmethod
    def create_chat_session_data(
        model_id: str,
        session_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        context_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Tạo dữ liệu cho chat session mới"""
        return {
            "session_id": ChatUtils.generate_session_id(),
            "model_id": model_id,
            "session_name": session_name or f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "system_prompt": system_prompt,
            "context_files": context_files or [],
            "message_count": 0,
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
    
    @staticmethod
    def create_chat_message_data(
        session_id: str,
        role: str,
        content: str,
        content_type: str = "text",
        attached_files: Optional[List[str]] = None,
        tokens_used: Optional[int] = None,
        response_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Tạo dữ liệu cho chat message"""
        return {
            "message_id": ChatUtils.generate_message_id(),
            "session_id": session_id,
            "role": role,
            "content": content,
            "content_type": content_type,
            "attached_files": attached_files or [],
            "timestamp": datetime.now(),
            "tokens_used": tokens_used,
            "response_time": response_time
        }
    
    @staticmethod
    def measure_response_time(func):
        """Decorator để đo thời gian response"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            response_time = end_time - start_time
            
            # Thêm response_time vào result nếu là dict
            if isinstance(result, dict):
                result["response_time"] = response_time
            
            return result
        return wrapper
    
    @staticmethod
    def sanitize_message(message: str) -> str:
        """Làm sạch message"""
        import re
        
        # Loại bỏ ký tự đặc biệt nguy hiểm
        message = re.sub(r'[<>]', '', message)
        
        # Loại bỏ script tags
        message = re.sub(r'<script.*?</script>', '', message, flags=re.IGNORECASE | re.DOTALL)
        
        # Giới hạn độ dài
        if len(message) > 5000:
            message = message[:5000] + "..."
        
        return message.strip()
    
    @staticmethod
    def format_chat_history(
        messages: List[Dict[str, Any]],
        max_messages: int = 50
    ) -> List[Dict[str, Any]]:
        """Format chat history"""
        # Chỉ lấy tin nhắn gần nhất
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
        
        formatted_messages = []
        for msg in recent_messages:
            formatted_msg = {
                "message_id": msg.get("message_id"),
                "role": msg.get("role"),
                "content": msg.get("content"),
                "content_type": msg.get("content_type", "text"),
                "timestamp": msg.get("timestamp"),
                "tokens_used": msg.get("tokens_used"),
                "response_time": msg.get("response_time")
            }
            formatted_messages.append(formatted_msg)
        
        return formatted_messages 