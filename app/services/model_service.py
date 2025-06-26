import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import logging

from app.models.model_info import ModelInfo
from app.schemas.model import ModelCreateRequest
from app.utils.model_utils import ModelUtils
from app.config import settings

logger = logging.getLogger(__name__)

class ModelService:
    """Service để quản lý models"""
    
    def create_model(
        self,
        db: Session,
        request: ModelCreateRequest
    ) -> ModelInfo:
        """Tạo model mới"""
        try:
            # Kiểm tra model path có tồn tại không
            if not os.path.exists(request.model_path):
                raise ValueError(f"Model path not found: {request.model_path}")
            
            # Tạo model ID
            model_id = ModelUtils.generate_model_id()
            
            # Tạo model record
            model = ModelInfo(
                model_id=model_id,
                model_name=request.model_name,
                model_type=request.model_type,
                base_model=request.base_model,
                model_path=request.model_path,
                hf_repo_url=request.hf_repo_url,
                model_size=request.model_size,
                description=request.description,
                tags=request.tags,
                config=request.config
            )
            
            db.add(model)
            db.commit()
            db.refresh(model)
            
            logger.info(f"Created model: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            db.rollback()
            raise
    
    def get_model(
        self,
        db: Session,
        model_id: str
    ) -> Optional[ModelInfo]:
        """Lấy model theo ID"""
        return db.query(ModelInfo).filter(ModelInfo.model_id == model_id).first()
    
    def get_models(
        self,
        db: Session,
        model_type: Optional[str] = None,
        is_active: Optional[bool] = None,
        page: int = 1,
        size: int = 10
    ) -> Dict[str, Any]:
        """Lấy danh sách models với filter"""
        offset = (page - 1) * size
        
        query = db.query(ModelInfo)
        
        # Apply filters
        if model_type:
            query = query.filter(ModelInfo.model_type == model_type)
        
        if is_active is not None:
            query = query.filter(ModelInfo.is_active == is_active)
        
        models = query.order_by(ModelInfo.created_at.desc()).offset(offset).limit(size).all()
        total = query.count()
        
        return {
            "models": models,
            "total": total,
            "page": page,
            "size": size
        }
    
    def update_model(
        self,
        db: Session,
        model_id: str,
        update_data: Dict[str, Any]
    ) -> Optional[ModelInfo]:
        """Cập nhật model"""
        try:
            model = self.get_model(db, model_id)
            if not model:
                return None
            
            # Cập nhật các trường được phép
            allowed_fields = [
                "model_name", "description", "tags", "config", 
                "accuracy", "loss", "perplexity", "is_active"
            ]
            
            for field, value in update_data.items():
                if field in allowed_fields and hasattr(model, field):
                    setattr(model, field, value)
            
            model.updated_at = datetime.now()
            db.commit()
            db.refresh(model)
            
            logger.info(f"Updated model: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Error updating model {model_id}: {e}")
            db.rollback()
            return None
    
    def delete_model(
        self,
        db: Session,
        model_id: str
    ) -> bool:
        """Xóa model"""
        try:
            model = self.get_model(db, model_id)
            if not model:
                return False
            
            # Soft delete - chỉ set is_active = False
            model.is_active = False
            model.updated_at = datetime.now()
            db.commit()
            
            logger.info(f"Deleted model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            db.rollback()
            return False
    
    def register_finetuned_model(
        self,
        db: Session,
        job_id: str,
        model_path: str,
        model_name: str,
        base_model: str = "Qwen/Qwen3-8B"
    ) -> ModelInfo:
        """Đăng ký model đã fine-tune"""
        try:
            # Kiểm tra model path
            if not os.path.exists(model_path):
                raise ValueError(f"Model path not found: {model_path}")
            
            # Đọc model info từ file
            model_info_path = os.path.join(model_path, "model_info.json")
            config = {}
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    config = json.load(f)
            
            # Tạo model record
            model_request = ModelCreateRequest(
                model_name=model_name,
                model_type="finetuned",
                base_model=base_model,
                model_path=model_path,
                model_size="8B",
                description=f"Fine-tuned model from job {job_id}",
                tags=["finetuned", "qwen"],
                config=config
            )
            
            model = self.create_model(db, model_request)
            
            logger.info(f"Registered finetuned model: {model.model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Error registering finetuned model: {e}")
            raise
    
    def get_model_performance(
        self,
        db: Session,
        model_id: str
    ) -> Dict[str, Any]:
        """Lấy thông tin performance của model"""
        model = self.get_model(db, model_id)
        if not model:
            return {}
        
        return {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "accuracy": model.accuracy,
            "loss": model.loss,
            "perplexity": model.perplexity,
            "training_data_size": model.training_data_size,
            "training_epochs": model.training_epochs,
            "training_steps": model.training_steps
        }
    
    def update_model_performance(
        self,
        db: Session,
        model_id: str,
        accuracy: Optional[float] = None,
        loss: Optional[float] = None,
        perplexity: Optional[float] = None
    ) -> bool:
        """Cập nhật performance metrics của model"""
        try:
            model = self.get_model(db, model_id)
            if not model:
                return False
            
            if accuracy is not None:
                model.accuracy = accuracy
            if loss is not None:
                model.loss = loss
            if perplexity is not None:
                model.perplexity = perplexity
            
            model.updated_at = datetime.now()
            db.commit()
            
            logger.info(f"Updated performance for model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
            db.rollback()
            return False
    
    def get_base_models(self) -> List[Dict[str, str]]:
        """Lấy danh sách base models có sẵn"""
        return [
            {
                "model_id": "qwen3-8b",
                "model_name": "Qwen3-8B",
                "model_path": "Qwen/Qwen3-8B",
                "description": "Base Qwen3-8B model from Alibaba"
            }
        ]
    
    def validate_model_path(self, model_path: str) -> Dict[str, Any]:
        """Validate model path"""
        result = {
            "valid": False,
            "errors": [],
            "model_info": {}
        }
        
        try:
            if not os.path.exists(model_path):
                result["errors"].append("Model path does not exist")
                return result
            
            # Kiểm tra các file cần thiết
            required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
            for file in required_files:
                file_path = os.path.join(model_path, file)
                if not os.path.exists(file_path):
                    result["errors"].append(f"Required file not found: {file}")
            
            # Đọc model info nếu có
            model_info_path = os.path.join(model_path, "model_info.json")
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    result["model_info"] = json.load(f)
            
            result["valid"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(f"Error validating model path: {str(e)}")
        
        return result
    
    def get_model_statistics(self, db: Session) -> Dict[str, Any]:
        """Lấy thống kê về models"""
        total_models = db.query(ModelInfo).count()
        base_models = db.query(ModelInfo).filter(ModelInfo.model_type == "base").count()
        finetuned_models = db.query(ModelInfo).filter(ModelInfo.model_type == "finetuned").count()
        active_models = db.query(ModelInfo).filter(ModelInfo.is_active == True).count()
        
        return {
            "total_models": total_models,
            "base_models": base_models,
            "finetuned_models": finetuned_models,
            "active_models": active_models
        } 