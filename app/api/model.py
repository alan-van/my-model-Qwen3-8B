from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.schemas.model import ModelInfoResponse, ModelListResponse, ModelCreateRequest
from app.services.model_service import ModelService

router = APIRouter(prefix="/api/models", tags=["Models"])

# Service instance
model_service = ModelService()

@router.get("/", response_model=ModelListResponse)
async def get_models(
    type: Optional[str] = None,
    is_active: Optional[bool] = None,
    page: int = 1,
    size: int = 10,
    db: Session = Depends(get_db)
):
    """Lấy danh sách models"""
    try:
        result = model_service.get_models(db, type, is_active, page, size)
        
        # Convert to response format
        models = []
        for model in result["models"]:
            models.append(ModelInfoResponse(
                id=model.id,
                name=model.name,
                type=model.type,
                base_model=model.base_model,
                path=model.path,
                hf_repo_url=model.hf_repo_url,
                size=model.size,
                accuracy=model.accuracy,
                loss=model.loss,
                perplexity=model.perplexity,
                training_data_size=model.training_data_size,
                training_epochs=model.training_epochs,
                training_steps=model.training_steps,
                description=model.description,
                tags=model.tags,
                is_active=model.is_active,
                created_at=model.created_at,
                updated_at=model.updated_at
            ))
        
        return ModelListResponse(
            models=models,
            total=result["total"],
            page=result["page"],
            size=result["size"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{id}", response_model=ModelInfoResponse)
async def get_model(
    id: str,
    db: Session = Depends(get_db)
):
    """Lấy thông tin model theo ID"""
    try:
        model = model_service.get_model(db, id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return ModelInfoResponse(
            id=model.id,
            name=model.name,
            type=model.type,
            base_model=model.base_model,
            path=model.path,
            hf_repo_url=model.hf_repo_url,
            size=model.size,
            accuracy=model.accuracy,
            loss=model.loss,
            perplexity=model.perplexity,
            training_data_size=model.training_data_size,
            training_epochs=model.training_epochs,
            training_steps=model.training_steps,
            description=model.description,
            tags=model.tags,
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=ModelInfoResponse)
async def create_model(
    request: ModelCreateRequest,
    db: Session = Depends(get_db)
):
    """Tạo model mới"""
    try:
        model = model_service.create_model(db, request)
        
        return ModelInfoResponse(
            id=model.id,
            name=model.name,
            type=model.type,
            base_model=model.base_model,
            path=model.path,
            hf_repo_url=model.hf_repo_url,
            size=model.size,
            accuracy=model.accuracy,
            loss=model.loss,
            perplexity=model.perplexity,
            training_data_size=model.training_data_size,
            training_epochs=model.training_epochs,
            training_steps=model.training_steps,
            description=model.description,
            tags=model.tags,
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{id}", response_model=ModelInfoResponse)
async def update_model(
    id: str,
    update_data: dict,
    db: Session = Depends(get_db)
):
    """Cập nhật model"""
    try:
        model = model_service.update_model(db, id, update_data)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return ModelInfoResponse(
            id=model.id,
            name=model.name,
            type=model.type,
            base_model=model.base_model,
            path=model.path,
            hf_repo_url=model.hf_repo_url,
            size=model.size,
            accuracy=model.accuracy,
            loss=model.loss,
            perplexity=model.perplexity,
            training_data_size=model.training_data_size,
            training_epochs=model.training_epochs,
            training_steps=model.training_steps,
            description=model.description,
            tags=model.tags,
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{id}")
async def delete_model(
    id: str,
    db: Session = Depends(get_db)
):
    """Xóa model"""
    try:
        success = model_service.delete_model(db, id)
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {"message": "Model deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{id}/performance")
async def get_model_performance(
    id: str,
    db: Session = Depends(get_db)
):
    """Lấy thông tin performance của model"""
    try:
        performance = model_service.get_model_performance(db, id)
        if not performance:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return performance
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{id}/performance")
async def update_model_performance(
    id: str,
    accuracy: Optional[float] = None,
    loss: Optional[float] = None,
    perplexity: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Cập nhật performance metrics của model"""
    try:
        success = model_service.update_model_performance(
            db, id, accuracy, loss, perplexity
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {"message": "Model performance updated successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/base/list")
async def get_base_models():
    """Lấy danh sách base models có sẵn"""
    try:
        return model_service.get_base_models()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate")
async def validate_model_path(path: str):
    """Validate model path"""
    try:
        result = model_service.validate_model_path(path)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_model_statistics(db: Session = Depends(get_db)):
    """Lấy thống kê về models"""
    try:
        return model_service.get_model_statistics(db)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 