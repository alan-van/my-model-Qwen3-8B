import os
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.schemas.finetune import (
    FineTuneRequest, FineTuneResponse, FineTuneStatusResponse, 
    FineTuneHistory, FineTuneHistoryItem
)
from app.services.finetune_service import FineTuneService
from app.services.model_service import ModelService
from app.config import settings

router = APIRouter(prefix="/api/finetune", tags=["Fine-tuning"])

# Service instances
finetune_service = FineTuneService()
model_service = ModelService()

@router.post("/upload", response_model=dict)
async def upload_files(files: List[str], db: Session = Depends(get_db)):
    """Upload files để fine-tuning"""
    try:
        # Validate files
        valid_files = []
        for file_path in files:
            if not os.path.exists(file_path):
                raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
            
            from app.utils.file_processor import FileProcessor
            if not FileProcessor.is_supported_file(file_path, settings.allowed_extensions):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file_path}"
                )
            valid_files.append(file_path)
        
        return {
            "message": "Files uploaded successfully",
            "files": valid_files,
            "count": len(valid_files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start", response_model=FineTuneResponse)
async def start_finetune(
    request: FineTuneRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Bắt đầu fine-tuning"""
    try:
        # Validate request
        if not request.data_files:
            raise HTTPException(status_code=400, detail="Data files are required")
        
        # Kiểm tra files tồn tại
        for file_path in request.data_files:
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Data file not found: {file_path}"
                )
        
        # Tạo job
        job = finetune_service.create_finetune_job(db, request)
        
        # Bắt đầu fine-tuning trong background
        background_tasks.add_task(finetune_service.start_finetune_job, db, job.job_id)
        
        return FineTuneResponse(
            job_id=job.job_id,
            status=job.status,
            message="Fine-tuning job created and started",
            created_at=job.created_at
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}", response_model=FineTuneStatusResponse)
async def get_finetune_status(job_id: str, db: Session = Depends(get_db)):
    """Lấy trạng thái fine-tuning job"""
    try:
        job = finetune_service.get_job_status(db, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return FineTuneStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            current_epoch=job.current_epoch,
            current_step=job.current_step,
            total_steps=job.total_steps,
            final_loss=job.final_loss,
            final_accuracy=job.final_accuracy,
            model_path=job.model_path,
            hf_repo_url=job.hf_repo_url,
            error_message=job.error_message,
            started_at=job.started_at,
            completed_at=job.completed_at
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=FineTuneHistory)
async def get_finetune_history(
    page: int = 1,
    size: int = 10,
    db: Session = Depends(get_db)
):
    """Lấy lịch sử fine-tuning jobs"""
    try:
        result = finetune_service.get_job_history(db, page, size)
        
        # Convert to response format
        history_items = []
        for job in result["jobs"]:
            history_items.append(FineTuneHistoryItem(
                id=job.id,
                job_id=job.job_id,
                model_name=job.model_name,
                base_model=job.base_model,
                status=job.status,
                progress=job.progress,
                final_loss=job.final_loss,
                final_accuracy=job.final_accuracy,
                created_at=job.created_at,
                completed_at=job.completed_at
            ))
        
        return FineTuneHistory(
            jobs=history_items,
            total=result["total"],
            page=result["page"],
            size=result["size"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cancel/{job_id}")
async def cancel_finetune_job(job_id: str, db: Session = Depends(get_db)):
    """Hủy fine-tuning job"""
    try:
        success = finetune_service.cancel_job(db, job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        
        return {"message": "Job cancelled successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register/{job_id}")
async def register_finetuned_model(job_id: str, db: Session = Depends(get_db)):
    """Đăng ký model đã fine-tune"""
    try:
        # Lấy job info
        job = finetune_service.get_job_status(db, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status != "completed":
            raise HTTPException(status_code=400, detail="Job is not completed")
        
        if not job.model_path:
            raise HTTPException(status_code=400, detail="Model path not found")
        
        # Đăng ký model
        model = model_service.register_finetuned_model(
            db=db,
            job_id=job_id,
            model_path=job.model_path,
            model_name=job.model_name,
            base_model=job.base_model
        )
        
        return {
            "message": "Model registered successfully",
            "model_id": model.model_id,
            "model_name": model.model_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_finetune_config():
    """Lấy cấu hình fine-tuning mặc định"""
    return {
        "default_learning_rate": settings.default_learning_rate,
        "default_batch_size": settings.default_batch_size,
        "default_epochs": settings.default_epochs,
        "default_max_length": settings.default_max_length,
        "default_warmup_steps": settings.default_warmup_steps,
        "allowed_extensions": settings.allowed_extensions,
        "max_file_size": settings.max_file_size
    } 