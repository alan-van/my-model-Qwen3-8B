import os
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import logging

from app.models.finetune_job import FineTuneJob
from app.schemas.finetune import FineTuneRequest, FineTuneStatus
from app.utils.model_utils import ModelUtils
from app.utils.data_processor import DataProcessor
from app.config import settings

logger = logging.getLogger(__name__)

class FineTuneService:
    """Service để xử lý fine-tuning"""
    
    def __init__(self):
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
    
    def create_finetune_job(
        self,
        db: Session,
        request: FineTuneRequest
    ) -> FineTuneJob:
        """Tạo job fine-tuning mới"""
        try:
            # Tạo job ID
            job_id = ModelUtils.generate_job_id()
            
            # Chuẩn bị dữ liệu
            dataset, data_size = DataProcessor.prepare_finetuning_data(
                request.data_files,
                max_length=request.max_length
            )
            
            # Tạo job record
            job = FineTuneJob(
                job_id=job_id,
                model_name=request.model_name,
                base_model=request.base_model,
                learning_rate=request.learning_rate,
                batch_size=request.batch_size,
                epochs=request.epochs,
                max_length=request.max_length,
                warmup_steps=request.warmup_steps,
                data_files=request.data_files,
                data_size=data_size,
                status=FineTuneStatus.PENDING,
                config=request.config
            )
            
            db.add(job)
            db.commit()
            db.refresh(job)
            
            logger.info(f"Created finetune job: {job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error creating finetune job: {e}")
            db.rollback()
            raise
    
    def start_finetune_job(
        self,
        db: Session,
        job_id: str
    ) -> bool:
        """Bắt đầu job fine-tuning"""
        try:
            # Lấy job từ database
            job = db.query(FineTuneJob).filter(FineTuneJob.job_id == job_id).first()
            if not job:
                raise ValueError(f"Job not found: {job_id}")
            
            if job.status != FineTuneStatus.PENDING:
                raise ValueError(f"Job {job_id} is not in pending status")
            
            # Cập nhật trạng thái
            job.status = FineTuneStatus.RUNNING
            job.started_at = datetime.now()
            db.commit()
            
            # Chạy fine-tuning trong thread riêng
            thread = threading.Thread(
                target=self._run_finetune_job,
                args=(db, job_id)
            )
            thread.daemon = True
            thread.start()
            
            # Lưu thông tin job active
            self.active_jobs[job_id] = {
                "thread": thread,
                "start_time": datetime.now()
            }
            
            logger.info(f"Started finetune job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting finetune job {job_id}: {e}")
            return False
    
    def _run_finetune_job(self, db: Session, job_id: str):
        """Chạy fine-tuning job trong background"""
        try:
            # Lấy job info
            job = db.query(FineTuneJob).filter(FineTuneJob.job_id == job_id).first()
            if not job:
                logger.error(f"Job not found: {job_id}")
                return
            
            logger.info(f"Starting finetune process for job: {job_id}")
            
            # Tải base model
            model, tokenizer = ModelUtils.load_base_model(job.base_model)
            
            # Thiết lập LoRA
            lora_config = ModelUtils.setup_lora_config()
            model = ModelUtils.prepare_model_for_training(model, lora_config)
            
            # Chuẩn bị dataset
            dataset, _ = DataProcessor.prepare_finetuning_data(
                job.data_files,
                max_length=job.max_length
            )
            
            # Tokenize dataset
            def tokenize_function(examples):
                return ModelUtils.tokenize_function(examples, tokenizer, job.max_length)
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Tạo training arguments
            output_dir = os.path.join(settings.finetuned_model_path, job_id)
            training_args = ModelUtils.create_training_arguments(
                output_dir=output_dir,
                learning_rate=job.learning_rate,
                batch_size=job.batch_size,
                epochs=job.epochs,
                warmup_steps=job.warmup_steps,
                max_length=job.max_length
            )
            
            # Tạo trainer
            trainer = ModelUtils.create_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=tokenized_dataset,
                eval_dataset=None,
                training_args=training_args
            )
            
            # Training callback để cập nhật progress
            class ProgressCallback:
                def __init__(self, service, job_id, db):
                    self.service = service
                    self.job_id = job_id
                    self.db = db
                
                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step % 10 == 0:  # Cập nhật mỗi 10 steps
                        self.service._update_job_progress(
                            self.db, self.job_id, state.global_step, state.max_steps
                        )
            
            trainer.add_callback(ProgressCallback(self, job_id, db))
            
            # Bắt đầu training
            trainer.train()
            
            # Lưu model
            model_path = os.path.join(output_dir, "final_model")
            ModelUtils.save_model(model, tokenizer, model_path, job.model_name)
            
            # Cập nhật job thành công
            self._update_job_completed(db, job_id, model_path)
            
            logger.info(f"Finetune job completed: {job_id}")
            
        except Exception as e:
            logger.error(f"Error in finetune job {job_id}: {e}")
            self._update_job_failed(db, job_id, str(e))
    
    def _update_job_progress(
        self,
        db: Session,
        job_id: str,
        current_step: int,
        total_steps: int
    ):
        """Cập nhật progress của job"""
        try:
            job = db.query(FineTuneJob).filter(FineTuneJob.job_id == job_id).first()
            if job:
                job.current_step = current_step
                job.total_steps = total_steps
                job.progress = min(current_step / total_steps, 1.0) if total_steps > 0 else 0.0
                db.commit()
        except Exception as e:
            logger.error(f"Error updating job progress: {e}")
    
    def _update_job_completed(
        self,
        db: Session,
        job_id: str,
        model_path: str
    ):
        """Cập nhật job hoàn thành"""
        try:
            job = db.query(FineTuneJob).filter(FineTuneJob.job_id == job_id).first()
            if job:
                job.status = FineTuneStatus.COMPLETED
                job.progress = 1.0
                job.model_path = model_path
                job.completed_at = datetime.now()
                db.commit()
                
                # Xóa khỏi active jobs
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
        except Exception as e:
            logger.error(f"Error updating job completed: {e}")
    
    def _update_job_failed(
        self,
        db: Session,
        job_id: str,
        error_message: str
    ):
        """Cập nhật job thất bại"""
        try:
            job = db.query(FineTuneJob).filter(FineTuneJob.job_id == job_id).first()
            if job:
                job.status = FineTuneStatus.FAILED
                job.error_message = error_message
                job.completed_at = datetime.now()
                db.commit()
                
                # Xóa khỏi active jobs
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
        except Exception as e:
            logger.error(f"Error updating job failed: {e}")
    
    def get_job_status(
        self,
        db: Session,
        job_id: str
    ) -> Optional[FineTuneJob]:
        """Lấy trạng thái job"""
        return db.query(FineTuneJob).filter(FineTuneJob.job_id == job_id).first()
    
    def get_job_history(
        self,
        db: Session,
        page: int = 1,
        size: int = 10
    ) -> Dict[str, Any]:
        """Lấy lịch sử jobs"""
        offset = (page - 1) * size
        
        jobs = db.query(FineTuneJob).order_by(
            FineTuneJob.created_at.desc()
        ).offset(offset).limit(size).all()
        
        total = db.query(FineTuneJob).count()
        
        return {
            "jobs": jobs,
            "total": total,
            "page": page,
            "size": size
        }
    
    def cancel_job(
        self,
        db: Session,
        job_id: str
    ) -> bool:
        """Hủy job"""
        try:
            job = db.query(FineTuneJob).filter(FineTuneJob.job_id == job_id).first()
            if not job:
                return False
            
            if job.status not in [FineTuneStatus.PENDING, FineTuneStatus.RUNNING]:
                return False
            
            # Cập nhật trạng thái
            job.status = FineTuneStatus.FAILED
            job.error_message = "Job cancelled by user"
            job.completed_at = datetime.now()
            db.commit()
            
            # Xóa khỏi active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False 