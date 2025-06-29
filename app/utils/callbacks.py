import logging
from typing import Dict, Any
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

class CustomProgressCallback(TrainerCallback):
    """Custom callback tương thích với phiên bản mới của transformers"""
    
    def __init__(self):
        self.training_step = 0
        self.eval_step = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        logger.info("Starting training...")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        logger.info("Training completed!")
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of a training step."""
        self.training_step += 1
        if self.training_step % args.logging_steps == 0:
            logger.info(f"Training step {self.training_step}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        self.eval_step += 1
        if metrics:
            logger.info(f"Evaluation step {self.eval_step}: {metrics}")
    
    def on_save(self, args, state, control, **kwargs):
        """Called after saving a checkpoint."""
        logger.info(f"Model saved at step {state.global_step}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the command is logged."""
        if logs:
            logger.info(f"Logs: {logs}")

class JobProgressCallback(TrainerCallback):
    """Callback để cập nhật tiến độ job vào database"""
    
    def __init__(self, service, job_id, db):
        self.service = service
        self.job_id = job_id
        self.db = db
        self.last_log_step = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        if hasattr(self.service, 'db') and hasattr(self.service.db, 'update_job_logs'):
            # Cho fine_tuning_service.py
            self.service.db.update_job_logs(self.job_id, "Training started...")
        else:
            # Cho finetune_service.py
            logger.info(f"Training started for job {self.job_id}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        if hasattr(self.service, 'db') and hasattr(self.service.db, 'update_job_logs'):
            # Cho fine_tuning_service.py
            self.service.db.update_job_logs(self.job_id, "Training completed!")
        else:
            # Cho finetune_service.py
            logger.info(f"Training completed for job {self.job_id}")
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of a training step."""
        if hasattr(self.service, 'db') and hasattr(self.service.db, 'update_job_logs'):
            # Cho fine_tuning_service.py
            if state.global_step - self.last_log_step >= args.logging_steps:
                self.service.db.update_job_logs(self.job_id, f"Training step {state.global_step}")
                self.last_log_step = state.global_step
        else:
            # Cho finetune_service.py
            if state.global_step % 10 == 0:  # Cập nhật mỗi 10 steps
                self.service._update_job_progress(
                    self.db, self.job_id, state.global_step, state.max_steps
                )
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of a training step."""
        if hasattr(self.service, 'db') and hasattr(self.service.db, 'update_job_logs'):
            # Cho fine_tuning_service.py
            if state.global_step - self.last_log_step >= args.logging_steps:
                self.service.db.update_job_logs(self.job_id, f"Training step {state.global_step}")
                self.last_log_step = state.global_step
        else:
            # Cho finetune_service.py
            if state.global_step % 10 == 0:  # Cập nhật mỗi 10 steps
                self.service._update_job_progress(
                    self.db, self.job_id, state.global_step, state.max_steps
                )
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics:
            eval_loss = metrics.get('eval_loss', 'N/A')
            if hasattr(self.service, 'db') and hasattr(self.service.db, 'update_job_logs'):
                # Cho fine_tuning_service.py
                self.service.db.update_job_logs(self.job_id, f"Evaluation loss: {eval_loss}")
            else:
                # Cho finetune_service.py
                logger.info(f"Evaluation loss for job {self.job_id}: {eval_loss}")
    
    def on_save(self, args, state, control, **kwargs):
        """Called after saving a checkpoint."""
        if hasattr(self.service, 'db') and hasattr(self.service.db, 'update_job_logs'):
            # Cho fine_tuning_service.py
            self.service.db.update_job_logs(self.job_id, f"Checkpoint saved at step {state.global_step}")
        else:
            # Cho finetune_service.py
            logger.info(f"Checkpoint saved for job {self.job_id} at step {state.global_step}") 