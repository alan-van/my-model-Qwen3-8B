import os
import threading
import torch
from typing import Optional, Tuple
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from log import logger
from utils.model_utils import ModelUtils
from utils.callbacks import JobProgressCallback
from database.database import Database

class FineTuningService:
    def __init__(self, db: Database):
        self.db = db
        self._training_threads = {}
    
    def start_fine_tuning(self, job_id: int) -> None:
        """Bắt đầu fine-tuning trong background thread"""
        try:
            # Lấy thông tin job
            job = self.db.get_job(job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                return
            
            # Cập nhật trạng thái
            self.db.update_job_status(job_id, "running")
            
            # Tạo output directory
            output_dir = f"models/fine_tuned_{job_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Load model và tokenizer
            try:
                model, tokenizer = self._load_model_and_tokenizer(job.model_path)
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.db.update_job_status(job_id, "failed")
                self.db.update_job_logs(job_id, f"Error loading model: {str(e)}")
                return
            
            # Load dataset
            try:
                dataset = self._load_dataset(job.dataset_path)
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                self.db.update_job_status(job_id, "failed")
                self.db.update_job_logs(job_id, f"Error loading dataset: {str(e)}")
                return
            
            # Prepare data
            try:
                train_dataset, eval_dataset = self._prepare_data(dataset, tokenizer, job.max_length)
            except Exception as e:
                logger.error(f"Error preparing data: {e}")
                self.db.update_job_status(job_id, "failed")
                self.db.update_job_logs(job_id, f"Error preparing data: {str(e)}")
                return
            
            # Create training arguments
            try:
                training_args = ModelUtils.create_training_arguments(
                    output_dir=output_dir,
                    learning_rate=job.learning_rate,
                    batch_size=job.batch_size,
                    epochs=job.epochs,
                    warmup_steps=job.warmup_steps,
                    max_length=job.max_length,
                    save_steps=job.save_steps,
                    eval_steps=job.eval_steps,
                    logging_steps=job.logging_steps
                )
            except Exception as e:
                logger.error(f"Error creating training arguments: {e}")
                self.db.update_job_status(job_id, "failed")
                self.db.update_job_logs(job_id, f"Error creating training arguments: {str(e)}")
                return
            
            # Create trainer
            try:
                trainer = self._create_trainer(model, tokenizer, training_args, train_dataset, eval_dataset, job_id)
            except Exception as e:
                logger.error(f"Error creating trainer: {e}")
                self.db.update_job_status(job_id, "failed")
                self.db.update_job_logs(job_id, f"Error creating trainer: {str(e)}")
                return
            
            # Start training
            try:
                logger.info(f"Starting training for job {job_id}")
                self.db.update_job_logs(job_id, "Starting training...")
                
                # Train model
                trainer.train()
                
                # Save model
                trainer.save_model()
                tokenizer.save_pretrained(output_dir)
                
                # Update job status
                self.db.update_job_status(job_id, "completed")
                self.db.update_job_logs(job_id, f"Training completed successfully. Model saved to {output_dir}")
                
                logger.info(f"Training completed for job {job_id}")
                
            except Exception as e:
                logger.error(f"Error during training: {e}")
                self.db.update_job_status(job_id, "failed")
                self.db.update_job_logs(job_id, f"Error during training: {str(e)}")
                
        except Exception as e:
            logger.error(f"Unexpected error in fine-tuning: {e}")
            self.db.update_job_status(job_id, "failed")
            self.db.update_job_logs(job_id, f"Unexpected error: {str(e)}")
    
    def start_fine_tuning_async(self, job_id: int) -> None:
        """Bắt đầu fine-tuning trong background thread"""
        if job_id in self._training_threads and self._training_threads[job_id].is_alive():
            logger.warning(f"Training for job {job_id} is already running")
            return
        
        thread = threading.Thread(target=self.start_fine_tuning, args=(job_id,))
        thread.daemon = True
        thread.start()
        self._training_threads[job_id] = thread
    
    def _load_model_and_tokenizer(self, model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model và tokenizer"""
        logger.info(f"Loading model from {model_path}")
        
        # Kiểm tra xem có GPU không
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Thêm padding token nếu chưa có
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model với quantization nếu có GPU
        if device == "cuda":
            try:
                # Thử load với 4-bit quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    load_in_4bit=True,
                    quantization_config={"load_in_4bit": True}
                )
                logger.info("Model loaded with 4-bit quantization")
            except Exception as e:
                logger.warning(f"4-bit quantization failed: {e}, trying 8-bit...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        load_in_8bit=True
                    )
                    logger.info("Model loaded with 8-bit quantization")
                except Exception as e2:
                    logger.warning(f"8-bit quantization failed: {e2}, loading without quantization...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    logger.info("Model loaded without quantization")
        else:
            # CPU mode
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            logger.info("Model loaded for CPU")
        
        return model, tokenizer
    
    def _load_dataset(self, dataset_path: str) -> Dataset:
        """Load dataset từ file"""
        logger.info(f"Loading dataset from {dataset_path}")
        
        if dataset_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(dataset_path)
            return Dataset.from_pandas(df)
        elif dataset_path.endswith('.json'):
            return Dataset.from_json(dataset_path)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    def _prepare_data(self, dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Tuple[Dataset, Dataset]:
        """Chuẩn bị dữ liệu cho training"""
        logger.info("Preparing data for training")
        
        def tokenize_function(examples):
            # Giả sử dataset có cột 'text'
            texts = examples.get('text', examples.get('content', examples.get('prompt', '')))
            if isinstance(texts, str):
                texts = [texts]
            
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split thành train và eval
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
        
        return split_dataset["train"], split_dataset["test"]
    
    def _create_trainer(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        training_args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        job_id: int
    ) -> Trainer:
        """Tạo trainer"""
        logger.info("Creating trainer")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Tạo custom callback để cập nhật tiến độ job
        job_callback = JobProgressCallback(self, job_id, self.db)
        
        # Tạo trainer với cấu hình tương thích phiên bản mới
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": data_collator,
            "callbacks": [job_callback],
        }
        
        # Chỉ thêm tokenizer nếu phiên bản transformers hỗ trợ
        try:
            from transformers import __version__ as transformers_version
            import packaging.version
            if packaging.version.parse(transformers_version) < packaging.version.parse("5.0.0"):
                trainer_kwargs["tokenizer"] = tokenizer
        except ImportError:
            # Nếu không thể kiểm tra version, thử thêm tokenizer
            try:
                trainer_kwargs["tokenizer"] = tokenizer
            except:
                pass
        
        trainer = Trainer(**trainer_kwargs)
        
        return trainer
    
    def get_training_status(self, job_id: int) -> Optional[str]:
        """Lấy trạng thái training"""
        if job_id in self._training_threads:
            thread = self._training_threads[job_id]
            if thread.is_alive():
                return "running"
            else:
                return "completed"
        return None 