import os
import uuid
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging
from typing import Dict, Any, Optional, Tuple
import json

logger = logging.getLogger(__name__)

try:
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from tokenization_qwen2 import Qwen2Tokenizer

class ModelUtils:
    """Utility class để quản lý model và fine-tuning"""
    
    @staticmethod
    def generate_job_id() -> str:
        """Tạo job ID duy nhất"""
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_model_id() -> str:
        """Tạo model ID duy nhất"""
        return f"model_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def load_base_model(model_name: str, device: str = "auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Tải base model và tokenizer, luôn ưu tiên Hugging Face Hub online cho Qwen3-8B"""
        try:
            logger.info(f"Loading base model: {model_name}")
            # Nếu là Qwen/Qwen3-8B thì luôn load từ Hugging Face Hub
            if model_name == "Qwen/Qwen3-8B":
                model_source = model_name
                logger.info(f"Always loading Qwen/Qwen3-8B from Hugging Face repo: {model_name}")
            else:
                # Ưu tiên local path nếu có
                local_dir = os.getenv("BASE_MODEL_LOCAL_DIR")
                use_local = local_dir and os.path.isdir(local_dir) and len(os.listdir(local_dir)) > 0
                model_source = local_dir if use_local else model_name
                if use_local:
                    logger.info(f"Loading model from local directory: {local_dir}")
                else:
                    logger.info(f"Loading model from Hugging Face repo: {model_name}")

            # Tải tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=True,
                padding_side="right"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Xác định có nên dùng quantization không
            quantization_config = None
            use_quant = False
            try:
                import bitsandbytes as bnb
                from packaging import version
                if torch.cuda.is_available() and version.parse(bnb.__version__) >= version.parse("0.43.1"):
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                    use_quant = True
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Quantization config setup failed: {e}")

            # Tải model
            if use_quant and quantization_config is not None:
                model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    torch_dtype=torch.float16,
                    device_map=device,
                    trust_remote_code=True,
                    quantization_config=quantization_config
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    torch_dtype=torch.float16,
                    device_map=device,
                    trust_remote_code=True
                )
            logger.info(f"Successfully loaded model: {model_name}")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    @staticmethod
    def setup_lora_config(
        r: int = 16,
        lora_alpha: int = 32,
        target_modules: Optional[list] = None,
        lora_dropout: float = 0.1
    ) -> LoraConfig:
        """Thiết lập cấu hình LoRA"""
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        
        return config
    
    @staticmethod
    def prepare_model_for_training(
        model: AutoModelForCausalLM,
        lora_config: LoraConfig
    ) -> AutoModelForCausalLM:
        """Chuẩn bị model cho training với LoRA"""
        try:
            logger.info("Preparing model for training with LoRA")
            
            # Áp dụng LoRA
            model = get_peft_model(model, lora_config)
            
            # In thông tin trainable parameters
            model.print_trainable_parameters()
            
            return model
            
        except Exception as e:
            logger.error(f"Error preparing model for training: {e}")
            raise
    
    @staticmethod
    def create_training_arguments(
        output_dir: str,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        epochs: int = 3,
        warmup_steps: int = 100,
        max_length: int = 512,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 10
    ) -> TrainingArguments:
        """Tạo training arguments, tự động bỏ qua các tham số không hỗ trợ nếu phát hiện lỗi"""
        import inspect
        from transformers import TrainingArguments
        args = dict(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            warmup_steps=warmup_steps,
            max_steps=-1,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Tắt wandb logging
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            dataloader_num_workers=0,
            group_by_length=True,
            length_column_name="length",
            max_length=max_length
        )
        while True:
            try:
                return TrainingArguments(**args)
            except TypeError as e:
                msg = str(e)
                # Tìm tham số không hợp lệ
                import re
                m = re.search(r"got an unexpected keyword argument '([^']+)'", msg)
                if not m:
                    raise
                bad_arg = m.group(1)
                logger.warning(f"TrainingArguments does not support argument '{bad_arg}', removing and retrying.")
                args.pop(bad_arg, None)
                # Nếu evaluation_strategy bị loại bỏ, cũng loại bỏ các tham số liên quan
                if bad_arg == "evaluation_strategy":
                    if "load_best_model_at_end" in args:
                        logger.warning("Removing 'load_best_model_at_end' as it requires evaluation_strategy.")
                        args.pop("load_best_model_at_end")
                    if "save_strategy" in args:
                        logger.warning("Removing 'save_strategy' as it requires evaluation_strategy.")
                        args.pop("save_strategy")
                    if "metric_for_best_model" in args:
                        logger.warning("Removing 'metric_for_best_model' as it requires evaluation_strategy.")
                        args.pop("metric_for_best_model")
                if not args:
                    raise RuntimeError("No valid arguments left for TrainingArguments!")
    
    @staticmethod
    def tokenize_function(
        examples: Dict[str, Any],
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """Tokenize function cho dataset"""
        # Kết hợp instruction và response
        texts = []
        for instruction, response in zip(examples["instruction"], examples["response"]):
            text = f"Instruction: {instruction}\n\nResponse: {response}"
            texts.append(text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Thêm length column
        tokenized["length"] = [len(ids) for ids in tokenized["input_ids"]]
        
        return tokenized
    
    @staticmethod
    def create_trainer(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        training_args: TrainingArguments
    ) -> Trainer:
        """Tạo trainer"""
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Tạo trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        return trainer
    
    @staticmethod
    def save_model(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        output_path: str,
        model_name: str
    ):
        """Lưu model đã fine-tune"""
        try:
            logger.info(f"Saving model to {output_path}")
            
            # Tạo thư mục output
            os.makedirs(output_path, exist_ok=True)
            
            # Lưu model
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            # Lưu thông tin model
            model_info = {
                "model_name": model_name,
                "base_model": "Qwen/Qwen3-8B",
                "model_type": "finetuned",
                "created_at": str(torch.datetime.now()),
                "config": {
                    "max_length": 512,
                    "lora_config": {
                        "r": 16,
                        "lora_alpha": 32,
                        "lora_dropout": 0.1
                    }
                }
            }
            
            with open(os.path.join(output_path, "model_info.json"), "w") as f:
                json.dump(model_info, f, indent=2)
            
            logger.info(f"Model saved successfully to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    @staticmethod
    def load_finetuned_model(
        model_path: str,
        device: str = "auto"
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Tải model đã fine-tune"""
        try:
            logger.info(f"Loading finetuned model from {model_path}")
            
            # Tải tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Tải model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
                load_in_8bit=True
            )
            
            logger.info(f"Successfully loaded finetuned model from {model_path}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading finetuned model from {model_path}: {e}")
            raise
    
    @staticmethod
    def generate_response(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Tạo response từ model"""
        try:
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            
            # Move to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Loại bỏ prompt từ response
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}" 