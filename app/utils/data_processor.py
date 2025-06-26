import json
import re
from typing import List, Dict, Any, Tuple
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Utility class để xử lý dữ liệu cho fine-tuning"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Làm sạch text"""
        # Loại bỏ ký tự đặc biệt không cần thiết
        text = re.sub(r'\s+', ' ', text)  # Thay thế nhiều khoảng trắng
        text = re.sub(r'[^\w\s\.,!?;:()\[\]{}"\'-]', '', text)  # Loại bỏ ký tự đặc biệt
        return text.strip()
    
    @staticmethod
    def create_instruction_prompt(instruction: str, context: str = "", response: str = "") -> str:
        """Tạo prompt theo format instruction"""
        if context:
            prompt = f"Context: {context}\n\nInstruction: {instruction}\n\nResponse: {response}"
        else:
            prompt = f"Instruction: {instruction}\n\nResponse: {response}"
        return prompt
    
    @staticmethod
    def create_conversation_prompt(messages: List[Dict[str, str]]) -> str:
        """Tạo prompt theo format conversation"""
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prompt += f"{role.capitalize()}: {content}\n"
        return prompt.strip()
    
    @staticmethod
    def extract_qa_pairs(text: str) -> List[Dict[str, str]]:
        """Trích xuất cặp Q&A từ text"""
        qa_pairs = []
        
        # Pattern để tìm câu hỏi và câu trả lời
        patterns = [
            r'Q[:\s]*([^?\n]+)\?[:\s]*A[:\s]*([^?\n]+)',
            r'Question[:\s]*([^?\n]+)\?[:\s]*Answer[:\s]*([^?\n]+)',
            r'([^?\n]+\?)[:\s]*([^?\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                question = DataProcessor.clean_text(match[0])
                answer = DataProcessor.clean_text(match[1])
                if question and answer:
                    qa_pairs.append({
                        "instruction": question,
                        "response": answer
                    })
        
        return qa_pairs
    
    @staticmethod
    def process_table_data(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Xử lý dữ liệu bảng thành instruction-response pairs"""
        processed_data = []
        
        for row in data:
            # Tạo instruction từ các cột
            instruction_parts = []
            response_parts = []
            
            for key, value in row.items():
                if key.lower() in ['question', 'instruction', 'prompt']:
                    instruction_parts.append(str(value))
                elif key.lower() in ['answer', 'response', 'output']:
                    response_parts.append(str(value))
                else:
                    # Thêm vào context
                    instruction_parts.append(f"{key}: {value}")
            
            if instruction_parts and response_parts:
                instruction = " ".join(instruction_parts)
                response = " ".join(response_parts)
                
                processed_data.append({
                    "instruction": DataProcessor.clean_text(instruction),
                    "response": DataProcessor.clean_text(response)
                })
        
        return processed_data
    
    @staticmethod
    def create_training_dataset(data_files: List[str], max_length: int = 512) -> Dataset:
        """Tạo dataset cho training từ các file"""
        all_data = []
        
        for file_path in data_files:
            try:
                # Đọc và xử lý file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Trích xuất Q&A pairs
                qa_pairs = DataProcessor.extract_qa_pairs(content)
                
                for qa in qa_pairs:
                    # Tạo prompt
                    prompt = DataProcessor.create_instruction_prompt(
                        qa["instruction"], 
                        response=qa["response"]
                    )
                    
                    # Cắt ngắn nếu cần
                    if len(prompt) > max_length:
                        prompt = prompt[:max_length]
                    
                    all_data.append({
                        "text": prompt,
                        "instruction": qa["instruction"],
                        "response": qa["response"]
                    })
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        # Tạo dataset
        if all_data:
            dataset = Dataset.from_list(all_data)
            return dataset
        else:
            raise ValueError("No valid data found in the provided files")
    
    @staticmethod
    def prepare_finetuning_data(
        data_files: List[str], 
        data_type: str = "instruction",
        max_length: int = 512
    ) -> Tuple[Dataset, int]:
        """Chuẩn bị dữ liệu cho fine-tuning"""
        
        all_data = []
        
        for file_path in data_files:
            try:
                # Xử lý file dựa trên extension
                from .file_processor import FileProcessor
                result = FileProcessor.process_file(file_path)
                
                if result["content_type"] == "text":
                    # Xử lý text
                    content = result["content"]
                    qa_pairs = DataProcessor.extract_qa_pairs(content)
                    
                    for qa in qa_pairs:
                        if data_type == "instruction":
                            prompt = DataProcessor.create_instruction_prompt(
                                qa["instruction"], 
                                response=qa["response"]
                            )
                        else:
                            prompt = DataProcessor.create_conversation_prompt([
                                {"role": "user", "content": qa["instruction"]},
                                {"role": "assistant", "content": qa["response"]}
                            ])
                        
                        if len(prompt) > max_length:
                            prompt = prompt[:max_length]
                        
                        all_data.append({
                            "text": prompt,
                            "instruction": qa["instruction"],
                            "response": qa["response"]
                        })
                
                elif result["content_type"] == "table":
                    # Xử lý bảng
                    table_data = DataProcessor.process_table_data(result["content"])
                    
                    for item in table_data:
                        if data_type == "instruction":
                            prompt = DataProcessor.create_instruction_prompt(
                                item["instruction"], 
                                response=item["response"]
                            )
                        else:
                            prompt = DataProcessor.create_conversation_prompt([
                                {"role": "user", "content": item["instruction"]},
                                {"role": "assistant", "content": item["response"]}
                            ])
                        
                        if len(prompt) > max_length:
                            prompt = prompt[:max_length]
                        
                        all_data.append({
                            "text": prompt,
                            "instruction": item["instruction"],
                            "response": item["response"]
                        })
                        
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid data found in the provided files")
        
        # Tạo dataset
        dataset = Dataset.from_list(all_data)
        
        return dataset, len(all_data)
    
    @staticmethod
    def save_processed_data(dataset: Dataset, output_path: str):
        """Lưu dataset đã xử lý"""
        dataset.save_to_disk(output_path)
        logger.info(f"Processed data saved to {output_path}")
    
    @staticmethod
    def load_processed_data(data_path: str) -> Dataset:
        """Tải dataset đã xử lý"""
        return Dataset.load_from_disk(data_path) 