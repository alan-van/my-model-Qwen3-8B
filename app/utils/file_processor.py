import os
import pandas as pd
import PyPDF2
from docx import Document
from PIL import Image
import io
import base64
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FileProcessor:
    """Utility class để xử lý các loại file khác nhau"""
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Lấy extension của file"""
        return os.path.splitext(filename)[1].lower()
    
    @staticmethod
    def is_supported_file(filename: str, allowed_extensions: List[str]) -> bool:
        """Kiểm tra file có được hỗ trợ không"""
        ext = FileProcessor.get_file_extension(filename)
        return ext.lstrip('.') in allowed_extensions
    
    @staticmethod
    def process_csv(file_path: str) -> List[Dict[str, Any]]:
        """Xử lý file CSV"""
        try:
            df = pd.read_csv(file_path)
            data = []
            
            # Chuyển đổi DataFrame thành list of dicts
            for _, row in df.iterrows():
                row_dict = {}
                for col in df.columns:
                    value = row[col]
                    if pd.isna(value):
                        row_dict[col] = ""
                    else:
                        row_dict[col] = str(value)
                data.append(row_dict)
            
            return data
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {e}")
            raise
    
    @staticmethod
    def process_txt(file_path: str) -> str:
        """Xử lý file TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Error processing TXT file {file_path}: {e}")
            raise
    
    @staticmethod
    def process_pdf(file_path: str) -> str:
        """Xử lý file PDF"""
        try:
            content = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            return content.strip()
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {e}")
            raise
    
    @staticmethod
    def process_docx(file_path: str) -> str:
        """Xử lý file DOCX"""
        try:
            doc = Document(file_path)
            content = ""
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            return content.strip()
        except Exception as e:
            logger.error(f"Error processing DOCX file {file_path}: {e}")
            raise
    
    @staticmethod
    def process_xlsx(file_path: str) -> List[Dict[str, Any]]:
        """Xử lý file XLSX"""
        try:
            df = pd.read_excel(file_path)
            data = []
            
            # Chuyển đổi DataFrame thành list of dicts
            for _, row in df.iterrows():
                row_dict = {}
                for col in df.columns:
                    value = row[col]
                    if pd.isna(value):
                        row_dict[col] = ""
                    else:
                        row_dict[col] = str(value)
                data.append(row_dict)
            
            return data
        except Exception as e:
            logger.error(f"Error processing XLSX file {file_path}: {e}")
            raise
    
    @staticmethod
    def process_image(file_path: str) -> str:
        """Xử lý file image và trả về base64 string"""
        try:
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large
                max_size = (800, 800)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {e}")
            raise
    
    @staticmethod
    def process_file(file_path: str) -> Dict[str, Any]:
        """Xử lý file dựa trên extension"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        filename = os.path.basename(file_path)
        extension = FileProcessor.get_file_extension(filename)
        
        result = {
            "filename": filename,
            "file_path": file_path,
            "extension": extension,
            "file_size": os.path.getsize(file_path),
            "content": None,
            "content_type": None
        }
        
        try:
            if extension == '.csv':
                result["content"] = FileProcessor.process_csv(file_path)
                result["content_type"] = "table"
            elif extension == '.txt':
                result["content"] = FileProcessor.process_txt(file_path)
                result["content_type"] = "text"
            elif extension == '.pdf':
                result["content"] = FileProcessor.process_pdf(file_path)
                result["content_type"] = "text"
            elif extension == '.docx':
                result["content"] = FileProcessor.process_docx(file_path)
                result["content_type"] = "text"
            elif extension == '.xlsx':
                result["content"] = FileProcessor.process_xlsx(file_path)
                result["content_type"] = "table"
            elif extension in ['.jpg', '.jpeg', '.png']:
                result["content"] = FileProcessor.process_image(file_path)
                result["content_type"] = "image"
            else:
                raise ValueError(f"Unsupported file type: {extension}")
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            result["error"] = str(e)
        
        return result
    
    @staticmethod
    def extract_text_from_files(file_paths: List[str]) -> str:
        """Trích xuất text từ danh sách file"""
        all_text = []
        
        for file_path in file_paths:
            try:
                result = FileProcessor.process_file(file_path)
                if result["content_type"] == "text":
                    all_text.append(result["content"])
                elif result["content_type"] == "table":
                    # Convert table to text
                    if isinstance(result["content"], list):
                        for row in result["content"]:
                            all_text.append(" ".join([f"{k}: {v}" for k, v in row.items()]))
            except Exception as e:
                logger.error(f"Error extracting text from {file_path}: {e}")
                continue
        
        return "\n\n".join(all_text) 