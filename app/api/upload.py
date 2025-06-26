import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from typing import List
import uuid
from sqlalchemy.orm import Session

from app.database import get_db
from app.config import settings
from app.utils.file_processor import FileProcessor

router = APIRouter(prefix="/api/upload", tags=["File Upload"])

@router.post("/file")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload một file"""
    try:
        # Validate file size
        if file.size and file.size > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {settings.max_file_size} bytes"
            )
        
        # Validate file extension
        filename = file.filename
        if not filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        if not FileProcessor.is_supported_file(filename, settings.allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(settings.allowed_extensions)}"
            )
        
        # Generate unique filename
        file_extension = os.path.splitext(filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        file_path = os.path.join(settings.upload_dir, unique_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "message": "File uploaded successfully",
            "filename": filename,
            "file_path": file_path,
            "file_size": os.path.getsize(file_path)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/files")
async def upload_files(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Upload nhiều files"""
    try:
        uploaded_files = []
        
        for file in files:
            # Validate file size
            if file.size and file.size > settings.max_file_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} too large. Maximum size is {settings.max_file_size} bytes"
                )
            
            # Validate file extension
            filename = file.filename
            if not filename:
                continue
            
            if not FileProcessor.is_supported_file(filename, settings.allowed_extensions):
                continue
            
            # Generate unique filename
            file_extension = os.path.splitext(filename)[1]
            unique_filename = f"{uuid.uuid4().hex}{file_extension}"
            file_path = os.path.join(settings.upload_dir, unique_filename)
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append({
                "original_filename": filename,
                "file_path": file_path,
                "file_size": os.path.getsize(file_path)
            })
        
        return {
            "message": f"Uploaded {len(uploaded_files)} files successfully",
            "files": uploaded_files,
            "count": len(uploaded_files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/file/{file_path:path}")
async def download_file(file_path: str):
    """Download file"""
    try:
        full_path = os.path.join(settings.upload_dir, file_path)
        
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(full_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/file/{file_path:path}")
async def delete_file(file_path: str):
    """Xóa file"""
    try:
        full_path = os.path.join(settings.upload_dir, file_path)
        
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        os.remove(full_path)
        
        return {"message": "File deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process")
async def process_file(file_path: str, db: Session = Depends(get_db)):
    """Xử lý file và trả về nội dung"""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Xử lý file
        result = FileProcessor.process_file(file_path)
        
        return {
            "filename": result["filename"],
            "file_path": result["file_path"],
            "extension": result["extension"],
            "file_size": result["file_size"],
            "content_type": result["content_type"],
            "content": result["content"],
            "error": result.get("error")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_uploaded_files():
    """Lấy danh sách files đã upload"""
    try:
        files = []
        
        if os.path.exists(settings.upload_dir):
            for filename in os.listdir(settings.upload_dir):
                file_path = os.path.join(settings.upload_dir, filename)
                if os.path.isfile(file_path):
                    files.append({
                        "filename": filename,
                        "file_path": file_path,
                        "file_size": os.path.getsize(file_path),
                        "extension": os.path.splitext(filename)[1]
                    })
        
        return {
            "files": files,
            "count": len(files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_upload_config():
    """Lấy cấu hình upload"""
    return {
        "max_file_size": settings.max_file_size,
        "allowed_extensions": settings.allowed_extensions,
        "upload_dir": settings.upload_dir
    } 