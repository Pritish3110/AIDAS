"""
FastAPI web application for animal disease classification
"""
import os
import io
import logging
import shutil
from pathlib import Path
from typing import List, Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from PIL import Image
import numpy as np

from config import Config
from inference import DiseasePredictor, save_predictions_to_file

# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL), format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Animal Disease Classification API",
    description="AI-powered animal disease identification system",
    version="1.0.0"
)

# Configuration
config = Config()

# Mount static files
app.mount("/static", StaticFiles(directory="../static"), name="static")

# Templates
templates = Jinja2Templates(directory="../templates")

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the disease predictor"""
    global predictor
    try:
        if os.path.exists(config.BEST_MODEL_PATH):
            predictor = DiseasePredictor(config.BEST_MODEL_PATH, config)
            logger.info("Disease predictor initialized successfully")
        else:
            logger.warning("No trained model found. Please train a model first.")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")

# Initialize predictor on startup
@app.on_event("startup")
async def startup_event():
    initialize_predictor()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with file upload interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    top_k: int = Form(3)
):
    """
    Predict disease from uploaded image
    
    Args:
        file: Uploaded image file
        top_k: Number of top predictions to return
    
    Returns:
        JSON response with predictions
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure a trained model is available."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    try:
        # Save uploaded file temporarily
        temp_dir = os.path.join(config.BASE_DIR, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # Save file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Make prediction
        result = predictor.get_prediction_explanation(temp_file_path)
        
        # Clean up temp file
        os.remove(temp_file_path)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "predictions": result['predictions'][:top_k],
            "top_prediction": result['top_prediction'],
            "explanation": result.get('explanation', {}),
            "preprocessing_info": result.get('preprocessing_info', {})
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    top_k: int = Form(3)
):
    """
    Predict diseases for multiple images
    
    Args:
        files: List of uploaded image files
        top_k: Number of top predictions to return for each image
    
    Returns:
        JSON response with batch predictions
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure a trained model is available."
        )
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400, 
            detail="Maximum 10 files allowed in batch prediction"
        )
    
    results = []
    temp_dir = os.path.join(config.BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        temp_files = []
        
        # Save all files temporarily
        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not an image"
                )
            
            temp_file_path = os.path.join(temp_dir, file.filename)
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_files.append(temp_file_path)
        
        # Make predictions
        predictions = predictor.predict_batch(temp_files, top_k)
        
        # Format results
        for i, (file, prediction) in enumerate(zip(files, predictions)):
            if 'error' in prediction:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": prediction['error']
                })
            else:
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "predictions": prediction['predictions'],
                    "top_prediction": prediction['top_prediction']
                })
        
        # Clean up temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return JSONResponse(content={
            "success": True,
            "total_files": len(files),
            "results": results
        })
        
    except Exception as e:
        # Clean up temp files on error
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """Get information about the loaded model"""
    if predictor is None:
        return {"model_loaded": False, "message": "No model loaded"}
    
    return {
        "model_loaded": True,
        "class_names": predictor.class_names,
        "num_classes": len(predictor.class_names),
        "input_shape": predictor.model.input_shape,
        "model_type": "TensorFlow/Keras"
    }

@app.get("/models/classes")
async def get_classes():
    """Get list of supported disease classes"""
    if predictor is None:
        return {"classes": config.DISEASE_CATEGORIES}
    
    return {"classes": predictor.class_names}

@app.post("/models/reload")
async def reload_model():
    """Reload the model"""
    global predictor
    try:
        initialize_predictor()
        if predictor is not None:
            return {"success": True, "message": "Model reloaded successfully"}
        else:
            return {"success": False, "message": "Failed to load model"}
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

@app.get("/api/docs")
async def get_api_documentation():
    """Get API documentation"""
    return {
        "title": "Animal Disease Classification API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "Home page with file upload interface",
            "GET /health": "Health check",
            "POST /predict": "Predict disease from single image",
            "POST /predict/batch": "Predict diseases from multiple images",
            "GET /models/info": "Get model information",
            "GET /models/classes": "Get supported disease classes",
            "POST /models/reload": "Reload the model"
        },
        "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
        "max_file_size": "10MB per file",
        "max_batch_size": "10 files"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG,
        log_level="info"
    )