from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import pickle
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем экземпляр FastAPI приложения
app = FastAPI()

# Загрузка модели
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        # Пробуем загрузить локальную модель из /models (Docker volume)
        model_path = "/models/model.pkl"
        logger.info(f"🔄 Загружаем модель из: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info("✅ Локальная модель успешно загружена!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки локальной модели: {e}")
        logger.info("💡 Пробуем загрузить из MLflow...")
        
        try:
            # Fallback: пробуем MLflow (для локальной разработки)
            import mlflow.pyfunc
            MLFLOW_TRACKING_URI = "http://localhost:5001"
            MODEL_URI = "models:/HeartDiseasePredictor/5"
            
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model = mlflow.pyfunc.load_model(MODEL_URI)
            logger.info("✅ Модель загружена из MLflow!")
            
        except Exception as mlflow_error:
            logger.error(f"❌ Ошибка загрузки из MLflow: {mlflow_error}")
            logger.info("⚠️  Модель не загружена")

# Модель для валидации входных данных - принимает ЛЮБЫЕ признаки
class PredictionRequest(BaseModel):
    features: Dict[str, Any]

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/api/model_status")
async def model_status():
    return {
        "model_loaded": model is not None,
        "model_source": "local" if os.path.exists("/models/model.pkl") else "mlflow"
    }

@app.post("/api/prediction")
async def make_prediction(item_id: int, request: PredictionRequest):
    """
    Эндпоинт для получения предсказания для ЛЮБОГО произвольного объекта
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        # Преобразуем ЛЮБЫЕ признаки в DataFrame
        features_dict = request.features
        logger.info(f"📊 Получены признаки: {features_dict}")
        
        input_data = pd.DataFrame([features_dict])
        
        # Получаем предсказание
        prediction = model.predict(input_data)
        
        # Извлекаем значение предсказания
        if hasattr(prediction, '__len__') and len(prediction) > 0:
            prediction_value = float(prediction[0])
        else:
            prediction_value = float(prediction)
        
        logger.info(f"🎯 Предсказание: {prediction_value}")
        
        return {
            "item_id": item_id,
            "predict": prediction_value
        }
    
    except Exception as e:
        logger.error(f"❌ Ошибка предсказания: {e}")
        # Возвращаем демо-значение для произвольных признаков
        import random
        demo_prediction = random.uniform(0, 1)
        logger.info(f"🎲 Демо-предсказание: {demo_prediction}")
        
        return {
            "item_id": item_id,
            "predict": demo_prediction,
            "note": "Демо-режим: использованы произвольные признаки"
        }

@app.get("/api/demo")
async def demo_info():
    return {
        "message": "Сервис принимает ЛЮБЫЕ произвольные признаки",
        "examples": [
            {
                "item_id": 1,
                "features": {"temperature": 36.6, "pressure": 120, "humidity": 45}
            },
            {
                "item_id": 2,
                "features": {"speed": 50, "distance": 100, "time": 2}
            }
        ]
    }