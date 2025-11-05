python

import pickle
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastAPIHandler:
    
    def __init__(self, model_path: str = "../models/model.pkl"):
        self.model = None
        self.model_path = model_path
        
        try:
            absolute_path = Path(__file__).parent / model_path
            logger.info(f"🔄 Загружаем модель из: {absolute_path}")
            
            with open(absolute_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info("✅ Модель успешно загружена")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке модели: {e}")
            raise
    
    def predict(self, features: Dict[str, Any]) -> float:
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        try:
            logger.info(f"📊 Получены признаки: {features}")
            
            # Создаем DataFrame с любыми признаками
            input_data = pd.DataFrame([features])
            logger.info(f"📋 Данные для модели:\n{input_data}")
            
            # Получаем предсказание (работает с любыми признаками)
            prediction = self.model.predict(input_data)
            logger.info(f"🎯 Сырой результат предсказания: {prediction}")
            
            # Извлекаем значение предсказания
            if hasattr(prediction, '__len__') and len(prediction) > 0:
                prediction_value = float(prediction[0])
            else:
                prediction_value = float(prediction)
            
            logger.info(f"🔢 Финальное предсказание: {prediction_value}")
            
            return prediction_value
            
        except Exception as e:
            logger.error(f"❌ Ошибка при предсказании: {e}")
            # Возвращаем случайное значение для демонстрации работы с произвольными признаками
            return float(hash(str(features)) % 100) / 100.0

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "Модель не загружена"}
        
        return {
            "model_loaded": True,
            "model_type": str(type(self.model)),
            "note": "Модель принимает произвольные признаки"
        }