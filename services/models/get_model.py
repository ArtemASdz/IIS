import mlflow
import mlflow.sklearn
import pickle
import os
from pathlib import Path

def download_model_by_run_id(run_id, output_path="services/models/model.pkl"):
    """
    Выгружает модель из MLflow по run_id и сохраняет в файл model.pkl
    """
    
    MLFLOW_TRACKING_URI = "http://localhost:5001"
    
    try:
        # Устанавливаем подключение к MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"🔗 Подключаемся к MLflow: {MLFLOW_TRACKING_URI}")
        
        # Пробуем загрузить модель разными способами
        print(f"📥 Загружаем модель с run_id: {run_id}")
        
        # Способ 1: Через runs URI
        try:
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)
            print("✅ Модель загружена через runs URI")
        except Exception as e:
            print(f"❌ Ошибка через runs URI: {e}")
            # Способ 2: Через models URI (если модель зарегистрирована)
            model_uri = f"models:/HeartDiseasePredictor/5"
            model = mlflow.pyfunc.load_model(model_uri)
            print("✅ Модель загружена через models URI")
        
        # Сохраняем модель в файл
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✅ Модель успешно сохранена в: {output_path}")
        print(f"📏 Размер файла: {os.path.getsize(output_path)} байт")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        return False

def download_model_direct_artifact(run_id, output_path="models/model.pkl"):
    """
    Альтернативный способ: прямое копирование артефактов
    """
    import shutil
    from mlflow.tracking import MlflowClient
    
    MLFLOW_TRACKING_URI = "http://localhost:5001"
    
    try:
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        print(f"📥 Скачиваем артефакты для run_id: {run_id}")
        
        # Скачиваем всю папку артефактов
        local_dir = "models/temp_artifacts"
        client.download_artifacts(run_id, "model", local_dir)
        
        # Ищем файл модели
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                if file.endswith('.pkl') or file == 'model.pkl' or file == 'model':
                    source_path = os.path.join(root, file)
                    shutil.copy2(source_path, output_path)
                    print(f"✅ Модель скопирована из: {source_path}")
                    
                    # Удаляем временную директорию
                    shutil.rmtree(local_dir)
                    return True
        
        print("❌ Файл модели не найден в артефактах")
        return False
        
    except Exception as e:
        print(f"❌ Ошибка при скачивании артефактов: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Скрипт загрузки модели из MLflow")
    print("=" * 50)
    
    run_id = "06e7ec5721f94aceb33c7a308d4d2f32"
    
    print(f"🎯 Используем run_id: {run_id}")
    
    # Пробуем основной способ
    success = download_model_by_run_id(run_id)
    
    if not success:
        print("\n🔄 Пробуем альтернативный способ...")
        success = download_model_direct_artifact(run_id)
    
    if success:
        print("\n🎉 Модель успешно выгружена и сохранена в models/model.pkl!")
    else:
        print("\n❌ Все способы не удались")
        print("💡 Убедитесь, что:")
        print("   - MLflow сервер запущен: ./start_mlflow.sh")
        print("   - Сервер доступен по http://localhost:5001")
        print("   - Run ID корректен")