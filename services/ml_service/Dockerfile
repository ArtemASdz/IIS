FROM python:3.11-slim

# Копируем содержимое текущей директории в /ml_service в контейнере
COPY . /ml_service

# Создаем директорию /models и делаем ее доступной снаружи
VOLUME /models

# Устанавливаем рабочую директорию
WORKDIR /ml_service

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt


# Открываем порт для доступа снаружи
EXPOSE 8000

# Запускаем сервер с явным указанием хоста и порта
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Команда для сборки образа:
# docker build -t heart-disease-predictor:1 .

# Команда для запуска контейнера:
# docker run -p 8000:8000 -v $(pwd)/../models:/models heart-disease-predictor:1