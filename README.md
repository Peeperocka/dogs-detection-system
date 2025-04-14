
## 📦 Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Peeperocka/dogs-detection-system
cd dogs-detection-system
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## 🚀 Использование

1. Поместите изображения для обработки в папку `input_images`
2. Запустите систему:
```bash
python main.py
```

**Что происходит:**
- Система мониторит директорию `input_images`
- Обнаруженные изображения добавляются в очередь обработки, если очередь не переполнена
- Результаты сохраняются в лог-файл `dog_detector.log`
- Обработанные файлы удаляются из исходной директории

## 🛠 Кастомизация

Чтобы добавить свою логику обработки результатов:

1. Редактируйте файл `processing/response.py`
2. Модифицируйте метод `handle_pack` (или добавьте свои):

```python
class Response:
    def handle_pack(self, clusters):
        if clusters:
            pack_sizes = [len(cluster) for cluster in clusters]

            # Ваша кастомная логика:
            # - Сохранение в БД
            # - Отправка уведомлений
            # - Визуализация результатов

            logger.info(f"Обнаружены стаи: {pack_sizes}")
```

## ⚙ Конфигурация

Настройки в файле `config.py`:

```python
# Модель
MODEL_PATH = "models/best.onnx"         # Путь к ONNX-модели
CONFIDENCE_THRESHOLD = 0.5              # Порог уверенности (0.0-1.0)

# Кластеризация
EPS = 0.6                               # Радиус поиска соседей (% от высоты)
MIN_SAMPLES = 3                         # Мин. собак для стаи

# Производительность
THREAD_POOL_MAX_WORKERS = 4             # Потоки для обработки
QUEUE_MAX_SIZE = 10                     # Макс. размер очереди

# Логирование
LOG_LEVEL = "INFO"                      # Уровень детализации
MAX_FILE_SIZE_MB = 10                   # Макс. размер лог-файла
```

## 📂 Структура проекта

```
.
├── config.py             # Конфигурация
├── main.py               # Точка входа
├── requirements.txt      # Зависимости
├── input_images/         # Исходные изображения
├── models/               # ONNX-модель
│
├── processing
│   ├── detector.py       # Детекция объектов
│   ├── cluster.py        # Алгоритмы кластеризации
│   ├── response.py       # Обработка результатов (кастомизируемый)
│   └── image_queue.py    # Управление очередью
│
└── utils
    └── logger.py         # Настройка логирования
```

## 📚 Рекомендации

1. Для дообучения модели используйте [YOLOv11](https://github.com/ultralytics/ultralytics)
2. Экспорт в ONNX:
```python
from ultralytics import YOLO
model = YOLO('yolov11n.pt')
model.export(format='onnx', imgsz=[640, 640])
```

3. Оптимальные настройки для CPU:
```python
# config.py
THREAD_POOL_MAX_WORKERS = os.cpu_count() // 2
```
