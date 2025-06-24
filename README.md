# 🍽️ Restaurant Object Detection with YOLOv11

> **📖 README Languages / Языки README**  
> This README is available in two languages:  
> • [🇷🇺 Russian Version](#russian-version) (Русская версия)  
> • [🇺🇸 English Version](#english-version) (English версия)

---

## 🛠️ **Technology Stack**

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Ultralytics](https://img.shields.io/badge/YOLOv11-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

</div>

---

# Russian Version

## 🧠 Профессиональная система детекции объектов в ресторанах

**Высокопроизводительная система детекции объектов с использованием YOLOv11 для ресторанной среды**

[![GitHub](https://img.shields.io/badge/GitHub-amir2628-181717?style=flat-square&logo=github)](https://github.com/amir2628/restaurant-object-detection)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## 📋 Описание проекта

Профессиональная система автоматической детекции объектов, специально разработанная для ресторанной среды. Система использует современную архитектуру YOLOv11 и включает полный пайплайн машинного обучения: от автоматической аннотации данных до развертывания готовой модели.

### 🎯 Ключевые особенности

- **🤖 Автоматическая аннотация** с использованием ансамбля моделей
- **🎯 Высокая точность** - mAP@0.5: 79.7%
- **⚡ Быстрый инференс** - ~2ms на изображение
- **🔧 Production-ready** - готово к внедрению
- **📊 Comprehensive мониторинг** - детальная аналитика

### 🍽️ Детектируемые объекты

- 👥 **Люди** (персонал, посетители)
- 🪑 **Мебель** (столы, стулья)
- 🍽️ **Посуда** (тарелки, чашки, бокалы)
- 🍴 **Приборы** (вилки, ножи, ложки)
- 🍕 **Еда** (пицца, торты, фрукты)
- 📱 **Предметы** (телефоны, ноутбуки, книги)

## 🏗️ Архитектура проекта

```
restaurant-object-detection/
├── 📁 config/
│   ├── pipeline_config.json           # Конфигурация пайплайна
│   └── model_config.yaml             # Параметры модели
├── 📁 scripts/
│   ├── fix_annotations.py            # 🔧 Исправление аннотаций
│   ├── prepare_data.py               # 📊 Подготовка данных
│   ├── train_model.py                # 🚀 Обучение модели
│   ├── run_inference.py              # 🎯 Инференс
│   └── generate_final_report.py      # 📋 Генерация отчетов
├── 📁 src/
│   ├── data/                         # Модули обработки данных
│   ├── models/                       # Модели и инференс
│   ├── utils/                        # Утилиты
│   └── api/                          # API интерфейсы
├── 📁 data/
│   ├── raw/                          # Исходные видео
│   ├── processed/dataset/            # Готовый датасет
│   └── annotations/                  # Аннотации
├── 📁 outputs/
│   ├── experiments/                  # Результаты обучения
│   ├── inference/                    # Результаты инференса
│   └── reports/                      # Отчеты
└── 📄 requirements.txt               # Зависимости
```

## 🚀 Быстрый старт

### 1. Установка

```bash
# Клонирование репозитория
git clone https://github.com/amir2628/restaurant-object-detection.git
cd restaurant-object-detection

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Подготовка исходных данных

**⚠️ ВАЖНО: Поместите ваши видеофайлы в правильную директорию!**

```bash
# Создайте директорию для исходных видео (если не существует)
mkdir -p data/raw

# Поместите ваши видеофайлы ресторана в data/raw/
# Поддерживаемые форматы: .mp4, .avi, .mov, .mkv, .wmv
# Пример структуры:
# data/raw/
# ├── restaurant_video_1.mp4
# ├── restaurant_video_2.mp4
# └── restaurant_video_3.avi
```

### 3. Подготовка данных (если нужна аннотация)

```bash
# Автоматическое исправление пустых аннотаций
python scripts/fix_annotations.py --dataset "data/processed/dataset"

# Полный пайплайн подготовки данных (извлечение кадров + аннотация)
python scripts/prepare_data.py --input "data/raw" --config "config/pipeline_config.json"
```

### 4. Обучение модели

```bash
# Обучение с готовым датасетом
python scripts/train_model.py --data "data/processed/dataset/dataset.yaml"

# Обучение с кастомной конфигурацией
python scripts/train_model.py --data "dataset.yaml" --config "config/train_config.json"

# Обучение с Weights & Biases мониторингом
python scripts/train_model.py --data "dataset.yaml" --wandb
```

### 4. Инференс

```bash
# Инференс на изображениях
python scripts/run_inference.py \
  --model "outputs/experiments/yolo_*/weights/best.pt" \
  --input-dir "path/to/images"

# Инференс на видео
python scripts/run_inference.py \
  --model "outputs/experiments/yolo_*/weights/best.pt" \
  --video "path/to/video.mp4"

# Real-time инференс с камеры
python scripts/run_inference.py \
  --model "outputs/experiments/yolo_*/weights/best.pt" \
  --realtime --camera 0
```

### 5. Генерация отчетов

```bash
# Полный отчет по проекту
python scripts/generate_final_report.py \
  --model-path "outputs/experiments/yolo_*/weights/best.pt" \
  --dataset-dir "data/processed/dataset" \
  --experiment-dir "outputs/experiments/yolo_*" \
  --output "final_report.md" \
  --project-time 8.5
```

## 📊 Результаты

### 🏆 Достигнутые метрики

| Метрика | Значение | Комментарий |
|---------|----------|-------------|
| **mAP@0.5** | **79.7%** | 🥇 Отличный результат |
| **mAP@0.5:0.95** | **74.2%** | 🥈 Высокая точность |
| **Скорость инференса** | **~2ms** | ⚡ Real-time обработка |
| **Размер модели** | **~6MB** | 📦 Компактная |
| **Время обучения** | **17.5 мин** | 🚀 Быстрое обучение |

### 📈 Особенности реализации

- **🤖 Ensemble аннотация** - Использование 3 моделей (YOLOv11n, s, m)
- **🎯 TTA (Test Time Augmentation)** - Повышение точности
- **🔍 Smart фильтрация** - Автоматическое удаление некачественных детекций
- **⚡ GPU оптимизация** - CUDA, AMP, оптимизированные батчи
- **📊 Comprehensive мониторинг** - Wandb, TensorBoard, кастомные метрики

## 🔧 Конфигурация

### Основные параметры (config/pipeline_config.json)

```json
{
  "annotation": {
    "confidence_threshold": 0.25,
    "ensemble_models": ["yolo11n", "yolo11s", "yolo11m"],
    "tta_enabled": true
  },
  "training": {
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.01
  }
}
```

## 🎯 API использование

```python
from src.api.detection_api import DetectionAPI

# Инициализация
api = DetectionAPI(model_path="path/to/best.pt")

# Детекция на изображении
results = api.detect_image("image.jpg")

# Детекция на видео
results = api.detect_video("video.mp4")

# Batch обработка
results = api.detect_batch(["img1.jpg", "img2.jpg"])
```

## 🛠️ Системные требования

- **Python:** 3.8+
- **GPU:** CUDA 11.0+ (рекомендуется)
- **RAM:** 8GB+
- **GPU память:** 4GB+ (рекомендуется)
- **Место на диске:** 10GB+

## 📝 Лицензия

MIT License - см. [LICENSE](LICENSE) файл.

## 👥 Автор

**Amir** - [@amir2628](https://github.com/amir2628)

---

# English Version

## 🧠 Professional Restaurant Object Detection System

**High-performance object detection system using YOLOv11 for restaurant environments**

[![GitHub](https://img.shields.io/badge/GitHub-amir2628-181717?style=flat-square&logo=github)](https://github.com/amir2628/restaurant-object-detection)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## 📋 Project Description

Professional automatic object detection system specifically designed for restaurant environments. The system uses state-of-the-art YOLOv11 architecture and includes a complete machine learning pipeline: from automatic data annotation to production-ready model deployment.

### 🎯 Key Features

- **🤖 Automatic annotation** using ensemble of models
- **🎯 High accuracy** - mAP@0.5: 79.7%
- **⚡ Fast inference** - ~2ms per image
- **🔧 Production-ready** - ready for deployment
- **📊 Comprehensive monitoring** - detailed analytics

### 🍽️ Detectable Objects

- 👥 **People** (staff, customers)
- 🪑 **Furniture** (tables, chairs)
- 🍽️ **Tableware** (plates, cups, glasses)
- 🍴 **Utensils** (forks, knives, spoons)
- 🍕 **Food** (pizza, cakes, fruits)
- 📱 **Objects** (phones, laptops, books)

## 🏗️ Project Architecture

```
restaurant-object-detection/
├── 📁 config/
│   ├── pipeline_config.json           # Pipeline configuration
│   └── model_config.yaml             # Model parameters
├── 📁 scripts/
│   ├── fix_annotations.py            # 🔧 Fix annotations
│   ├── prepare_data.py               # 📊 Data preparation
│   ├── train_model.py                # 🚀 Model training
│   ├── run_inference.py              # 🎯 Inference
│   └── generate_final_report.py      # 📋 Report generation
├── 📁 src/
│   ├── data/                         # Data processing modules
│   ├── models/                       # Models and inference
│   ├── utils/                        # Utilities
│   └── api/                          # API interfaces
├── 📁 data/
│   ├── raw/                          # Source videos
│   ├── processed/dataset/            # Ready dataset
│   └── annotations/                  # Annotations
├── 📁 outputs/
│   ├── experiments/                  # Training results
│   ├── inference/                    # Inference results
│   └── reports/                      # Reports
└── 📄 requirements.txt               # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/amir2628/restaurant-object-detection.git
cd restaurant-object-detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Source Data

**⚠️ IMPORTANT: Place your video files in the correct directory!**

```bash
# Create directory for source videos (if it doesn't exist)
mkdir -p data/raw

# Place your restaurant video files in data/raw/
# Supported formats: .mp4, .avi, .mov, .mkv, .wmv
# Example structure:
# data/raw/
# ├── restaurant_video_1.mp4
# ├── restaurant_video_2.mp4
# └── restaurant_video_3.avi
```

### 3. Data Preparation (if annotation needed)

```bash
# Automatic fix for empty annotations
python scripts/fix_annotations.py --dataset "data/processed/dataset"

# Full data preparation pipeline (frame extraction + annotation)
python scripts/prepare_data.py --input "data/raw" --config "config/pipeline_config.json"
```

### 4. Model Training

```bash
# Training with ready dataset
python scripts/train_model.py --data "data/processed/dataset/dataset.yaml"

# Training with custom configuration
python scripts/train_model.py --data "dataset.yaml" --config "config/train_config.json"

# Training with Weights & Biases monitoring
python scripts/train_model.py --data "dataset.yaml" --wandb
```

### 4. Inference

```bash
# Inference on images
python scripts/run_inference.py \
  --model "outputs/experiments/yolo_*/weights/best.pt" \
  --input-dir "path/to/images"

# Inference on video
python scripts/run_inference.py \
  --model "outputs/experiments/yolo_*/weights/best.pt" \
  --video "path/to/video.mp4"

# Real-time inference from camera
python scripts/run_inference.py \
  --model "outputs/experiments/yolo_*/weights/best.pt" \
  --realtime --camera 0
```

### 5. Report Generation

```bash
# Complete project report
python scripts/generate_final_report.py \
  --model-path "outputs/experiments/yolo_*/weights/best.pt" \
  --dataset-dir "data/processed/dataset" \
  --experiment-dir "outputs/experiments/yolo_*" \
  --output "final_report.md" \
  --project-time 8.5
```

## 📊 Results

### 🏆 Achieved Metrics

| Metric | Value | Comment |
|--------|-------|---------|
| **mAP@0.5** | **79.7%** | 🥇 Excellent result |
| **mAP@0.5:0.95** | **74.2%** | 🥈 High accuracy |
| **Inference Speed** | **~2ms** | ⚡ Real-time processing |
| **Model Size** | **~6MB** | 📦 Compact |
| **Training Time** | **17.5 min** | 🚀 Fast training |

### 📈 Implementation Features

- **🤖 Ensemble annotation** - Using 3 models (YOLOv11n, s, m)
- **🎯 TTA (Test Time Augmentation)** - Improved accuracy
- **🔍 Smart filtering** - Automatic removal of low-quality detections
- **⚡ GPU optimization** - CUDA, AMP, optimized batching
- **📊 Comprehensive monitoring** - Wandb, TensorBoard, custom metrics

## 🔧 Configuration

### Main Parameters (config/pipeline_config.json)

```json
{
  "annotation": {
    "confidence_threshold": 0.25,
    "ensemble_models": ["yolo11n", "yolo11s", "yolo11m"],
    "tta_enabled": true
  },
  "training": {
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.01
  }
}
```

## 🎯 API Usage

```python
from src.api.detection_api import DetectionAPI

# Initialization
api = DetectionAPI(model_path="path/to/best.pt")

# Image detection
results = api.detect_image("image.jpg")

# Video detection
results = api.detect_video("video.mp4")

# Batch processing
results = api.detect_batch(["img1.jpg", "img2.jpg"])
```

## 🛠️ System Requirements

- **Python:** 3.8+
- **GPU:** CUDA 11.0+ (recommended)
- **RAM:** 8GB+
- **GPU Memory:** 4GB+ (recommended)
- **Disk Space:** 10GB+

## 📈 Performance Benchmarks

- **Real-time processing:** ✅ 30+ FPS
- **Batch processing:** ✅ 500+ images/minute
- **Memory usage:** ✅ <2GB GPU memory
- **Model accuracy:** ✅ Production-ready (79.7% mAP@0.5)

## 🚀 Deployment Options

### Docker Deployment
```bash
# Build container
docker build -t restaurant-detector .

# Run inference service
docker run -p 8000:8000 restaurant-detector
```

### API Service
```bash
# Start FastAPI service
python src/api/main.py

# Access at http://localhost:8000/docs
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file.

## 👥 Author

**Amir** - [@amir2628](https://github.com/amir2628)

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11
- Restaurant industry for inspiration
- Open-source community for tools and libraries

---

<div align="center">

**🌟 If this project helped you, please give it a star! 🌟**

[![GitHub stars](https://img.shields.io/github/stars/amir2628/restaurant-object-detection?style=social)](https://github.com/amir2628/restaurant-object-detection)
[![GitHub forks](https://img.shields.io/github/forks/amir2628/restaurant-object-detection?style=social)](https://github.com/amir2628/restaurant-object-detection)

</div>