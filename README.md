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
![GroundingDINO](https://img.shields.io/badge/GroundingDINO-FF6B35?style=for-the-badge&logo=ai&logoColor=white)

</div>

---

# Russian Version

## 🧠 Профессиональная система детекции объектов в ресторанах

**Высокопроизводительная система детекции объектов с использованием YOLOv11 и автоматической аннотации GroundingDINO для ресторанной среды**

**Найдите отчет [здесь](/final_report.md)**

[![GitHub](https://img.shields.io/badge/GitHub-amir2628-181717?style=flat-square&logo=github)](https://github.com/amir2628/restaurant-object-detection)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![GroundingDINO](https://img.shields.io/badge/GroundingDINO-IDEA--Research-FF6B35?style=flat-square)](https://github.com/IDEA-Research/GroundingDINO)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## 📋 Описание проекта

Профессиональная система автоматической детекции объектов, специально разработанная для ресторанной среды. Система использует современную архитектуру YOLOv11 и включает **революционный механизм автоматической аннотации с GroundingDINO**, что позволяет обрабатывать тысячи кадров видео без ручной разметки данных.

### 🌟 Ключевые особенности

- **🤖 Автоматическая аннотация GroundingDINO**: Устраняет необходимость ручной разметки тысяч кадров
- **⚡ YOLOv11**: Последняя версия архитектуры YOLO для высокой точности и скорости
- **🎯 Специализация на ресторанах**: Оптимизировано для детекции еды и посуды
- **📊 Полный ML пайплайн**: От видео до готовой модели
- **🚀 GPU ускорение**: Поддержка CUDA для быстрой обработки

### 🍕 Детектируемые объекты

Система обучена распознавать 10 ключевых категорий ресторанных объектов:
- **Еда**: `chicken` (курица), `meat` (мясо), `salad` (салат), `soup` (суп)
- **Посуда**: `cup` (чашка), `plate` (тарелка), `bowl` (миска)
- **Приборы**: `spoon` (ложка), `fork` (вилка), `knife` (нож)

## 📁 Структура проекта

```
restaurant-object-detection/
├── 📁 config/
│   ├── pipeline_config.json           # Конфигурация пайплайна
│   └── model_config.yaml             # Параметры модели
├── 📁 scripts/
│   ├── fix_annotations.py            # 🔧 Исправление аннотаций
│   ├── prepare_data.py               # 📊 Подготовка данных с GroundingDINO
│   ├── train_model.py                # 🚀 Обучение модели
│   └── run_inference.py              # 🎯 Инференс
├── 📁 src/
│   ├── data/                         # Модули обработки данных
│   ├── models/                       # Модели и инференс
│   └── utils/                        # Утилиты
├── 📁 data/
│   ├── raw/                          # Исходные видео
│   ├── processed/dataset/            # Готовый датасет с аннотациями
│   └── annotations/                  # Аннотации GroundingDINO
├── 📁 outputs/
│   ├── experiments/                  # Результаты обучения
│   └── inference/                    # Результаты инференса
├── 📄 groundingdino_swinb_cogcoor.pth # Модель GroundingDINO
├── 📁 GroundingDINO/                 # Исходный код GroundingDINO
└── 📄 requirements.txt               # Зависимости
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Клонирование репозитория
git clone https://github.com/amir2628/restaurant-object-detection.git
cd restaurant-object-detection

# Установка основных зависимостей
pip install -r requirements.txt

# Установка GroundingDINO
pip install groundingdino-py

# Клонирование исходного кода GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git

# Скачивание предобученной модели GroundingDINO
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
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

**⚠️ ВАЖНО: Модели YOLO11l (Large: `yolo_restaurant_detection_1750941635`) и YOLO11n (nano: `yolo_restaurant_detection_1750973996`) уже были обучены на основе предоставленных тестовых данных. Вы можете пропустить подготовку данных и выполнить инференс. Если вы хотите обучить модель YOLO11 самостоятельно, то вам необходимо выполнить подготовку данных и запустить обучение заново.**

### 3. Подготовка данных с автоматической аннотацией GroundingDINO

```bash
# Полный пайплайн: извлечение кадров + автоматическая аннотация
python scripts/prepare_data.py --input "data/raw" --config "config/pipeline_config.json" --confidence 0.1

# С настройкой FPS для извлечения кадров
python scripts/prepare_data.py --input "data/raw" --config "config/pipeline_config.json" --fps 1.5 --confidence 0.2

# Увеличение порога уверенности для более качественных аннотаций
python scripts/prepare_data.py --input "data/raw" --config "config/pipeline_config.json" --confidence 0.3
```

**🧠 Что происходит на этапе подготовки данных:**
1. **Извлечение кадров** из видеофайлов с заданным FPS
2. **Автоматическая аннотация** каждого кадра с помощью GroundingDINO
3. **Фильтрация детекций** по порогу уверенности
4. **Разделение на train/val/test** splits (70%/20%/10%)
5. **Генерация dataset.yaml** для обучения YOLO
6. **Создание отчета** о качестве аннотаций

### 4. Обучение модели YOLOv11

```bash
# Базовое обучение с автоматически созданным датасетом
python scripts/train_model.py --data "data\processed\dataset\dataset.yaml" --device cuda

# Обучение с настройкой количества эпох
python scripts/train_model.py --data "data\processed\dataset\dataset.yaml" --device cuda --epochs 200

# Обучение с мониторингом Weights & Biases
python scripts/train_model.py --data "data\processed\dataset\dataset.yaml" --device cuda --wandb

# Обучение с настройкой размера батча
python scripts/train_model.py --data "data\processed\dataset\dataset.yaml" --device cuda --batch-size 32
```

### 5. Запуск инференса

```bash
# Инференс на видео с сохранением результатов
python scripts/run_inference.py --model "outputs\experiments\yolo_restaurant_detection_[EXPERIMENT_ID]\weights\best.pt" --video "[SOME_VIDEO_OR_DIRECTORY_OF_VIDEOS]" --output "outputs\final_demo" --device cuda

# Инференс на изображениях
python scripts/run_inference.py --model "outputs\experiments\yolo_restaurant_detection_[EXPERIMENT_ID]\weights\best.pt" --input-dir "path/to/images" --output "outputs\inference_results"

# Real-time инференс с веб-камеры
python scripts/run_inference.py --model "outputs\experiments\yolo_restaurant_detection_[EXPERIMENT_ID]\weights\best.pt" --realtime --camera 0

# Инференс с настройкой порога уверенности
python scripts/run_inference.py --model "outputs\experiments\yolo_restaurant_detection_[EXPERIMENT_ID]\weights\best.pt" --video "test_video.mp4" --confidence 0.3 --iou 0.5
```

## 🤖 GroundingDINO: Революция в автоматической аннотации

### Почему GroundingDINO?

Традиционная аннотация видеоданных для детекции объектов - это **крайне трудозатратный процесс**:
- 📹 Один час видео = ~108,000 кадров (при 30 FPS)
- ⏱️ Ручная разметка одного кадра = 2-5 минут
- 💰 Общее время: **3,600-9,000 часов** на один час видео!

**GroundingDINO решает эту проблему полностью автоматически:**

### 🔧 Как работает интеграция GroundingDINO

1. **Текстовые промпты**: Система использует естественные описания объектов
   ```
   "chicken . meat . salad . soup . cup . plate . bowl . spoon . fork . knife ."
   ```

2. **Автоматическая детекция**: GroundingDINO находит объекты по текстовому описанию

3. **Фильтрация качества**: Удаление детекций с низкой уверенностью

4. **YOLO формат**: Автоматическое преобразование в формат аннотаций YOLO

### ⚙️ Настройки GroundingDINO (config/pipeline_config.json)

```json
{
  "annotation": {
    "method": "groundingdino",
    "confidence_threshold": 0.25,
    "text_threshold": 0.25,
    "box_threshold": 0.25,
    "detection_prompt": "chicken . meat . salad . soup . cup . plate . bowl . spoon . fork . knife .",
    "iou_threshold": 0.6
  },
  "groundingdino": {
    "checkpoint_path": "groundingdino_swinb_cogcoor.pth",
    "config_paths": [
      "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    ]
  }
}
```

### 🎯 Преимущества автоматической аннотации

- **⚡ Скорость**: 1000+ кадров в час вместо 10-20 при ручной разметке
- **💰 Экономия**: Снижение затрат на аннотацию в 100+ раз
- **🎯 Консистентность**: Единые стандарты разметки для всего датасета  
- **📈 Масштабируемость**: Легкое добавление новых видео и классов

## 📊 Результаты обучения

### 🏆 Достигнутые метрики

Результаты обучения YOLOv11 на GPU (500 эпох, 87.3 минуты):

```
============================================================
🎯 ИТОГИ ОБУЧЕНИЯ
============================================================
📁 Эксперимент: yolo_restaurant_detection_1750973996
⏱️ Время обучения: 87.3 минут
🔄 Эпох завершено: 500
💻 Устройство: 0
📊 Финальный mAP@0.5: 0.7478
📊 Финальный mAP@0.5:0.95: 0.7055
💎 Лучшая модель: outputs\experiments\yolo_restaurant_detection_1750973996\weights\best.pt
============================================================

==================================================
🎉 ОБУЧЕНИЕ YOLO11 УСПЕШНО ЗАВЕРШЕНО!
==================================================
💎 Лучшая модель: outputs\experiments\yolo_restaurant_detection_1750973996\weights\best.pt
📊 mAP@0.5: 74.8%
⏱️ Время обучения: 87.3 мин
```

| Метрика | Значение | Комментарий |
|---------|----------|-------------|
| **mAP@0.5** | **74.8%** | 🥇 Отличный результат |
| **mAP@0.5:0.95** | **70.6%** | 🥈 Высокая точность |
| **Скорость инференса** | **~2ms** | ⚡ Real-time обработка |
| **Размер модели** | **~6MB** | 📦 Компактная |
| **Время обучения** | **87.3 мин** | 🚀 Быстрое обучение |

<img width="1280" alt="Image" src="https://github.com/user-attachments/assets/acc152e7-2ed4-485e-a824-da97a6c7bef3" />

### 📈 Особенности реализации

- **🤖 GroundingDINO аннотация** - Полностью автоматическая разметка данных
- **🎯 Умная фильтрация** - Автоматическое удаление некачественных детекций
- **⚡ GPU оптимизация** - CUDA, AMP, оптимизированные батчи
- **📊 Comprehensive мониторинг** - Детальные отчеты о процессе

## 🛠️ Системные требования

- **Python:** 3.8+
- **GPU:** CUDA 11.0+ (рекомендуется)
- **RAM:** 8GB+
- **GPU память:** 4GB+ (рекомендуется)
- **Место на диске:** 10GB+
- **GroundingDINO модель:** ~1.8GB

## 📝 Лицензия

MIT License - см. [LICENSE](LICENSE) файл.

## 👥 Автор

**Amir** - [@amir2628](https://github.com/amir2628)

---

# English Version

## 🧠 Professional Restaurant Object Detection System

**High-performance object detection system using YOLOv11 with automated GroundingDINO annotation for restaurant environments**

**Find the report [here](/final_report.md)**

[![GitHub](https://img.shields.io/badge/GitHub-amir2628-181717?style=flat-square&logo=github)](https://github.com/amir2628/restaurant-object-detection)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![GroundingDINO](https://img.shields.io/badge/GroundingDINO-IDEA--Research-FF6B35?style=flat-square)](https://github.com/IDEA-Research/GroundingDINO)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## 📋 Project Description

Professional automatic object detection system specifically designed for restaurant environments. The system uses the modern YOLOv11 architecture and includes a **revolutionary automatic annotation mechanism with GroundingDINO**, eliminating the need for manual annotation of thousands of video frames.

### 🌟 Key Features

- **🤖 Automatic GroundingDINO Annotation**: Eliminates the need for manual annotation of thousands of frames
- **⚡ YOLOv11**: Latest YOLO architecture for high accuracy and speed
- **🎯 Restaurant Specialization**: Optimized for food and tableware detection
- **📊 Complete ML Pipeline**: From video to ready-to-use model
- **🚀 GPU Acceleration**: CUDA support for fast processing

### 🍕 Detectable Objects

The system is trained to recognize 10 key categories of restaurant objects:
- **Food**: `chicken`, `meat`, `salad`, `soup`
- **Tableware**: `cup`, `plate`, `bowl`
- **Utensils**: `spoon`, `fork`, `knife`

## 📁 Project Structure

```
restaurant-object-detection/
├── 📁 config/
│   ├── pipeline_config.json           # Pipeline configuration
│   └── model_config.yaml             # Model parameters
├── 📁 scripts/
│   ├── fix_annotations.py            # 🔧 Annotation fixing
│   ├── prepare_data.py               # 📊 Data preparation with GroundingDINO
│   ├── train_model.py                # 🚀 Model training
│   └── run_inference.py              # 🎯 Inference
├── 📁 src/
│   ├── data/                         # Data processing modules
│   ├── models/                       # Models and inference
│   └── utils/                        # Utilities
├── 📁 data/
│   ├── raw/                          # Source videos
│   ├── processed/dataset/            # Ready dataset with annotations
│   └── annotations/                  # GroundingDINO annotations
├── 📁 outputs/
│   ├── experiments/                  # Training results
│   └── inference/                    # Inference results
├── 📄 groundingdino_swinb_cogcoor.pth # GroundingDINO model
├── 📁 GroundingDINO/                 # GroundingDINO source code
└── 📄 requirements.txt               # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/amir2628/restaurant-object-detection.git
cd restaurant-object-detection

# Install main dependencies
pip install -r requirements.txt

# Install GroundingDINO
pip install groundingdino-py

# Clone GroundingDINO source code
git clone https://github.com/IDEA-Research/GroundingDINO.git

# Download pre-trained GroundingDINO model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
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

**⚠️ IMPORTANT: A YOLO11l (Large: `yolo_restaurant_detection_1750941635`) and a YOLO11n (nano: `yolo_restaurant_detection_1750973996`) has already been trained on the test data which was provided. You can skip the data preparation and run the inference. If you want to train a YOLO 11 model yourself, then you have to do the data preparation and run the training again.**

### 3. Data Preparation with Automatic GroundingDINO Annotation

```bash
# Full pipeline: frame extraction + automatic annotation
python scripts/prepare_data.py --input "data/raw" --config "config/pipeline_config.json" --confidence 0.1

# With FPS setting for frame extraction
python scripts/prepare_data.py --input "data/raw" --config "config/pipeline_config.json" --fps 1.5 --confidence 0.2

# Increase confidence threshold for higher quality annotations
python scripts/prepare_data.py --input "data/raw" --config "config/pipeline_config.json" --confidence 0.3
```

**🧠 What happens during data preparation:**
1. **Frame extraction** from video files at specified FPS
2. **Automatic annotation** of each frame using GroundingDINO
3. **Detection filtering** by confidence threshold
4. **Train/val/test split** (70%/20%/10%)
5. **dataset.yaml generation** for YOLO training
6. **Quality report creation** about annotations

### 4. YOLOv11 Model Training

```bash
# Basic training with automatically created dataset
python scripts/train_model.py --data "data\processed\dataset\dataset.yaml" --device cuda

# Training with custom epoch count
python scripts/train_model.py --data "data\processed\dataset\dataset.yaml" --device cuda --epochs 200

# Training with Weights & Biases monitoring
python scripts/train_model.py --data "data\processed\dataset\dataset.yaml" --device cuda --wandb

# Training with custom batch size
python scripts/train_model.py --data "data\processed\dataset\dataset.yaml" --device cuda --batch-size 32
```

### 5. Run Inference

```bash
# Video inference with result saving
python scripts/run_inference.py --model "outputs\experiments\yolo_restaurant_detection_[EXPERIMENT_ID]\weights\best.pt" --video "[SOME_VIDEO_OR_DIRECTORY_OF_VIDEOS]" --output "outputs\final_demo" --device cuda

# Image inference
python scripts/run_inference.py --model "outputs\experiments\yolo_restaurant_detection_[EXPERIMENT_ID]\weights\best.pt" --input-dir "path/to/images" --output "outputs\inference_results"

# Real-time inference from webcam
python scripts/run_inference.py --model "outputs\experiments\yolo_restaurant_detection_[EXPERIMENT_ID]\weights\best.pt" --realtime --camera 0

# Inference with confidence threshold settings
python scripts/run_inference.py --model "outputs\experiments\yolo_restaurant_detection_[EXPERIMENT_ID]\weights\best.pt" --video "test_video.mp4" --confidence 0.3 --iou 0.5
```

## 🤖 GroundingDINO: Revolution in Automatic Annotation

### Why GroundingDINO?

Traditional video data annotation for object detection is an **extremely labor-intensive process**:
- 📹 One hour of video = ~108,000 frames (at 30 FPS)
- ⏱️ Manual annotation of one frame = 2-5 minutes
- 💰 Total time: **3,600-9,000 hours** for one hour of video!

**GroundingDINO solves this problem completely automatically:**

### 🔧 How GroundingDINO Integration Works

1. **Text Prompts**: System uses natural object descriptions
   ```
   "chicken . meat . salad . soup . cup . plate . bowl . spoon . fork . knife ."
   ```

2. **Automatic Detection**: GroundingDINO finds objects based on text descriptions

3. **Quality Filtering**: Removal of low-confidence detections

4. **YOLO Format**: Automatic conversion to YOLO annotation format

### ⚙️ GroundingDINO Settings (config/pipeline_config.json)

```json
{
  "annotation": {
    "method": "groundingdino",
    "confidence_threshold": 0.25,
    "text_threshold": 0.25,
    "box_threshold": 0.25,
    "detection_prompt": "chicken . meat . salad . soup . cup . plate . bowl . spoon . fork . knife .",
    "iou_threshold": 0.6
  },
  "groundingdino": {
    "checkpoint_path": "groundingdino_swinb_cogcoor.pth",
    "config_paths": [
      "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    ]
  }
}
```

### 🎯 Advantages of Automatic Annotation

- **⚡ Speed**: 1000+ frames per hour instead of 10-20 with manual annotation
- **💰 Cost Savings**: 100+ times reduction in annotation costs
- **🎯 Consistency**: Uniform annotation standards across the entire dataset
- **📈 Scalability**: Easy addition of new videos and classes

## 📊 Training Results

### 🏆 Achieved Metrics

YOLOv11 training results on GPU (500 epochs, 87.3 minutes):

```
============================================================
🎯 TRAINING SUMMARY
============================================================
📁 Experiment: yolo_restaurant_detection_1750973996
⏱️ Training time: 87.3 minutes
🔄 Epochs completed: 500
💻 Device: 0
📊 Final mAP@0.5: 0.7478
📊 Final mAP@0.5:0.95: 0.7055
💎 Best model: outputs\experiments\yolo_restaurant_detection_1750973996\weights\best.pt
============================================================

==================================================
🎉 YOLO11 TRAINING COMPLETED SUCCESSFULLY!
==================================================
💎 Best model: outputs\experiments\yolo_restaurant_detection_1750973996\weights\best.pt
📊 mAP@0.5: 74.8%
⏱️ Training time: 87.3 min
```

| Metric | Value | Comment |
|--------|-------|---------|
| **mAP@0.5** | **74.8%** | 🥇 Excellent result |
| **mAP@0.5:0.95** | **70.6%** | 🥈 High accuracy |
| **Inference Speed** | **~2ms** | ⚡ Real-time processing |
| **Model Size** | **~6MB** | 📦 Compact |
| **Training Time** | **87.3 min** | 🚀 Fast training |

<img width="1280" alt="Image" src="https://github.com/user-attachments/assets/acc152e7-2ed4-485e-a824-da97a6c7bef3" />

### 📈 Implementation Features

- **🤖 GroundingDINO Annotation** - Fully automatic data annotation
- **🎯 Smart Filtering** - Automatic removal of low-quality detections
- **⚡ GPU Optimization** - CUDA, AMP, optimized batching
- **📊 Comprehensive Monitoring** - Detailed process reports

## 🛠️ System Requirements

- **Python:** 3.8+
- **GPU:** CUDA 11.0+ (recommended)
- **RAM:** 8GB+
- **GPU Memory:** 4GB+ (recommended)
- **Disk Space:** 10GB+
- **GroundingDINO Model:** ~1.8GB

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11
- [IDEA-Research](https://github.com/IDEA-Research/GroundingDINO) for GroundingDINO
- Open-source community for tools and libraries

## 👥 Author

**Amir** - [@amir2628](https://github.com/amir2628)

## 🚀 Advanced Usage

### Additional Command Options

#### Data Preparation Advanced Options

```bash
# Extract frames at different FPS rates
python scripts/prepare_data.py --input "data/raw" --config "config/pipeline_config.json" --fps 0.5  # Lower FPS for fewer frames
python scripts/prepare_data.py --input "data/raw" --config "config/pipeline_config.json" --fps 5.0  # Higher FPS for more frames

# Adjust confidence threshold for annotation quality
python scripts/prepare_data.py --input "data/raw" --config "config/pipeline_config.json" --confidence 0.05  # More detections, lower quality
python scripts/prepare_data.py --input "data/raw" --config "config/pipeline_config.json" --confidence 0.4   # Fewer detections, higher quality

# Custom output directory
python scripts/prepare_data.py --input "data/raw" --config "config/pipeline_config.json" --output "data/custom_dataset"
```

#### Training Advanced Options

```bash
# Training with different model sizes
python scripts/train_model.py --data "data/processed/dataset/dataset.yaml" --device cuda --model yolo11n  # Nano (fastest)
python scripts/train_model.py --data "data/processed/dataset/dataset.yaml" --device cuda --model yolo11s  # Small
python scripts/train_model.py --data "data/processed/dataset/dataset.yaml" --device cuda --model yolo11m  # Medium
python scripts/train_model.py --data "data/processed/dataset/dataset.yaml" --device cuda --model yolo11l  # Large
python scripts/train_model.py --data "data/processed/dataset/dataset.yaml" --device cuda --model yolo11x  # Extra Large

# Training with custom image size
python scripts/train_model.py --data "data/processed/dataset/dataset.yaml" --device cuda --imgsz 512   # Smaller images, faster training
python scripts/train_model.py --data "data/processed/dataset/dataset.yaml" --device cuda --imgsz 1024  # Larger images, better accuracy

# Training with custom learning rate and optimization
python scripts/train_model.py --data "data/processed/dataset/dataset.yaml" --device cuda --lr0 0.001 --optimizer AdamW

# Resume training from checkpoint
python scripts/train_model.py --data "data/processed/dataset/dataset.yaml" --device cuda --resume "outputs/experiments/yolo_*/weights/last.pt"

# Training with data augmentation settings
python scripts/train_model.py --data "data/processed/dataset/dataset.yaml" --device cuda --augment --mixup 0.2 --copy_paste 0.1
```

#### Inference Advanced Options

```bash
# Batch inference on multiple videos
python scripts/run_inference.py --model "outputs/experiments/yolo_*/weights/best.pt" --input-dir "path/to/videos" --output "results" --device cuda

# Inference with custom confidence and IoU thresholds
python scripts/run_inference.py --model "outputs/experiments/yolo_*/weights/best.pt" --video "test.mp4" --confidence 0.1 --iou 0.3 --output "low_confidence_results"

# Inference with specific classes only
python scripts/run_inference.py --model "outputs/experiments/yolo_*/weights/best.pt" --video "test.mp4" --classes 0 1 2  # Only detect first 3 classes

# Save inference results in different formats
python scripts/run_inference.py --model "outputs/experiments/yolo_*/weights/best.pt" --video "test.mp4" --save-json --save-txt --save-crop

# Inference with video output settings
python scripts/run_inference.py --model "outputs/experiments/yolo_*/weights/best.pt" --video "test.mp4" --output "results" --fps 15 --quality high
```

#### Annotation Fixing Advanced Options

```bash
# Fix annotations for specific splits only
python scripts/fix_annotations.py --dataset "data/processed/dataset" --splits train val --auto-annotate

# Fix with different confidence thresholds
python scripts/fix_annotations.py --dataset "data/processed/dataset" --auto-annotate --confidence 0.15

# Create dataset structure and annotate in one step
python scripts/fix_annotations.py --dataset "data/new_dataset" --create-structure --auto-annotate --confidence 0.2

# Overwrite existing annotations
python scripts/fix_annotations.py --dataset "data/processed/dataset" --auto-annotate --overwrite --confidence 0.3
```

### 🔧 Configuration Customization

#### Custom Detection Classes

To detect different objects, modify `config/pipeline_config.json`:

```json
{
  "annotation": {
    "detection_prompt": "pizza . burger . fries . drink . napkin . menu .",
    "target_classes": ["pizza", "burger", "fries", "drink", "napkin", "menu"]
  },
  "dataset": {
    "class_names": ["pizza", "burger", "fries", "drink", "napkin", "menu"]
  }
}
```

#### Performance Optimization Settings

```json
{
  "video_processing": {
    "fps_extraction": 1.0,
    "max_frames_per_video": 500,
    "target_size": [416, 416]
  },
  "annotation": {
    "confidence_threshold": 0.3,
    "iou_threshold": 0.5
  },
  "quality_control": {
    "min_detection_size": 0.02,
    "max_detection_size": 0.9
  }
}
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. GroundingDINO Model Not Found
```bash
Error: groundingdino_swinb_cogcoor.pth not found
```
**Solution:**
```bash
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

#### 2. CUDA Out of Memory
```bash
Error: CUDA out of memory
```
**Solutions:**
- Reduce batch size: `--batch-size 8`
- Use smaller image size: `--imgsz 416`
- Use CPU: `--device cpu`

#### 3. No Video Files Found
```bash
Error: No supported video files found in data/raw/
```
**Solution:**
- Check file formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`
- Verify files are in `data/raw/` directory

#### 4. Empty Annotations
```bash
Warning: Many empty annotation files created
```
**Solutions:**
- Lower confidence threshold: `--confidence 0.1`
- Check video quality and lighting
- Verify detection prompt matches objects in videos

### 📊 Performance Monitoring

Monitor training progress with:
- **TensorBoard**: `tensorboard --logdir outputs/experiments/`
- **Weights & Biases**: Add `--wandb` flag to training
- **Live plots**: Check `outputs/experiments/*/` for training curves

### 💡 Tips for Better Results

1. **Video Quality**: Use well-lit, clear videos for better annotations
2. **Frame Rate**: Start with 1-2 FPS for initial experiments
3. **Confidence Tuning**: Lower thresholds (0.1-0.2) for more detections
4. **Class Balance**: Ensure diverse examples of all object types
5. **Validation**: Always check a sample of annotations manually

---

<div align="center">

**🌟 If this project helped you, please give it a star! 🌟**

[![GitHub stars](https://img.shields.io/github/stars/amir2628/restaurant-object-detection?style=social)](https://github.com/amir2628/restaurant-object-detection)
[![GitHub forks](https://img.shields.io/github/forks/amir2628/restaurant-object-detection?style=social)](https://github.com/amir2628/restaurant-object-detection)

</div>