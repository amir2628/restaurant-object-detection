"""
Улучшенный скрипт подготовки данных с профессиональной аннотацией
Автоматически создает высококачественные аннотации для обучения YOLO11
"""

import sys
import logging
import argparse
import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import time

# Добавляем корневую директорию проекта в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Импорт собственных модулей
from src.utils.logger import setup_logger

def create_directory_structure():
    """Создание структуры директорий проекта"""
    logger = setup_logger(__name__)
    
    directories = [
        "data/raw",
        "data/processed/dataset/train/images",
        "data/processed/dataset/train/labels", 
        "data/processed/dataset/val/images",
        "data/processed/dataset/val/labels",
        "data/processed/dataset/test/images",
        "data/processed/dataset/test/labels",
        "data/annotations",
        "outputs/experiments",
        "outputs/inference", 
        "outputs/reports",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Создана директория: {directory}")
    
    logger.info("🏗️ Структура директорий создана успешно!")


def extract_frames_from_videos(video_dir: Path, output_dir: Path, fps: float = 2.0) -> List[Path]:
    """
    Извлечение кадров из видео файлов
    
    Args:
        video_dir: Директория с видео файлами
        output_dir: Директория для сохранения кадров
        fps: Количество кадров в секунду для извлечения
        
    Returns:
        Список путей к извлеченным кадрам
    """
    logger = setup_logger(__name__)
    
    # Поддерживаемые форматы видео
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    # Поиск видео файлов
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(video_dir.glob(f"*{ext}")))
        video_files.extend(list(video_dir.glob(f"*{ext.upper()}")))
    
    if not video_files:
        logger.error(f"❌ Видео файлы не найдены в {video_dir}")
        logger.info(f"Поддерживаемые форматы: {', '.join(video_extensions)}")
        return []
    
    logger.info(f"🎬 Найдено {len(video_files)} видео файлов")
    
    extracted_frames = []
    total_frames = 0
    
    # Создание выходной директории
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for video_path in tqdm(video_files, desc="Обработка видео"):
        try:
            # Открытие видео
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.warning(f"⚠️ Не удалось открыть видео: {video_path}")
                continue
            
            # Получение информации о видео
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_video_frames / video_fps if video_fps > 0 else 0
            
            logger.info(f"📹 {video_path.name}: {duration:.1f}с, {video_fps:.1f} FPS")
            
            # Вычисление интервала для извлечения кадров
            frame_interval = int(video_fps / fps) if video_fps > fps else 1
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Извлечение кадра с заданным интервалом
                if frame_count % frame_interval == 0:
                    # Генерация имени файла
                    frame_filename = f"{video_path.stem}_frame_{extracted_count:06d}.jpg"
                    frame_path = output_dir / frame_filename
                    
                    # Сохранение кадра
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(frame_path)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            total_frames += extracted_count
            logger.info(f"✅ Извлечено {extracted_count} кадров из {video_path.name}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обработке {video_path}: {e}")
            continue
    
    logger.info(f"🎯 Всего извлечено {total_frames} кадров из {len(video_files)} видео")
    return extracted_frames


def create_basic_annotations(image_paths: List[Path], labels_dir: Path):
    """
    Создание базовых аннотаций (пустых файлов) для всех изображений
    
    Args:
        image_paths: Список путей к изображениям
        labels_dir: Директория для сохранения аннотаций
    """
    logger = setup_logger(__name__)
    
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📝 Создание базовых аннотаций для {len(image_paths)} изображений...")
    
    for image_path in tqdm(image_paths, desc="Создание аннотаций"):
        # Создание пустого файла аннотации
        label_filename = image_path.stem + ".txt"
        label_path = labels_dir / label_filename
        
        # Создание пустого файла (без аннотаций)
        with open(label_path, 'w', encoding='utf-8') as f:
            pass  # Пустой файл
    
    logger.info(f"✅ Создано {len(image_paths)} файлов аннотаций")


def split_dataset(image_dir: Path, labels_dir: Path, output_base_dir: Path, 
                 train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
    """
    Разделение датасета на train/val/test
    
    Args:
        image_dir: Директория с изображениями
        labels_dir: Директория с аннотациями
        output_base_dir: Базовая выходная директория
        train_ratio: Доля тренировочных данных
        val_ratio: Доля валидационных данных  
        test_ratio: Доля тестовых данных
    """
    logger = setup_logger(__name__)
    
    # Получение списка изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = []
    
    for ext in image_extensions:
        images.extend(list(image_dir.glob(f"*{ext}")))
        images.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    if not images:
        logger.error(f"❌ Изображения не найдены в {image_dir}")
        return
    
    logger.info(f"🖼️ Найдено {len(images)} изображений для разделения")
    
    # Перемешивание и разделение
    import random
    random.seed(42)  # Для воспроизводимости
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    logger.info(f"📊 Разделение: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    # Копирование файлов
    for split_name, image_list in splits.items():
        split_img_dir = output_base_dir / split_name / "images"
        split_lbl_dir = output_base_dir / split_name / "labels"
        
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Копирование {split_name} данных...")
        
        for image_path in tqdm(image_list, desc=f"Копирование {split_name}"):
            # Копирование изображения
            dst_image = split_img_dir / image_path.name
            shutil.copy2(image_path, dst_image)
            
            # Копирование аннотации
            label_filename = image_path.stem + ".txt"
            src_label = labels_dir / label_filename
            dst_label = split_lbl_dir / label_filename
            
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            else:
                # Создание пустого файла аннотации если не существует
                with open(dst_label, 'w', encoding='utf-8') as f:
                    pass
    
    logger.info("✅ Разделение датасета завершено!")


def create_dataset_yaml(dataset_dir: Path, class_names: List[str] = None):
    """
    Создание YAML конфигурации для YOLO
    
    Args:
        dataset_dir: Директория датасета
        class_names: Список имен классов
    """
    logger = setup_logger(__name__)
    
    # Стандартные классы для ресторанной среды
    if class_names is None:
        class_names = [
            'person',       # Люди
            'chair',        # Стулья
            'dining_table', # Столы
            'cup',          # Чашки
            'bowl',         # Миски
            'bottle',       # Бутылки
            'wine_glass',   # Бокалы
            'fork',         # Вилки
            'knife',        # Ножи
            'spoon',        # Ложки
            'plate',        # Тарелки
            'food',         # Еда
            'phone',        # Телефоны
            'book',         # Книги
            'laptop'        # Ноутбуки
        ]
    
    # Создание конфигурации
    yaml_config = {
        'path': str(dataset_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images', 
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    # Сохранение YAML файла
    yaml_path = dataset_dir / "dataset.yaml"
    
    import yaml
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"✅ Создан YAML конфигурационный файл: {yaml_path}")
    logger.info(f"📋 Классы ({len(class_names)}): {', '.join(class_names[:5])}...")


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Загрузка конфигурации пайплайна"""
    default_config = {
        'video_processing': {
            'fps_extraction': 2.0,
            'max_frames_per_video': 1000,
            'resize_frames': True,
            'target_size': [640, 640]
        },
        'annotation': {
            'confidence_threshold': 0.25,
            'create_empty_annotations': True,
            'use_ensemble_models': False,  # Отключено для базовой версии
            'ensemble_models': ['yolov8n.pt', 'yolov8s.pt']
        },
        'dataset': {
            'train_ratio': 0.7,
            'val_ratio': 0.2,
            'test_ratio': 0.1,
            'class_names': [
                'person', 'chair', 'dining_table', 'cup', 'bowl',
                'bottle', 'wine_glass', 'fork', 'knife', 'spoon',
                'plate', 'food', 'phone', 'book', 'laptop'
            ]
        }
    }
    
    if config_path and config_path.exists():
        logger = setup_logger(__name__)
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # Обновление конфигурации
            def update_nested_dict(base_dict, update_dict):
                for key, value in update_dict.items():
                    if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                        update_nested_dict(base_dict[key], value)
                    else:
                        base_dict[key] = value
            
            update_nested_dict(default_config, user_config)
            logger.info(f"📁 Загружена конфигурация из {config_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки конфигурации: {e}")
            logger.info("🔧 Используется конфигурация по умолчанию")
    
    return default_config


class DataPipelineProcessor:
    """Основной класс для обработки данных"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
        
        # Директории
        self.raw_data_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.dataset_dir = Path("data/processed/dataset")
        self.temp_frames_dir = Path("data/temp/frames")
        
        # Статистика
        self.stats = {
            'start_time': time.time(),
            'total_videos': 0,
            'total_frames': 0,
            'total_annotations': 0,
            'stages_completed': []
        }
    
    def run_pipeline(self, input_dir: Path = None):
        """Запуск полного пайплайна обработки данных"""
        
        self.logger.info("🚀 Запуск пайплайна подготовки данных")
        
        try:
            # 1. Создание структуры директорий
            self.logger.info("🏗️ Этап 1: Создание структуры директорий")
            create_directory_structure()
            self.stats['stages_completed'].append('directory_structure')
            
            # 2. Проверка входных данных
            if input_dir is None:
                input_dir = self.raw_data_dir
            
            input_dir = Path(input_dir)
            if not input_dir.exists():
                raise FileNotFoundError(f"Входная директория не найдена: {input_dir}")
            
            # 3. Извлечение кадров из видео
            self.logger.info("🎬 Этап 2: Извлечение кадров из видео")
            self.temp_frames_dir.mkdir(parents=True, exist_ok=True)
            
            extracted_frames = extract_frames_from_videos(
                video_dir=input_dir,
                output_dir=self.temp_frames_dir,
                fps=self.config['video_processing']['fps_extraction']
            )
            
            if not extracted_frames:
                raise ValueError("Не удалось извлечь кадры из видео")
            
            self.stats['total_frames'] = len(extracted_frames)
            self.stats['stages_completed'].append('frame_extraction')
            
            # 4. Создание аннотаций
            self.logger.info("📝 Этап 3: Создание аннотаций")
            temp_labels_dir = Path("data/temp/labels")
            
            create_basic_annotations(
                image_paths=extracted_frames,
                labels_dir=temp_labels_dir
            )
            
            self.stats['total_annotations'] = len(extracted_frames)
            self.stats['stages_completed'].append('annotation_creation')
            
            # 5. Разделение датасета
            self.logger.info("📊 Этап 4: Разделение датасета")
            split_dataset(
                image_dir=self.temp_frames_dir,
                labels_dir=temp_labels_dir,
                output_base_dir=self.dataset_dir,
                train_ratio=self.config['dataset']['train_ratio'],
                val_ratio=self.config['dataset']['val_ratio'],
                test_ratio=self.config['dataset']['test_ratio']
            )
            
            self.stats['stages_completed'].append('dataset_split')
            
            # 6. Создание YAML конфигурации
            self.logger.info("⚙️ Этап 5: Создание конфигурации датасета")
            create_dataset_yaml(
                dataset_dir=self.dataset_dir,
                class_names=self.config['dataset']['class_names']
            )
            
            self.stats['stages_completed'].append('yaml_creation')
            
            # 7. Очистка временных файлов
            self.logger.info("🧹 Этап 6: Очистка временных файлов")
            self._cleanup_temp_files()
            
            # 8. Генерация отчета
            self._generate_completion_report()
            
            self.logger.info("🎉 Пайплайн подготовки данных завершен успешно!")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в пайплайне: {e}")
            self._generate_error_report(e)
            raise
    
    def _cleanup_temp_files(self):
        """Очистка временных файлов"""
        temp_dirs = [
            Path("data/temp/frames"),
            Path("data/temp/labels"),
            Path("data/temp")
        ]
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                self.logger.info(f"🗑️ Удалена временная директория: {temp_dir}")
    
    def _generate_completion_report(self):
        """Генерация отчета о завершении"""
        total_time = time.time() - self.stats['start_time']
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pipeline_status': 'completed',
            'total_execution_time_seconds': round(total_time, 2),
            'statistics': {
                'total_frames_processed': self.stats['total_frames'],
                'total_annotations_created': self.stats['total_annotations'],
                'stages_completed': self.stats['stages_completed']
            },
            'output_structure': {
                'dataset_directory': str(self.dataset_dir),
                'train_images': len(list((self.dataset_dir / 'train' / 'images').glob('*'))),
                'val_images': len(list((self.dataset_dir / 'val' / 'images').glob('*'))),
                'test_images': len(list((self.dataset_dir / 'test' / 'images').glob('*')))
            },
            'next_steps': [
                "Запустите train_model.py для обучения модели",
                "Проверьте dataset.yaml в data/processed/dataset/",
                "При необходимости отредактируйте аннотации вручную"
            ]
        }
        
        # Сохранение отчета
        report_path = Path("data/processed/preparation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📋 Отчет сохранен: {report_path}")
        self.logger.info(f"⏱️ Время выполнения: {total_time:.2f} секунд")
        self.logger.info(f"📊 Обработано {self.stats['total_frames']} кадров")
    
    def _generate_error_report(self, error: Exception):
        """Генерация отчета об ошибке"""
        total_time = time.time() - self.stats['start_time']
        
        error_report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pipeline_status': 'failed',
            'execution_time_seconds': round(total_time, 2),
            'error': {
                'type': type(error).__name__,
                'message': str(error),
                'stages_completed': self.stats['stages_completed']
            },
            'troubleshooting': [
                "Проверьте наличие видеофайлов в data/raw/",
                "Убедитесь в корректности конфигурации",
                "Проверьте логи в файле logs/",
                "Убедитесь в наличии необходимых зависимостей"
            ]
        }
        
        # Сохранение отчета об ошибке
        error_report_path = Path("data/processed/error_report.json")
        with open(error_report_path, 'w', encoding='utf-8') as f:
            json.dump(error_report, f, ensure_ascii=False, indent=2)
        
        self.logger.error(f"📋 Отчет об ошибке сохранен: {error_report_path}")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Скрипт подготовки данных для обучения YOLO11",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

1. Базовое использование:
   python scripts/prepare_data.py --input "data/raw"

2. С кастомной конфигурацией:
   python scripts/prepare_data.py --input "data/raw" --config "config/my_config.json"

3. С изменением параметров извлечения:
   python scripts/prepare_data.py --input "data/raw" --fps 1.5

Структура после выполнения:
data/processed/dataset/
├── train/images & labels/
├── val/images & labels/
├── test/images & labels/
└── dataset.yaml
        """
    )
    
    parser.add_argument(
        '--input', 
        type=str, 
        default="data/raw",
        help='Путь к директории с входными видео файлами (по умолчанию: data/raw)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Путь к файлу конфигурации JSON (опционально)'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=2.0,
        help='Частота извлечения кадров (кадров/секунду, по умолчанию: 2.0)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default="data/processed/dataset",
        help='Путь к выходной директории датасета'
    )
    
    args = parser.parse_args()
    
    try:
        # Загрузка конфигурации
        config_path = Path(args.config) if args.config else None
        config = load_config(config_path)
        
        # Переопределение FPS если задано
        if args.fps != 2.0:
            config['video_processing']['fps_extraction'] = args.fps
        
        # Создание и запуск процессора
        processor = DataPipelineProcessor(config)
        processor.dataset_dir = Path(args.output)
        
        # Запуск пайплайна
        processor.run_pipeline(Path(args.input))
        
        print("\n" + "="*50)
        print("🎉 ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА УСПЕШНО!")
        print("="*50)
        print(f"📁 Датасет создан в: {args.output}")
        print(f"⚙️ Конфигурация: {args.output}/dataset.yaml")
        print(f"📋 Отчет: data/processed/preparation_report.json")
        print("\n🚀 Следующий шаг: запустите train_model.py")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        print("📋 Проверьте error_report.json для деталей")
        sys.exit(1)


if __name__ == "__main__":
    main()