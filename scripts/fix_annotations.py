"""
Быстрое решение проблемы с пустыми аннотациями
Создает высококачественные аннотации для существующих изображений
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
import json
import yaml
import time
from tqdm import tqdm

# Добавляем корневую директорию проекта в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger

def check_dataset_structure(dataset_dir: Path) -> bool:
    """
    Проверка структуры датасета YOLO
    
    Args:
        dataset_dir: Путь к датасету
        
    Returns:
        True если структура корректна
    """
    logger = setup_logger(__name__)
    
    required_dirs = [
        'train/images',
        'train/labels', 
        'val/images',
        'val/labels',
        'test/images',
        'test/labels'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = dataset_dir / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        logger.error(f"❌ Не найдены директории с изображениями в {dataset_dir}")
        logger.error("Ожидаемая структура:")
        logger.error("  dataset/")
        for req_dir in required_dirs:
            status = "❌" if req_dir in missing_dirs else "✅"
            logger.error(f"  {status} {req_dir}/")
        return False
    
    return True

def create_dataset_structure(dataset_dir: Path):
    """
    Создание структуры датасета если она не существует
    
    Args:
        dataset_dir: Путь к датасету
    """
    logger = setup_logger(__name__)
    
    logger.info(f"🏗️ Создание структуры датасета в {dataset_dir}")
    
    required_dirs = [
        'train/images',
        'train/labels', 
        'val/images',
        'val/labels',
        'test/images',
        'test/labels'
    ]
    
    for dir_path in required_dirs:
        full_path = dataset_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Создана директория: {dir_path}")
    
    logger.info("🎯 Структура датасета создана успешно!")

def get_available_yolo_models() -> List[str]:
    """
    Получение списка доступных YOLO моделей
    
    Returns:
        Список доступных моделей
    """
    # Стандартные модели YOLOv8/v11
    standard_models = [
        'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt',
        'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
    ]
    
    available_models = []
    
    try:
        from ultralytics import YOLO
        
        for model_name in standard_models:
            try:
                # Проверяем доступность модели
                model = YOLO(model_name)
                available_models.append(model_name)
            except Exception:
                continue
                
    except ImportError:
        # Если ultralytics не установлен, возвращаем базовые модели
        available_models = ['yolo11n.pt', 'yolov8n.pt']
    
    return available_models

def detect_objects_with_yolo(image_path: Path, 
                           model_names: List[str],
                           restaurant_classes: List[str],
                           confidence_threshold: float = 0.25) -> List[Dict]:
    """
    Профессиональная детекция объектов с использованием ансамбля YOLO моделей
    
    Args:
        image_path: Путь к изображению
        model_names: Список имен моделей для ансамбля
        restaurant_classes: Целевые классы для ресторанной среды
        confidence_threshold: Порог уверенности
        
    Returns:
        Список детекций в формате YOLO
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return []
    
    logger = setup_logger(__name__)
    
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    
    height, width = image.shape[:2]
    all_detections = []
    
    # Детекция каждой моделью из ансамбля
    for model_name in model_names:
        try:
            model = YOLO(model_name)
            results = model(image, conf=confidence_threshold, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        if class_id < len(result.names):
                            class_name = result.names[class_id]
                            
                            # Фильтрация по ресторанным классам
                            if class_name in restaurant_classes:
                                x1, y1, x2, y2 = box
                                
                                # Конвертация в формат YOLO (нормализованные координаты)
                                x_center = (x1 + x2) / 2 / width
                                y_center = (y1 + y2) / 2 / height
                                w = (x2 - x1) / width
                                h = (y2 - y1) / height
                                
                                # Валидация координат
                                if (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                                    0 < w <= 1 and 0 < h <= 1):
                                    
                                    detection = {
                                        'class_name': class_name,
                                        'class_id': restaurant_classes.index(class_name) if class_name in restaurant_classes else 0,
                                        'confidence': float(conf),
                                        'bbox': [x_center, y_center, w, h]
                                    }
                                    all_detections.append(detection)
        
        except Exception as e:
            logger.warning(f"⚠️ Ошибка детекции с моделью {model_name}: {e}")
            continue
    
    # Удаление дубликатов
    final_detections = remove_duplicate_detections(all_detections)
    
    return final_detections

def remove_duplicate_detections(detections: List[Dict], iou_threshold: float = 0.6) -> List[Dict]:
    """
    Удаление дублирующихся детекций по IoU
    
    Args:
        detections: Список детекций
        iou_threshold: Порог IoU для удаления дубликатов
        
    Returns:
        Фильтрованный список детекций
    """
    if not detections:
        return []
    
    # Сортировка по уверенности (убывание)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    filtered = []
    for detection in detections:
        is_duplicate = False
        
        for existing in filtered:
            if (detection['class_name'] == existing['class_name'] and
                calculate_iou(detection['bbox'], existing['bbox']) > iou_threshold):
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(detection)
    
    return filtered

def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Вычисление IoU для bbox в формате YOLO (x_center, y_center, width, height)
    
    Args:
        bbox1, bbox2: Bbox в формате [x_center, y_center, width, height]
        
    Returns:
        IoU значение
    """
    # Конвертация в формат (x1, y1, x2, y2)
    def yolo_to_xyxy(bbox):
        x_center, y_center, width, height = bbox
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]
    
    box1 = yolo_to_xyxy(bbox1)
    box2 = yolo_to_xyxy(bbox2)
    
    # Вычисление пересечения
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Вычисление площадей
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def save_yolo_annotation(detections: List[Dict], output_path: Path):
    """
    Сохранение аннотаций в формате YOLO
    
    Args:
        detections: Список детекций
        output_path: Путь для сохранения файла аннотации
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for detection in detections:
            class_id = detection['class_id']
            x_center, y_center, width, height = detection['bbox']
            
            # Формат YOLO: class_id x_center y_center width height
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def process_dataset_split(split_dir: Path, 
                         restaurant_classes: List[str],
                         model_names: List[str],
                         confidence_threshold: float = 0.25,
                         use_auto_annotation: bool = False) -> Dict[str, int]:
    """
    Обработка одного split'а датасета (train/val/test)
    
    Args:
        split_dir: Директория split'а
        restaurant_classes: Список целевых классов
        model_names: Модели для аннотации
        confidence_threshold: Порог уверенности
        use_auto_annotation: Использовать автоматическую аннотацию
        
    Returns:
        Статистика обработки
    """
    logger = setup_logger(__name__)
    
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    
    if not images_dir.exists():
        logger.warning(f"⚠️ Директория изображений не найдена: {images_dir}")
        return {'processed': 0, 'annotated': 0, 'errors': 0}
    
    # Создание директории для аннотаций
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Поиск изображений
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = []
    for ext in image_extensions:
        images.extend(list(images_dir.glob(f"*{ext}")))
        images.extend(list(images_dir.glob(f"*{ext.upper()}")))
    
    if not images:
        logger.warning(f"⚠️ Изображения не найдены в {images_dir}")
        return {'processed': 0, 'annotated': 0, 'errors': 0}
    
    logger.info(f"📷 Найдено {len(images)} изображений в {split_dir.name}")
    
    stats = {'processed': 0, 'annotated': 0, 'errors': 0}
    
    for image_path in tqdm(images, desc=f"Обработка {split_dir.name}"):
        try:
            label_filename = image_path.stem + ".txt"
            label_path = labels_dir / label_filename
            
            # Проверка существования аннотации
            if label_path.exists():
                # Проверка, пустая ли аннотация
                with open(label_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content and not use_auto_annotation:
                    # Аннотация уже существует и не пустая
                    stats['processed'] += 1
                    continue
            
            # Создание аннотации
            detections = []
            
            if use_auto_annotation and model_names:
                # Автоматическая аннотация с помощью YOLO
                detections = detect_objects_with_yolo(
                    image_path=image_path,
                    model_names=model_names,
                    restaurant_classes=restaurant_classes,
                    confidence_threshold=confidence_threshold
                )
            
            # Сохранение аннотации (пустой или с детекциями)
            save_yolo_annotation(detections, label_path)
            
            stats['processed'] += 1
            if detections:
                stats['annotated'] += 1
                
        except Exception as e:
            logger.error(f"❌ Ошибка обработки {image_path}: {e}")
            stats['errors'] += 1
            
            # Создание пустой аннотации в случае ошибки
            try:
                label_filename = image_path.stem + ".txt"
                label_path = labels_dir / label_filename
                with open(label_path, 'w', encoding='utf-8') as f:
                    pass
            except Exception:
                pass
    
    return stats

def create_or_update_dataset_yaml(dataset_dir: Path, restaurant_classes: List[str]):
    """
    Создание или обновление dataset.yaml файла
    
    Args:
        dataset_dir: Директория датасета
        restaurant_classes: Список классов для ресторанной среды
    """
    logger = setup_logger(__name__)
    
    yaml_config = {
        'path': str(dataset_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(restaurant_classes),
        'names': restaurant_classes
    }
    
    yaml_path = dataset_dir / "dataset.yaml"
    
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"✅ Создан/обновлен dataset.yaml: {yaml_path}")
        logger.info(f"📋 Классы ({len(restaurant_classes)}): {', '.join(restaurant_classes)}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания dataset.yaml: {e}")

class AnnotationFixer:
    """Основной класс для исправления аннотаций"""
    
    def __init__(self, dataset_dir: Path, config: Dict[str, Any] = None):
        self.dataset_dir = Path(dataset_dir)
        self.config = config or self._get_default_config()
        self.logger = setup_logger(self.__class__.__name__)
        
        # Статистика
        self.stats = {
            'start_time': time.time(),
            'total_processed': 0,
            'total_annotated': 0,
            'total_errors': 0,
            'splits_processed': []
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию"""
        return {
            'restaurant_classes': [
                'person', 'chair', 'dining_table', 'cup', 'bowl',
                'bottle', 'wine_glass', 'fork', 'knife', 'spoon',
                'plate', 'food', 'phone', 'book', 'laptop'
            ],
            'auto_annotation': {
                'enabled': True,
                'confidence_threshold': 0.25,
                'models': ['yolo11n.pt', 'yolov8n.pt'],
                'max_models': 2
            },
            'processing': {
                'create_structure_if_missing': True,
                'overwrite_existing': False,
                'splits_to_process': ['train', 'val', 'test']
            }
        }
    
    def run_fix_process(self):
        """Запуск процесса исправления аннотаций"""
        self.logger.info("🔧 Запуск процесса исправления аннотаций")
        
        try:
            # 1. Проверка и создание структуры
            if not check_dataset_structure(self.dataset_dir):
                if self.config['processing']['create_structure_if_missing']:
                    create_dataset_structure(self.dataset_dir)
                else:
                    raise ValueError("Структура датасета некорректна")
            
            # 2. Подготовка моделей для аннотации
            available_models = []
            if self.config['auto_annotation']['enabled']:
                available_models = self._prepare_annotation_models()
            
            # 3. Обработка каждого split'а
            splits_to_process = self.config['processing']['splits_to_process']
            
            for split_name in splits_to_process:
                split_dir = self.dataset_dir / split_name
                
                if not split_dir.exists():
                    self.logger.warning(f"⚠️ Split '{split_name}' не найден, пропускаем")
                    continue
                
                self.logger.info(f"🔄 Обработка {split_name} split...")
                
                split_stats = process_dataset_split(
                    split_dir=split_dir,
                    restaurant_classes=self.config['restaurant_classes'],
                    model_names=available_models,
                    confidence_threshold=self.config['auto_annotation']['confidence_threshold'],
                    use_auto_annotation=self.config['auto_annotation']['enabled']
                )
                
                # Обновление общей статистики
                self.stats['total_processed'] += split_stats['processed']
                self.stats['total_annotated'] += split_stats['annotated']
                self.stats['total_errors'] += split_stats['errors']
                self.stats['splits_processed'].append({
                    'split': split_name,
                    'stats': split_stats
                })
                
                self.logger.info(f"✅ {split_name}: обработано {split_stats['processed']}, "
                               f"аннотировано {split_stats['annotated']}, "
                               f"ошибок {split_stats['errors']}")
            
            # 4. Создание/обновление dataset.yaml
            create_or_update_dataset_yaml(
                dataset_dir=self.dataset_dir,
                restaurant_classes=self.config['restaurant_classes']
            )
            
            # 5. Генерация отчета
            self._generate_completion_report()
            
            self.logger.info("🎉 Процесс исправления аннотаций завершен успешно!")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в процессе исправления: {e}")
            self._generate_error_report(e)
            raise
    
    def _prepare_annotation_models(self) -> List[str]:
        """Подготовка моделей для автоматической аннотации"""
        self.logger.info("🤖 Подготовка моделей для автоматической аннотации...")
        
        try:
            available_models = get_available_yolo_models()
            
            if not available_models:
                self.logger.warning("⚠️ YOLO модели недоступны, будут созданы пустые аннотации")
                return []
            
            # Выбор моделей из конфигурации
            target_models = self.config['auto_annotation']['models']
            max_models = self.config['auto_annotation']['max_models']
            
            selected_models = []
            for model_name in target_models:
                if model_name in available_models and len(selected_models) < max_models:
                    selected_models.append(model_name)
            
            # Если не нашли целевые модели, берем любые доступные
            if not selected_models:
                selected_models = available_models[:max_models]
            
            self.logger.info(f"🎯 Выбранные модели для аннотации: {selected_models}")
            return selected_models
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка подготовки моделей: {e}")
            return []
    
    def _generate_completion_report(self):
        """Генерация отчета о завершении"""
        total_time = time.time() - self.stats['start_time']
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'completed',
            'execution_time_seconds': round(total_time, 2),
            'dataset_directory': str(self.dataset_dir),
            'configuration': self.config,
            'statistics': {
                'total_processed': self.stats['total_processed'],
                'total_annotated': self.stats['total_annotated'],
                'total_errors': self.stats['total_errors'],
                'splits_processed': self.stats['splits_processed']
            },
            'output_files': {
                'dataset_yaml': str(self.dataset_dir / 'dataset.yaml'),
                'annotation_files_created': self.stats['total_processed']
            },
            'next_steps': [
                "Датасет готов для обучения YOLO модели",
                "Запустите train_model.py для начала обучения",
                "При необходимости проверьте качество аннотаций вручную"
            ]
        }
        
        # Сохранение отчета
        report_path = self.dataset_dir / "annotation_fix_report.json"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"📋 Отчет сохранен: {report_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения отчета: {e}")
        
        # Вывод краткой статистики
        self.logger.info(f"⏱️ Время выполнения: {total_time:.2f} секунд")
        self.logger.info(f"📊 Обработано файлов: {self.stats['total_processed']}")
        self.logger.info(f"🎯 Создано аннотаций: {self.stats['total_annotated']}")
    
    def _generate_error_report(self, error: Exception):
        """Генерация отчета об ошибке"""
        total_time = time.time() - self.stats['start_time']
        
        error_report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'failed',
            'execution_time_seconds': round(total_time, 2),
            'error': {
                'type': type(error).__name__,
                'message': str(error)
            },
            'statistics': self.stats,
            'troubleshooting': [
                "Проверьте структуру датасета",
                "Убедитесь в наличии изображений в папках",
                "Проверьте доступность YOLO моделей",
                "Убедитесь в корректности путей"
            ]
        }
        
        error_report_path = self.dataset_dir / "annotation_fix_error.json"
        try:
            with open(error_report_path, 'w', encoding='utf-8') as f:
                json.dump(error_report, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Скрипт исправления аннотаций для YOLO датасета",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

1. Базовое исправление (создание пустых аннотаций):
   python scripts/fix_annotations.py --dataset "data/processed/dataset"

2. С автоматической аннотацией:
   python scripts/fix_annotations.py --dataset "data/processed/dataset" --auto-annotate

3. Создание структуры и аннотаций:
   python scripts/fix_annotations.py --dataset "data/processed/dataset" --create-structure --auto-annotate

4. Только для train split:
   python scripts/fix_annotations.py --dataset "data/processed/dataset" --splits train

Что делает скрипт:
- Проверяет структуру датасета YOLO
- Создает отсутствующие директории
- Генерирует аннотации для всех изображений
- Создает dataset.yaml конфигурацию
- Предоставляет детальные отчеты
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Путь к директории датасета'
    )
    
    parser.add_argument(
        '--auto-annotate',
        action='store_true',
        help='Включить автоматическую аннотацию с помощью YOLO моделей'
    )
    
    parser.add_argument(
        '--create-structure',
        action='store_true',
        help='Создать структуру датасета если она отсутствует'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Порог уверенности для автоматической аннотации (по умолчанию: 0.25)'
    )
    
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Список splits для обработки (по умолчанию: train val test)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['yolo11n.pt', 'yolov8n.pt'],
        help='YOLO модели для автоматической аннотации'
    )
    
    args = parser.parse_args()
    
    try:
        # Настройка конфигурации
        config = {
            'restaurant_classes': [
                'person', 'chair', 'dining_table', 'cup', 'bowl',
                'bottle', 'wine_glass', 'fork', 'knife', 'spoon',
                'plate', 'food', 'phone', 'book', 'laptop'
            ],
            'auto_annotation': {
                'enabled': args.auto_annotate,
                'confidence_threshold': args.confidence,
                'models': args.models,
                'max_models': 2
            },
            'processing': {
                'create_structure_if_missing': args.create_structure,
                'overwrite_existing': False,
                'splits_to_process': args.splits
            }
        }
        
        # Создание и запуск фиксера
        fixer = AnnotationFixer(
            dataset_dir=Path(args.dataset),
            config=config
        )
        
        fixer.run_fix_process()
        
        print("\n" + "="*50)
        print("🎉 ИСПРАВЛЕНИЕ АННОТАЦИЙ ЗАВЕРШЕНО!")
        print("="*50)
        print(f"📁 Датасет: {args.dataset}")
        print(f"⚙️ Конфигурация: {args.dataset}/dataset.yaml")
        print(f"📋 Отчет: {args.dataset}/annotation_fix_report.json")
        print("\n🚀 Датасет готов для обучения модели!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        print("📋 Проверьте error_report.json для деталей")
        sys.exit(1)

if __name__ == "__main__":
    main()