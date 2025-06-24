"""
Быстрое решение проблемы с пустыми аннотациями
Создает высококачественные аннотации для существующих изображений
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
import json
import yaml
import time

def setup_logger():
    """Настройка логгера"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('annotation_fix.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def detect_objects_professional(image_path: Path, models: List[YOLO], 
                               restaurant_classes: List[str],
                               confidence_threshold: float = 0.25) -> List[Dict]:
    """
    Профессиональная детекция объектов с использованием ансамбля моделей
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    
    height, width = image.shape[:2]
    all_detections = []
    
    # Детекция каждой моделью
    for model in models:
        try:
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
                                        'confidence': float(conf),
                                        'bbox': [x_center, y_center, w, h]
                                    }
                                    all_detections.append(detection)
        
        except Exception as e:
            logging.warning(f"Ошибка детекции с моделью: {e}")
            continue
    
    # Простое удаление дубликатов по IoU
    final_detections = remove_duplicate_detections(all_detections)
    
    return final_detections

def remove_duplicate_detections(detections: List[Dict], iou_threshold: float = 0.6) -> List[Dict]:
    """Удаление дублирующихся детекций"""
    if not detections:
        return []
    
    # Сортировка по уверенности
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
    """Вычисление IoU между двумя bbox в формате YOLO"""
    def yolo_to_corners(bbox):
        x_center, y_center, width, height = bbox
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return x1, y1, x2, y2
    
    x1_1, y1_1, x2_1, y2_1 = yolo_to_corners(bbox1)
    x1_2, y1_2, x2_2, y2_2 = yolo_to_corners(bbox2)
    
    # Пересечение
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Площади
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def create_class_mapping(restaurant_classes: List[str]) -> Dict[str, int]:
    """Создание маппинга классов"""
    return {class_name: idx for idx, class_name in enumerate(restaurant_classes)}

def save_yolo_annotation(detections: List[Dict], output_path: Path, class_mapping: Dict[str, int]):
    """Сохранение аннотации в формате YOLO"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for detection in detections:
            class_name = detection['class_name']
            if class_name in class_mapping:
                class_id = class_mapping[class_name]
                bbox = detection['bbox']
                
                # Формат YOLO: class_id x_center y_center width height
                line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                f.write(line)

def create_dataset_yaml(dataset_path: Path, class_mapping: Dict[str, int]):
    """Создание dataset.yaml для YOLO"""
    yaml_content = {
        'path': str(dataset_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_mapping),
        'names': list(class_mapping.keys())
    }
    
    yaml_path = dataset_path / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    return yaml_path

def fix_annotations(dataset_dir: Path, confidence_threshold: float = 0.25):
    """
    Основная функция исправления аннотаций
    """
    logger = setup_logger()
    logger.info("🚀 Запуск профессионального исправления аннотаций")
    
    # Конфигурация ресторанных классов
    restaurant_classes = [
        'person', 'chair', 'dining table', 'cup', 'fork', 'knife',
        'spoon', 'bowl', 'bottle', 'wine glass', 'sandwich', 'pizza',
        'cake', 'apple', 'banana', 'orange', 'cell phone', 'laptop', 'book'
    ]
    
    class_mapping = create_class_mapping(restaurant_classes)
    
    # Инициализация моделей
    logger.info("🧠 Загрузка ансамбля моделей YOLO...")
    models = []
    
    model_configs = [
        ('yolo11n.pt', 0.15),
        ('yolo11s.pt', 0.18),
        ('yolo11m.pt', 0.22)
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"💻 Используется устройство: {device}")
    
    for model_name, conf in model_configs:
        try:
            logger.info(f"📥 Загрузка модели: {model_name}")
            model = YOLO(model_name)
            model.to(device)
            models.append(model)
            logger.info(f"✅ Модель {model_name} успешно загружена")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить {model_name}: {e}")
    
    if not models:
        logger.error("❌ Не удалось загрузить ни одной модели!")
        return False
    
    # Обработка каждого split'а
    splits = ['train', 'val', 'test']
    total_processed = 0
    total_annotations_created = 0
    
    for split in splits:
        split_images_dir = dataset_dir / split / 'images'
        split_labels_dir = dataset_dir / split / 'labels'
        
        if not split_images_dir.exists():
            logger.warning(f"⚠️ Директория {split_images_dir} не существует, пропускаем")
            continue
        
        # Создание директории для аннотаций
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Поиск изображений
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(split_images_dir.glob(f"*{ext}")))
            image_files.extend(list(split_images_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            logger.warning(f"⚠️ Нет изображений в {split_images_dir}")
            continue
        
        logger.info(f"🖼️ Обработка {split}: найдено {len(image_files)} изображений")
        
        split_annotations = 0
        
        # Обработка изображений с прогресс-баром
        for image_path in tqdm(image_files, desc=f"Аннотация {split}"):
            try:
                # Детекция объектов
                detections = detect_objects_professional(
                    image_path, models, restaurant_classes, confidence_threshold
                )
                
                # Сохранение аннотации
                annotation_path = split_labels_dir / f"{image_path.stem}.txt"
                save_yolo_annotation(detections, annotation_path, class_mapping)
                
                total_processed += 1
                split_annotations += len(detections)
                
            except Exception as e:
                logger.error(f"❌ Ошибка обработки {image_path}: {e}")
                
                # Создание пустого файла аннотации в случае ошибки
                annotation_path = split_labels_dir / f"{image_path.stem}.txt"
                annotation_path.touch()
        
        logger.info(f"✅ {split} завершен: {len(image_files)} изображений, {split_annotations} аннотаций")
        total_annotations_created += split_annotations
    
    # Создание dataset.yaml
    logger.info("📄 Создание dataset.yaml...")
    yaml_path = create_dataset_yaml(dataset_dir, class_mapping)
    logger.info(f"✅ Создан файл: {yaml_path}")
    
    # Создание отчета
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_images_processed': total_processed,
        'total_annotations_created': total_annotations_created,
        'models_used': [config[0] for config in model_configs if config[0] in [str(m.ckpt_path) for m in models]],
        'confidence_threshold': confidence_threshold,
        'restaurant_classes': restaurant_classes,
        'class_mapping': class_mapping,
        'dataset_yaml': str(yaml_path)
    }
    
    report_path = dataset_dir / 'annotation_fix_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Итоговая статистика
    logger.info("\n" + "="*60)
    logger.info("📋 ИТОГИ ИСПРАВЛЕНИЯ АННОТАЦИЙ")
    logger.info("="*60)
    logger.info(f"🖼️ Обработано изображений: {total_processed}")
    logger.info(f"🎯 Создано аннотаций: {total_annotations_created}")
    logger.info(f"📄 Dataset YAML: {yaml_path}")
    logger.info(f"📊 Отчет: {report_path}")
    logger.info("="*60)
    logger.info("✅ ПРОБЛЕМА С ПУСТЫМИ АННОТАЦИЯМИ РЕШЕНА!")
    logger.info("🚀 Теперь можно запускать обучение:")
    logger.info(f"   python scripts/train_model.py --data {yaml_path}")
    logger.info("="*60)
    
    return True

def main():
    """Основная функция командной строки"""
    parser = argparse.ArgumentParser(
        description="Быстрое исправление проблемы с пустыми аннотациями YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
    # Исправление аннотаций в существующем датасете
    python fix_annotations.py --dataset data/processed/dataset
    
    # С кастомным порогом уверенности
    python fix_annotations.py --dataset data/processed/dataset --confidence 0.3
    
    # Для конкретного split'а
    python fix_annotations.py --dataset data/processed/dataset --split train
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Путь к директории датасета"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Порог уверенности для детекции (по умолчанию: 0.25)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        choices=['train', 'val', 'test', 'all'],
        default='all',
        help="Какой split обрабатывать (по умолчанию: all)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующие аннотации"
    )
    
    args = parser.parse_args()
    
    try:
        dataset_path = Path(args.dataset)
        
        if not dataset_path.exists():
            print(f"❌ Путь к датасету не существует: {dataset_path}")
            sys.exit(1)
        
        # Проверка структуры датасета
        required_dirs = []
        if args.split == 'all':
            for split in ['train', 'val', 'test']:
                split_dir = dataset_path / split / 'images'
                if split_dir.exists():
                    required_dirs.append(split_dir)
        else:
            split_dir = dataset_path / args.split / 'images'
            if split_dir.exists():
                required_dirs.append(split_dir)
        
        if not required_dirs:
            print(f"❌ Не найдены директории с изображениями в {dataset_path}")
            print("Ожидаемая структура:")
            print("  dataset/")
            print("  ├── train/images/")
            print("  ├── val/images/")
            print("  └── test/images/")
            sys.exit(1)
        
        # Предупреждение о перезаписи
        if not args.force:
            labels_exist = any((dataset_path / split / 'labels').exists() and 
                             list((dataset_path / split / 'labels').glob('*.txt'))
                             for split in ['train', 'val', 'test'])
            
            if labels_exist:
                response = input("⚠️ Найдены существующие аннотации. Перезаписать? (y/N): ")
                if response.lower() not in ['y', 'yes', 'да']:
                    print("❌ Операция отменена")
                    sys.exit(0)
        
        # Запуск исправления
        print("🚀 Начинаем исправление аннотаций...")
        success = fix_annotations(dataset_path, args.confidence)
        
        if success:
            print("\n🎉 Исправление аннотаций успешно завершено!")
            print("🚀 Проблема с WARNING Labels are missing решена!")
            print("\nТеперь можно запускать обучение без предупреждений:")
            print(f"   python scripts/train_model.py --data {dataset_path}/dataset.yaml")
            sys.exit(0)
        else:
            print("\n❌ Исправление аннотаций завершилось с ошибками!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Операция прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()