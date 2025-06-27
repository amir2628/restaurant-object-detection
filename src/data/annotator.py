"""
Профессиональная система автоматической аннотации для YOLOv11
Использует GroundingDINO для создания высококачественных аннотаций
"""

import logging
import json
import cv2
import numpy as np
import os
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import albumentations as A
from PIL import Image, ImageDraw, ImageFont

# Импорт GroundingDINO-py
try:
    from groundingdino.models import build_groundingdino
    from groundingdino.util.inference import load_model, predict, load_image
    USE_GROUNDINGDINO_PY = True
    print("✓ Используется groundingdino-py")
except ImportError:
    print("✗ Не удалось импортировать groundingdino-py")
    print("Убедитесь, что он установлен: pip install groundingdino-py")
    USE_GROUNDINGDINO_PY = False


@dataclass
class DetectionResult:
    """Результат детекции объекта"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x_center, y_center, width, height] нормализованные
    original_bbox: List[int]  # [x1, y1, x2, y2] в пикселях


class SmartAnnotator:
    """
    Умная система автоматической аннотации с использованием GroundingDINO
    для создания точных аннотаций объектов в ресторанной среде
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        
        # Инициализация GroundingDINO
        self.groundingdino_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_groundingdino()
        
        # Система валидации
        self.validator = AnnotationValidator()
        
        # Трекер для статистики
        self.stats = {
            'total_images': 0,
            'total_detections': 0,
            'filtered_detections': 0,
            'class_distribution': {},
            'confidence_distribution': {},
            'processing_time': 0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger(f"{__name__}.SmartAnnotator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Загрузка конфигурации аннотатора"""
        default_config = {
            'groundingdino': {
                'checkpoint_path': 'groundingdino_swinb_cogcoor.pth',
                'config_paths': [
                    "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                    "groundingdino_config.py"
                ],
                'detection_threshold': 0.25,
                'prompt': "chicken . meat . salad . soup . cup . plate . bowl . spoon . fork . knife ."
            },
            'restaurant_classes': [
                'chicken',     # Курица
                'meat',        # Мясо  
                'salad',       # Салат
                'soup',        # Суп
                'cup',         # Чашка
                'plate',       # Тарелка
                'bowl',        # Миска
                'spoon',       # Ложка
                'fork',        # Вилка
                'knife'        # Нож
            ],
            'processing': {
                'batch_size': 8,
                'num_workers': 4,
                'confidence_threshold': 0.25,
                'nms_threshold': 0.6,
                'enable_tta': False
            },
            'validation': {
                'min_bbox_size': 0.01,
                'max_bbox_size': 0.9,
                'min_confidence': 0.15,
                'aspect_ratio_range': [0.1, 10.0]
            },
            'output': {
                'save_annotated_images': True,
                'create_visualization': False
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # Обновление конфигурации
                def update_dict(base, update):
                    for key, value in update.items():
                        if isinstance(value, dict) and key in base:
                            update_dict(base[key], value)
                        else:
                            base[key] = value
                
                update_dict(default_config, user_config)
                self.logger.info(f"Загружена конфигурация из: {config_path}")
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки конфигурации: {e}")
        
        return default_config
    
    def _init_groundingdino(self):
        """Инициализация модели GroundingDINO"""
        if not USE_GROUNDINGDINO_PY:
            self.logger.error("GroundingDINO-py не доступен!")
            raise ImportError("Требуется установка groundingdino-py")
        
        self.logger.info("Инициализация GroundingDINO...")
        
        try:
            checkpoint_path = self.config['groundingdino']['checkpoint_path']
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Файл модели не найден: {checkpoint_path}")
            
            # Попытка загрузки с конфигурацией
            config_path = None
            for config in self.config['groundingdino']['config_paths']:
                if os.path.exists(config):
                    config_path = config
                    break
            
            if config_path:
                try:
                    self.groundingdino_model = load_model(config_path, checkpoint_path)
                    self.logger.info(f"✓ GroundingDINO загружен с конфигурацией: {config_path}")
                except Exception as config_error:
                    self.logger.warning(f"Ошибка загрузки с конфигом: {config_error}")
                    # Попытка использовать build_groundingdino
                    try:
                        self.groundingdino_model = build_groundingdino(checkpoint_path)
                        self.logger.info("✓ GroundingDINO загружен через build_groundingdino")
                    except Exception as build_error:
                        self.logger.error(f"Ошибка build_groundingdino: {build_error}")
                        raise
            else:
                self.logger.info("Конфигурационный файл не найден, попытка загрузки без конфига...")
                try:
                    # Попытка использовать build_groundingdino напрямую
                    self.groundingdino_model = build_groundingdino(checkpoint_path)
                    self.logger.info("✓ GroundingDINO загружен через build_groundingdino")
                except Exception as build_error:
                    self.logger.error(f"Ошибка build_groundingdino: {build_error}")
                    # Последняя попытка - может быть load_model работает с одним параметром
                    try:
                        # Некоторые версии могут работать только с путем к чекпоинту
                        import torch
                        self.groundingdino_model = torch.load(checkpoint_path, map_location=self.device)
                        self.logger.info("✓ GroundingDINO загружен через torch.load")
                    except Exception as torch_error:
                        self.logger.error(f"Ошибка torch.load: {torch_error}")
                        raise
            
            self.logger.info(f"✓ Используется устройство: {self.device}")
            
        except Exception as e:
            self.logger.error(f"✗ Ошибка загрузки GroundingDINO: {e}")
            self.logger.error("Попробуйте скачать конфигурационный файл или проверьте версию groundingdino-py")
            raise
    
    def annotate_dataset(self, images_dir: Path, output_dir: Path, 
                        batch_size: int = 8, num_workers: int = 4) -> Dict[str, Any]:
        """
        Аннотация датасета с использованием GroundingDINO
        
        Args:
            images_dir: Директория с изображениями
            output_dir: Директория для аннотаций
            batch_size: Размер батча
            num_workers: Количество потоков
            
        Returns:
            Статистика обработки
        """
        self.logger.info(f"Начало аннотации датасета: {images_dir}")
        
        # Создание выходной директории
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Поиск изображений
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            self.logger.warning("Изображения не найдены!")
            return {'processed_images': 0, 'total_detections': 0, 'success_rate': 0.0}
        
        self.logger.info(f"Найдено {len(image_files)} изображений")
        
        # Сброс статистики
        self.stats = {
            'total_images': 0,
            'total_detections': 0,
            'filtered_detections': 0,
            'class_distribution': {},
            'confidence_distribution': {},
            'processing_time': 0
        }
        
        # Обработка изображений
        processed_count = 0
        total_detections = 0
        
        for image_path in tqdm(image_files, desc="Аннотация изображений"):
            try:
                # Аннотация одного изображения
                detections = self._annotate_single_image(image_path)
                
                if detections:
                    # Сохранение аннотации в формате YOLO
                    annotation_path = output_dir / f"{image_path.stem}.txt"
                    self._save_yolo_annotation(detections, annotation_path, image_path)
                    
                    # Обновление статистики
                    processed_count += 1
                    total_detections += len(detections)
                    
                    for detection in detections:
                        class_name = detection.class_name
                        self.stats['class_distribution'][class_name] = \
                            self.stats['class_distribution'].get(class_name, 0) + 1
                else:
                    # Создание пустой аннотации
                    annotation_path = output_dir / f"{image_path.stem}.txt"
                    annotation_path.touch()
                
                self.stats['total_images'] += 1
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки {image_path}: {e}")
                continue
        
        # Сохранение статистики
        self._save_annotation_stats(output_dir)
        
        success_rate = processed_count / len(image_files) if image_files else 0
        
        result = {
            'processed_images': processed_count,
            'total_detections': total_detections,
            'success_rate': success_rate,
            'total_images': len(image_files)
        }
        
        self.logger.info(f"Аннотация завершена: {processed_count}/{len(image_files)} изображений")
        self.logger.info(f"Всего детекций: {total_detections}")
        self.logger.info(f"Успешность: {success_rate:.1%}")
        
        return result
    
    def _annotate_single_image(self, image_path: Path) -> List[DetectionResult]:
        """Аннотация одного изображения с использованием GroundingDINO"""
        
        try:
            # Загрузка изображения
            image_source, image = load_image(str(image_path))
            
            # Получение промпта
            prompt = self.config['groundingdino']['prompt']
            detection_threshold = self.config['groundingdino']['detection_threshold']
            
            # Выполнение детекции
            boxes, logits, phrases = predict(
                model=self.groundingdino_model,
                image=image,
                caption=prompt,
                box_threshold=detection_threshold,
                text_threshold=detection_threshold,
                device=self.device
            )
            
            # Обработка результатов
            detections = []
            
            if len(boxes) > 0:
                # Конвертация тензоров в списки
                if hasattr(boxes, 'cpu'):
                    boxes = boxes.cpu().numpy()
                if hasattr(logits, 'cpu'):
                    logits = logits.cpu().numpy()
                
                # Создание объектов детекции
                for i, (box, confidence, phrase) in enumerate(zip(boxes, logits, phrases)):
                    # Маппинг обнаруженной фразы на наши классы
                    mapped_class = self._map_to_food_classes(phrase)
                    
                    if mapped_class and confidence >= self.config['processing']['confidence_threshold']:
                        # Получение ID класса
                        class_id = self._get_class_mapping().get(mapped_class, 0)
                        
                        # Создание объекта детекции
                        detection = DetectionResult(
                            class_id=class_id,
                            class_name=mapped_class,
                            confidence=float(confidence),
                            bbox=box.tolist(),  # Уже в нормализованном формате XYXY центр
                            original_bbox=[]  # Заполним при необходимости
                        )
                        
                        detections.append(detection)
            
            # Фильтрация и валидация
            filtered_detections = self._filter_detections(detections)
            
            return filtered_detections
            
        except Exception as e:
            self.logger.error(f"Ошибка детекции на изображении {image_path}: {e}")
            return []
    
    def _map_to_food_classes(self, detected_label: str) -> Optional[str]:
        """Маппинг обнаруженных меток на наши классы еды"""
        label_lower = str(detected_label).lower().strip()
        
        # Простое маппинг на основе вхождения
        for class_name in self.config['restaurant_classes']:
            if class_name.lower() in label_lower:
                return class_name
        
        return None
    
    def _filter_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Фильтрация детекций по качеству"""
        filtered = []
        
        for detection in detections:
            # Проверка уверенности
            if detection.confidence < self.config['validation']['min_confidence']:
                continue
            
            # Проверка размера bbox
            bbox = detection.bbox
            if len(bbox) >= 4:
                width = bbox[2] if len(bbox) > 2 else 0
                height = bbox[3] if len(bbox) > 3 else 0
                
                bbox_area = width * height
                
                if (bbox_area < self.config['validation']['min_bbox_size'] or 
                    bbox_area > self.config['validation']['max_bbox_size']):
                    continue
                
                # Проверка соотношения сторон
                if height > 0:
                    aspect_ratio = width / height
                    min_ratio, max_ratio = self.config['validation']['aspect_ratio_range']
                    
                    if not (min_ratio <= aspect_ratio <= max_ratio):
                        continue
            
            filtered.append(detection)
        
        return filtered
    
    def _save_yolo_annotation(self, detections: List[DetectionResult], 
                             annotation_path: Path, image_path: Path):
        """Сохранение аннотации в формате YOLO"""
        
        # Получение размеров изображения
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
            img.close()
        except Exception as e:
            self.logger.error(f"Ошибка чтения размеров изображения {image_path}: {e}")
            return
        
        with open(annotation_path, 'w', encoding='utf-8') as f:
            for detection in detections:
                # Формат YOLO: class_id x_center y_center width height (нормализованные)
                # bbox от GroundingDINO уже в правильном формате
                bbox = detection.bbox
                
                if len(bbox) >= 4:
                    x_center, y_center, width, height = bbox[:4]
                    
                    # Убеждаемся, что координаты нормализованы
                    if x_center > 1 or y_center > 1 or width > 1 or height > 1:
                        # Нормализация, если это пиксельные координаты
                        x_center = x_center / img_width
                        y_center = y_center / img_height  
                        width = width / img_width
                        height = height / img_height
                    
                    # Запись в файл
                    line = f"{detection.class_id} {x_center:.6f} {y_center:.6f} " \
                           f"{width:.6f} {height:.6f}\n"
                    f.write(line)
    
    def _get_class_mapping(self) -> Dict[str, int]:
        """Получение маппинга классов для ресторанной сцены"""
        restaurant_classes = self.config['restaurant_classes']
        return {class_name: idx for idx, class_name in enumerate(restaurant_classes)}
    
    def _save_annotation_stats(self, output_dir: Path):
        """Сохранение статистики аннотации"""
        stats_path = output_dir / 'annotation_stats.json'
        
        # Добавление маппинга классов к статистике
        self.stats['class_mapping'] = self._get_class_mapping()
        self.stats['total_classes'] = len(self.config['restaurant_classes'])
        self.stats['detection_method'] = 'GroundingDINO'
        self.stats['prompt_used'] = self.config['groundingdino']['prompt']
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Статистика аннотации сохранена: {stats_path}")


class AnnotationValidator:
    """Валидатор качества аннотаций"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AnnotationValidator")
    
    def validate_annotation_file(self, annotation_path: Path, 
                                image_path: Optional[Path] = None) -> Dict[str, Any]:
        """Валидация файла аннотации"""
        validation_result = {
            'valid': True,
            'issues': [],
            'line_count': 0,
            'bbox_count': 0
        }
        
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            validation_result['line_count'] = len(lines)
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                if not line:  # Пустая строка
                    continue
                
                parts = line.split()
                
                if len(parts) != 5:
                    validation_result['valid'] = False
                    validation_result['issues'].append(
                        f"Строка {line_num}: ожидается 5 значений, получено {len(parts)}"
                    )
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Проверка диапазонов
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                            0 < width <= 1 and 0 < height <= 1):
                        validation_result['valid'] = False
                        validation_result['issues'].append(
                            f"Строка {line_num}: координаты вне допустимого диапазона"
                        )
                    
                    validation_result['bbox_count'] += 1
                    
                except ValueError:
                    validation_result['valid'] = False
                    validation_result['issues'].append(
                        f"Строка {line_num}: неверный формат чисел"
                    )
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['issues'].append(f"Ошибка чтения файла: {e}")
        
        return validation_result


def create_dataset_yaml(dataset_dir: Path, class_mapping: Dict[str, int]):
    """Создание файла dataset.yaml для YOLO"""
    yaml_content = f"""# Датасет для детекции объектов в ресторанной среде
# Создан автоматически с использованием GroundingDINO

path: {dataset_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Классы
nc: {len(class_mapping)}
names: {list(class_mapping.keys())}

# Дополнительная информация
description: "Профессиональный датасет для детекции объектов еды и посуды в ресторанной среде"
version: "2.0"
license: "Custom"
annotation_method: "GroundingDINO"
"""
    
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    return yaml_path


def main():
    """Основная функция для запуска аннотации"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Профессиональная автоматическая аннотация для YOLO11 с GroundingDINO")
    parser.add_argument("--images_dir", type=str, required=True, 
                       help="Директория с изображениями")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Директория для сохранения аннотаций")
    parser.add_argument("--config", type=str, default=None,
                       help="Путь к файлу конфигурации")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Размер батча")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Количество потоков")
    
    args = parser.parse_args()
    
    # Инициализация аннотатора
    config_path = Path(args.config) if args.config else None
    annotator = SmartAnnotator(config_path)
    
    # Запуск аннотации
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    stats = annotator.annotate_dataset(
        images_dir=images_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Создание dataset.yaml
    class_mapping = annotator._get_class_mapping()
    yaml_path = create_dataset_yaml(output_dir.parent, class_mapping)
    
    print(f"\n✅ Аннотация завершена!")
    print(f"📁 Аннотации сохранены в: {output_dir}")
    print(f"📄 Конфигурация YOLO: {yaml_path}")
    print(f"📊 Обработано изображений: {stats['processed_images']}")
    print(f"🎯 Всего детекций: {stats['total_detections']}")
    print(f"📈 Успешность: {stats['success_rate']:.1%}")


if __name__ == "__main__":
    main()