"""
Профессиональная система автоматической аннотации для YOLOv11
Использует предобученные модели для создания высококачественных аннотаций
"""

import logging
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from ultralytics import YOLO
import supervision as sv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import albumentations as A


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
    Умная система автоматической аннотации с использованием ансамбля моделей
    и продвинутых методов очистки данных
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        
        # Инициализация моделей
        self.models = {}
        self.ensemble_weights = {}
        self._init_models()
        
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
            'models': {
                'yolo11n': {'weight': 0.3, 'confidence': 0.15},
                'yolo11s': {'weight': 0.4, 'confidence': 0.2},
                'yolo11m': {'weight': 0.3, 'confidence': 0.25}
            },
            'ensemble': {
                'consensus_threshold': 0.6,  # Минимальное согласие моделей
                'confidence_boost': 0.1,     # Бонус за согласие
                'iou_threshold': 0.5
            },
            'filtering': {
                'min_confidence': 0.25,
                'min_area': 200,            # Минимальная площадь bbox
                'max_area_ratio': 0.9,      # Максимальное отношение к изображению
                'min_aspect_ratio': 0.1,    # Минимальное соотношение сторон
                'max_aspect_ratio': 10.0,   # Максимальное соотношение сторон
                'edge_threshold': 10        # Отступ от края изображения
            },
            'restaurant_classes': [
                'person', 'chair', 'dining table', 'cup', 'fork', 'knife',
                'spoon', 'bowl', 'bottle', 'wine glass', 'sandwich', 'pizza',
                'cake', 'apple', 'banana', 'orange', 'cell phone', 'laptop', 'book'
            ],
            'augmentation': {
                'enable_tta': True,  # Test Time Augmentation
                'tta_transforms': ['flip', 'rotate', 'scale']
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _init_models(self):
        """Инициализация ансамбля моделей"""
        self.logger.info("Инициализация ансамбля моделей YOLO...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Используется устройство: {device}")
        
        for model_name, model_config in self.config['models'].items():
            try:
                # Загрузка предобученной модели
                model_path = f"{model_name}.pt"
                self.logger.info(f"Загрузка модели: {model_path}")
                
                model = YOLO(model_path)
                model.to(device)
                
                self.models[model_name] = {
                    'model': model,
                    'confidence': model_config['confidence'],
                    'weight': model_config['weight']
                }
                
                self.logger.info(f"✅ Модель {model_name} успешно загружена")
                
            except Exception as e:
                self.logger.warning(f"❌ Не удалось загрузить модель {model_name}: {e}")
        
        if not self.models:
            # Если ни одна модель не загрузилась, используем базовую
            self.logger.info("Использование базовой модели YOLOv8n...")
            model = YOLO('yolov8n.pt')
            model.to(device)
            self.models['yolov8n'] = {
                'model': model,
                'confidence': 0.2,
                'weight': 1.0
            }
    
    def annotate_dataset(self, 
                        images_dir: Path, 
                        output_dir: Path,
                        batch_size: int = 8,
                        num_workers: int = 4) -> Dict[str, Any]:
        """
        Аннотация всего датасета с использованием ансамбля моделей
        
        Args:
            images_dir: Директория с изображениями
            output_dir: Директория для сохранения аннотаций
            batch_size: Размер батча для обработки
            num_workers: Количество потоков
            
        Returns:
            Статистика аннотации
        """
        self.logger.info(f"Начало аннотации датасета: {images_dir}")
        
        # Подготовка директорий
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Поиск изображений
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            self.logger.error(f"Не найдено изображений в {images_dir}")
            return self.stats
        
        self.logger.info(f"Найдено {len(image_files)} изображений для аннотации")
        self.stats['total_images'] = len(image_files)
        
        # Обработка батчами
        processed_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Создание батчей
            batches = [image_files[i:i + batch_size] 
                      for i in range(0, len(image_files), batch_size)]
            
            # Отправка задач на выполнение
            future_to_batch = {}
            for batch in batches:
                future = executor.submit(self._process_batch, batch, output_dir)
                future_to_batch[future] = batch
            
            # Обработка результатов
            with tqdm(total=len(image_files), desc="Аннотация изображений") as pbar:
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        processed_count += batch_results['processed']
                        failed_count += batch_results['failed']
                        
                        # Обновление статистики
                        for class_name, count in batch_results['class_distribution'].items():
                            self.stats['class_distribution'][class_name] = \
                                self.stats['class_distribution'].get(class_name, 0) + count
                        
                        self.stats['total_detections'] += batch_results['total_detections']
                        self.stats['filtered_detections'] += batch_results['filtered_detections']
                        
                    except Exception as e:
                        self.logger.error(f"Ошибка обработки батча: {e}")
                        failed_count += len(batch)
                    
                    pbar.update(len(batch))
        
        # Финальная статистика
        self.stats['processed_images'] = processed_count
        self.stats['failed_images'] = failed_count
        self.stats['success_rate'] = processed_count / len(image_files) if image_files else 0
        
        self.logger.info(f"Аннотация завершена:")
        self.logger.info(f"  - Обработано: {processed_count}/{len(image_files)}")
        self.logger.info(f"  - Неудачно: {failed_count}")
        self.logger.info(f"  - Всего детекций: {self.stats['total_detections']}")
        self.logger.info(f"  - После фильтрации: {self.stats['total_detections'] - self.stats['filtered_detections']}")
        
        # Сохранение статистики
        self._save_annotation_stats(output_dir)
        
        return self.stats
    
    def _process_batch(self, image_batch: List[Path], output_dir: Path) -> Dict[str, Any]:
        """Обработка батча изображений"""
        batch_stats = {
            'processed': 0,
            'failed': 0,
            'total_detections': 0,
            'filtered_detections': 0,
            'class_distribution': {}
        }
        
        for image_path in image_batch:
            try:
                # Загрузка изображения
                image = cv2.imread(str(image_path))
                if image is None:
                    self.logger.warning(f"Не удалось загрузить изображение: {image_path}")
                    batch_stats['failed'] += 1
                    continue
                
                # Аннотация изображения
                detections = self._annotate_single_image(image, image_path)
                
                # Сохранение аннотации
                annotation_path = output_dir / f"{image_path.stem}.txt"
                self._save_yolo_annotation(detections, annotation_path, image.shape)
                
                # Обновление статистики
                batch_stats['processed'] += 1
                batch_stats['total_detections'] += len(detections)
                
                for detection in detections:
                    class_name = detection.class_name
                    batch_stats['class_distribution'][class_name] = \
                        batch_stats['class_distribution'].get(class_name, 0) + 1
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки {image_path}: {e}")
                batch_stats['failed'] += 1
        
        return batch_stats
    
    def _annotate_single_image(self, image: np.ndarray, image_path: Path) -> List[DetectionResult]:
        """Аннотация одного изображения с использованием ансамбля"""
        
        # Получение детекций от всех моделей
        all_detections = []
        
        for model_name, model_info in self.models.items():
            try:
                model = model_info['model']
                confidence = model_info['confidence']
                
                # Базовая детекция
                results = model(image, conf=confidence, verbose=False)
                model_detections = self._parse_yolo_results(results, model_name)
                
                # Test Time Augmentation (TTA)
                if self.config['augmentation']['enable_tta']:
                    tta_detections = self._apply_tta(image, model, confidence)
                    model_detections.extend(tta_detections)
                
                all_detections.extend(model_detections)
                
            except Exception as e:
                self.logger.warning(f"Ошибка детекции с моделью {model_name}: {e}")
        
        # Применение ансамбля и фильтрации
        consensus_detections = self._apply_ensemble_consensus(all_detections, image.shape)
        filtered_detections = self._filter_detections(consensus_detections, image.shape)
        
        # Фильтрация по классам ресторана
        restaurant_detections = self._filter_restaurant_classes(filtered_detections)
        
        return restaurant_detections
    
    def _parse_yolo_results(self, results, model_name: str) -> List[DetectionResult]:
        """Парсинг результатов YOLO"""
        detections = []
        
        if not results or len(results) == 0:
            return detections
        
        result = results[0]  # Первое изображение в батче
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        height, width = result.orig_shape
        
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if class_id >= len(result.names):
                continue
                
            class_name = result.names[class_id]
            
            # Конвертация в формат YOLO (нормализованные координаты)
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            
            detection = DetectionResult(
                class_id=class_id,
                class_name=class_name,
                confidence=float(conf),
                bbox=[x_center, y_center, w, h],
                original_bbox=[int(x1), int(y1), int(x2), int(y2)]
            )
            
            detections.append(detection)
        
        return detections
    
    def _apply_tta(self, image: np.ndarray, model, confidence: float) -> List[DetectionResult]:
        """Применение Test Time Augmentation"""
        tta_detections = []
        
        # Горизонтальное отражение
        if 'flip' in self.config['augmentation']['tta_transforms']:
            flipped = cv2.flip(image, 1)
            results = model(flipped, conf=confidence, verbose=False)
            flipped_detections = self._parse_yolo_results(results, 'tta_flip')
            
            # Корректировка координат для отраженного изображения
            for det in flipped_detections:
                det.bbox[0] = 1.0 - det.bbox[0]  # Инвертируем x_center
                det.confidence *= 0.9  # Небольшое снижение уверенности для TTA
            
            tta_detections.extend(flipped_detections)
        
        # Поворот
        if 'rotate' in self.config['augmentation']['tta_transforms']:
            # Незначительный поворот на ±5 градусов
            for angle in [-5, 5]:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, matrix, (w, h))
                
                results = model(rotated, conf=confidence, verbose=False)
                rotated_detections = self._parse_yolo_results(results, f'tta_rotate_{angle}')
                
                # Снижение уверенности для повернутых изображений
                for det in rotated_detections:
                    det.confidence *= 0.85
                
                tta_detections.extend(rotated_detections)
        
        return tta_detections
    
    def _apply_ensemble_consensus(self, all_detections: List[DetectionResult], 
                                image_shape: Tuple[int, int, int]) -> List[DetectionResult]:
        """Применение консенсуса ансамбля"""
        if not all_detections:
            return []
        
        # Группировка детекций по IoU
        consensus_detections = []
        used_indices = set()
        
        for i, detection in enumerate(all_detections):
            if i in used_indices:
                continue
            
            # Поиск схожих детекций
            similar_detections = [detection]
            used_indices.add(i)
            
            for j, other_detection in enumerate(all_detections[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Проверка IoU и класса
                if (detection.class_name == other_detection.class_name and
                    self._calculate_iou(detection.bbox, other_detection.bbox) > 
                    self.config['ensemble']['iou_threshold']):
                    
                    similar_detections.append(other_detection)
                    used_indices.add(j)
            
            # Создание консенсусной детекции
            if len(similar_detections) >= 1:  # Минимум одна детекция
                consensus_det = self._create_consensus_detection(similar_detections)
                
                # Проверка порога консенсуса
                consensus_ratio = len(similar_detections) / len(self.models)
                if consensus_ratio >= self.config['ensemble']['consensus_threshold'] / len(self.models):
                    # Бонус за консенсус
                    consensus_det.confidence = min(1.0, 
                        consensus_det.confidence + 
                        self.config['ensemble']['confidence_boost'] * consensus_ratio)
                    
                    consensus_detections.append(consensus_det)
        
        return consensus_detections
    
    def _create_consensus_detection(self, detections: List[DetectionResult]) -> DetectionResult:
        """Создание консенсусной детекции из группы схожих"""
        # Взвешенное усреднение координат по уверенности
        total_weight = sum(det.confidence for det in detections)
        
        if total_weight == 0:
            return detections[0]
        
        weighted_bbox = [0, 0, 0, 0]
        for detection in detections:
            weight = detection.confidence / total_weight
            for i in range(4):
                weighted_bbox[i] += detection.bbox[i] * weight
        
        # Максимальная уверенность
        max_confidence = max(det.confidence for det in detections)
        
        # Наиболее частый класс
        class_votes = {}
        for det in detections:
            key = (det.class_id, det.class_name)
            class_votes[key] = class_votes.get(key, 0) + det.confidence
        
        best_class = max(class_votes.items(), key=lambda x: x[1])[0]
        
        return DetectionResult(
            class_id=best_class[0],
            class_name=best_class[1],
            confidence=max_confidence,
            bbox=weighted_bbox,
            original_bbox=[]  # Будет пересчитан при необходимости
        )
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Вычисление IoU между двумя bbox в формате YOLO"""
        # Конвертация из YOLO формата в углы
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
    
    def _filter_detections(self, detections: List[DetectionResult], 
                          image_shape: Tuple[int, int, int]) -> List[DetectionResult]:
        """Фильтрация детекций по качеству"""
        filtered = []
        height, width = image_shape[:2]
        
        for detection in detections:
            # Проверка уверенности
            if detection.confidence < self.config['filtering']['min_confidence']:
                self.stats['filtered_detections'] += 1
                continue
            
            # Проверка размера
            w_pixels = detection.bbox[2] * width
            h_pixels = detection.bbox[3] * height
            area = w_pixels * h_pixels
            
            if area < self.config['filtering']['min_area']:
                self.stats['filtered_detections'] += 1
                continue
            
            # Проверка отношения к изображению
            image_area = width * height
            if area / image_area > self.config['filtering']['max_area_ratio']:
                self.stats['filtered_detections'] += 1
                continue
            
            # Проверка соотношения сторон
            aspect_ratio = w_pixels / h_pixels if h_pixels > 0 else float('inf')
            if (aspect_ratio < self.config['filtering']['min_aspect_ratio'] or
                aspect_ratio > self.config['filtering']['max_aspect_ratio']):
                self.stats['filtered_detections'] += 1
                continue
            
            # Проверка близости к краям
            edge_threshold = self.config['filtering']['edge_threshold'] / width
            x_center, y_center = detection.bbox[0], detection.bbox[1]
            half_w, half_h = detection.bbox[2] / 2, detection.bbox[3] / 2
            
            if (x_center - half_w < edge_threshold or
                x_center + half_w > 1 - edge_threshold or
                y_center - half_h < edge_threshold or
                y_center + half_h > 1 - edge_threshold):
                # Не отфильтровываем полностью, но снижаем уверенность
                detection.confidence *= 0.8
            
            filtered.append(detection)
        
        return filtered
    
    def _filter_restaurant_classes(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Фильтрация по релевантным для ресторана классам"""
        restaurant_classes = set(self.config['restaurant_classes'])
        
        filtered = []
        for detection in detections:
            if detection.class_name in restaurant_classes:
                filtered.append(detection)
            else:
                self.stats['filtered_detections'] += 1
        
        return filtered
    
    def _save_yolo_annotation(self, detections: List[DetectionResult], 
                             output_path: Path, image_shape: Tuple[int, int, int]):
        """Сохранение аннотации в формате YOLO"""
        
        # Создание маппинга классов
        class_mapping = self._get_class_mapping()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for detection in detections:
                # Получение ID класса в нашей системе
                if detection.class_name in class_mapping:
                    class_id = class_mapping[detection.class_name]
                    
                    # Формат YOLO: class_id x_center y_center width height
                    line = f"{class_id} {detection.bbox[0]:.6f} {detection.bbox[1]:.6f} " \
                           f"{detection.bbox[2]:.6f} {detection.bbox[3]:.6f}\n"
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
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Статистика аннотации сохранена: {stats_path}")


class AnnotationValidator:
    """Валидатор качества аннотаций"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AnnotationValidator")
    
    def validate_annotation_file(self, annotation_path: Path, 
                                image_path: Optional[Path] = None) -> Dict[str, Any]:
        """Валидация одного файла аннотации"""
        validation_result = {
            'valid': True,
            'issues': [],
            'bbox_count': 0,
            'class_distribution': {}
        }
        
        if not annotation_path.exists():
            validation_result['valid'] = False
            validation_result['issues'].append("Файл аннотации не существует")
            return validation_result
        
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                # Пустой файл - это нормально (нет объектов)
                return validation_result
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    validation_result['valid'] = False
                    validation_result['issues'].append(
                        f"Строка {line_num}: неверный формат (ожидается 5 значений)"
                    )
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    # Проверка границ
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                           0 <= width <= 1 and 0 <= height <= 1):
                        validation_result['valid'] = False
                        validation_result['issues'].append(
                            f"Строка {line_num}: координаты вне допустимых границ [0,1]"
                        )
                    
                    # Проверка логичности размеров
                    if width <= 0 or height <= 0:
                        validation_result['valid'] = False
                        validation_result['issues'].append(
                            f"Строка {line_num}: ширина или высота <= 0"
                        )
                    
                    validation_result['bbox_count'] += 1
                    validation_result['class_distribution'][class_id] = \
                        validation_result['class_distribution'].get(class_id, 0) + 1
                    
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
    yaml_content = f"""# Датасет для детекции объектов в ресторане
# Создан автоматически профессиональной системой аннотации

path: {dataset_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Классы
nc: {len(class_mapping)}
names: {list(class_mapping.keys())}

# Дополнительная информация
description: "Профессиональный датасет для детекции объектов в ресторанной среде"
version: "1.0"
license: "Custom"
"""
    
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    return yaml_path


def main():
    """Основная функция для запуска аннотации"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Профессиональная автоматическая аннотация для YOLO11")
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