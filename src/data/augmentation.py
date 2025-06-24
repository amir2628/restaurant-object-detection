"""
Модуль для аугментации данных при обучении YOLOv11
Включает геометрические и цветовые преобразования с сохранением аннотаций
"""
import cv2
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json
import shutil
from dataclasses import dataclass
from tqdm import tqdm

from src.utils.logger import get_logger, log_execution_time
from config.config import config

# Попытка импорта albumentations с обработкой ошибки
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

@dataclass
class AugmentationConfig:
    """Конфигурация для аугментации"""
    # Геометрические преобразования
    rotation_limit: int = 15
    scale_limit: float = 0.2
    translate_limit: float = 0.1
    shear_limit: int = 10
    
    # Цветовые преобразования
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    saturation_limit: float = 0.3
    hue_limit: int = 20
    
    # Размытие и шум
    blur_limit: int = 3
    noise_limit: float = 0.1
    
    # Вероятности применения
    geometric_prob: float = 0.7
    color_prob: float = 0.8
    blur_prob: float = 0.3
    noise_prob: float = 0.2
    
    # Специальные преобразования
    cutout_prob: float = 0.1
    mixup_prob: float = 0.1
    mosaic_prob: float = 0.2

class SimpleAugmentator:
    """Простой аугментатор без зависимости от albumentations"""
    
    def __init__(self, config: AugmentationConfig = None, seed: int = 42):
        """
        Инициализация простого аугментатора
        
        Args:
            config: Конфигурация аугментации
            seed: Seed для воспроизводимости
        """
        self.logger = get_logger(__name__)
        self.config = config or AugmentationConfig()
        
        # Установка seed
        random.seed(seed)
        np.random.seed(seed)
        
        self.logger.info("Инициализирован SimpleAugmentator (без albumentations)")
    
    def augment_image_simple(self, image: np.ndarray, bboxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Простая аугментация изображения с bbox
        
        Args:
            image: Входное изображение
            bboxes: Список bbox в формате YOLO [x_center, y_center, width, height]
            
        Returns:
            Кортеж (аугментированное изображение, аугментированные bbox)
        """
        height, width = image.shape[:2]
        augmented_image = image.copy()
        augmented_bboxes = bboxes.copy()
        
        # Горизонтальное отражение
        if random.random() < 0.5:
            augmented_image = cv2.flip(augmented_image, 1)
            # Преобразование bbox для горизонтального отражения
            for i, bbox in enumerate(augmented_bboxes):
                x_center, y_center, w, h = bbox
                augmented_bboxes[i] = [1.0 - x_center, y_center, w, h]
        
        # Изменение яркости
        if random.random() < self.config.color_prob:
            brightness_factor = random.uniform(0.8, 1.2)
            augmented_image = np.clip(augmented_image * brightness_factor, 0, 255).astype(np.uint8)
        
        # Изменение контрастности
        if random.random() < self.config.color_prob:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = np.mean(augmented_image)
            augmented_image = np.clip((augmented_image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        
        # Размытие
        if random.random() < self.config.blur_prob:
            kernel_size = random.choice([3, 5])
            augmented_image = cv2.GaussianBlur(augmented_image, (kernel_size, kernel_size), 0)
        
        return augmented_image, augmented_bboxes

class DataAugmentator:
    """Класс для аугментации данных с поддержкой bounding boxes"""
    
    def __init__(self, config: AugmentationConfig = None, seed: int = 42):
        """
        Инициализация аугментатора
        
        Args:
            config: Конфигурация аугментации
            seed: Seed для воспроизводимости
        """
        self.logger = get_logger(__name__)
        self.config = config or AugmentationConfig()
        
        # Установка seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Выбор метода аугментации
        if ALBUMENTATIONS_AVAILABLE:
            self.transform = self._create_augmentation_pipeline()
            self.use_albumentations = True
            self.logger.info("Инициализирован DataAugmentator с albumentations")
        else:
            self.simple_augmentator = SimpleAugmentator(config, seed)
            self.use_albumentations = False
            self.logger.warning("Albumentations недоступен, используется простая аугментация")
        
        self.logger.info("Конфигурация аугментации:")
        self.logger.info(f"  - Геометрические преобразования: {self.config.geometric_prob}")
        self.logger.info(f"  - Цветовые преобразования: {self.config.color_prob}")
    
    def _create_augmentation_pipeline(self):
        """Создание пайплайна аугментации (только если albumentations доступен)"""
        if not ALBUMENTATIONS_AVAILABLE:
            return None
        
        transforms = []
        
        # Геометрические преобразования
        geometric_transforms = [
            A.Rotate(
                limit=self.config.rotation_limit, 
                p=self.config.geometric_prob,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.ShiftScaleRotate(
                shift_limit=self.config.translate_limit,
                scale_limit=self.config.scale_limit,
                rotate_limit=0,  # Поворот уже добавлен выше
                p=self.config.geometric_prob,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
        ]
        
        # Цветовые преобразования
        color_transforms = [
            A.RandomBrightnessContrast(
                brightness_limit=self.config.brightness_limit,
                contrast_limit=self.config.contrast_limit,
                p=self.config.color_prob
            ),
            A.HueSaturationValue(
                hue_shift_limit=self.config.hue_limit,
                sat_shift_limit=int(self.config.saturation_limit * 100),
                val_shift_limit=int(self.config.brightness_limit * 100),
                p=self.config.color_prob
            ),
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.3
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.2
            ),
        ]
        
        # Преобразования качества изображения
        quality_transforms = [
            A.OneOf([
                A.Blur(blur_limit=self.config.blur_limit, p=1.0),
                A.GaussianBlur(blur_limit=self.config.blur_limit, p=1.0),
                A.MotionBlur(blur_limit=self.config.blur_limit, p=1.0),
            ], p=self.config.blur_prob),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=self.config.noise_prob),
            
            A.OneOf([
                A.ImageCompression(quality_lower=85, quality_upper=100, p=1.0),
                A.Downscale(scale_min=0.8, scale_max=0.9, p=1.0),
            ], p=0.2),
        ]
        
        # Специальные преобразования
        special_transforms = [
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=self.config.cutout_prob
            ),
        ]
        
        # Объединение всех преобразований
        all_transforms = geometric_transforms + color_transforms + quality_transforms + special_transforms
        
        # Создание композиции с поддержкой bounding boxes
        return A.Compose(
            all_transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_area=100,
                min_visibility=0.3
            ),
            p=1.0
        )
    
    def _load_yolo_annotation(self, annotation_path: str) -> Tuple[List[float], List[int]]:
        """
        Загрузка аннотации в формате YOLO
        
        Args:
            annotation_path: Путь к файлу аннотации
            
        Returns:
            Кортеж (bboxes, class_labels)
        """
        bboxes = []
        class_labels = []
        
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        
                        # Проверка валидности координат
                        if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                            bboxes.append([x, y, w, h])
                            class_labels.append(class_id)
        
        except Exception as e:
            self.logger.warning(f"Ошибка при загрузке аннотации {annotation_path}: {e}")
        
        return bboxes, class_labels
    
    def _save_yolo_annotation(self, annotation_path: str, bboxes: List[List[float]], 
                            class_labels: List[int]):
        """
        Сохранение аннотации в формате YOLO
        
        Args:
            annotation_path: Путь для сохранения
            bboxes: Список bounding boxes
            class_labels: Список меток классов
        """
        try:
            with open(annotation_path, 'w', encoding='utf-8') as f:
                for bbox, class_label in zip(bboxes, class_labels):
                    x, y, w, h = bbox
                    f.write(f"{class_label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении аннотации {annotation_path}: {e}")
    
    @log_execution_time()
    def augment_image_with_annotation(self, 
                                    image_path: str, 
                                    annotation_path: str,
                                    output_image_path: str, 
                                    output_annotation_path: str) -> bool:
        """
        Аугментация изображения с аннотацией
        
        Args:
            image_path: Путь к исходному изображению
            annotation_path: Путь к исходной аннотации
            output_image_path: Путь для сохранения аугментированного изображения
            output_annotation_path: Путь для сохранения аугментированной аннотации
            
        Returns:
            True, если аугментация прошла успешно
        """
        try:
            # Загрузка изображения
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Не удалось загрузить изображение: {image_path}")
                return False
            
            # Загрузка аннотации
            bboxes, class_labels = self._load_yolo_annotation(annotation_path)
            
            if not bboxes:
                # Если нет аннотаций, сохраняем изображение с пустой аннотацией
                if self.use_albumentations:
                    # Применяем аугментацию только к изображению
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Создаем простую трансформацию без bbox
                    simple_transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.RandomBrightnessContrast(p=0.5),
                        A.HueSaturationValue(p=0.3)
                    ])
                    transformed = simple_transform(image=image_rgb)
                    augmented_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                else:
                    # Простая аугментация
                    augmented_image, _ = self.simple_augmentator.augment_image_simple(image, [])
                
                cv2.imwrite(output_image_path, augmented_image)
                Path(output_annotation_path).touch()  # Создаем пустой файл аннотации
                return True
            
            # Применение аугментации
            if self.use_albumentations:
                # Конвертация BGR -> RGB для albumentations
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                try:
                    transformed = self.transform(
                        image=image_rgb,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    # Получение результатов
                    augmented_image = transformed['image']
                    augmented_bboxes = transformed['bboxes']
                    augmented_labels = transformed['class_labels']
                    
                    # Конвертация RGB -> BGR для сохранения
                    augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                    
                except Exception as e:
                    self.logger.warning(f"Ошибка albumentations для {image_path}: {e}, используем простую аугментацию")
                    augmented_image_bgr, temp_bboxes = self.simple_augmentator.augment_image_simple(image, bboxes)
                    augmented_bboxes = temp_bboxes
                    augmented_labels = class_labels
            else:
                # Простая аугментация
                augmented_image_bgr, augmented_bboxes = self.simple_augmentator.augment_image_simple(image, bboxes)
                augmented_labels = class_labels
            
            # Сохранение результатов
            cv2.imwrite(output_image_path, augmented_image_bgr)
            self._save_yolo_annotation(output_annotation_path, augmented_bboxes, augmented_labels)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при аугментации {image_path}: {e}")
            return False
    
    def augment_dataset(self, 
                       images_dir: Path, 
                       annotations_dir: Path,
                       output_images_dir: Path, 
                       output_annotations_dir: Path,
                       augment_factor: int = 3) -> Dict[str, int]:
        """
        Аугментация всего датасета
        
        Args:
            images_dir: Директория с исходными изображениями
            annotations_dir: Директория с исходными аннотациями
            output_images_dir: Директория для аугментированных изображений
            output_annotations_dir: Директория для аугментированных аннотаций
            augment_factor: Количество аугментированных версий на изображение
            
        Returns:
            Статистика аугментации
        """
        self.logger.info(f"Начинается аугментация датасета с фактором {augment_factor}")
        
        # Создание выходных директорий
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Поиск пар изображение-аннотация
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        statistics = {
            'original_images': len(image_files),
            'augmented_images': 0,
            'successful_augmentations': 0,
            'failed_augmentations': 0
        }
        
        for image_path in tqdm(image_files, desc="Аугментация изображений"):
            annotation_path = annotations_dir / f"{image_path.stem}.txt"
            
            # Если аннотация не существует, создаем пустую
            if not annotation_path.exists():
                annotation_path.touch()
            
            # Создание аугментированных версий
            for i in range(augment_factor):
                output_image_path = output_images_dir / f"{image_path.stem}_aug_{i}{image_path.suffix}"
                output_annotation_path = output_annotations_dir / f"{image_path.stem}_aug_{i}.txt"
                
                success = self.augment_image_with_annotation(
                    str(image_path), str(annotation_path),
                    str(output_image_path), str(output_annotation_path)
                )
                
                if success:
                    statistics['successful_augmentations'] += 1
                else:
                    statistics['failed_augmentations'] += 1
                
                statistics['augmented_images'] += 1
        
        # Сохранение статистики
        stats_file = output_images_dir.parent / 'augmentation_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Аугментация завершена: {statistics['successful_augmentations']}/{statistics['augmented_images']} успешно")
        
        return statistics
    
    def create_mosaic_augmentation(self, 
                                 image_paths: List[str], 
                                 annotation_paths: List[str],
                                 output_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, List[Tuple[int, List[float]]]]:
        """
        Создание мозаичной аугментации (объединение 4 изображений)
        
        Args:
            image_paths: Пути к 4 изображениям
            annotation_paths: Пути к соответствующим аннотациям
            output_size: Размер выходного изображения
            
        Returns:
            Кортеж (мозаичное изображение, объединенные аннотации)
        """
        if len(image_paths) != 4 or len(annotation_paths) != 4:
            raise ValueError("Для мозаики нужно ровно 4 изображения и 4 аннотации")
        
        output_w, output_h = output_size
        half_w, half_h = output_w // 2, output_h // 2
        
        # Создание пустого изображения
        mosaic_image = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        all_annotations = []
        
        # Позиции для размещения изображений
        positions = [
            (0, 0, half_w, half_h),      # Верхний левый
            (half_w, 0, output_w, half_h),  # Верхний правый
            (0, half_h, half_w, output_h),  # Нижний левый
            (half_w, half_h, output_w, output_h)  # Нижний правый
        ]
        
        for i, (img_path, ann_path) in enumerate(zip(image_paths, annotation_paths)):
            try:
                # Загрузка изображения
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Изменение размера до четверти мозаики
                resized_image = cv2.resize(image, (half_w, half_h))
                
                # Размещение в мозаике
                x1, y1, x2, y2 = positions[i]
                mosaic_image[y1:y2, x1:x2] = resized_image
                
                # Загрузка и преобразование аннотаций
                bboxes, class_labels = self._load_yolo_annotation(ann_path)
                
                for bbox, class_label in zip(bboxes, class_labels):
                    # Преобразование координат для мозаики
                    orig_x, orig_y, orig_w, orig_h = bbox
                    
                    # Масштабирование к размеру четверти
                    new_x = orig_x * (half_w / output_w) + (x1 / output_w)
                    new_y = orig_y * (half_h / output_h) + (y1 / output_h)
                    new_w = orig_w * (half_w / output_w)
                    new_h = orig_h * (half_h / output_h)
                    
                    # Проверка валидности
                    if 0 <= new_x <= 1 and 0 <= new_y <= 1 and 0 < new_w <= 1 and 0 < new_h <= 1:
                        all_annotations.append((class_label, [new_x, new_y, new_w, new_h]))
                
            except Exception as e:
                self.logger.warning(f"Ошибка при обработке {img_path} для мозаики: {e}")
        
        return mosaic_image, all_annotations
    
    def create_mixup_augmentation(self, 
                                image1_path: str, 
                                image2_path: str,
                                alpha: float = 0.5) -> Tuple[np.ndarray, float]:
        """
        Создание MixUp аугментации (смешивание двух изображений)
        
        Args:
            image1_path: Путь к первому изображению
            image2_path: Путь ко второму изображению
            alpha: Коэффициент смешивания
            
        Returns:
            Кортеж (смешанное изображение, использованный alpha)
        """
        try:
            # Загрузка изображений
            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)
            
            if image1 is None or image2 is None:
                raise ValueError("Не удалось загрузить одно из изображений")
            
            # Приведение к одному размеру
            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]
            
            target_h, target_w = max(h1, h2), max(w1, w2)
            
            image1_resized = cv2.resize(image1, (target_w, target_h))
            image2_resized = cv2.resize(image2, (target_w, target_h))
            
            # Генерация случайного alpha из бета-распределения
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1
            
            # Смешивание изображений
            mixed_image = (lam * image1_resized + (1 - lam) * image2_resized).astype(np.uint8)
            
            return mixed_image, lam
            
        except Exception as e:
            self.logger.error(f"Ошибка при создании MixUp: {e}")
            return None, 0.0
    
    def apply_test_time_augmentation(self, 
                                   image: np.ndarray, 
                                   n_augmentations: int = 5) -> List[np.ndarray]:
        """
        Применение Test Time Augmentation (TTA)
        
        Args:
            image: Исходное изображение
            n_augmentations: Количество аугментированных версий
            
        Returns:
            Список аугментированных изображений
        """
        augmented_images = [image]  # Включаем оригинальное изображение
        
        if self.use_albumentations:
            # Создание более мягких преобразований для TTA
            tta_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, 
                    contrast_limit=0.1, 
                    p=0.5
                ),
                A.Rotate(limit=5, p=0.3),
                A.RandomScale(scale_limit=0.1, p=0.3),
            ])
            
            # Конвертация для albumentations
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            for _ in range(n_augmentations - 1):
                try:
                    transformed = tta_transform(image=image_rgb)
                    augmented_image = transformed['image']
                    
                    # Конвертация обратно в BGR если нужно
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                    
                    augmented_images.append(augmented_image)
                    
                except Exception as e:
                    self.logger.warning(f"Ошибка при TTA: {e}")
        else:
            # Простая TTA без albumentations
            for _ in range(n_augmentations - 1):
                try:
                    augmented_image, _ = self.simple_augmentator.augment_image_simple(image, [])
                    augmented_images.append(augmented_image)
                except Exception as e:
                    self.logger.warning(f"Ошибка при простой TTA: {e}")
        
        return augmented_images

# Утилиты для аугментации
def create_balanced_augmentation_strategy(class_distribution: Dict[str, int], 
                                        target_samples_per_class: int = 1000) -> Dict[str, int]:
    """
    Создание стратегии аугментации для балансировки классов
    
    Args:
        class_distribution: Текущее распределение классов
        target_samples_per_class: Целевое количество образцов на класс
        
    Returns:
        Словарь с коэффициентами аугментации для каждого класса
    """
    augmentation_strategy = {}
    
    for class_name, current_count in class_distribution.items():
        if current_count < target_samples_per_class:
            augment_factor = max(1, target_samples_per_class // current_count)
            augmentation_strategy[class_name] = min(augment_factor, 10)  # Максимум 10x
        else:
            augmentation_strategy[class_name] = 1  # Без аугментации
    
    return augmentation_strategy

def validate_augmented_annotations(original_ann_path: str, 
                                 augmented_ann_path: str) -> bool:
    """
    Валидация аугментированных аннотаций
    
    Args:
        original_ann_path: Путь к оригинальной аннотации
        augmented_ann_path: Путь к аугментированной аннотации
        
    Returns:
        True, если аугментированная аннотация валидна
    """
    try:
        # Загрузка аннотаций
        with open(original_ann_path, 'r') as f:
            original_lines = f.readlines()
        
        with open(augmented_ann_path, 'r') as f:
            augmented_lines = f.readlines()
        
        # Базовые проверки
        if len(augmented_lines) == 0 and len(original_lines) > 0:
            return False  # Потеряли все объекты
        
        # Проверка валидности координат
        for line in augmented_lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    x, y, w, h = map(float, parts[1:5])
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        return False
                except ValueError:
                    return False
        
        return True
        
    except Exception:
        return False

def apply_smart_augmentation(images_dir: Path, 
                           annotations_dir: Path,
                           output_dir: Path,
                           class_mapping: Dict[int, str],
                           target_balance: Dict[str, int] = None):
    """
    Применение умной аугментации с учетом баланса классов
    
    Args:
        images_dir: Директория с изображениями
        annotations_dir: Директория с аннотациями
        output_dir: Выходная директория
        class_mapping: Маппинг классов
        target_balance: Желаемый баланс классов
    """
    logger = get_logger(__name__)
    
    # Анализ текущего распределения классов
    class_counts = {}
    for ann_file in annotations_dir.glob("*.txt"):
        with open(ann_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                class_name = class_mapping.get(class_id, f"class_{class_id}")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Определение стратегии аугментации
    if target_balance is None:
        # Автоматическое определение целевого баланса
        max_count = max(class_counts.values()) if class_counts else 1000
        target_balance = {name: max_count for name in class_counts.keys()}
    
    strategy = create_balanced_augmentation_strategy(class_counts, max(target_balance.values()))
    
    logger.info("Стратегия аугментации:")
    for class_name, factor in strategy.items():
        current = class_counts.get(class_name, 0)
        target = target_balance.get(class_name, current)
        logger.info(f"  {class_name}: {current} -> {target} (фактор: {factor})")
    
    # Применение аугментации
    augmentator = DataAugmentator()
    
    output_images_dir = output_dir / "images"
    output_annotations_dir = output_dir / "annotations"
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Обработка каждого изображения
    for image_file in images_dir.glob("*"):
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            ann_file = annotations_dir / f"{image_file.stem}.txt"
            
            if ann_file.exists():
                # Определение классов в изображении
                with open(ann_file, 'r') as f:
                    lines = f.readlines()
                
                image_classes = set()
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_name = class_mapping.get(class_id, f"class_{class_id}")
                        image_classes.add(class_name)
                
                # Определение максимального фактора аугментации для этого изображения
                max_factor = max([strategy.get(cls, 1) for cls in image_classes]) if image_classes else 1
                
                # Копирование оригинала
                shutil.copy2(image_file, output_images_dir / image_file.name)
                shutil.copy2(ann_file, output_annotations_dir / ann_file.name)
                
                # Создание аугментированных версий
                for i in range(max_factor - 1):
                    output_img_path = output_images_dir / f"{image_file.stem}_aug_{i}{image_file.suffix}"
                    output_ann_path = output_annotations_dir / f"{image_file.stem}_aug_{i}.txt"
                    
                    success = augmentator.augment_image_with_annotation(
                        str(image_file), str(ann_file),
                        str(output_img_path), str(output_ann_path)
                    )
                    
                    if not success:
                        logger.warning(f"Не удалось аугментировать {image_file}")
    
    logger.info(f"Умная аугментация завершена. Результаты в: {output_dir}")