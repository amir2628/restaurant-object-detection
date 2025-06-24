"""
Модуль для создания и организации датасета для обучения YOLOv11
Включает разделение данных, валидацию и создание конфигурационных файлов
"""
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json
import yaml
import logging
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
import cv2
from dataclasses import dataclass

from src.utils.logger import get_logger, log_execution_time
from config.config import config

@dataclass
class DatasetStatistics:
    """Статистика датасета"""
    total_images: int
    total_annotations: int
    class_distribution: Dict[str, int]
    split_distribution: Dict[str, int]
    image_sizes: List[Tuple[int, int]]
    annotation_quality: Dict[str, float]

class DatasetBuilder:
    """Класс для создания и организации датасета YOLO"""
    
    def __init__(self,
                 train_split: float = None,
                 val_split: float = None,
                 test_split: float = None,
                 min_images_per_class: int = None,
                 max_images_per_class: int = None,
                 seed: int = 42):
        """
        Инициализация билдера датасета
        
        Args:
            train_split: Доля данных для обучения
            val_split: Доля данных для валидации
            test_split: Доля данных для тестирования
            min_images_per_class: Минимальное количество изображений на класс
            max_images_per_class: Максимальное количество изображений на класс
            seed: Seed для воспроизводимости
        """
        self.logger = get_logger(__name__)
        
        # Использование конфигурации по умолчанию или переданных параметров
        self.train_split = train_split or config.dataset.train_split
        self.val_split = val_split or config.dataset.val_split
        self.test_split = test_split or config.dataset.test_split
        self.min_images_per_class = min_images_per_class or config.dataset.min_images_per_class
        self.max_images_per_class = max_images_per_class or config.dataset.max_images_per_class
        
        # Проверка корректности разделения
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Сумма разделений должна быть равна 1.0, получено: {total_split}")
        
        # Установка seed для воспроизводимости
        random.seed(seed)
        np.random.seed(seed)
        
        self.logger.info(f"Инициализирован DatasetBuilder:")
        self.logger.info(f"  - Разделение: train={self.train_split}, val={self.val_split}, test={self.test_split}")
        self.logger.info(f"  - Диапазон изображений на класс: {self.min_images_per_class}-{self.max_images_per_class}")
    
    def analyze_dataset(self, images_dir: Path, annotations_dir: Path,
                       class_mapping_file: Path = None) -> DatasetStatistics:
        """
        Анализ существующего датасета
        
        Args:
            images_dir: Директория с изображениями
            annotations_dir: Директория с аннотациями
            class_mapping_file: Файл с маппингом классов
            
        Returns:
            Статистика датасета
        """
        self.logger.info("Начинаем анализ датасета...")
        
        # Загрузка маппинга классов
        class_mapping = self._load_class_mapping(class_mapping_file)
        
        # Поиск пар изображение-аннотация
        image_annotation_pairs = self._find_image_annotation_pairs(images_dir, annotations_dir)
        
        # Анализ статистики
        class_distribution = defaultdict(int)
        image_sizes = []
        annotation_quality_scores = []
        total_annotations = 0
        
        for image_path, annotation_path in tqdm(image_annotation_pairs, desc="Анализ файлов"):
            try:
                # Анализ изображения
                image = cv2.imread(str(image_path))
                if image is not None:
                    h, w = image.shape[:2]
                    image_sizes.append((w, h))
                
                # Анализ аннотации
                if annotation_path and annotation_path.exists():
                    annotations = self._load_yolo_annotation(annotation_path)
                    total_annotations += len(annotations)
                    
                    # Подсчет классов
                    for class_id, bbox in annotations:
                        if class_id in class_mapping:
                            class_name = class_mapping[class_id]
                            class_distribution[class_name] += 1
                        
                        # Оценка качества аннотации
                        quality_score = self._evaluate_annotation_quality(bbox, w, h)
                        annotation_quality_scores.append(quality_score)
                
            except Exception as e:
                self.logger.warning(f"Ошибка при анализе {image_path}: {e}")
        
        # Создание статистики
        statistics = DatasetStatistics(
            total_images=len(image_annotation_pairs),
            total_annotations=total_annotations,
            class_distribution=dict(class_distribution),
            split_distribution={},  # Будет заполнено позже
            image_sizes=image_sizes,
            annotation_quality={
                'mean_quality': np.mean(annotation_quality_scores) if annotation_quality_scores else 0,
                'std_quality': np.std(annotation_quality_scores) if annotation_quality_scores else 0,
                'min_quality': np.min(annotation_quality_scores) if annotation_quality_scores else 0,
                'max_quality': np.max(annotation_quality_scores) if annotation_quality_scores else 0
            }
        )
        
        self._log_dataset_statistics(statistics)
        return statistics
    
    def _load_class_mapping(self, class_mapping_file: Path = None) -> Dict[int, str]:
        """Загрузка маппинга классов"""
        if class_mapping_file and class_mapping_file.exists():
            with open(class_mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
                # Проверяем разные форматы маппинга
                if 'direct_mapping' in mapping_data:
                    return {int(k): v for k, v in mapping_data['direct_mapping'].items()}
                elif 'reverse_mapping' in mapping_data:
                    return {v: k for k, v in mapping_data['reverse_mapping'].items()}
                else:
                    # Предполагаем, что это прямой маппинг
                    return {int(k): v for k, v in mapping_data.items()}
        else:
            # Создание маппинга по умолчанию на основе конфигурации
            return {i: class_name for i, class_name in enumerate(config.annotation.target_classes)}
    
    def _find_image_annotation_pairs(self, images_dir: Path, annotations_dir: Path) -> List[Tuple[Path, Path]]:
        """Поиск пар изображение-аннотация"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        pairs = []
        
        # Поиск всех изображений
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        # Поиск соответствующих аннотаций
        for image_path in image_files:
            annotation_path = annotations_dir / f"{image_path.stem}.txt"
            pairs.append((image_path, annotation_path if annotation_path.exists() else None))
        
        self.logger.info(f"Найдено {len(pairs)} пар изображение-аннотация")
        return pairs
    
    def _load_yolo_annotation(self, annotation_path: Path) -> List[Tuple[int, List[float]]]:
        """Загрузка аннотации в формате YOLO"""
        annotations = []
        
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        annotations.append((class_id, bbox))
        except Exception as e:
            self.logger.warning(f"Ошибка чтения аннотации {annotation_path}: {e}")
        
        return annotations
    
    def _evaluate_annotation_quality(self, bbox: List[float], img_width: int, img_height: int) -> float:
        """Оценка качества аннотации"""
        x, y, w, h = bbox
        
        # Проверка валидности координат
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return 0.0
        
        # Вычисление площади в пикселях
        pixel_area = w * h * img_width * img_height
        
        # Оценка качества на основе размера объекта
        if pixel_area < config.annotation.min_bbox_area:
            return 0.2  # Очень маленький объект
        elif pixel_area > img_width * img_height * 0.8:
            return 0.3  # Слишком большой объект
        else:
            # Нормальный размер объекта
            size_score = min(pixel_area / (img_width * img_height * 0.1), 1.0)
            return 0.5 + 0.5 * size_score
    
    def _log_dataset_statistics(self, statistics: DatasetStatistics):
        """Логирование статистики датасета"""
        self.logger.info("=== СТАТИСТИКА ДАТАСЕТА ===")
        self.logger.info(f"Общее количество изображений: {statistics.total_images}")
        self.logger.info(f"Общее количество аннотаций: {statistics.total_annotations}")
        
        if statistics.total_images > 0:
            self.logger.info(f"Среднее количество объектов на изображение: "
                           f"{statistics.total_annotations / statistics.total_images:.2f}")
        
        # Распределение классов
        self.logger.info("Распределение классов:")
        for class_name, count in sorted(statistics.class_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / statistics.total_annotations) * 100 if statistics.total_annotations > 0 else 0
            self.logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Размеры изображений
        if statistics.image_sizes:
            widths, heights = zip(*statistics.image_sizes)
            self.logger.info(f"Размеры изображений:")
            self.logger.info(f"  Ширина: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
            self.logger.info(f"  Высота: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")
        
        # Качество аннотаций
        quality = statistics.annotation_quality
        self.logger.info(f"Качество аннотаций: mean={quality['mean_quality']:.3f}, "
                        f"std={quality['std_quality']:.3f}")
        self.logger.info("=" * 50)
    
    @log_execution_time()
    def create_dataset_splits(self, 
                            images_dir: Path, 
                            annotations_dir: Path,
                            output_dir: Path,
                            class_mapping_file: Path = None) -> Dict[str, Any]:
        """
        Создание разделений датасета (train/val/test)
        
        Args:
            images_dir: Директория с изображениями
            annotations_dir: Директория с аннотациями
            output_dir: Выходная директория для датасета
            class_mapping_file: Файл с маппингом классов
            
        Returns:
            Информация о созданном датасете
        """
        self.logger.info("Создание разделений датасета...")
        
        # Создание структуры директорий
        dataset_structure = self._create_dataset_structure(output_dir)
        
        # Анализ исходного датасета
        statistics = self.analyze_dataset(images_dir, annotations_dir, class_mapping_file)
        
        # Поиск валидных пар
        pairs = self._find_image_annotation_pairs(images_dir, annotations_dir)
        valid_pairs = []
        
        for img_path, ann_path in pairs:
            if img_path.exists():
                # Если аннотация не существует, создаем пустую
                if ann_path is None or not ann_path.exists():
                    ann_path = annotations_dir / f"{img_path.stem}.txt"
                    if not ann_path.exists():
                        ann_path.touch()  # Создаем пустой файл аннотации
                valid_pairs.append((img_path, ann_path))
        
        self.logger.info(f"Найдено {len(valid_pairs)} валидных пар для разделения")
        
        if not valid_pairs:
            raise ValueError("Не найдено валидных пар изображение-аннотация")
        
        # Стратифицированное разделение по классам
        splits = self._create_stratified_splits(valid_pairs, annotations_dir, class_mapping_file)
        
        # Копирование файлов в соответствующие директории
        split_info = {}
        for split_name, split_pairs in splits.items():
            self.logger.info(f"Обработка разделения {split_name}: {len(split_pairs)} файлов")
            
            images_count = self._copy_split_files(
                split_pairs, 
                dataset_structure[split_name]['images'],
                dataset_structure[split_name]['labels']
            )
            
            split_info[split_name] = {
                'images_count': images_count,
                'pairs_count': len(split_pairs)
            }
        
        # Копирование маппинга классов
        if class_mapping_file and class_mapping_file.exists():
            shutil.copy2(class_mapping_file, output_dir / 'class_mapping.json')
        
        # Создание YAML конфигурации для YOLO
        self._create_yolo_config(output_dir, class_mapping_file)
        
        # Сохранение статистики датасета
        dataset_info = {
            'creation_timestamp': str(self._get_current_timestamp()),
            'source_images_dir': str(images_dir),
            'source_annotations_dir': str(annotations_dir),
            'splits': split_info,
            'statistics': {
                'total_images': statistics.total_images,
                'total_annotations': statistics.total_annotations,
                'class_distribution': statistics.class_distribution
            },
            'split_ratios': {
                'train': self.train_split,
                'val': self.val_split,
                'test': self.test_split
            }
        }
        
        # Сохранение информации о датасете
        with open(output_dir / 'dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Датасет успешно создан в: {output_dir}")
        return dataset_info
    
    def _get_current_timestamp(self):
        """Получение текущего времени"""
        from datetime import datetime
        return datetime.now()
    
    def _create_dataset_structure(self, output_dir: Path) -> Dict[str, Dict[str, Path]]:
        """Создание структуры директорий датасета"""
        structure = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = output_dir / split
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            structure[split] = {
                'images': images_dir,
                'labels': labels_dir
            }
        
        return structure
    
    def _create_stratified_splits(self, pairs: List[Tuple[Path, Path]], 
                                annotations_dir: Path,
                                class_mapping_file: Path = None) -> Dict[str, List[Tuple[Path, Path]]]:
        """Создание стратифицированных разделений по классам"""
        # Загрузка маппинга классов
        class_mapping = self._load_class_mapping(class_mapping_file)
        
        # Если у нас мало данных, используем простое разделение
        if len(pairs) < 10:
            return self._create_simple_splits(pairs)
        
        # Группировка по классам
        class_to_pairs = defaultdict(list)
        pairs_without_objects = []
        
        for img_path, ann_path in pairs:
            try:
                if ann_path and ann_path.exists():
                    annotations = self._load_yolo_annotation(ann_path)
                    
                    if annotations:
                        # Определение классов в изображении
                        image_classes = set()
                        for class_id, _ in annotations:
                            if class_id in class_mapping:
                                image_classes.add(class_mapping[class_id])
                            else:
                                image_classes.add(f"class_{class_id}")
                        
                        # Добавление к каждому классу (для стратификации)
                        for class_name in image_classes:
                            class_to_pairs[class_name].append((img_path, ann_path))
                    else:
                        pairs_without_objects.append((img_path, ann_path))
                else:
                    pairs_without_objects.append((img_path, ann_path))
                    
            except Exception as e:
                self.logger.warning(f"Ошибка при анализе {ann_path}: {e}")
                pairs_without_objects.append((img_path, ann_path))
        
        # Создание разделений для каждого класса
        splits = {'train': [], 'val': [], 'test': []}
        
        # Обработка изображений с объектами
        for class_name, class_pairs in class_to_pairs.items():
            # Перемешивание
            random.shuffle(class_pairs)
            
            # Вычисление индексов разделения
            n_total = len(class_pairs)
            n_train = max(1, int(n_total * self.train_split))
            n_val = max(1, int(n_total * self.val_split))
            
            # Разделение
            train_pairs = class_pairs[:n_train]
            val_pairs = class_pairs[n_train:n_train + n_val]
            test_pairs = class_pairs[n_train + n_val:]
            
            splits['train'].extend(train_pairs)
            splits['val'].extend(val_pairs)
            splits['test'].extend(test_pairs)
            
            self.logger.info(f"Класс {class_name}: train={len(train_pairs)}, "
                           f"val={len(val_pairs)}, test={len(test_pairs)}")
        
        # Обработка изображений без объектов
        if pairs_without_objects:
            random.shuffle(pairs_without_objects)
            n_total = len(pairs_without_objects)
            n_train = int(n_total * self.train_split)
            n_val = int(n_total * self.val_split)
            
            splits['train'].extend(pairs_without_objects[:n_train])
            splits['val'].extend(pairs_without_objects[n_train:n_train + n_val])
            splits['test'].extend(pairs_without_objects[n_train + n_val:])
            
            self.logger.info(f"Изображения без объектов: train={n_train}, "
                           f"val={n_val}, test={len(pairs_without_objects) - n_train - n_val}")
        
        # Удаление дубликатов (так как изображение может содержать несколько классов)
        for split_name in splits:
            splits[split_name] = list(set(splits[split_name]))
            random.shuffle(splits[split_name])
        
        return splits
    
    def _create_simple_splits(self, pairs: List[Tuple[Path, Path]]) -> Dict[str, List[Tuple[Path, Path]]]:
        """Простое разделение данных когда мало образцов"""
        random.shuffle(pairs)
        
        n_total = len(pairs)
        n_train = max(1, int(n_total * self.train_split))
        n_val = max(1, int(n_total * self.val_split))
        
        splits = {
            'train': pairs[:n_train],
            'val': pairs[n_train:n_train + n_val],
            'test': pairs[n_train + n_val:]
        }
        
        # Убедимся, что каждый split содержит хотя бы одно изображение
        if not splits['val'] and splits['train']:
            splits['val'].append(splits['train'].pop())
        if not splits['test'] and splits['train']:
            splits['test'].append(splits['train'].pop())
        
        return splits
    
    def _copy_split_files(self, pairs: List[Tuple[Path, Path]], 
                         target_images_dir: Path, target_labels_dir: Path) -> int:
        """Копирование файлов в целевые директории"""
        copied_count = 0
        
        for img_path, ann_path in tqdm(pairs, desc="Копирование файлов"):
            try:
                # Копирование изображения
                target_img_path = target_images_dir / img_path.name
                shutil.copy2(img_path, target_img_path)
                
                # Копирование аннотации
                if ann_path and ann_path.exists():
                    target_ann_path = target_labels_dir / ann_path.name
                    shutil.copy2(ann_path, target_ann_path)
                else:
                    # Создание пустого файла аннотации
                    target_ann_path = target_labels_dir / f"{img_path.stem}.txt"
                    target_ann_path.touch()
                
                copied_count += 1
                
            except Exception as e:
                self.logger.error(f"Ошибка при копировании {img_path}: {e}")
        
        return copied_count
    
    def _create_yolo_config(self, output_dir: Path, class_mapping_file: Path = None):
        """Создание конфигурационного файла для YOLO"""
        # Загрузка маппинга классов
        class_mapping = self._load_class_mapping(class_mapping_file)
        
        # Создание списка имен классов в правильном порядке
        class_names = []
        if class_mapping:
            # Получаем максимальный ID класса
            max_class_id = max(class_mapping.keys()) if class_mapping else 0
            
            # Создаем список имен классов по порядку ID
            for i in range(max_class_id + 1):
                if i in class_mapping:
                    class_names.append(class_mapping[i])
                else:
                    class_names.append(f"class_{i}")
        
        # Если маппинг пустой или неполный, используем конфигурацию по умолчанию
        if not class_names:
            class_names = config.annotation.target_classes
        
        # Создание YAML конфигурации
        yolo_config = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        # Сохранение конфигурации
        config_file = output_dir / 'dataset.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(yolo_config, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"YOLO конфигурация сохранена в: {config_file}")
        self.logger.info(f"Классы в датасете ({len(class_names)}): {class_names}")

# Дополнительные утилиты

def merge_datasets(dataset_dirs: List[Path], output_dir: Path, 
                  dataset_name: str = "merged_dataset") -> Path:
    """
    Объединение нескольких датасетов в один
    
    Args:
        dataset_dirs: Список директорий с датасетами
        output_dir: Выходная директория
        dataset_name: Имя нового датасета
        
    Returns:
        Путь к объединенному датасету
    """
    logger = get_logger(__name__)
    
    merged_dataset_dir = output_dir / dataset_name
    builder = DatasetBuilder()
    
    # Временная директория для сбора всех изображений и аннотаций
    temp_images_dir = merged_dataset_dir / "temp_images"
    temp_annotations_dir = merged_dataset_dir / "temp_annotations"
    
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    temp_annotations_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Объединение {len(dataset_dirs)} датасетов...")
    
    # Сбор всех файлов
    for i, dataset_dir in enumerate(dataset_dirs):
        logger.info(f"Обработка датасета {i+1}/{len(dataset_dirs)}: {dataset_dir}")
        
        # Поиск изображений и аннотаций во всех split'ах
        for split in ['train', 'val', 'test']:
            images_dir = dataset_dir / split / 'images'
            labels_dir = dataset_dir / split / 'labels'
            
            if images_dir.exists():
                for img_file in images_dir.glob("*"):
                    if img_file.is_file():
                        # Создание уникального имени файла
                        new_name = f"dataset{i}_{img_file.name}"
                        shutil.copy2(img_file, temp_images_dir / new_name)
                        
                        # Копирование соответствующей аннотации
                        ann_file = labels_dir / f"{img_file.stem}.txt"
                        if ann_file.exists():
                            shutil.copy2(ann_file, temp_annotations_dir / f"dataset{i}_{ann_file.name}")
    
    # Создание нового датасета из собранных файлов
    dataset_info = builder.create_dataset_splits(
        temp_images_dir, temp_annotations_dir, merged_dataset_dir
    )
    
    # Удаление временных директорий
    shutil.rmtree(temp_images_dir)
    shutil.rmtree(temp_annotations_dir)
    
    logger.info(f"Объединенный датасет создан в: {merged_dataset_dir}")
    return merged_dataset_dir