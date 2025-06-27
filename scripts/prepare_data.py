"""
Улучшенный скрипт подготовки данных с GroundingDINO аннотацией
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


def create_dataset_yaml(dataset_dir: Path, class_names: List[str] = None):
    """
    Создание YAML конфигурации для YOLO с GroundingDINO классами
    
    Args:
        dataset_dir: Директория датасета
        class_names: Список имен классов
    """
    logger = setup_logger(__name__)
    
    # Специализированные классы для ресторанной среды (GroundingDINO)
    if class_names is None:
        class_names = [
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
            'method': 'groundingdino',
            'confidence_threshold': 0.25,
            'text_threshold': 0.25,
            'box_threshold': 0.25,
            'create_empty_annotations': True,
            'groundingdino_checkpoint': 'groundingdino_swinb_cogcoor.pth',
            'detection_prompt': 'chicken . meat . salad . soup . cup . plate . bowl . spoon . fork . knife .'
        },
        'dataset': {
            'train_ratio': 0.7,
            'val_ratio': 0.2,
            'test_ratio': 0.1,
            'class_names': [
                'chicken', 'meat', 'salad', 'soup', 'cup',
                'plate', 'bowl', 'spoon', 'fork', 'knife'
            ]
        },
        'quality_control': {
            'min_detection_size': 0.01,
            'max_detection_size': 0.8,
            'min_confidence': 0.15
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
            logger.info(f"✅ Конфигурация загружена из: {config_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки конфигурации: {e}")
            logger.info("Используется конфигурация по умолчанию")
    
    return default_config


class DataPipelineProcessor:
    """Основной процессор пайплайна подготовки данных с GroundingDINO"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
        self.dataset_dir = Path("data/processed/dataset")
        
        # Статистика
        self.stats = {
            'start_time': time.time(),
            'total_videos': 0,
            'total_frames': 0,
            'total_annotations': 0,
            'stages_completed': []
        }
    
    def run_pipeline(self, input_dir: Path):
        """Запуск полного пайплайна подготовки данных"""
        self.logger.info("🚀 Запуск пайплайна подготовки данных с GroundingDINO")
        self.stats['start_time'] = time.time()
        
        try:
            # 1. Создание структуры директорий
            self.logger.info("🏗️ Этап 1: Создание структуры директорий")
            create_directory_structure()
            self.stats['stages_completed'].append('directory_structure')
            
            # 2. Поиск и валидация видео файлов
            self.logger.info("🔍 Этап 2: Поиск видео файлов")
            video_files = self._find_video_files(input_dir)
            self.stats['total_videos'] = len(video_files)
            
            if not video_files:
                raise ValueError(f"Видео файлы не найдены в директории: {input_dir}")
            
            # 3. Извлечение кадров из видео
            self.logger.info("🎬 Этап 3: Извлечение кадров")
            all_frames = self._extract_frames_from_videos(video_files)
            self.stats['total_frames'] = len(all_frames)
            self.stats['stages_completed'].append('frame_extraction')
            
            # 4. Разделение на train/val/test
            self.logger.info("📂 Этап 4: Разделение на train/val/test")
            self._split_and_organize_dataset(all_frames)
            self.stats['stages_completed'].append('dataset_split')
            
            # 5. Аннотация с GroundingDINO
            self.logger.info("🧠 Этап 5: Аннотация с GroundingDINO")
            self._annotate_with_groundingdino()
            self.stats['stages_completed'].append('annotation')
            
            # 6. Создание YAML конфигурации
            self.logger.info("📄 Этап 6: Создание dataset.yaml")
            create_dataset_yaml(self.dataset_dir, self.config['dataset']['class_names'])
            self.stats['stages_completed'].append('yaml_creation')
            
            # 7. Очистка временных файлов
            self.logger.info("🧹 Этап 7: Очистка временных файлов")
            self._cleanup_temp_files()
            
            # 8. Генерация отчета
            self._generate_completion_report()
            
            self.logger.info("🎉 Пайплайн подготовки данных завершен успешно!")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в пайплайне: {e}")
            self._generate_error_report(e)
            raise
    
    def _find_video_files(self, input_dir: Path) -> List[Path]:
        """Поиск видео файлов в директории"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        video_files = []
        
        # Используем set для избежания дубликатов
        found_files = set()
        
        for ext in video_extensions:
            # Поиск файлов с нижним регистром расширения
            for file_path in input_dir.glob(f"*{ext}"):
                found_files.add(file_path)
            # Поиск файлов с верхним регистром расширения
            for file_path in input_dir.glob(f"*{ext.upper()}"):
                found_files.add(file_path)
        
        # Конвертация set в отсортированный список
        video_files = sorted(list(found_files))
        
        self.logger.info(f"Найдено {len(video_files)} уникальных видео файлов")
        for video_file in video_files:
            self.logger.info(f"  - {video_file.name}")
        
        return video_files
    
    def _extract_frames_from_videos(self, video_files: List[Path]) -> List[Path]:
        """Извлечение кадров из всех видео"""
        all_frames = []
        fps_extraction = self.config['video_processing']['fps_extraction']
        max_frames = self.config['video_processing']['max_frames_per_video']
        
        temp_frames_dir = Path("data/temp/frames")
        temp_frames_dir.mkdir(parents=True, exist_ok=True)
        
        for video_file in tqdm(video_files, desc="Обработка видео"):
            try:
                frames = self._extract_frames_from_single_video(
                    video_file, temp_frames_dir, fps_extraction, max_frames
                )
                all_frames.extend(frames)
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки видео {video_file}: {e}")
                continue
        
        self.logger.info(f"Извлечено {len(all_frames)} кадров")
        return all_frames
    
    def _extract_frames_from_single_video(self, video_path: Path, output_dir: Path, 
                                        fps: float, max_frames: int) -> List[Path]:
        """Извлечение кадров из одного видео"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        # Получение информации о видео
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Расчет интервала извлечения
        frame_interval = int(video_fps / fps) if fps > 0 else 1
        
        extracted_frames = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or saved_count >= max_frames:
                break
            
            if frame_count % frame_interval == 0:
                # Сохранение кадра
                frame_filename = f"{video_path.stem}_frame_{frame_count:06d}.jpg"
                frame_path = output_dir / frame_filename
                
                # Изменение размера если необходимо
                if self.config['video_processing']['resize_frames']:
                    target_size = self.config['video_processing']['target_size']
                    frame = cv2.resize(frame, target_size)
                
                cv2.imwrite(str(frame_path), frame)
                extracted_frames.append(frame_path)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        return extracted_frames
    
    def _split_and_organize_dataset(self, frame_paths: List[Path]):
        """Разделение кадров на train/val/test и организация в структуру YOLO"""
        import random
        
        # Перемешивание кадров
        frame_paths = frame_paths.copy()
        random.shuffle(frame_paths)
        
        # Расчет размеров splits
        total_frames = len(frame_paths)
        train_size = int(total_frames * self.config['dataset']['train_ratio'])
        val_size = int(total_frames * self.config['dataset']['val_ratio'])
        test_size = total_frames - train_size - val_size
        
        # Разделение кадров
        train_frames = frame_paths[:train_size]
        val_frames = frame_paths[train_size:train_size + val_size]
        test_frames = frame_paths[train_size + val_size:]
        
        self.logger.info(f"Разделение: train={len(train_frames)}, val={len(val_frames)}, test={len(test_frames)}")
        
        # Копирование файлов в соответствующие директории
        splits = {
            'train': train_frames,
            'val': val_frames,
            'test': test_frames
        }
        
        for split_name, frames in splits.items():
            split_images_dir = self.dataset_dir / split_name / "images"
            split_labels_dir = self.dataset_dir / split_name / "labels"
            
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Создание директории для визуализаций
            split_visualizations_dir = self.dataset_dir / split_name / "visualizations"
            split_visualizations_dir.mkdir(parents=True, exist_ok=True)
            
            for frame_path in tqdm(frames, desc=f"Копирование {split_name}"):
                # Копирование изображения
                target_image_path = split_images_dir / frame_path.name
                shutil.copy2(frame_path, target_image_path)
                
                # Создание пустого файла аннотации (будет заполнен на следующем этапе)
                target_label_path = split_labels_dir / f"{frame_path.stem}.txt"
                target_label_path.touch()
    
    def _apply_massive_augmentation(self):
        """
        Применение массивной аугментации для увеличения размера датасета
        Использует аннотации оригинальных изображений для аугментированных
        """
        from src.data.augmentation import DataAugmentator
        
        self.logger.info("🎨 Этап 5.5: Применение массивной аугментации с сохранением аннотаций")
        
        # Инициализация аугментатора
        augmentator = DataAugmentator()
        
        # Конфигурация аугментации для разных splits
        augmentation_config = {
            'train': {
                'factor': 6,  # Уменьшаем до 6 для лучшего качества
                'description': 'Интенсивная аугментация для тренировочного набора'
            },
            'val': {
                'factor': 3,  # Умеренная аугментация
                'description': 'Умеренная аугментация для валидационного набора'
            },
            'test': {
                'factor': 2,  # Легкая аугментация
                'description': 'Легкая аугментация для тестового набора'
            }
        }
        
        total_original_images = 0
        total_augmented_images = 0
        
        for split_name, aug_config in augmentation_config.items():
            self.logger.info(f"🔄 Аугментация {split_name} набора (фактор: {aug_config['factor']})")
            
            # Пути к директориям
            images_dir = self.dataset_dir / split_name / "images"
            labels_dir = self.dataset_dir / split_name / "labels"
            
            if not images_dir.exists():
                self.logger.warning(f"Директория {images_dir} не найдена, пропуск...")
                continue
            
            # Поиск только оригинальных изображений (без _aug_ в названии)
            all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            original_images = [img for img in all_images if "_aug_" not in img.name]
            original_count = len(original_images)
            total_original_images += original_count
            
            self.logger.info(f"  Найдено {original_count} оригинальных изображений")
            
            if original_count == 0:
                continue
            
            # Применение аугментации с сохранением аннотаций
            augmented_count = 0
            factor = aug_config['factor']
            
            for original_image in tqdm(original_images, desc=f"Аугментация {split_name}"):
                original_annotation = labels_dir / f"{original_image.stem}.txt"
                
                # Проверяем, есть ли аннотация для оригинального изображения
                if not original_annotation.exists():
                    self.logger.warning(f"Аннотация не найдена для {original_image}")
                    continue
                
                # Создание multiple аугментированных версий
                for aug_idx in range(factor - 1):  # -1 потому что оригинал уже есть
                    aug_image_name = f"{original_image.stem}_aug_{aug_idx:03d}{original_image.suffix}"
                    aug_annotation_name = f"{original_image.stem}_aug_{aug_idx:03d}.txt"
                    
                    aug_image_path = images_dir / aug_image_name
                    aug_annotation_path = labels_dir / aug_annotation_name
                    
                    try:
                        # Применение аугментации с трансформацией аннотаций
                        success = augmentator.augment_image_with_annotation(
                            str(original_image),
                            str(original_annotation),
                            str(aug_image_path),
                            str(aug_annotation_path)
                        )
                        
                        if success:
                            # Проверяем, что аннотация действительно создана и не пуста
                            if aug_annotation_path.exists() and aug_annotation_path.stat().st_size > 0:
                                augmented_count += 1
                            else:
                                # Если аннотация пуста, копируем оригинальную напрямую
                                self.logger.warning(f"Аннотация после аугментации пуста для {original_image}, копируем оригинальную")
                                shutil.copy2(original_annotation, aug_annotation_path)
                                augmented_count += 1
                        else:
                            # Если аугментация не удалась, используем fallback подход
                            self._fallback_augmentation(original_image, original_annotation, 
                                                       aug_image_path, aug_annotation_path)
                            augmented_count += 1
                            
                    except Exception as e:
                        self.logger.warning(f"Ошибка аугментации {original_image}: {e}")
                        # Используем fallback подход
                        try:
                            self._fallback_augmentation(original_image, original_annotation, 
                                                       aug_image_path, aug_annotation_path)
                            augmented_count += 1
                        except Exception as fallback_error:
                            self.logger.error(f"Fallback аугментация также не удалась для {original_image}: {fallback_error}")
                            continue
            
            final_count = original_count + augmented_count
            total_augmented_images += final_count
            
            self.logger.info(f"  ✅ {split_name}: {original_count} → {final_count} изображений "
                           f"(увеличение в {final_count/original_count:.1f} раз)")
        
        # Обновление статистики
        self.stats['total_frames'] = total_original_images
        self.stats['total_augmented_images'] = total_augmented_images
        self.stats['augmentation_factor'] = total_augmented_images / total_original_images if total_original_images > 0 else 1
        
        self.logger.info(f"🎨 Массивная аугментация завершена!")
        self.logger.info(f"📊 Общая статистика:")
        self.logger.info(f"  - Оригинальных изображений: {total_original_images}")
        self.logger.info(f"  - Итоговых изображений: {total_augmented_images}")
        self.logger.info(f"  - Коэффициент увеличения: {self.stats['augmentation_factor']:.1f}x")
    
    def _fallback_augmentation(self, original_image: Path, original_annotation: Path,
                              aug_image_path: Path, aug_annotation_path: Path):
        """
        Простая fallback аугментация с гарантированным сохранением аннотаций
        """
        try:
            # Загрузка и простая аугментация изображения
            import cv2
            import random
            
            image = cv2.imread(str(original_image))
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {original_image}")
            
            # Простые аугментации, которые не нарушают аннотации
            augmented_image = image.copy()
            
            # Случайная яркость
            if random.random() < 0.5:
                brightness = random.uniform(0.8, 1.2)
                augmented_image = np.clip(augmented_image * brightness, 0, 255).astype(np.uint8)
            
            # Случайный контраст
            if random.random() < 0.5:
                contrast = random.uniform(0.8, 1.2)
                mean = np.mean(augmented_image)
                augmented_image = np.clip((augmented_image - mean) * contrast + mean, 0, 255).astype(np.uint8)
            
            # Небольшое размытие
            if random.random() < 0.3:
                augmented_image = cv2.GaussianBlur(augmented_image, (3, 3), 0)
            
            # Горизонтальное отражение с трансформацией аннотаций
            flip_horizontal = random.random() < 0.5
            if flip_horizontal:
                augmented_image = cv2.flip(augmented_image, 1)
                # Трансформируем аннотации для горизонтального отражения
                self._flip_annotations_horizontal(original_annotation, aug_annotation_path)
            else:
                # Просто копируем аннотацию
                shutil.copy2(original_annotation, aug_annotation_path)
            
            # Сохранение аугментированного изображения
            cv2.imwrite(str(aug_image_path), augmented_image)
            
            self.logger.debug(f"Fallback аугментация выполнена для {original_image}")
            
        except Exception as e:
            self.logger.error(f"Ошибка в fallback аугментации: {e}")
            # В крайнем случае просто копируем файлы
            shutil.copy2(original_image, aug_image_path)
            shutil.copy2(original_annotation, aug_annotation_path)
    
    def _flip_annotations_horizontal(self, original_annotation: Path, output_annotation: Path):
        """
        Трансформация аннотаций для горизонтального отражения
        """
        try:
            with open(original_annotation, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            transformed_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        # Убедимся, что class_id это целое число
                        try:
                            class_id = int(float(parts[0]))  # Сначала float, потом int для обработки "4.0"
                        except ValueError:
                            class_id = int(parts[0])  # Прямое преобразование если это уже int
                            
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Трансформация для горизонтального отражения
                        new_x_center = 1.0 - x_center
                        
                        # Сохраняем class_id как целое число (без .0)
                        transformed_line = f"{class_id} {new_x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        transformed_lines.append(transformed_line)
            
            # Сохранение трансформированных аннотаций
            with open(output_annotation, 'w', encoding='utf-8') as f:
                f.writelines(transformed_lines)
                
            self.logger.debug(f"Трансформировано {len(transformed_lines)} аннотаций для {output_annotation}")
                
        except Exception as e:
            self.logger.warning(f"Ошибка трансформации аннотаций: {e}, копируем оригинал")
            shutil.copy2(original_annotation, output_annotation)
    
    def _validate_annotations(self):
        """
        Валидация аннотаций после аугментации
        """
        self.logger.info("🔍 Проверка качества аннотаций...")
        
        splits = ['train', 'val', 'test']
        validation_stats = {
            'total_images': 0,
            'images_with_annotations': 0,
            'empty_annotations': 0,
            'augmented_with_annotations': 0,
            'original_with_annotations': 0
        }
        
        for split in splits:
            images_dir = self.dataset_dir / split / "images"
            labels_dir = self.dataset_dir / split / "labels"
            
            if not images_dir.exists():
                continue
            
            # Поиск всех изображений
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            
            for image_file in image_files:
                validation_stats['total_images'] += 1
                annotation_file = labels_dir / f"{image_file.stem}.txt"
                
                if annotation_file.exists() and annotation_file.stat().st_size > 0:
                    validation_stats['images_with_annotations'] += 1
                    
                    # Проверяем, аугментированное это изображение или оригинальное
                    if "_aug_" in image_file.name:
                        validation_stats['augmented_with_annotations'] += 1
                    else:
                        validation_stats['original_with_annotations'] += 1
                else:
                    validation_stats['empty_annotations'] += 1
                    
                    # Если это аугментированное изображение без аннотации - проблема
                    if "_aug_" in image_file.name:
                        self.logger.warning(f"Аугментированное изображение без аннотации: {image_file}")
        
        # Логирование результатов валидации
        total = validation_stats['total_images']
        if total > 0:
            ann_rate = (validation_stats['images_with_annotations'] / total) * 100
            self.logger.info(f"📊 Результаты валидации аннотаций:")
            self.logger.info(f"  - Всего изображений: {total}")
            self.logger.info(f"  - С аннотациями: {validation_stats['images_with_annotations']} ({ann_rate:.1f}%)")
            self.logger.info(f"  - Оригинальных с аннотациями: {validation_stats['original_with_annotations']}")
            self.logger.info(f"  - Аугментированных с аннотациями: {validation_stats['augmented_with_annotations']}")
            self.logger.info(f"  - Пустых аннотаций: {validation_stats['empty_annotations']}")
            
            if ann_rate < 50:
                self.logger.warning("⚠️ Низкий процент изображений с аннотациями!")
            
        self.stats['validation_stats'] = validation_stats
    
    def _fix_annotation_format(self):
        """
        Исправление формата аннотаций (float class_id -> int class_id)
        """
        self.logger.info("🔧 Исправление формата аннотаций...")
        
        splits = ['train', 'val', 'test']
        fixed_files = 0
        
        for split in splits:
            labels_dir = self.dataset_dir / split / "labels"
            
            if not labels_dir.exists():
                continue
            
            # Поиск всех файлов аннотаций
            annotation_files = list(labels_dir.glob("*.txt"))
            
            for annotation_file in annotation_files:
                try:
                    if annotation_file.stat().st_size == 0:
                        continue  # Пропускаем пустые файлы
                    
                    # Чтение текущих аннотаций
                    with open(annotation_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Проверка и исправление формата
                    fixed_lines = []
                    needs_fix = False
                    
                    for line in lines:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                try:
                                    # Проверяем, является ли class_id float (например "4.0")
                                    class_id_str = parts[0]
                                    if '.' in class_id_str:
                                        # Конвертируем float в int
                                        class_id = int(float(class_id_str))
                                        needs_fix = True
                                    else:
                                        class_id = int(class_id_str)
                                    
                                    # Остальные координаты
                                    x_center = float(parts[1])
                                    y_center = float(parts[2])
                                    width = float(parts[3])
                                    height = float(parts[4])
                                    
                                    # Создаем исправленную строку
                                    fixed_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                                    fixed_lines.append(fixed_line)
                                    
                                except ValueError as e:
                                    self.logger.warning(f"Не удалось исправить строку '{line}' в {annotation_file}: {e}")
                                    fixed_lines.append(line + '\n' if not line.endswith('\n') else line)
                    
                    # Сохранение исправленного файла если были изменения
                    if needs_fix:
                        with open(annotation_file, 'w', encoding='utf-8') as f:
                            f.writelines(fixed_lines)
                        fixed_files += 1
                        self.logger.debug(f"Исправлен формат аннотаций в {annotation_file}")
                        
                except Exception as e:
                    self.logger.warning(f"Ошибка исправления {annotation_file}: {e}")
                    continue
        
        if fixed_files > 0:
            self.logger.info(f"🔧 Исправлен формат в {fixed_files} файлах аннотаций")
        else:
            self.logger.info("✅ Все аннотации уже в правильном формате")
    
    def _annotate_with_groundingdino(self):
        """Аннотация всех splits с использованием GroundingDINO"""
        from src.data.annotator import SmartAnnotator
        
        # Инициализация аннотатора
        annotator = SmartAnnotator()
        
        splits = ['train', 'val', 'test']
        total_annotations = 0
        
        for split in splits:
            self.logger.info(f"Аннотация {split} набора...")
            
            images_dir = self.dataset_dir / split / "images"
            labels_dir = self.dataset_dir / split / "labels"
            
            if not images_dir.exists():
                self.logger.warning(f"Директория {images_dir} не найдена, пропуск...")
                continue
            
            # Запуск аннотации
            try:
                stats = annotator.annotate_dataset(
                    images_dir=images_dir,
                    output_dir=labels_dir,
                    batch_size=self.config.get('performance', {}).get('batch_size', 1),
                    num_workers=self.config.get('performance', {}).get('num_workers', 1)
                )
                
                annotations_count = stats.get('total_detections', 0)
                total_annotations += annotations_count
                
                self.logger.info(f"✅ {split}: {stats['processed_images']} изображений, "
                               f"{annotations_count} аннотаций")
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка аннотации {split}: {e}")
                continue
        
        self.stats['total_annotations'] = total_annotations
        self.logger.info(f"🎯 Всего создано аннотаций: {total_annotations}")
    
    def run_pipeline(self, input_dir: Path):
        """Запуск полного пайплайна подготовки данных с массивной аугментацией"""
        self.logger.info("🚀 Запуск пайплайна подготовки данных с GroundingDINO и массивной аугментацией")
        self.stats['start_time'] = time.time()
        
        try:
            # 1. Создание структуры директорий
            self.logger.info("🏗️ Этап 1: Создание структуры директорий")
            create_directory_structure()
            self.stats['stages_completed'].append('directory_structure')
            
            # 2. Поиск и валидация видео файлов
            self.logger.info("🔍 Этап 2: Поиск видео файлов")
            video_files = self._find_video_files(input_dir)
            self.stats['total_videos'] = len(video_files)
            
            if not video_files:
                raise ValueError(f"Видео файлы не найдены в директории: {input_dir}")
            
            # 3. Извлечение кадров из видео
            self.logger.info("🎬 Этап 3: Извлечение кадров")
            all_frames = self._extract_frames_from_videos(video_files)
            self.stats['total_frames'] = len(all_frames)
            self.stats['stages_completed'].append('frame_extraction')
            
            # 4. Разделение на train/val/test
            self.logger.info("📂 Этап 4: Разделение на train/val/test")
            self._split_and_organize_dataset(all_frames)
            self.stats['stages_completed'].append('dataset_split')
            
            # 5. Аннотация с GroundingDINO (первоначальная)
            self.logger.info("🧠 Этап 5: Аннотация с GroundingDINO")
            self._annotate_with_groundingdino()
            self.stats['stages_completed'].append('annotation')
            
            # 5.5. МАССИВНАЯ АУГМЕНТАЦИЯ (новый этап)
            self._apply_massive_augmentation()
            self.stats['stages_completed'].append('massive_augmentation')
            
            # 6. Исправление формата аннотаций
            self.logger.info("🔧 Этап 6: Исправление формата аннотаций")
            self._fix_annotation_format()
            self.stats['stages_completed'].append('format_fix')
            
            # 7. Валидация аннотаций после аугментации
            self.logger.info("🔍 Этап 7: Валидация аннотаций")
            self._validate_annotations()
            self.stats['stages_completed'].append('validation')
            
            # 8. Генерация визуализаций с bounding boxes
            self.logger.info("🖼️ Этап 8: Генерация визуализаций")
            self._generate_visualizations()
            self._create_visualization_summary()
            self.stats['stages_completed'].append('visualizations')
            
            # 9. Создание YAML конфигурации
            self.logger.info("📄 Этап 9: Создание dataset.yaml")
            create_dataset_yaml(self.dataset_dir, self.config['dataset']['class_names'])
            self.stats['stages_completed'].append('yaml_creation')
            
            # 8. Очистка временных файлов
            self.logger.info("🧹 Этап 8: Очистка временных файлов")
            self._cleanup_temp_files()
            
            # 9. Генерация отчета
            self._generate_completion_report()
            
            self.logger.info("🎉 Пайплайн подготовки данных с массивной аугментацией завершен успешно!")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в пайплайне: {e}")
            self._generate_error_report(e)
            raise
    
    def _annotate_augmented_images(self):
        """
        Аннотация только новых аугментированных изображений (оптимизация)
        """
        from src.data.annotator import SmartAnnotator
        
        self.logger.info("🔄 Аннотация аугментированных изображений...")
        
        # Инициализация аннотатора
        annotator = SmartAnnotator()
        
        splits = ['train', 'val', 'test']
        total_new_annotations = 0
        
        for split in splits:
            images_dir = self.dataset_dir / split / "images"
            labels_dir = self.dataset_dir / split / "labels"
            
            if not images_dir.exists():
                continue
            
            # Поиск только аугментированных изображений
            augmented_images = list(images_dir.glob("*_aug_*.jpg")) + list(images_dir.glob("*_aug_*.png"))
            
            if not augmented_images:
                self.logger.info(f"  {split}: нет аугментированных изображений для аннотации")
                continue
            
            self.logger.info(f"  {split}: найдено {len(augmented_images)} аугментированных изображений")
            
            # Аннотация только аугментированных изображений
            annotated_count = 0
            
            for aug_image in tqdm(augmented_images, desc=f"Аннотация {split} аугментированных"):
                aug_annotation = labels_dir / f"{aug_image.stem}.txt"
                
                # Проверяем, есть ли уже аннотация
                if aug_annotation.exists() and aug_annotation.stat().st_size > 0:
                    continue  # Аннотация уже есть
                
                try:
                    # Аннотация одного изображения
                    detections = annotator._process_single_frame(aug_image)
                    
                    if detections:
                        # Сохранение аннотации
                        annotator._save_yolo_annotation(detections, aug_annotation, aug_image)
                        annotated_count += 1
                        total_new_annotations += len(detections)
                    else:
                        # Создание пустой аннотации
                        aug_annotation.touch()
                        
                except Exception as e:
                    self.logger.warning(f"Ошибка аннотации {aug_image}: {e}")
                    # Создание пустой аннотации в случае ошибки
                    aug_annotation.touch()
            
            self.logger.info(f"  ✅ {split}: аннотировано {annotated_count} аугментированных изображений")
        
        self.stats['total_annotations'] += total_new_annotations
        self.logger.info(f"🎯 Создано дополнительных аннотаций: {total_new_annotations}")
    
    def _generate_visualizations(self):
        """
        Генерация изображений с визуализированными bounding boxes для всех splits
        """
        self.logger.info("🖼️ Этап 6.5: Генерация визуализаций с bounding boxes")
        
        # Цвета для разных классов (RGB)
        class_colors = {
            'chicken': (255, 165, 0),    # Оранжевый
            'meat': (139, 69, 19),       # Коричневый
            'salad': (0, 128, 0),        # Зеленый
            'soup': (255, 215, 0),       # Золотой
            'cup': (70, 130, 180),       # Стальной голубой
            'plate': (220, 220, 220),    # Светло-серый
            'bowl': (255, 192, 203),     # Розовый
            'spoon': (192, 192, 192),    # Серебряный
            'fork': (169, 169, 169),     # Темно-серый
            'knife': (105, 105, 105)     # Тускло-серый
        }
        
        class_names = self.config['dataset']['class_names']
        splits = ['train', 'val', 'test']
        total_visualizations = 0
        
        for split in splits:
            self.logger.info(f"🎨 Создание визуализаций для {split} набора...")
            
            images_dir = self.dataset_dir / split / "images"
            labels_dir = self.dataset_dir / split / "labels"
            visualizations_dir = self.dataset_dir / split / "visualizations"
            
            if not images_dir.exists():
                continue
            
            # Поиск всех изображений
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            split_visualizations = 0
            
            for image_file in tqdm(image_files, desc=f"Визуализация {split}"):
                try:
                    # Соответствующий файл аннотации
                    annotation_file = labels_dir / f"{image_file.stem}.txt"
                    
                    # Загрузка изображения
                    image = cv2.imread(str(image_file))
                    if image is None:
                        continue
                    
                    height, width = image.shape[:2]
                    
                    # Чтение аннотаций
                    if annotation_file.exists() and annotation_file.stat().st_size > 0:
                        with open(annotation_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        # Отрисовка каждого bounding box
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                parts = line.split()
                                if len(parts) != 5:
                                    continue
                                
                                # Правильная обработка class_id для избежания float->int ошибок
                                try:
                                    class_id = int(float(parts[0]))  # Обрабатываем "4.0" как 4
                                except ValueError:
                                    class_id = int(parts[0])  # Прямое преобразование если это уже int
                                    
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                bbox_width = float(parts[3])
                                bbox_height = float(parts[4])
                                
                                # Валидация class_id
                                if 0 <= class_id < len(class_names):
                                    class_name = class_names[class_id]
                                    color = class_colors.get(class_name, (255, 255, 255))  # Белый по умолчанию
                                    
                                    # Конвертация из нормализованных координат в пиксели
                                    x1 = int((x_center - bbox_width/2) * width)
                                    y1 = int((y_center - bbox_height/2) * height)
                                    x2 = int((x_center + bbox_width/2) * width)
                                    y2 = int((y_center + bbox_height/2) * height)
                                    
                                    # Обеспечение границ изображения
                                    x1 = max(0, min(x1, width-1))
                                    y1 = max(0, min(y1, height-1))
                                    x2 = max(0, min(x2, width-1))
                                    y2 = max(0, min(y2, height-1))
                                    
                                    # Отрисовка прямоугольника
                                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                                    
                                    # Подготовка текста с названием класса
                                    label_text = f"{class_name}"
                                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                    
                                    # Фон для текста
                                    cv2.rectangle(image, 
                                                (x1, y1 - label_size[1] - 10), 
                                                (x1 + label_size[0], y1), 
                                                color, -1)
                                    
                                    # Текст
                                    cv2.putText(image, label_text, 
                                              (x1, y1 - 5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                              (255, 255, 255), 2)
                                else:
                                    self.logger.warning(f"Неверный class_id {class_id} в {annotation_file}")
                                
                            except (ValueError, IndexError) as e:
                                self.logger.warning(f"Ошибка парсинга аннотации в {annotation_file}: {e}")
                                continue
                    
                    # Сохранение визуализированного изображения
                    visualization_path = visualizations_dir / f"{image_file.stem}_visualized{image_file.suffix}"
                    cv2.imwrite(str(visualization_path), image)
                    split_visualizations += 1
                    
                except Exception as e:
                    self.logger.warning(f"Ошибка создания визуализации для {image_file}: {e}")
                    continue
            
            total_visualizations += split_visualizations
            self.logger.info(f"  ✅ {split}: создано {split_visualizations} визуализаций")
        
        self.stats['total_visualizations'] = total_visualizations
        self.logger.info(f"🖼️ Всего создано визуализаций: {total_visualizations}")
        self.logger.info(f"📁 Визуализации сохранены в директориях visualizations/ для каждого split")
    
    def _create_visualization_summary(self):
        """
        Создание сводной информации о визуализациях
        """
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'visualization_info': {
                'total_visualizations': self.stats.get('total_visualizations', 0),
                'class_colors': {
                    'chicken': '#FFA500',    # Оранжевый
                    'meat': '#8B4513',       # Коричневый
                    'salad': '#008000',      # Зеленый
                    'soup': '#FFD700',       # Золотой
                    'cup': '#4682B4',        # Стальной голубой
                    'plate': '#DCDCDC',      # Светло-серый
                    'bowl': '#FFC0CB',       # Розовый
                    'spoon': '#C0C0C0',      # Серебряный
                    'fork': '#A9A9A9',       # Темно-серый
                    'knife': '#696969'       # Тускло-серый
                },
                'splits': {}
            },
            'usage_instructions': [
                "Откройте директории train/visualizations/, val/visualizations/, test/visualizations/",
                "Каждое изображение показывает оригинал с отрисованными bounding boxes",
                "Цвета соответствуют разным классам объектов",
                "Используйте для проверки качества аннотаций",
                "Файлы названы как: original_name_visualized.jpg"
            ]
        }
        
        # Подсчет визуализаций по splits
        for split in ['train', 'val', 'test']:
            vis_dir = self.dataset_dir / split / "visualizations"
            if vis_dir.exists():
                vis_count = len(list(vis_dir.glob("*_visualized.*")))
                summary['visualization_info']['splits'][split] = vis_count
        
        # Сохранение сводки
        summary_path = self.dataset_dir / "visualization_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📋 Сводка по визуализациям сохранена: {summary_path}")
    
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
            'annotation_method': 'GroundingDINO',
            'statistics': {
                'total_videos_processed': self.stats['total_videos'],
                'total_frames_processed': self.stats['total_frames'],
                'total_annotations_created': self.stats['total_annotations'],
                'total_augmented_images': self.stats.get('total_augmented_images', 0),
                'total_visualizations': self.stats.get('total_visualizations', 0),
                'augmentation_factor': self.stats.get('augmentation_factor', 1.0),
                'stages_completed': self.stats['stages_completed']
            },
            'output_structure': {
                'dataset_directory': str(self.dataset_dir),
                'train_images': len(list((self.dataset_dir / 'train' / 'images').glob('*'))),
                'val_images': len(list((self.dataset_dir / 'val' / 'images').glob('*'))),
                'test_images': len(list((self.dataset_dir / 'test' / 'images').glob('*'))),
                'train_visualizations': len(list((self.dataset_dir / 'train' / 'visualizations').glob('*'))) if (self.dataset_dir / 'train' / 'visualizations').exists() else 0,
                'val_visualizations': len(list((self.dataset_dir / 'val' / 'visualizations').glob('*'))) if (self.dataset_dir / 'val' / 'visualizations').exists() else 0,
                'test_visualizations': len(list((self.dataset_dir / 'test' / 'visualizations').glob('*'))) if (self.dataset_dir / 'test' / 'visualizations').exists() else 0
            },
            'classes_used': self.config['dataset']['class_names'],
            'groundingdino_config': {
                'checkpoint': self.config['annotation']['groundingdino_checkpoint'],
                'prompt': self.config['annotation']['detection_prompt'],
                'confidence_threshold': self.config['annotation']['confidence_threshold']
            },
            'next_steps': [
                "Запустите train_model.py для обучения модели",
                "Проверьте dataset.yaml в data/processed/dataset/",
                "При необходимости отредактируйте аннотации вручную",
                "Убедитесь, что файл groundingdino_swinb_cogcoor.pth находится в корне проекта"
            ]
        }
        
        # Сохранение отчета
        report_path = Path("data/processed/preparation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📋 Отчет сохранен: {report_path}")
        self.logger.info(f"⏱️ Время выполнения: {total_time:.2f} секунд")
        self.logger.info(f"📊 Обработано {self.stats['total_frames']} кадров")
        self.logger.info(f"🎯 Создано {self.stats['total_annotations']} аннотаций")
    
    def _generate_error_report(self, error: Exception):
        """Генерация отчета об ошибке"""
        total_time = time.time() - self.stats['start_time']
        
        error_report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pipeline_status': 'failed',
            'error_message': str(error),
            'error_type': type(error).__name__,
            'execution_time_seconds': round(total_time, 2),
            'stages_completed': self.stats['stages_completed'],
            'statistics_before_error': {
                'total_videos': self.stats['total_videos'],
                'total_frames': self.stats['total_frames'],
                'total_annotations': self.stats['total_annotations']
            },
            'troubleshooting_tips': [
                "Убедитесь, что файл groundingdino_swinb_cogcoor.pth находится в корне проекта",
                "Проверьте установку groundingdino-py: pip install groundingdino-py",
                "Убедитесь, что есть видео файлы в директории data/raw/",
                "Проверьте наличие свободного места на диске",
                "При проблемах с GPU попробуйте использовать CPU режим"
            ]
        }
        
        # Сохранение отчета об ошибке
        error_report_path = Path("data/processed/error_report.json")
        with open(error_report_path, 'w', encoding='utf-8') as f:
            json.dump(error_report, f, ensure_ascii=False, indent=2)
        
        self.logger.error(f"💥 Отчет об ошибке сохранен: {error_report_path}")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="Подготовка данных для YOLO11 с GroundingDINO аннотацией",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

1. Базовое использование:
   python scripts/prepare_data.py --input "data/raw"

2. С кастомной конфигурацией:
   python scripts/prepare_data.py --input "data/raw" --config "config/pipeline_config.json"

3. С изменением параметров извлечения:
   python scripts/prepare_data.py --input "data/raw" --fps 1.5

Требования:
- Файл groundingdino_swinb_cogcoor.pth в корне проекта
- Установленный groundingdino-py: pip install groundingdino-py
- Видео файлы в указанной input директории

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
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Порог уверенности для GroundingDINO (по умолчанию: 0.25)'
    )
    
    args = parser.parse_args()
    
    try:
        # Проверка наличия GroundingDINO модели
        groundingdino_path = Path("groundingdino_swinb_cogcoor.pth")
        if not groundingdino_path.exists():
            print("\n❌ ОШИБКА: Файл модели GroundingDINO не найден!")
            print(f"Ожидается файл: {groundingdino_path.absolute()}")
            print("\nСкачайте модель:")
            print("wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swinb_cogcoor.pth")
            sys.exit(1)
        
        # Проверка входной директории
        input_dir = Path(args.input)
        if not input_dir.exists():
            print(f"\n❌ ОШИБКА: Входная директория не найдена: {input_dir}")
            print("Создайте директорию и поместите в нее видео файлы")
            sys.exit(1)
        
        # Загрузка конфигурации
        config_path = Path(args.config) if args.config else None
        config = load_config(config_path)
        
        # Переопределение параметров из аргументов
        if args.fps != 2.0:
            config['video_processing']['fps_extraction'] = args.fps
        
        if args.confidence != 0.25:
            config['annotation']['confidence_threshold'] = args.confidence
            config['annotation']['text_threshold'] = args.confidence
            config['annotation']['box_threshold'] = args.confidence
        
        # Включение массивной аугментации по умолчанию
        if 'augmentation' not in config:
            config['augmentation'] = {'enabled': True}
        
        config['dataset']['enable_massive_augmentation'] = True
        
        # Создание и запуск процессора
        processor = DataPipelineProcessor(config)
        processor.dataset_dir = Path(args.output)
        
        # Запуск пайплайна
        processor.run_pipeline(input_dir)
        
        print("\n" + "="*60)
        print("🎉 ПОДГОТОВКА ДАННЫХ С GROUNDINGDINO ЗАВЕРШЕНА УСПЕШНО!")
        print("="*60)
        print(f"📁 Датасет создан в: {args.output}")
        print(f"⚙️ Конфигурация: {args.output}/dataset.yaml")
        print(f"📋 Отчет: data/processed/preparation_report.json")
        print(f"🧠 Метод аннотации: GroundingDINO")
        print(f"🎯 Классы: chicken, meat, salad, soup, cup, plate, bowl, spoon, fork, knife")
        print("\n🚀 Следующий шаг: запустите train_model.py")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        print("📋 Проверьте error_report.json для деталей")
        print("\n💡 Частые проблемы:")
        print("- Отсутствует файл groundingdino_swinb_cogcoor.pth")
        print("- Не установлен groundingdino-py")
        print("- Нет видео файлов в входной директории")
        print("- Недостаточно места на диске")
        sys.exit(1)


if __name__ == "__main__":
    main()