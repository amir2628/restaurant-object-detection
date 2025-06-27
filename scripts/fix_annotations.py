"""
Скрипт для исправления и создания аннотаций с использованием GroundingDINO
Автоматически создает высококачественные аннотации для объектов еды и посуды
"""

import sys
import argparse
import time
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

# Добавление корневой директории в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger


def check_dataset_structure(dataset_dir: Path) -> bool:
    """Проверка корректности структуры датасета YOLO"""
    logger = setup_logger(__name__)
    
    required_dirs = [
        dataset_dir / "train" / "images",
        dataset_dir / "train" / "labels",
        dataset_dir / "val" / "images", 
        dataset_dir / "val" / "labels",
        dataset_dir / "test" / "images",
        dataset_dir / "test" / "labels"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not dir_path.exists():
            missing_dirs.append(str(dir_path))
    
    if missing_dirs:
        logger.warning(f"Отсутствующие директории: {missing_dirs}")
        return False
    
    logger.info("✅ Структура датасета корректна")
    return True


def create_dataset_structure(dataset_dir: Path):
    """Создание структуры директорий датасета"""
    logger = setup_logger(__name__)
    
    directories = [
        "train/images", "train/labels",
        "val/images", "val/labels", 
        "test/images", "test/labels"
    ]
    
    for directory in directories:
        dir_path = dataset_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Создана директория: {dir_path}")
    
    logger.info("🏗️ Структура датасета создана")


def create_dataset_yaml_with_groundingdino_classes(dataset_dir: Path):
    """Создание dataset.yaml с классами GroundingDINO"""
    logger = setup_logger(__name__)
    
    # Классы для GroundingDINO (специализированные для ресторанной среды)
    restaurant_classes = [
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
        logger.info(f"📋 Классы GroundingDINO ({len(restaurant_classes)}): {', '.join(restaurant_classes)}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания dataset.yaml: {e}")


def auto_annotate_with_groundingdino(image_path: Path, restaurant_classes: List[str], 
                                    confidence_threshold: float = 0.25) -> List[Dict]:
    """
    Автоматическая аннотация изображения с использованием GroundingDINO
    
    Args:
        image_path: Путь к изображению
        restaurant_classes: Список классов для детекции
        confidence_threshold: Порог уверенности
        
    Returns:
        Список детекций в формате YOLO
    """
    logger = setup_logger(__name__)
    
    try:
        # Импорт GroundingDINO
        from groundingdino.util.inference import load_model, predict, load_image
        import torch
        import os
        
        # Проверка наличия модели
        checkpoint_path = "groundingdino_swinb_cogcoor.pth"
        if not os.path.exists(checkpoint_path):
            logger.warning(f"❌ Файл модели GroundingDINO не найден: {checkpoint_path}")
            return []
        
        # Загрузка модели (кешируем глобально для производительности)
        if not hasattr(auto_annotate_with_groundingdino, 'model'):
            # Попытка загрузки с конфигурацией
            config_paths = [
                "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                "groundingdino_config.py"
            ]
            
            model = None
            for config_path in config_paths:
                if os.path.exists(config_path):
                    try:
                        model = load_model(config_path, checkpoint_path)
                        logger.info(f"✅ GroundingDINO загружен с конфигом: {config_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Ошибка загрузки с конфигом {config_path}: {e}")
                        continue
            
            if model is None:
                # Загрузка без конфигурации
                try:
                    model = load_model(checkpoint_path)
                    logger.info("✅ GroundingDINO загружен без конфигурации")
                except Exception as e:
                    logger.error(f"❌ Не удалось загрузить GroundingDINO: {e}")
                    return []
            
            auto_annotate_with_groundingdino.model = model
        
        model = auto_annotate_with_groundingdino.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Создание промпта
        prompt = " . ".join(restaurant_classes) + " ."
        
        # Загрузка и обработка изображения
        image_source, image = load_image(str(image_path))
        
        # Выполнение детекции
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=confidence_threshold,
            text_threshold=confidence_threshold,
            device=device
        )
        
        # Обработка результатов
        detections = []
        
        if len(boxes) > 0:
            # Конвертация тензоров
            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu().numpy()
            if hasattr(logits, 'cpu'):
                logits = logits.cpu().numpy()
            
            # Получение размеров изображения
            from PIL import Image
            img = Image.open(image_path)
            img_width, img_height = img.size
            img.close()
            
            for i, (box, confidence, phrase) in enumerate(zip(boxes, logits, phrases)):
                # Маппинг фразы на класс
                phrase_lower = str(phrase).lower().strip()
                mapped_class = None
                
                for class_name in restaurant_classes:
                    if class_name.lower() in phrase_lower:
                        mapped_class = class_name
                        break
                
                if mapped_class and confidence >= confidence_threshold:
                    # Получение ID класса
                    class_id = restaurant_classes.index(mapped_class)
                    
                    # Нормализация координат (GroundingDINO возвращает нормализованные координаты)
                    x_center, y_center, width, height = box[:4]
                    
                    # Валидация координат
                    if (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                        0 < width <= 1 and 0 < height <= 1):
                        
                        detection = {
                            'class_name': mapped_class,
                            'class_id': class_id,
                            'confidence': float(confidence),
                            'bbox': [x_center, y_center, width, height]
                        }
                        detections.append(detection)
        
        return detections
        
    except ImportError:
        logger.error("❌ GroundingDINO не установлен. Установите: pip install groundingdino-py")
        return []
    except Exception as e:
        logger.error(f"❌ Ошибка в автоматической аннотации: {e}")
        return []


def remove_duplicate_detections(detections: List[Dict], iou_threshold: float = 0.6) -> List[Dict]:
    """Удаление дублирующихся детекций"""
    if len(detections) <= 1:
        return detections
    
    # Простой алгоритм удаления дубликатов по IoU
    filtered_detections = []
    
    for i, detection in enumerate(detections):
        is_duplicate = False
        
        for j, other_detection in enumerate(filtered_detections):
            # Расчет IoU между боксами
            box1 = detection['bbox']
            box2 = other_detection['bbox']
            
            # Конвертация из центр + размер в углы
            def center_to_corners(bbox):
                x_center, y_center, width, height = bbox
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                return [x1, y1, x2, y2]
            
            corners1 = center_to_corners(box1)
            corners2 = center_to_corners(box2)
            
            # Расчет пересечения
            x1 = max(corners1[0], corners2[0])
            y1 = max(corners1[1], corners2[1])
            x2 = min(corners1[2], corners2[2])
            y2 = min(corners1[3], corners2[3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = box1[2] * box1[3]
                area2 = box2[2] * box2[3]
                union = area1 + area2 - intersection
                
                iou = intersection / union if union > 0 else 0
                
                if iou > iou_threshold:
                    # Оставляем детекцию с большей уверенностью
                    if detection['confidence'] <= other_detection['confidence']:
                        is_duplicate = True
                        break
                    else:
                        # Заменяем старую детекцию новой
                        filtered_detections[j] = detection
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            filtered_detections.append(detection)
    
    return filtered_detections


class AnnotationFixer:
    """Основной класс для исправления аннотаций с GroundingDINO"""
    
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
        """Конфигурация по умолчанию с классами GroundingDINO"""
        return {
            'restaurant_classes': [
                'chicken', 'meat', 'salad', 'soup', 'cup',
                'plate', 'bowl', 'spoon', 'fork', 'knife'
            ],
            'auto_annotation': {
                'enabled': True,
                'confidence_threshold': 0.25,
                'method': 'groundingdino',
                'checkpoint_path': 'groundingdino_swinb_cogcoor.pth'
            },
            'processing': {
                'create_structure_if_missing': True,
                'overwrite_existing': False,
                'splits_to_process': ['train', 'val', 'test']
            }
        }
    
    def run_fix_process(self):
        """Запуск процесса исправления аннотаций"""
        self.logger.info("🔧 Запуск процесса исправления аннотаций с GroundingDINO")
        
        try:
            # 1. Проверка и создание структуры
            if not check_dataset_structure(self.dataset_dir):
                if self.config['processing']['create_structure_if_missing']:
                    create_dataset_structure(self.dataset_dir)
                else:
                    raise ValueError("Структура датасета некорректна")
            
            # 2. Создание/обновление dataset.yaml
            create_dataset_yaml_with_groundingdino_classes(self.dataset_dir)
            
            # 3. Обработка каждого split
            for split in self.config['processing']['splits_to_process']:
                self._process_split(split)
            
            # 4. Генерация финального отчета
            self._generate_report()
            
            self.logger.info("✅ Процесс исправления аннотаций завершен успешно!")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в процессе исправления: {e}")
            raise
    
    def _process_split(self, split: str):
        """Обработка одного split (train/val/test)"""
        self.logger.info(f"📂 Обработка {split} набора...")
        
        images_dir = self.dataset_dir / split / "images"
        labels_dir = self.dataset_dir / split / "labels"
        
        if not images_dir.exists():
            self.logger.warning(f"Директория изображений не найдена: {images_dir}")
            return
        
        # Поиск всех изображений
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            self.logger.warning(f"Изображения не найдены в {images_dir}")
            return
        
        self.logger.info(f"Найдено {len(image_files)} изображений в {split}")
        
        # Обработка каждого изображения
        processed_count = 0
        annotated_count = 0
        
        for image_path in image_files:
            try:
                annotation_path = labels_dir / f"{image_path.stem}.txt"
                
                # Проверка существования аннотации
                needs_annotation = (
                    not annotation_path.exists() or
                    annotation_path.stat().st_size == 0 or
                    self.config['processing']['overwrite_existing']
                )
                
                if needs_annotation and self.config['auto_annotation']['enabled']:
                    # Автоматическая аннотация с GroundingDINO
                    detections = auto_annotate_with_groundingdino(
                        image_path,
                        self.config['restaurant_classes'],
                        self.config['auto_annotation']['confidence_threshold']
                    )
                    
                    # Удаление дубликатов
                    detections = remove_duplicate_detections(detections)
                    
                    # Сохранение аннотации
                    self._save_yolo_annotation(detections, annotation_path)
                    
                    if detections:
                        annotated_count += 1
                        self.logger.debug(f"✅ Аннотировано {len(detections)} объектов в {image_path.name}")
                    else:
                        # Создание пустого файла аннотации
                        annotation_path.touch()
                        self.logger.debug(f"📝 Создана пустая аннотация для {image_path.name}")
                
                elif not annotation_path.exists():
                    # Создание пустого файла если автоаннотация отключена
                    annotation_path.touch()
                    self.logger.debug(f"📝 Создана пустая аннотация для {image_path.name}")
                
                processed_count += 1
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка обработки {image_path}: {e}")
                self.stats['total_errors'] += 1
                continue
        
        self.stats['total_processed'] += processed_count
        self.stats['total_annotated'] += annotated_count
        self.stats['splits_processed'].append(split)
        
        self.logger.info(f"✅ {split}: обработано {processed_count}, аннотировано {annotated_count}")
    
    def _save_yolo_annotation(self, detections: List[Dict], annotation_path: Path):
        """Сохранение аннотации в формате YOLO"""
        try:
            with open(annotation_path, 'w', encoding='utf-8') as f:
                for detection in detections:
                    bbox = detection['bbox']
                    class_id = detection['class_id']
                    
                    # Формат YOLO: class_id x_center y_center width height
                    line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                    f.write(line)
                    
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения аннотации {annotation_path}: {e}")
            # Создание пустого файла в случае ошибки
            annotation_path.touch()
    
    def _generate_report(self):
        """Генерация отчета о работе"""
        total_time = time.time() - self.stats['start_time']
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'method': 'GroundingDINO',
            'execution_time_seconds': round(total_time, 2),
            'statistics': {
                'total_processed': self.stats['total_processed'],
                'total_annotated': self.stats['total_annotated'],
                'total_errors': self.stats['total_errors'],
                'splits_processed': self.stats['splits_processed']
            },
            'configuration': {
                'classes_used': self.config['restaurant_classes'],
                'confidence_threshold': self.config['auto_annotation']['confidence_threshold'],
                'auto_annotation_enabled': self.config['auto_annotation']['enabled']
            },
            'dataset_structure': {
                'dataset_directory': str(self.dataset_dir),
                'yaml_config': str(self.dataset_dir / "dataset.yaml")
            }
        }
        
        # Сохранение отчета
        report_path = self.dataset_dir / "annotation_fix_report.json"
        
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📋 Отчет сохранен: {report_path}")
        self.logger.info(f"⏱️ Время выполнения: {total_time:.2f} секунд")
        self.logger.info(f"📊 Статистика: обработано {self.stats['total_processed']}, "
                        f"аннотировано {self.stats['total_annotated']}")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="Исправление аннотаций с использованием GroundingDINO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

1. Базовое исправление (создание пустых аннотаций):
   python scripts/fix_annotations.py --dataset "data/processed/dataset"

2. С автоматической аннотацией GroundingDINO:
   python scripts/fix_annotations.py --dataset "data/processed/dataset" --auto-annotate

3. Создание структуры и аннотаций:
   python scripts/fix_annotations.py --dataset "data/processed/dataset" --create-structure --auto-annotate

4. Только для train split:
   python scripts/fix_annotations.py --dataset "data/processed/dataset" --splits train

5. С настройкой порога уверенности:
   python scripts/fix_annotations.py --dataset "data/processed/dataset" --auto-annotate --confidence 0.3

Требования для автоаннотации:
- Файл groundingdino_swinb_cogcoor.pth в корне проекта
- Установленный groundingdino-py: pip install groundingdino-py

Что делает скрипт:
- Проверяет структуру датасета YOLO
- Создает отсутствующие директории
- Генерирует аннотации для всех изображений с GroundingDINO
- Создает dataset.yaml с классами еды и посуды
- Предоставляет детальные отчеты

Классы для GroundingDINO:
chicken, meat, salad, soup, cup, plate, bowl, spoon, fork, knife
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
        help='Включить автоматическую аннотацию с помощью GroundingDINO'
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
        help='Порог уверенности для GroundingDINO (по умолчанию: 0.25)'
    )
    
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Список splits для обработки (по умолчанию: train val test)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Перезаписать существующие аннотации'
    )
    
    args = parser.parse_args()
    
    try:
        # Проверка наличия модели GroundingDINO если требуется автоаннотация
        if args.auto_annotate:
            groundingdino_path = Path("groundingdino_swinb_cogcoor.pth")
            if not groundingdino_path.exists():
                print("\n❌ ОШИБКА: Для автоаннотации требуется файл модели GroundingDINO!")
                print(f"Ожидается файл: {groundingdino_path.absolute()}")
                print("\nСкачайте модель:")
                print("wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swinb_cogcoor.pth")
                sys.exit(1)
        
        # Настройка конфигурации
        config = {
            'restaurant_classes': [
                'chicken', 'meat', 'salad', 'soup', 'cup',
                'plate', 'bowl', 'spoon', 'fork', 'knife'
            ],
            'auto_annotation': {
                'enabled': args.auto_annotate,
                'confidence_threshold': args.confidence,
                'method': 'groundingdino',
                'checkpoint_path': 'groundingdino_swinb_cogcoor.pth'
            },
            'processing': {
                'create_structure_if_missing': args.create_structure,
                'overwrite_existing': args.overwrite,
                'splits_to_process': args.splits
            }
        }
        
        # Создание и запуск фиксера
        fixer = AnnotationFixer(
            dataset_dir=Path(args.dataset),
            config=config
        )
        
        fixer.run_fix_process()
        
        print("\n" + "="*60)
        print("🎉 ИСПРАВЛЕНИЕ АННОТАЦИЙ С GROUNDINGDINO ЗАВЕРШЕНО!")
        print("="*60)
        print(f"📁 Датасет: {args.dataset}")
        print(f"📄 Конфигурация: {args.dataset}/dataset.yaml")
        print(f"📋 Отчет: {args.dataset}/annotation_fix_report.json")
        
        if args.auto_annotate:
            print(f"🧠 Метод аннотации: GroundingDINO")
            print(f"🎯 Классы: chicken, meat, salad, soup, cup, plate, bowl, spoon, fork, knife")
            print(f"📊 Порог уверенности: {args.confidence}")
        else:
            print("📝 Созданы пустые файлы аннотаций")
        
        print(f"📂 Обработанные splits: {', '.join(args.splits)}")
        print("\n🚀 Следующий шаг: запустите train_model.py")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        print("\n💡 Частые проблемы:")
        print("- Отсутствует файл groundingdino_swinb_cogcoor.pth")
        print("- Не установлен groundingdino-py")
        print("- Некорректная структура датасета")
        print("- Недостаточно прав доступа к файлам")
        sys.exit(1)


if __name__ == "__main__":
    main()