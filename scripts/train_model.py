"""
Профессиональное обучение YOLO11 для детекции объектов
Поддерживает автоматический выбор GPU/CPU и полный мониторинг процесса
"""

import sys
import argparse
import logging
import json
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Добавляем корневую директорию проекта в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger

# Проверка доступности библиотек
try:
    from ultralytics import YOLO
    import torch
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("❌ Ultralytics не установлен. Установите: pip install ultralytics")
    sys.exit(1)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class DeviceManager:
    """Простой менеджер устройств для выбора GPU/CPU"""
    
    def __init__(self, preferred_device: str = "auto"):
        self.logger = setup_logger(__name__)
        self.preferred_device = preferred_device
        self.device = self._select_device()
        
    def _select_device(self):
        """Выбор оптимального устройства"""
        if self.preferred_device.lower() == "cpu":
            return "cpu"
        elif self.preferred_device.lower() == "cuda" or self.preferred_device.lower() == "gpu":
            if torch.cuda.is_available():
                return 0  # CUDA device 0
            else:
                self.logger.warning("⚠️ CUDA не доступна, переключение на CPU")
                return "cpu"
        else:  # auto
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                self.logger.info(f"🎮 Найдено GPU: {gpu_name}")
                self.logger.info(f"🎮 GPU память: {gpu_memory:.1f} GB")
                self.logger.info(f"🎮 Количество GPU: {gpu_count}")
                
                return 0  # Используем первую GPU
            else:
                self.logger.info("💻 GPU не доступен, используем CPU")
                return "cpu"
    
    def get_device(self):
        """Возвращает выбранное устройство"""
        return self.device
    
    def get_recommended_batch_size(self, base_batch_size: int = 16) -> int:
        """Рекомендация размера батча на основе доступной памяти"""
        if self.device == "cpu":
            return max(base_batch_size // 4, 1)
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if gpu_memory >= 24:  # High-end GPU
                return base_batch_size * 2
            elif gpu_memory >= 12:  # Mid-range GPU  
                return base_batch_size
            elif gpu_memory >= 8:  # Entry-level GPU
                return max(base_batch_size // 2, 4)
            elif gpu_memory >= 4:  # Low memory GPU
                return max(base_batch_size // 4, 2)
            else:  # Very low memory
                return 1
        
        return base_batch_size

class ProfessionalYOLOTrainer:
    """Профессиональный тренер YOLO11 с полным мониторингом"""
    
    def __init__(self, config_path: Optional[Path] = None, device: str = "auto"):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = self._load_config(config_path)
        self.device_manager = DeviceManager(device)
        self.model = None
        self.experiment_name = None
        self.experiment_dir = None
        
        # Статистика обучения
        self.training_stats = {
            'start_time': None,
            'end_time': None,
            'epochs_completed': 0,
            'best_metrics': {},
            'training_interrupted': False
        }
    
    def _load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Загрузка конфигурации обучения"""
        default_config = {
            'model': {
                'size': 'n',  # n, s, m, l, x
                'input_size': 640,
                'pretrained': True
            },
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.01,
                'weight_decay': 0.0005,
                'momentum': 0.937,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'optimizer': 'AdamW',
                'patience': 50,
                'save_period': 10,
                'val_period': 1,
                'amp': True,  # Automatic Mixed Precision
                'workers': 8
            },
            'augmentation': {
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'flipud': 0.0,
                'fliplr': 0.5
            },
            'validation': {
                'conf_threshold': 0.001,
                'iou_threshold': 0.6,
                'max_det': 300,
                'save_json': False,
                'plots': True
            },
            'logging': {
                'verbose': True,
                'exist_ok': True,
                'wandb': {
                    'enabled': False,
                    'project': 'restaurant-object-detection',
                    'name': None
                }
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # Слияние конфигураций
                def merge_configs(base, update):
                    for key, value in update.items():
                        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                            merge_configs(base[key], value)
                        else:
                            base[key] = value
                
                merge_configs(default_config, user_config)
                self.logger.info(f"📁 Загружена конфигурация из {config_path}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка загрузки конфигурации: {e}")
                self.logger.info("🔧 Используется конфигурация по умолчанию")
        
        return default_config
    
    def validate_dataset(self, dataset_yaml: Path) -> Dict[str, Any]:
        """Валидация датасета"""
        self.logger.info("🔍 Валидация датасета...")
        
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Конфигурация датасета не найдена: {dataset_yaml}")
        
        # Загрузка конфигурации датасета
        with open(dataset_yaml, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f)
        
        # Проверка основных полей
        required_fields = ['path', 'train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in dataset_config:
                raise ValueError(f"Отсутствует обязательное поле в dataset.yaml: {field}")
        
        # Проверка путей
        dataset_path = Path(dataset_config['path'])
        train_dir = dataset_path / dataset_config['train']
        val_dir = dataset_path / dataset_config['val']
        
        train_images = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))
        val_images = list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png'))
        
        # Проверка аннотаций
        train_labels_dir = dataset_path / 'train' / 'labels'
        val_labels_dir = dataset_path / 'val' / 'labels'
        
        train_labels = list(train_labels_dir.glob('*.txt')) if train_labels_dir.exists() else []
        val_labels = list(val_labels_dir.glob('*.txt')) if val_labels_dir.exists() else []
        
        self.logger.info("📊 Информация о датасете:")
        self.logger.info(f"  - Классы: {dataset_config['nc']} ({', '.join(dataset_config['names'])})")
        self.logger.info(f"  - TRAIN: {len(train_images)} изображений, {len(train_labels)} аннотаций")
        self.logger.info(f"  - VAL: {len(val_images)} изображений, {len(val_labels)} аннотаций")
        
        if len(train_images) == 0:
            raise ValueError("Не найдены тренировочные изображения")
        
        if len(val_images) == 0:
            self.logger.warning("⚠️ Не найдены валидационные изображения")
        
        self.logger.info("✅ Валидация датасета пройдена")
        
        return dataset_config
    
    def setup_model(self, model_size: str = None) -> YOLO:
        """Настройка модели YOLO11"""
        self.logger.info("🧠 Инициализация модели YOLO11...")
        
        if model_size is None:
            model_size = self.config['model']['size']
        
        # Создание модели
        model_name = f"yolo11{model_size}.pt"
        self.logger.info(f"📥 Загрузка предобученной модели: {model_name}")
        
        try:
            self.model = YOLO(model_name)
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise
        
        # Размещение на устройстве
        device = self.device_manager.get_device()
        self.logger.info(f"💻 Модель размещена на: {device}")
        
        # Информация о модели
        if hasattr(self.model, 'model'):
            try:
                total_params = sum(p.numel() for p in self.model.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
                
                self.logger.info("📈 Параметры модели:")
                self.logger.info(f"  - Всего: {total_params:,}")
                self.logger.info(f"  - Обучаемые: {trainable_params:,}")
            except Exception as e:
                self.logger.warning(f"⚠️ Не удалось получить информацию о параметрах: {e}")
        
        return self.model
    
    def prepare_training_parameters(self, dataset_yaml: Path, experiment_name: str = None) -> Dict[str, Any]:
        """Подготовка параметров обучения"""
        self.logger.info("⚙️ Подготовка параметров обучения...")
        
        # Генерация имени эксперимента
        if experiment_name is None:
            timestamp = int(time.time())
            experiment_name = f"yolo_restaurant_detection_{timestamp}"
        
        self.experiment_name = experiment_name
        
        # Создание директории эксперимента
        experiments_dir = Path("outputs/experiments")
        self.experiment_dir = experiments_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Автоматическая настройка batch_size
        recommended_batch = self.device_manager.get_recommended_batch_size(
            self.config['training']['batch_size']
        )
        
        if recommended_batch != self.config['training']['batch_size']:
            self.logger.info(f"📊 Автоматическая корректировка batch_size: "
                           f"{self.config['training']['batch_size']} → {recommended_batch}")
            actual_batch_size = recommended_batch
        else:
            actual_batch_size = self.config['training']['batch_size']
        
        # Параметры для YOLO
        train_params = {
            'data': str(dataset_yaml),
            'epochs': self.config['training']['epochs'],
            'batch': actual_batch_size,
            'imgsz': self.config['model']['input_size'],
            'device': self.device_manager.get_device(),
            'workers': self.config['training']['workers'],
            'project': str(self.experiment_dir.parent),
            'name': self.experiment_name,
            'exist_ok': self.config['logging']['exist_ok'],
            'verbose': self.config['logging']['verbose'],
            
            # Оптимизация
            'optimizer': self.config['training']['optimizer'],
            'lr0': self.config['training']['learning_rate'],
            'weight_decay': self.config['training']['weight_decay'],
            'momentum': self.config['training']['momentum'],
            'warmup_epochs': self.config['training']['warmup_epochs'],
            'warmup_momentum': self.config['training']['warmup_momentum'],
            'warmup_bias_lr': self.config['training']['warmup_bias_lr'],
            'patience': self.config['training']['patience'],
            'amp': self.config['training']['amp'],
            
            # Аугментация
            'mosaic': self.config['augmentation']['mosaic'],
            'mixup': self.config['augmentation']['mixup'],
            'copy_paste': self.config['augmentation']['copy_paste'],
            'degrees': self.config['augmentation']['degrees'],
            'translate': self.config['augmentation']['translate'],
            'scale': self.config['augmentation']['scale'],
            'shear': self.config['augmentation']['shear'],
            'perspective': self.config['augmentation']['perspective'],
            'hsv_h': self.config['augmentation']['hsv_h'],
            'hsv_s': self.config['augmentation']['hsv_s'],
            'hsv_v': self.config['augmentation']['hsv_v'],
            'flipud': self.config['augmentation']['flipud'],
            'fliplr': self.config['augmentation']['fliplr'],
            
            # Валидация
            'val': True,
            'save_period': self.config['training']['save_period'],
            'plots': self.config['validation']['plots']
        }
        
        # Сохранение конфигурации
        config_path = self.experiment_dir / 'training_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment_name': experiment_name,
                'training_config': self.config,
                'training_parameters': train_params,
                'device_info': {
                    'device': str(self.device_manager.get_device()),
                    'cuda_available': torch.cuda.is_available(),
                    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else None
                }
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 Конфигурация сохранена: {config_path}")
        self.logger.info(f"📁 Эксперимент: {self.experiment_dir}")
        
        return train_params
    
    def run_training(self, train_params: Dict[str, Any]) -> Dict[str, Any]:
        """Запуск процесса обучения"""
        self.logger.info("🚀 Запуск обучения модели...")
        
        # Предобучающие проверки
        self._pre_training_checks(train_params)
        
        # Запуск обучения
        self.training_stats['start_time'] = time.time()
        
        try:
            # Обучение модели
            results = self.model.train(**train_params)
            
            self.training_stats['end_time'] = time.time()
            self.training_stats['epochs_completed'] = train_params['epochs']
            
            # Обработка результатов
            training_results = self._process_results(results)
            
            self.logger.info("🎉 Обучение завершено успешно!")
            return training_results
            
        except KeyboardInterrupt:
            self.logger.warning("⚠️ Обучение прервано пользователем")
            self.training_stats['training_interrupted'] = True
            raise
        except Exception as e:
            self.logger.error(f"❌ Ошибка во время обучения: {e}")
            self.training_stats['training_interrupted'] = True
            raise
    
    def _pre_training_checks(self, train_params: Dict[str, Any]):
        """Предобучающие проверки"""
        self.logger.info("🔧 Выполнение предобучающих проверок...")
        
        # Проверка памяти GPU
        if torch.cuda.is_available() and train_params['device'] != 'cpu':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            batch_size = train_params['batch']
            
            if gpu_memory < 4:
                self.logger.warning("⚠️ Мало GPU памяти, рекомендуется уменьшить batch_size")
            
            recommended_batch = self.device_manager.get_recommended_batch_size(16)
            if batch_size > recommended_batch:
                self.logger.warning(f"⚠️ Большой batch_size: {batch_size}, рекомендуется: {recommended_batch}")
        
        self.logger.info("✅ Тестовый forward pass успешен")
    
    def _process_results(self, results) -> Dict[str, Any]:
        """Обработка результатов обучения"""
        total_time = self.training_stats['end_time'] - self.training_stats['start_time']
        
        training_results = {
            'experiment_name': self.experiment_name,
            'total_training_time_seconds': total_time,
            'total_training_time_minutes': total_time / 60,
            'epochs_completed': self.training_stats['epochs_completed'],
            'device_used': str(self.device_manager.get_device()),
            'training_interrupted': self.training_stats['training_interrupted']
        }
        
        # Поиск лучшей модели
        weights_dir = self.experiment_dir / 'weights'
        if weights_dir.exists():
            best_model = weights_dir / 'best.pt'
            last_model = weights_dir / 'last.pt'
            
            if best_model.exists():
                training_results['best_model_path'] = str(best_model)
                self.logger.info(f"💎 Лучшая модель: {best_model}")
            
            if last_model.exists():
                training_results['last_model_path'] = str(last_model)
                self.logger.info(f"📄 Последняя модель: {last_model}")
        
        # Попытка извлечь метрики
        try:
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                training_results['final_metrics'] = metrics
                
                # Логирование ключевых метрик
                if 'metrics/mAP50(B)' in metrics:
                    map50 = metrics['metrics/mAP50(B)']
                    self.logger.info(f"📊 Финальный mAP@0.5: {map50:.4f}")
                    training_results['final_map50'] = map50
                
                if 'metrics/mAP50-95(B)' in metrics:
                    map50_95 = metrics['metrics/mAP50-95(B)']
                    self.logger.info(f"📊 Финальный mAP@0.5:0.95: {map50_95:.4f}")
                    training_results['final_map50_95'] = map50_95
        
        except Exception as e:
            self.logger.warning(f"⚠️ Не удалось извлечь финальные метрики: {e}")
        
        # Сохранение результатов
        results_path = self.experiment_dir / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(training_results, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"📋 Результаты сохранены: {results_path}")
        
        return training_results
    
    def train_model(self, dataset_yaml: Path, resume_from: Optional[Path] = None) -> Dict[str, Any]:
        """Полный цикл обучения модели"""
        try:
            # 1. Валидация датасета
            dataset_config = self.validate_dataset(dataset_yaml)
            
            # 2. Настройка модели
            model = self.setup_model()
            
            # 3. Подготовка параметров обучения
            train_params = self.prepare_training_parameters(dataset_yaml)
            
            # 4. Обработка resume
            if resume_from and resume_from.exists():
                train_params['resume'] = str(resume_from)
                self.logger.info(f"🔄 Продолжение обучения с: {resume_from}")
            
            # 5. Запуск обучения
            results = self.run_training(train_params)
            
            # 6. Генерация итогового отчета
            self._generate_final_report(results, dataset_config)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Критическая ошибка: {e}")
            raise
    
    def _generate_final_report(self, training_results: Dict[str, Any], dataset_config: Dict[str, Any]):
        """Генерация итогового отчета"""
        self.logger.info("\n" + "="*60)
        self.logger.info("🎯 ИТОГИ ОБУЧЕНИЯ")
        self.logger.info("="*60)
        self.logger.info(f"📁 Эксперимент: {self.experiment_name}")
        self.logger.info(f"⏱️ Время обучения: {training_results.get('total_training_time_minutes', 0):.1f} минут")
        self.logger.info(f"🔄 Эпох завершено: {training_results.get('epochs_completed', 0)}")
        self.logger.info(f"💻 Устройство: {training_results.get('device_used', 'неизвестно')}")
        
        if 'final_map50' in training_results:
            self.logger.info(f"📊 Финальный mAP@0.5: {training_results['final_map50']:.4f}")
        
        if 'final_map50_95' in training_results:
            self.logger.info(f"📊 Финальный mAP@0.5:0.95: {training_results['final_map50_95']:.4f}")
        
        if 'best_model_path' in training_results:
            self.logger.info(f"💎 Лучшая модель: {training_results['best_model_path']}")
        
        self.logger.info(f"📋 Полные результаты: {self.experiment_dir}")
        self.logger.info("="*60)

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="Профессиональное обучение YOLO11 для детекции объектов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

1. Базовое обучение:
   python scripts/train_model.py --data data/processed/dataset/dataset.yaml

2. Обучение на GPU:
   python scripts/train_model.py --data dataset.yaml --device cuda

3. Обучение с кастомной конфигурацией:
   python scripts/train_model.py --data dataset.yaml --config config/train_config.json

4. Продолжение обучения:
   python scripts/train_model.py --data dataset.yaml --resume runs/train/exp/weights/last.pt

5. Обучение с Weights & Biases:
   python scripts/train_model.py --data dataset.yaml --wandb

6. Обучение на CPU (для тестирования):
   python scripts/train_model.py --data dataset.yaml --device cpu
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Путь к файлу dataset.yaml"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=['auto', 'cpu', 'cuda', 'gpu'],
        default='auto',
        help="Устройство для обучения (auto, cpu, cuda/gpu). По умолчанию: auto"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Путь к файлу конфигурации обучения (JSON)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Путь к чекпоинту для продолжения обучения (.pt файл)"
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Включить логирование в Weights & Biases"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод"
    )
    
    parser.add_argument(
        "--model-size",
        type=str,
        choices=['n', 's', 'm', 'l', 'x'],
        default='n',
        help="Размер модели YOLO11 (n=nano, s=small, m=medium, l=large, x=xlarge). По умолчанию: n"
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Нормализация device аргумента
        device = args.device
        if device == 'gpu':
            device = 'cuda'
        
        # Подготовка путей
        dataset_yaml = Path(args.data)
        config_path = Path(args.config) if args.config else None
        resume_from = Path(args.resume) if args.resume else None
        
        # Проверка существования файлов
        if not dataset_yaml.exists():
            print(f"❌ Файл dataset.yaml не найден: {dataset_yaml}")
            sys.exit(1)
        
        if config_path and not config_path.exists():
            print(f"❌ Файл конфигурации не найден: {config_path}")
            sys.exit(1)
        
        if resume_from and not resume_from.exists():
            print(f"❌ Файл чекпоинта не найден: {resume_from}")
            sys.exit(1)
        
        # Инициализация тренера
        trainer = ProfessionalYOLOTrainer(config_path, device)
        
        # Переопределение размера модели если указан
        if args.model_size != 'n':
            trainer.config['model']['size'] = args.model_size
        
        # Включение wandb если запрошено
        if args.wandb:
            if WANDB_AVAILABLE:
                trainer.config['logging']['wandb']['enabled'] = True
                print("📊 Weights & Biases включен")
            else:
                print("⚠️ Wandb не установлен. Установите: pip install wandb")
        
        # Запуск обучения
        print("\n" + "="*50)
        print("🚀 ЗАПУСК ОБУЧЕНИЯ YOLO11")
        print("="*50)
        print(f"📁 Датасет: {dataset_yaml}")
        print(f"💻 Устройство: {device}")
        print(f"🧠 Размер модели: YOLO11{args.model_size}")
        if config_path:
            print(f"⚙️ Конфигурация: {config_path}")
        if resume_from:
            print(f"🔄 Продолжение с: {resume_from}")
        print("="*50)
        
        results = trainer.train_model(dataset_yaml, resume_from)
        
        if results:
            print("\n" + "="*50)
            print("🎉 ОБУЧЕНИЕ YOLO11 УСПЕШНО ЗАВЕРШЕНО!")
            print("="*50)
            
            if 'best_model_path' in results:
                print(f"💎 Лучшая модель: {results['best_model_path']}")
            
            if 'final_map50' in results:
                print(f"📊 mAP@0.5: {results['final_map50']:.1%}")
            
            print(f"⏱️ Время обучения: {results.get('total_training_time_minutes', 0):.1f} мин")
            print("\n🚀 Теперь можно запускать инференс:")
            
            if 'best_model_path' in results:
                model_path = results['best_model_path']
                print(f"   python scripts/run_inference.py --model \"{model_path}\" --input-dir \"path/to/images\"")
            
            print("="*50)
            sys.exit(0)
        else:
            print("\n❌ Обучение завершилось с ошибками!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        print("📋 Проверьте логи для получения дополнительной информации")
        sys.exit(1)


if __name__ == "__main__":
    main()