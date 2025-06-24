"""
Улучшенный скрипт обучения YOLO11 с профессиональным мониторингом
Автоматически решает проблемы с пустыми аннотациями и оптимизирует обучение
"""

import sys
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import time
import json
import torch
import numpy as np
from ultralytics import YOLO
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ Wandb не установлен. Логирование в Wandb будет отключено.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Matplotlib/Seaborn не установлены. Визуализации будут отключены.")

def setup_logger(name: str) -> logging.Logger:
    """Простая настройка логгера"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

class SimpleDeviceManager:
    """Простой менеджер устройств"""
    @staticmethod
    def get_optimal_device():
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"

class SimpleMetrics:
    """Простой класс для метрик"""
    def __init__(self):
        pass


class ProfessionalYOLOTrainer:
    """
    Профессиональная система обучения YOLO11 с продвинутым мониторингом
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = self._load_config(config_path)
        
        # Инициализация компонентов
        self.device_manager = SimpleDeviceManager()
        self.metrics = SimpleMetrics()
        
        # Состояние обучения
        self.training_state = {
            'start_time': None,
            'end_time': None,
            'best_map50': 0.0,
            'best_map50_95': 0.0,
            'epochs_completed': 0,
            'early_stopping_counter': 0,
            'training_interrupted': False
        }
        
        # Настройка wandb если включен и доступен
        self.use_wandb = (WANDB_AVAILABLE and 
                         self.config.get('logging', {}).get('wandb', {}).get('enabled', False))
        if self.use_wandb:
            self._init_wandb()
    
    def _load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Загрузка конфигурации обучения"""
        default_config = {
            'model': {
                'size': 'n',  # n, s, m, l, x
                'input_size': 640,
                'pretrained': True,
                'freeze_backbone': False,
                'freeze_epochs': 0
            },
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.01,
                'patience': 15,
                'min_lr': 1e-6,
                'optimizer': 'AdamW',
                'weight_decay': 0.0005,
                'momentum': 0.937,
                'scheduler': 'cosine',
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'save_period': 10,
                'val_period': 1
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
                'save_json': True,
                'save_hybrid': False,
                'plots': True
            },
            'optimization': {
                'amp': True,  # Automatic Mixed Precision
                'single_cls': False,
                'rect': False,
                'cos_lr': False,
                'close_mosaic': 10,
                'resume': False,
                'overlap_mask': True,
                'mask_ratio': 4
            },
            'callbacks': {
                'early_stopping': True,
                'model_checkpoint': True,
                'lr_scheduler': True,
                'tensorboard': True
            },
            'logging': {
                'verbose': True,
                'save_dir': 'outputs/experiments',
                'name': 'yolo_restaurant_detection',
                'exist_ok': True,
                'wandb': {
                    'enabled': False,
                    'project': 'restaurant-object-detection',
                    'entity': None,
                    'tags': ['yolo11', 'restaurant', 'detection']
                }
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)
                self._deep_update(default_config, user_config)
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Глубокое обновление словаря"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _init_wandb(self):
        """Инициализация Weights & Biases"""
        if not WANDB_AVAILABLE:
            self.logger.warning("⚠️ Wandb не доступен, отключаем логирование")
            self.use_wandb = False
            return
            
        try:
            wandb_config = self.config['logging']['wandb']
            wandb.init(
                project=wandb_config['project'],
                entity=wandb_config.get('entity'),
                tags=wandb_config.get('tags', []),
                config=self.config,
                name=f"{self.config['logging']['name']}_{int(time.time())}"
            )
            self.logger.info("✅ Weights & Biases инициализирован")
        except Exception as e:
            self.logger.warning(f"⚠️ Не удалось инициализировать wandb: {e}")
            self.use_wandb = False
    
    def train_model(self, dataset_yaml: Path, resume_from: Optional[Path] = None) -> Dict[str, Any]:
        """
        Основная функция обучения модели
        
        Args:
            dataset_yaml: Путь к конфигурации датасета
            resume_from: Путь к чекпоинту для продолжения обучения
            
        Returns:
            Результаты обучения
        """
        self.training_state['start_time'] = time.time()
        self.logger.info("🚀 Начало профессионального обучения YOLO11")
        
        try:
            # Проверка датасета
            self._validate_dataset(dataset_yaml)
            
            # Инициализация модели
            model = self._initialize_model(resume_from)
            
            # Настройка параметров обучения
            train_params = self._prepare_training_params(dataset_yaml)
            
            # Предобучающие проверки
            self._pre_training_checks(model, dataset_yaml)
            
            # Запуск обучения
            results = self._run_training(model, train_params)
            
            # Постобработка результатов
            self._post_training_analysis(model, results)
            
            return results
            
        except KeyboardInterrupt:
            self.logger.warning("⚠️ Обучение прервано пользователем")
            self.training_state['training_interrupted'] = True
            return {}
        except Exception as e:
            self.logger.error(f"❌ Ошибка при обучении: {e}")
            raise
        finally:
            self.training_state['end_time'] = time.time()
            self._log_training_summary()
    
    def _validate_dataset(self, dataset_yaml: Path):
        """Валидация датасета перед обучением"""
        self.logger.info("🔍 Валидация датасета...")
        
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Файл датасета не найден: {dataset_yaml}")
        
        # Загрузка конфигурации датасета
        with open(dataset_yaml, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f)
        
        # Проверка обязательных полей
        required_fields = ['path', 'train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in dataset_config:
                raise ValueError(f"Отсутствует обязательное поле в dataset.yaml: {field}")
        
        # Проверка путей
        dataset_path = Path(dataset_config['path'])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Директория датасета не найдена: {dataset_path}")
        
        # Проверка наличия изображений и аннотаций
        splits_info = {}
        for split in ['train', 'val']:
            if split in dataset_config:
                split_path = dataset_path / dataset_config[split]
                if split_path.is_dir():
                    # Если путь указывает на директорию images
                    images_dir = split_path
                    labels_dir = split_path.parent / 'labels'
                else:
                    # Если путь относительный
                    images_dir = dataset_path / split / 'images'
                    labels_dir = dataset_path / split / 'labels'
                
                if not images_dir.exists():
                    raise FileNotFoundError(f"Директория изображений не найдена: {images_dir}")
                
                if not labels_dir.exists():
                    self.logger.warning(f"⚠️ Директория аннотаций не найдена: {labels_dir}")
                    # Создаем пустую директорию для labels
                    labels_dir.mkdir(parents=True, exist_ok=True)
                
                # Подсчет файлов
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_files.extend(list(images_dir.glob(f"*{ext}")))
                    image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
                
                label_files = list(labels_dir.glob("*.txt"))
                
                # Создание пустых файлов аннотаций если их нет
                if len(label_files) == 0 and len(image_files) > 0:
                    self.logger.warning(f"⚠️ Создание пустых файлов аннотаций для {split}")
                    for image_file in image_files:
                        label_file = labels_dir / f"{image_file.stem}.txt"
                        if not label_file.exists():
                            label_file.touch()  # Создаем пустой файл
                    label_files = list(labels_dir.glob("*.txt"))
                
                splits_info[split] = {
                    'images': len(image_files),
                    'labels': len(label_files),
                    'images_dir': str(images_dir),
                    'labels_dir': str(labels_dir)
                }
        
        # Логирование информации о датасете
        self.logger.info(f"📊 Информация о датасете:")
        self.logger.info(f"  - Классы: {dataset_config['nc']} ({', '.join(dataset_config['names'])})")
        for split, info in splits_info.items():
            self.logger.info(f"  - {split.upper()}: {info['images']} изображений, {info['labels']} аннотаций")
        
        # Проверка на критические проблемы
        if splits_info.get('train', {}).get('images', 0) == 0:
            raise ValueError("Нет изображений для обучения!")
        
        if splits_info.get('val', {}).get('images', 0) == 0:
            self.logger.warning("⚠️ Нет изображений для валидации!")
        
        self.logger.info("✅ Валидация датасета пройдена")
    
    def _initialize_model(self, resume_from: Optional[Path] = None) -> YOLO:
        """Инициализация модели YOLO"""
        self.logger.info("🧠 Инициализация модели YOLO11...")
        
        model_config = self.config['model']
        
        if resume_from and resume_from.exists():
            self.logger.info(f"🔄 Продолжение обучения с: {resume_from}")
            model = YOLO(str(resume_from))
        else:
            if model_config['pretrained']:
                model_name = f"yolo11{model_config['size']}.pt"
                self.logger.info(f"📥 Загрузка предобученной модели: {model_name}")
                model = YOLO(model_name)
            else:
                model_name = f"yolo11{model_config['size']}.yaml"
                self.logger.info(f"🏗️ Создание модели с нуля: {model_name}")
                model = YOLO(model_name)
        
        # Настройка устройства
        device = self.device_manager.get_optimal_device()
        model.to(device)
        
        self.logger.info(f"💻 Модель размещена на: {device}")
        
        # Логирование архитектуры
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        self.logger.info(f"📈 Параметры модели:")
        self.logger.info(f"  - Всего: {total_params:,}")
        self.logger.info(f"  - Обучаемые: {trainable_params:,}")
        
        return model
    
    def _prepare_training_params(self, dataset_yaml: Path) -> Dict[str, Any]:
        """Подготовка параметров обучения"""
        self.logger.info("⚙️ Подготовка параметров обучения...")
        
        # Создание выходной директории
        save_dir = Path(self.config['logging']['save_dir'])
        experiment_name = f"{self.config['logging']['name']}_{int(time.time())}"
        experiment_dir = save_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Базовые параметры
        train_params = {
            'data': str(dataset_yaml),
            'epochs': self.config['training']['epochs'],
            'batch': self.config['training']['batch_size'],
            'imgsz': self.config['model']['input_size'],
            'lr0': self.config['training']['learning_rate'],
            'lrf': self.config['training']['min_lr'] / self.config['training']['learning_rate'],
            'momentum': self.config['training']['momentum'],
            'weight_decay': self.config['training']['weight_decay'],
            'warmup_epochs': self.config['training']['warmup_epochs'],
            'warmup_momentum': self.config['training']['warmup_momentum'],
            'warmup_bias_lr': self.config['training']['warmup_bias_lr'],
            'optimizer': self.config['training']['optimizer'],
            'patience': self.config['training']['patience'],
            'save_period': self.config['training']['save_period'],
            'val': self.config['training']['val_period'] == 1,
            'project': str(save_dir),
            'name': experiment_name,
            'exist_ok': self.config['logging']['exist_ok'],
            'verbose': self.config['logging']['verbose']
        }
        
        # Аугментация
        augmentation = self.config['augmentation']
        train_params.update({
            'mosaic': augmentation['mosaic'],
            'mixup': augmentation['mixup'],
            'copy_paste': augmentation['copy_paste'],
            'degrees': augmentation['degrees'],
            'translate': augmentation['translate'],
            'scale': augmentation['scale'],
            'shear': augmentation['shear'],
            'perspective': augmentation['perspective'],
            'hsv_h': augmentation['hsv_h'],
            'hsv_s': augmentation['hsv_s'],
            'hsv_v': augmentation['hsv_v'],
            'flipud': augmentation['flipud'],
            'fliplr': augmentation['fliplr']
        })
        
        # Валидация
        validation = self.config['validation']
        train_params.update({
            'conf': validation['conf_threshold'],
            'iou': validation['iou_threshold'],
            'max_det': validation['max_det'],
            'save_json': validation['save_json'],
            'plots': validation['plots']
        })
        
        # Оптимизация
        optimization = self.config['optimization']
        train_params.update({
            'amp': optimization['amp'],
            'single_cls': optimization['single_cls'],
            'rect': optimization['rect'],
            'cos_lr': optimization['cos_lr'],
            'close_mosaic': optimization['close_mosaic'],
            'resume': optimization['resume'],
            'overlap_mask': optimization['overlap_mask'],
            'mask_ratio': optimization['mask_ratio']
        })
        
        # Сохранение конфигурации
        config_path = experiment_dir / 'training_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 Конфигурация сохранена: {config_path}")
        self.logger.info(f"📁 Эксперимент: {experiment_dir}")
        
        return train_params
    
    def _pre_training_checks(self, model: YOLO, dataset_yaml: Path):
        """Предобучающие проверки"""
        self.logger.info("🔧 Выполнение предобучающих проверок...")
        
        # Проверка памяти GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"🎮 GPU память: {gpu_memory:.1f} GB")
            
            if gpu_memory < 4:
                self.logger.warning("⚠️ Мало GPU памяти, рекомендуется уменьшить batch_size")
        
        # Проверка batch_size
        recommended_batch = self._estimate_optimal_batch_size()
        current_batch = self.config['training']['batch_size']
        
        if current_batch > recommended_batch:
            self.logger.warning(f"⚠️ Большой batch_size: {current_batch}, рекомендуется: {recommended_batch}")
        
        # Тестовый forward pass
        try:
            test_input = torch.randn(1, 3, self.config['model']['input_size'], self.config['model']['input_size'])
            if torch.cuda.is_available():
                test_input = test_input.cuda()
            
            with torch.no_grad():
                _ = model.model(test_input)
            
            self.logger.info("✅ Тестовый forward pass успешен")
        except Exception as e:
            self.logger.error(f"❌ Ошибка в тестовом forward pass: {e}")
            raise
    
    def _estimate_optimal_batch_size(self) -> int:
        """Оценка оптимального размера батча"""
        if not torch.cuda.is_available():
            return 4
        
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        input_size = self.config['model']['input_size']
        
        # Эмпирическая формула
        if input_size <= 640:
            if gpu_memory_gb >= 12:
                return 32
            elif gpu_memory_gb >= 8:
                return 16
            elif gpu_memory_gb >= 6:
                return 8
            else:
                return 4
        else:
            return max(2, int(gpu_memory_gb / 2))
    
    def _run_training(self, model: YOLO, train_params: Dict[str, Any]) -> Dict[str, Any]:
        """Запуск обучения"""
        self.logger.info("🚀 Запуск обучения модели...")
        
        # Callback для мониторинга
        def on_train_epoch_end(trainer):
            self.training_state['epochs_completed'] = trainer.epoch + 1
            
            # Логирование в wandb
            if self.use_wandb and hasattr(trainer, 'metrics'):
                wandb.log({
                    'epoch': trainer.epoch,
                    'train/box_loss': trainer.loss.item() if hasattr(trainer, 'loss') else 0,
                    'lr': trainer.optimizer.param_groups[0]['lr'] if hasattr(trainer, 'optimizer') else 0
                })
        
        def on_val_end(trainer):
            if hasattr(trainer, 'metrics') and trainer.metrics:
                metrics = trainer.metrics
                
                # Обновление лучших метрик
                if hasattr(metrics, 'box') and hasattr(metrics.box, 'map50'):
                    current_map50 = metrics.box.map50
                    if current_map50 > self.training_state['best_map50']:
                        self.training_state['best_map50'] = current_map50
                        self.logger.info(f"🎯 Новый лучший mAP@0.5: {current_map50:.4f}")
                
                if hasattr(metrics, 'box') and hasattr(metrics.box, 'map'):
                    current_map50_95 = metrics.box.map
                    if current_map50_95 > self.training_state['best_map50_95']:
                        self.training_state['best_map50_95'] = current_map50_95
                        self.logger.info(f"🎯 Новый лучший mAP@0.5:0.95: {current_map50_95:.4f}")
                
                # Логирование в wandb
                if self.use_wandb:
                    wandb_metrics = {
                        'val/mAP50': getattr(metrics.box, 'map50', 0),
                        'val/mAP50-95': getattr(metrics.box, 'map', 0),
                        'val/precision': getattr(metrics.box, 'mp', 0),
                        'val/recall': getattr(metrics.box, 'mr', 0)
                    }
                    wandb.log(wandb_metrics)
        
        # Добавление callback'ов
        if hasattr(model, 'add_callback'):
            model.add_callback('on_train_epoch_end', on_train_epoch_end)
            model.add_callback('on_val_end', on_val_end)
        
        try:
            # Запуск обучения
            results = model.train(**train_params)
            
            self.logger.info("✅ Обучение успешно завершено")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка во время обучения: {e}")
            raise
    
    def _post_training_analysis(self, model: YOLO, results):
        """Анализ результатов после обучения"""
        self.logger.info("📊 Анализ результатов обучения...")
        
        try:
            # Путь к результатам
            if hasattr(results, 'save_dir'):
                results_dir = Path(results.save_dir)
            else:
                results_dir = Path(self.config['logging']['save_dir']) / self.config['logging']['name']
            
            # Анализ метрик
            self._analyze_training_metrics(results_dir)
            
            # Создание дополнительных визуализаций
            self._create_custom_visualizations(results_dir)
            
            # Анализ модели
            self._analyze_model_performance(model, results_dir)
            
            # Создание отчета
            self._create_training_report(results_dir)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка в постанализе: {e}")
    
    def _analyze_training_metrics(self, results_dir: Path):
        """Анализ метрик обучения"""
        self.logger.info("📈 Анализ метрик обучения...")
        
        # Поиск файла с результатами
        results_csv = results_dir / 'results.csv'
        if not results_csv.exists():
            self.logger.warning("⚠️ Файл results.csv не найден")
            return
        
        try:
            import pandas as pd
            
            # Загрузка метрик
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()  # Удаление пробелов
            
            # Анализ сходимости
            if 'train/box_loss' in df.columns:
                final_train_loss = df['train/box_loss'].iloc[-1]
                self.logger.info(f"📉 Финальная train loss: {final_train_loss:.4f}")
            
            if 'val/box_loss' in df.columns:
                final_val_loss = df['val/box_loss'].iloc[-1]
                self.logger.info(f"📉 Финальная val loss: {final_val_loss:.4f}")
            
            # Лучшие метрики
            if 'metrics/mAP50(B)' in df.columns:
                best_map50 = df['metrics/mAP50(B)'].max()
                self.logger.info(f"🎯 Лучший mAP@0.5: {best_map50:.4f}")
            
            if 'metrics/mAP50-95(B)' in df.columns:
                best_map50_95 = df['metrics/mAP50-95(B)'].max()
                self.logger.info(f"🎯 Лучший mAP@0.5:0.95: {best_map50_95:.4f}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка анализа метрик: {e}")
    
    def _create_custom_visualizations(self, results_dir: Path):
        """Создание дополнительных визуализаций"""
        if not PLOTTING_AVAILABLE:
            self.logger.warning("⚠️ Matplotlib не доступен, пропускаем визуализации")
            return
            
        self.logger.info("🎨 Создание дополнительных визуализаций...")
        
        try:
            # Настройка стиля
            plt.style.use('default')  # Используем базовый стиль
            
            # График обучения
            self._plot_training_curves(results_dir)
            
            # Анализ классов
            self._plot_class_distribution(results_dir)
            
            # График потерь
            self._plot_loss_analysis(results_dir)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка создания визуализаций: {e}")
    
    def _plot_training_curves(self, results_dir: Path):
        """График кривых обучения"""
        if not PLOTTING_AVAILABLE:
            return
            
        results_csv = results_dir / 'results.csv'
        if not results_csv.exists():
            return
        
        try:
            import pandas as pd
            
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Кривые обучения YOLO11', fontsize=16, fontweight='bold')
            
            # mAP графики
            if 'metrics/mAP50(B)' in df.columns:
                axes[0, 0].plot(df.index, df['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2)
                axes[0, 0].set_title('mAP@0.5')
                axes[0, 0].set_xlabel('Эпоха')
                axes[0, 0].set_ylabel('mAP')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()
            
            if 'metrics/mAP50-95(B)' in df.columns:
                axes[0, 1].plot(df.index, df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2, color='orange')
                axes[0, 1].set_title('mAP@0.5:0.95')
                axes[0, 1].set_xlabel('Эпоха')
                axes[0, 1].set_ylabel('mAP')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
            
            # Loss графики
            if 'train/box_loss' in df.columns:
                axes[1, 0].plot(df.index, df['train/box_loss'], label='Train Box Loss', linewidth=2, color='red')
                if 'val/box_loss' in df.columns:
                    axes[1, 0].plot(df.index, df['val/box_loss'], label='Val Box Loss', linewidth=2, color='blue')
                axes[1, 0].set_title('Box Loss')
                axes[1, 0].set_xlabel('Эпоха')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
            
            # Learning Rate
            if 'lr/pg0' in df.columns:
                axes[1, 1].plot(df.index, df['lr/pg0'], label='Learning Rate', linewidth=2, color='green')
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Эпоха')
                axes[1, 1].set_ylabel('LR')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Сохранение
            output_path = results_dir / 'training_curves_custom.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 График обучения сохранен: {output_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка создания графика обучения: {e}")
    
    def _plot_class_distribution(self, results_dir: Path):
        """График распределения классов"""
        # Поиск файла с аннотациями для анализа
        try:
            # Попытка найти информацию о классах из confusion matrix
            confusion_matrix_path = results_dir / 'confusion_matrix.png'
            if confusion_matrix_path.exists():
                self.logger.info("📊 Confusion matrix уже создана YOLO")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка анализа классов: {e}")
    
    def _plot_loss_analysis(self, results_dir: Path):
        """Детальный анализ потерь"""
        if not PLOTTING_AVAILABLE:
            return
            
        results_csv = results_dir / 'results.csv'
        if not results_csv.exists():
            return
        
        try:
            import pandas as pd
            
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            
            # Поиск всех loss колонок
            loss_columns = [col for col in df.columns if 'loss' in col.lower()]
            
            if not loss_columns:
                return
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Простые цвета вместо seaborn
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i, col in enumerate(loss_columns):
                color = colors[i % len(colors)]
                ax.plot(df.index, df[col], label=col, linewidth=2, color=color)
            
            ax.set_title('Анализ всех потерь', fontsize=14, fontweight='bold')
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            # Сохранение
            output_path = results_dir / 'loss_analysis.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Анализ потерь сохранен: {output_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка анализа потерь: {e}")
    
    def _analyze_model_performance(self, model: YOLO, results_dir: Path):
        """Анализ производительности модели"""
        self.logger.info("⚡ Анализ производительности модели...")
        
        try:
            # Подсчет параметров
            if hasattr(model, 'model'):
                total_params = sum(p.numel() for p in model.model.parameters())
                trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                
                performance_info = {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_size_mb': total_params * 4 / (1024 * 1024),  # Приблизительно
                    'training_time_minutes': (self.training_state.get('end_time', time.time()) - 
                                            self.training_state.get('start_time', 0)) / 60,
                    'epochs_completed': self.training_state.get('epochs_completed', 0),
                    'best_map50': self.training_state.get('best_map50', 0),
                    'best_map50_95': self.training_state.get('best_map50_95', 0)
                }
                
                # Сохранение информации о производительности
                perf_path = results_dir / 'performance_analysis.json'
                with open(perf_path, 'w', encoding='utf-8') as f:
                    json.dump(performance_info, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"📊 Анализ производительности сохранен: {perf_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка анализа производительности: {e}")
    
    def _create_training_report(self, results_dir: Path):
        """Создание отчета об обучении"""
        self.logger.info("📋 Создание отчета об обучении...")
        
        try:
            report = {
                'training_summary': {
                    'start_time': time.strftime('%Y-%m-%d %H:%M:%S', 
                                               time.localtime(self.training_state.get('start_time', 0))),
                    'end_time': time.strftime('%Y-%m-%d %H:%M:%S', 
                                             time.localtime(self.training_state.get('end_time', 0))),
                    'total_duration_minutes': (self.training_state.get('end_time', time.time()) - 
                                             self.training_state.get('start_time', 0)) / 60,
                    'epochs_completed': self.training_state.get('epochs_completed', 0),
                    'training_interrupted': self.training_state.get('training_interrupted', False)
                },
                'best_metrics': {
                    'best_map50': self.training_state.get('best_map50', 0),
                    'best_map50_95': self.training_state.get('best_map50_95', 0)
                },
                'configuration': self.config,
                'files_generated': {
                    'best_weights': str(results_dir / 'weights' / 'best.pt'),
                    'last_weights': str(results_dir / 'weights' / 'last.pt'),
                    'results_csv': str(results_dir / 'results.csv'),
                    'training_curves': str(results_dir / 'results.png')
                }
            }
            
            # Сохранение отчета
            report_path = results_dir / 'training_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"📋 Отчет об обучении сохранен: {report_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка создания отчета: {e}")
    
    def _log_training_summary(self):
        """Логирование итогов обучения"""
        total_time = (self.training_state.get('end_time', time.time()) - 
                     self.training_state.get('start_time', 0)) / 60
        
        self.logger.info("\n" + "="*60)
        self.logger.info("📋 ИТОГИ ОБУЧЕНИЯ YOLO11")
        self.logger.info("="*60)
        self.logger.info(f"⏱️ Общее время обучения: {total_time:.1f} минут")
        self.logger.info(f"🔄 Завершено эпох: {self.training_state.get('epochs_completed', 0)}")
        self.logger.info(f"🎯 Лучший mAP@0.5: {self.training_state.get('best_map50', 0):.4f}")
        self.logger.info(f"🎯 Лучший mAP@0.5:0.95: {self.training_state.get('best_map50_95', 0):.4f}")
        
        if self.training_state.get('training_interrupted'):
            self.logger.warning("⚠️ Обучение было прервано")
        else:
            self.logger.info("✅ Обучение успешно завершено")
        
        self.logger.info("="*60)


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="Профессиональное обучение YOLO11 для детекции объектов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
    # Базовое обучение
    python scripts/train_model.py --data data/datasets/restaurant_detection/dataset.yaml
    
    # Обучение с кастомной конфигурацией
    python scripts/train_model.py --data dataset.yaml --config config/train_config.json
    
    # Продолжение обучения
    python scripts/train_model.py --data dataset.yaml --resume runs/train/exp/weights/last.pt
    
    # Обучение с Weights & Biases
    python scripts/train_model.py --data dataset.yaml --wandb
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Путь к файлу dataset.yaml"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Путь к файлу конфигурации обучения"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Путь к чекпоинту для продолжения обучения"
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
    
    args = parser.parse_args()
    
    # Настройка логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Подготовка путей
        dataset_yaml = Path(args.data)
        config_path = Path(args.config) if args.config else None
        resume_from = Path(args.resume) if args.resume else None
        
        # Инициализация тренера
        trainer = ProfessionalYOLOTrainer(config_path)
        
        # Включение wandb если запрошено и доступно
        if args.wandb and WANDB_AVAILABLE:
            trainer.config['logging']['wandb']['enabled'] = True
            trainer.use_wandb = True
            trainer._init_wandb()
        elif args.wandb and not WANDB_AVAILABLE:
            print("⚠️ Wandb не установлен, но был запрошен. Устанавливаем: pip install wandb")
        
        # Запуск обучения
        results = trainer.train_model(dataset_yaml, resume_from)
        
        if results:
            print("\n🎉 Обучение YOLO11 успешно завершено!")
            print(f"🎯 Лучший mAP@0.5: {trainer.training_state.get('best_map50', 0):.4f}")
            print(f"🎯 Лучший mAP@0.5:0.95: {trainer.training_state.get('best_map50_95', 0):.4f}")
            print("\n📁 Результаты сохранены в outputs/experiments/")
            print("🚀 Можно запускать inference:")
            print("   python scripts/run_inference.py --weights runs/train/exp/weights/best.pt")
            sys.exit(0)
        else:
            print("\n❌ Обучение завершилось с ошибками!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()