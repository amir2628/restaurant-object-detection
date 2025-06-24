"""
Модуль для обучения модели YOLOv11
Включает полный цикл обучения с мониторингом метрик и сохранением чекпоинтов
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
import json
import time
import shutil
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb
from collections import defaultdict

from src.utils.logger import get_logger, log_execution_time
from src.utils.device_manager import get_device_manager
from src.utils.metrics import MetricsCalculator
from src.models.model_manager import ModelManager
from config.config import config

class TrainingCallback:
    """Колбэк для мониторинга обучения"""
    
    def __init__(self, trainer):
        """
        Инициализация колбэка
        
        Args:
            trainer: Экземпляр YOLOTrainer
        """
        self.trainer = trainer
        self.logger = get_logger(f"{__name__}.callback")
        
        # История метрик
        self.metrics_history = defaultdict(list)
        self.epoch_times = []
        self.best_metrics = {}
        
        # Для early stopping
        self.patience_counter = 0
        self.best_metric_value = -float('inf')
        self.early_stop_triggered = False
    
    def on_epoch_start(self, epoch: int):
        """Вызывается в начале эпохи"""
        self.epoch_start_time = time.time()
        self.logger.debug(f"Начало эпохи {epoch}")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Вызывается в конце эпохи"""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # Сохранение метрик
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append(value)
        
        # Логирование
        self.logger.info(f"Эпоха {epoch} завершена за {epoch_time:.2f}с")
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")
        
        # Обновление лучших метрик
        primary_metric = metrics.get('val_map50', metrics.get('map50', 0))
        if primary_metric > self.best_metric_value:
            self.best_metric_value = primary_metric
            self.best_metrics = metrics.copy()
            self.patience_counter = 0
            self.logger.info(f"Новый лучший результат: {primary_metric:.4f}")
        else:
            self.patience_counter += 1
        
        # Проверка early stopping
        if self.patience_counter >= self.trainer.patience:
            self.early_stop_triggered = True
            self.logger.info(f"Early stopping: метрика не улучшается {self.patience_counter} эпох")
        
        # Сохранение промежуточных графиков
        if epoch % 10 == 0:
            self._save_training_plots(epoch)
    
    def _save_training_plots(self, epoch: int):
        """Сохранение графиков обучения"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # График потерь
            if 'train_loss' in self.metrics_history and 'val_loss' in self.metrics_history:
                axes[0, 0].plot(self.metrics_history['train_loss'], label='Train Loss')
                axes[0, 0].plot(self.metrics_history['val_loss'], label='Val Loss')
                axes[0, 0].set_title('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # График mAP
            if 'train_map50' in self.metrics_history and 'val_map50' in self.metrics_history:
                axes[0, 1].plot(self.metrics_history['train_map50'], label='Train mAP@0.5')
                axes[0, 1].plot(self.metrics_history['val_map50'], label='Val mAP@0.5')
                axes[0, 1].set_title('mAP@0.5')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # График precision/recall
            if 'precision' in self.metrics_history and 'recall' in self.metrics_history:
                axes[1, 0].plot(self.metrics_history['precision'], label='Precision')
                axes[1, 0].plot(self.metrics_history['recall'], label='Recall')
                axes[1, 0].set_title('Precision/Recall')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # График времени эпох
            if self.epoch_times:
                axes[1, 1].plot(self.epoch_times)
                axes[1, 1].set_title('Время эпохи (сек)')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Сохранение
            plots_dir = self.trainer.experiment_dir / "training_plots"
            plots_dir.mkdir(exist_ok=True)
            plt.savefig(plots_dir / f"training_progress_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Ошибка при сохранении графиков: {e}")

class YOLOTrainer:
    """Класс для обучения модели YOLOv11"""
    
    def __init__(self, 
                 model_size: str = None,
                 experiment_name: str = None,
                 output_dir: Path = None,
                 config_override: Dict[str, Any] = None):
        """
        Инициализация тренера
        
        Args:
            model_size: Размер модели (n, s, m, l, x)
            experiment_name: Имя эксперимента
            output_dir: Директория для вывода
            config_override: Переопределение конфигурации
        """
        self.logger = get_logger(__name__)
        self.device_manager = get_device_manager()
        
        # Конфигурация
        self.model_size = model_size or config.model.model_size
        self.config = self._prepare_config(config_override)
        
        # Директории
        self.output_dir = output_dir or config.paths.trained_models_dir
        self.experiment_name = experiment_name or f"yolo_{self.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = self.output_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Модель и тренировочные параметры
        self.model = None
        self.dataset_config = None
        self.training_args = {}
        
        # Колбэки и мониторинг
        self.callback = TrainingCallback(self)
        self.model_manager = ModelManager()
        
        # Параметры early stopping
        self.patience = self.config.get('patience', config.training.patience)
        
        # WandB интеграция
        self.use_wandb = self.config.get('use_wandb', False)
        
        self.logger.info(f"Инициализирован YOLOTrainer:")
        self.logger.info(f"  - Размер модели: {self.model_size}")
        self.logger.info(f"  - Эксперимент: {self.experiment_name}")
        self.logger.info(f"  - Устройство: {self.device_manager.get_device()}")
        self.logger.info(f"  - Выходная директория: {self.experiment_dir}")
    
    def _prepare_config(self, config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """Подготовка конфигурации обучения"""
        training_config = {
            # Основные параметры
            'epochs': config.model.epochs,
            'batch_size': config.model.batch_size,
            'learning_rate': config.model.learning_rate,
            'patience': config.training.patience,
            
            # Оптимизатор
            'optimizer': config.model.optimizer,
            'weight_decay': config.model.weight_decay,
            'momentum': config.model.momentum,
            
            # Scheduler
            'scheduler': config.model.scheduler,
            'warmup_epochs': config.model.warmup_epochs,
            
            # Аугментация
            'mosaic': config.model.mosaic,
            'mixup': config.model.mixup,
            'copy_paste': config.model.copy_paste,
            
            # Устройство
            'device': str(self.device_manager.get_device()),
            'workers': config.training.workers,
            
            # Сохранение
            'save_period': config.training.save_period,
            'save_best_only': config.training.save_best_only,
            
            # Логирование
            'log_metrics': config.training.log_metrics,
            'log_images': config.training.log_images,
            
            # Дополнительные параметры
            'input_size': config.model.input_size,
            'dropout': config.model.dropout,
            'label_smoothing': config.model.label_smoothing
        }
        
        # Переопределение конфигурации
        if config_override:
            training_config.update(config_override)
        
        return training_config
    
    def prepare_model(self, pretrained: bool = True, weights_path: Optional[Path] = None) -> YOLO:
        """
        Подготовка модели для обучения
        
        Args:
            pretrained: Использовать предобученные веса
            weights_path: Путь к кастомным весам
            
        Returns:
            Подготовленная модель YOLO
        """
        self.logger.info("Подготовка модели...")
        
        if weights_path and weights_path.exists():
            # Загрузка кастомных весов
            self.logger.info(f"Загрузка весов из: {weights_path}")
            self.model = YOLO(str(weights_path))
        elif pretrained:
            # Использование предобученной модели
            model_name = f"yolov8{self.model_size}.pt"
            self.logger.info(f"Загрузка предобученной модели: {model_name}")
            self.model = YOLO(model_name)
        else:
            # Инициализация с нуля
            model_name = f"yolov8{self.model_size}.yaml"
            self.logger.info(f"Создание модели с нуля: {model_name}")
            self.model = YOLO(model_name)
        
        # Перемещение модели на устройство
        self.model.to(self.device_manager.get_device())
        
        # Получение информации о модели
        model_info = self._get_model_info()
        self.logger.info(f"Модель подготовлена:")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")
        
        return self.model
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        if not self.model:
            return {}
        
        try:
            total_params = sum(p.numel() for p in self.model.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            
            return {
                'Размер модели': self.model_size,
                'Общее количество параметров': f"{total_params:,}",
                'Обучаемых параметров': f"{trainable_params:,}",
                'Размер входа': self.config.get('input_size', 640),
                'Устройство': str(self.device_manager.get_device())
            }
        except Exception as e:
            self.logger.warning(f"Не удалось получить информацию о модели: {e}")
            return {}
    
    def prepare_dataset(self, dataset_yaml: Path) -> Dict[str, Any]:
        """
        Подготовка датасета для обучения
        
        Args:
            dataset_yaml: Путь к конфигурации датасета
            
        Returns:
            Конфигурация датасета
        """
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Конфигурация датасета не найдена: {dataset_yaml}")
        self.dataset_yaml = dataset_yaml
        # Загрузка конфигурации датасета
        with open(dataset_yaml, 'r', encoding='utf-8') as f:
            self.dataset_config = yaml.safe_load(f)
        
        self.logger.info(f"Датасет подготовлен:")
        self.logger.info(f"  - Путь: {self.dataset_config.get('path', 'не указан')}")
        self.logger.info(f"  - Классов: {self.dataset_config.get('nc', 'неизвестно')}")
        self.logger.info(f"  - Имена классов: {self.dataset_config.get('names', [])}")
        
        # Валидация путей датасета
        self._validate_dataset_paths()
        
        return self.dataset_config
    
    def _validate_dataset_paths(self):
        """Валидация путей в конфигурации датасета"""
        if not self.dataset_config:
            return
        
        dataset_path = Path(self.dataset_config.get('path', ''))
        
        for split in ['train', 'val', 'test']:
            if split in self.dataset_config:
                split_path = dataset_path / self.dataset_config[split]
                if not split_path.exists():
                    self.logger.warning(f"Путь для {split} не найден: {split_path}")
                else:
                    # Подсчет изображений
                    image_count = len(list(split_path.glob("*.jpg")) + list(split_path.glob("*.png")))
                    self.logger.info(f"  - {split}: {image_count} изображений")
    
    def setup_training_args(self, **kwargs) -> Dict[str, Any]:
        """
        Настройка аргументов для обучения
        
        Args:
            **kwargs: Дополнительные аргументы
            
        Returns:
            Словарь аргументов для обучения
        """
        self.training_args = {
            # 'data': str(self.dataset_config) if isinstance(self.dataset_config, dict) else self.dataset_config,
            'data': str(self.dataset_yaml),
            'epochs': self.config['epochs'],
            'batch': self.config['batch_size'],
            'imgsz': self.config['input_size'],
            'lr0': self.config['learning_rate'],
            'optimizer': self.config['optimizer'],
            'weight_decay': self.config['weight_decay'],
            'warmup_epochs': self.config['warmup_epochs'],
            'mosaic': self.config['mosaic'],
            'mixup': self.config['mixup'],
            'copy_paste': self.config['copy_paste'],
            'device': self.config['device'],
            'workers': self.config['workers'],
            'project': str(self.experiment_dir.parent),
            'name': self.experiment_name,
            'save': True,
            'save_period': self.config['save_period'],
            'verbose': True,
            'plots': True,
            'val': True,
            'patience': self.patience,
            'resume': False,  # Будет установлено отдельно при необходимости
        }
        
        # Добавление дополнительных аргументов
        self.training_args.update(kwargs)
        
        # Логирование настроек
        self.logger.info("Настройки обучения:")
        for key, value in self.training_args.items():
            self.logger.info(f"  {key}: {value}")
        
        return self.training_args
    
    def setup_wandb(self, project_name: str = "yolo_training"):
        """Настройка Weights & Biases"""
        if not self.use_wandb:
            return
        
        try:
            wandb.init(
                project=project_name,
                name=self.experiment_name,
                config=self.config,
                dir=str(self.experiment_dir)
            )
            self.logger.info("WandB инициализирован")
        except Exception as e:
            self.logger.warning(f"Ошибка инициализации WandB: {e}")
            self.use_wandb = False
    
    @log_execution_time()
    def train(self, 
             dataset_yaml: Path,
             pretrained: bool = True,
             weights_path: Optional[Path] = None,
             resume_from: Optional[Path] = None,
             **training_kwargs) -> Dict[str, Any]:
        """
        Запуск обучения модели
        
        Args:
            dataset_yaml: Путь к конфигурации датасета
            pretrained: Использовать предобученные веса
            weights_path: Путь к кастомным весам
            resume_from: Путь к чекпоинту для продолжения обучения
            **training_kwargs: Дополнительные аргументы обучения
            
        Returns:
            Результаты обучения
        """
        start_time = time.time()
        
        self.logger.info("=" * 50)
        self.logger.info("НАЧАЛО ОБУЧЕНИЯ YOLOV11")
        self.logger.info("=" * 50)
        
        try:
            # Подготовка модели
            if resume_from and resume_from.exists():
                self.logger.info(f"Продолжение обучения с чекпоинта: {resume_from}")
                self.model = YOLO(str(resume_from))
                self.training_args['resume'] = True
            else:
                self.prepare_model(pretrained, weights_path)
            
            # Подготовка датасета
            self.prepare_dataset(dataset_yaml)
            
            # Настройка аргументов обучения
            self.setup_training_args(**training_kwargs)
            
            # Настройка WandB
            self.setup_wandb()
            
            # Сохранение конфигурации эксперимента
            self._save_experiment_config()
            
            # Запуск обучения
            self.logger.info("Запуск обучения...")
            results = self.model.train(**self.training_args)
            
            # Обработка результатов
            training_results = self._process_training_results(results, start_time)
            
            # Сохранение результатов
            self._save_training_results(training_results)
            
            # Регистрация модели в менеджере
            self._register_trained_model(training_results)
            
            # Генерация финального отчета
            self._generate_training_report(training_results)
            
            self.logger.info("=" * 50)
            self.logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
            self.logger.info("=" * 50)
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Ошибка во время обучения: {e}")
            raise
        finally:
            if self.use_wandb:
                wandb.finish()
    
    def _save_experiment_config(self):
        """Сохранение конфигурации эксперимента"""
        config_data = {
            'experiment_name': self.experiment_name,
            'model_size': self.model_size,
            'training_config': self.config,
            'training_args': self.training_args,
            'dataset_config': self.dataset_config,
            'device_info': {
                'device': str(self.device_manager.get_device()),
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        config_file = self.experiment_dir / 'experiment_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Конфигурация эксперимента сохранена: {config_file}")
    
    def _process_training_results(self, results, start_time: float) -> Dict[str, Any]:
        """Обработка результатов обучения"""
        total_time = time.time() - start_time
        
        training_results = {
            'experiment_name': self.experiment_name,
            'model_size': self.model_size,
            'total_training_time': total_time,
            'epochs_completed': self.training_args['epochs'],
            'best_model_path': None,
            'last_model_path': None,
            'metrics': {},
            'training_history': self.callback.metrics_history,
            'best_metrics': self.callback.best_metrics,
            'early_stopped': self.callback.early_stop_triggered,
            'device_used': str(self.device_manager.get_device())
        }
        
        # Поиск сохраненных моделей
        weights_dir = self.experiment_dir / 'weights'
        if weights_dir.exists():
            best_pt = weights_dir / 'best.pt'
            last_pt = weights_dir / 'last.pt'
            
            if best_pt.exists():
                training_results['best_model_path'] = str(best_pt)
            if last_pt.exists():
                training_results['last_model_path'] = str(last_pt)
        
        # Извлечение финальных метрик из результатов YOLO
        if hasattr(results, 'results_dict'):
            training_results['metrics'] = results.results_dict
        
        return training_results
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Сохранение результатов обучения"""
        results_file = self.experiment_dir / 'training_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Результаты обучения сохранены: {results_file}")
    
    def _register_trained_model(self, training_results: Dict[str, Any]):
        """Регистрация обученной модели в менеджере"""
        if training_results.get('best_model_path'):
            try:
                model_id = self.model_manager.register_model(
                    model_path=Path(training_results['best_model_path']),
                    model_name=f"yolo_{self.model_size}_{self.experiment_name}",
                    model_type="trained",
                    description=f"Обученная модель YOLOv11-{self.model_size}",
                    metrics=training_results.get('best_metrics', {}),
                    training_config=self.config
                )
                
                training_results['registered_model_id'] = model_id
                self.logger.info(f"Модель зарегистрирована с ID: {model_id}")
                
            except Exception as e:
                self.logger.warning(f"Ошибка регистрации модели: {e}")
    
    def _generate_training_report(self, training_results: Dict[str, Any]):
        """Генерация детального отчета об обучении"""
        try:
            from src.utils.visualization import ReportGenerator
            
            report_generator = ReportGenerator()
            report_path = report_generator.generate_training_report(
                training_results, self.experiment_dir
            )
            
            training_results['report_path'] = str(report_path)
            self.logger.info(f"Отчет об обучении создан: {report_path}")
            
        except Exception as e:
            self.logger.warning(f"Ошибка создания отчета: {e}")
    
    def evaluate_model(self, 
                      model_path: Optional[Path] = None,
                      test_data: Optional[Path] = None) -> Dict[str, Any]:
        """
        Оценка обученной модели
        
        Args:
            model_path: Путь к модели для оценки
            test_data: Путь к тестовым данным
            
        Returns:
            Результаты оценки
        """
        if model_path:
            model = YOLO(str(model_path))
        elif self.model:
            model = self.model
        else:
            raise ValueError("Модель не указана и не обучена")
        
        self.logger.info("Начинается оценка модели...")
        
        # Оценка на валидационных данных
        val_results = model.val(
            data=str(self.dataset_config) if self.dataset_config else None,
            verbose=True,
            plots=True,
            save_json=True,
            project=str(self.experiment_dir),
            name='evaluation'
        )
        
        # Обработка результатов оценки
        evaluation_results = {
            'model_path': str(model_path) if model_path else 'current_model',
            'validation_metrics': val_results.results_dict if hasattr(val_results, 'results_dict') else {},
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Сохранение результатов оценки
        eval_file = self.experiment_dir / 'evaluation_results.json'
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info("Оценка модели завершена")
        return evaluation_results
    
    def hyperparameter_tuning(self, 
                            param_grid: Dict[str, List],
                            n_trials: int = 10,
                            metric_name: str = 'val_map50') -> Dict[str, Any]:
        """
        Подбор гиперпараметров
        
        Args:
            param_grid: Сетка параметров для поиска
            n_trials: Количество испытаний
            metric_name: Метрика для оптимизации
            
        Returns:
            Результаты подбора гиперпараметров
        """
        self.logger.info(f"Начинается подбор гиперпараметров ({n_trials} испытаний)...")
        
        best_score = -float('inf')
        best_params = None
        trials_results = []
        
        for trial in range(n_trials):
            self.logger.info(f"Испытание {trial + 1}/{n_trials}")
            
            # Случайный выбор параметров
            trial_params = {}
            for param_name, param_values in param_grid.items():
                trial_params[param_name] = np.random.choice(param_values)
            
            self.logger.info(f"Параметры испытания: {trial_params}")
            
            try:
                # Создание нового эксперимента для испытания
                trial_name = f"{self.experiment_name}_trial_{trial + 1}"
                trial_trainer = YOLOTrainer(
                    model_size=self.model_size,
                    experiment_name=trial_name,
                    output_dir=self.output_dir,
                    config_override=trial_params
                )
                
                # Обучение с текущими параметрами
                trial_results = trial_trainer.train(
                    dataset_yaml=Path(str(self.dataset_config)),
                    pretrained=True,
                    epochs=max(10, trial_params.get('epochs', 50) // 5)  # Сокращенное обучение
                )
                
                # Извлечение метрики
                score = trial_results.get('best_metrics', {}).get(metric_name, 0)
                
                trial_info = {
                    'trial': trial + 1,
                    'params': trial_params,
                    'score': score,
                    'results': trial_results
                }
                
                trials_results.append(trial_info)
                
                # Обновление лучшего результата
                if score > best_score:
                    best_score = score
                    best_params = trial_params.copy()
                    self.logger.info(f"Новый лучший результат: {score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Ошибка в испытании {trial + 1}: {e}")
                continue
        
        # Сохранение результатов подбора
        tuning_results = {
            'best_params': best_params,
            'best_score': best_score,
            'metric_optimized': metric_name,
            'trials_results': trials_results,
            'param_grid': param_grid,
            'n_trials': n_trials
        }
        
        tuning_file = self.experiment_dir / 'hyperparameter_tuning.json'
        with open(tuning_file, 'w', encoding='utf-8') as f:
            json.dump(tuning_results, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Подбор гиперпараметров завершен. Лучшие параметры: {best_params}")
        return tuning_results
    
    def create_training_summary(self) -> Dict[str, Any]:
        """Создание сводки обучения"""
        summary = {
            'experiment_info': {
                'name': self.experiment_name,
                'model_size': self.model_size,
                'output_directory': str(self.experiment_dir)
            },
            'configuration': self.config,
            'dataset_info': self.dataset_config,
            'device_info': {
                'device': str(self.device_manager.get_device()),
                'cuda_available': torch.cuda.is_available(),
                'memory_info': self.device_manager.get_memory_usage()
            },
            'training_history': dict(self.callback.metrics_history),
            'best_metrics': self.callback.best_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary

# Утилиты для обучения
def create_trainer_from_config(config_path: Path) -> YOLOTrainer:
    """
    Создание тренера из конфигурационного файла
    
    Args:
        config_path: Путь к конфигурации
        
    Returns:
        Настроенный YOLOTrainer
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            config_data = yaml.safe_load(f)
        else:
            config_data = json.load(f)
    
    return YOLOTrainer(
        model_size=config_data.get('model_size', 'n'),
        experiment_name=config_data.get('experiment_name'),
        config_override=config_data.get('training_config', {})
    )

def resume_training(checkpoint_path: Path, 
                   dataset_yaml: Path,
                   additional_epochs: int = 50) -> Dict[str, Any]:
    """
    Продолжение обучения с чекпоинта
    
    Args:
        checkpoint_path: Путь к чекпоинту
        dataset_yaml: Конфигурация датасета
        additional_epochs: Дополнительные эпохи
        
    Returns:
        Результаты продолжения обучения
    """
    logger = get_logger(__name__)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")
    
    # Создание тренера для продолжения
    trainer = YOLOTrainer(
        experiment_name=f"resumed_{checkpoint_path.stem}_{datetime.now().strftime('%H%M%S')}"
    )
    
    # Продолжение обучения
    results = trainer.train(
        dataset_yaml=dataset_yaml,
        resume_from=checkpoint_path,
        epochs=additional_epochs
    )
    
    logger.info(f"Обучение продолжено с {checkpoint_path}")
    return results

def compare_training_runs(experiment_dirs: List[Path]) -> Dict[str, Any]:
    """
    Сравнение нескольких прогонов обучения
    
    Args:
        experiment_dirs: Список директорий экспериментов
        
    Returns:
        Результаты сравнения
    """
    logger = get_logger(__name__)
    
    comparison_data = {}
    
    for exp_dir in experiment_dirs:
        results_file = exp_dir / 'training_results.json'
        
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                comparison_data[exp_dir.name] = results
        else:
            logger.warning(f"Результаты не найдены для {exp_dir}")
    
    # Анализ сравнения
    comparison = {
        'experiments': comparison_data,
        'summary': {},
        'best_experiment': None
    }
    
    if comparison_data:
        # Поиск лучшего эксперимента по mAP
        best_map = -1
        best_exp = None
        
        for exp_name, data in comparison_data.items():
            map_score = data.get('best_metrics', {}).get('map50', 0)
            if map_score > best_map:
                best_map = map_score
                best_exp = exp_name
        
        comparison['best_experiment'] = {
            'name': best_exp,
            'map50': best_map
        }
        
        # Статистика времени обучения
        training_times = [data.get('total_training_time', 0) for data in comparison_data.values()]
        comparison['summary']['training_time'] = {
            'min': min(training_times),
            'max': max(training_times),
            'mean': np.mean(training_times)
        }
    
    return comparison