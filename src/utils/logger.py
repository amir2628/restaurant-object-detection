"""
Система логирования для проекта детекции объектов
Поддерживает различные уровни логирования и форматы вывода
"""
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import traceback

class ColoredFormatter(logging.Formatter):
    """Цветной форматтер для консольного вывода"""
    
    # ANSI коды цветов
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Добавление цвета к уровню логирования
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        
        # Добавление цвета к имени модуля
        if hasattr(record, 'name'):
            record.name = f"\033[34m{record.name}\033[0m"  # Blue
        
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """JSON форматтер для структурированного логирования"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
            'filename': record.filename,
            'line_number': record.lineno,
            'function': record.funcName
        }
        
        # Добавление дополнительных полей, если они есть
        if hasattr(record, 'extra_data'):
            log_entry['extra_data'] = record.extra_data
        
        # Добавление информации об исключении
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False, indent=None)

class ModelLogger:
    """Специализированный логгер для отслеживания метрик модели"""
    
    def __init__(self, log_dir: Path, experiment_name: str = None):
        """
        Инициализация логгера модели
        
        Args:
            log_dir: Директория для сохранения логов
            experiment_name: Имя эксперимента
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Файлы для различных типов логов
        self.metrics_file = self.experiment_dir / "metrics.jsonl"
        self.config_file = self.experiment_dir / "config.json"
        self.events_file = self.experiment_dir / "events.log"
        
        # Инициализация логгера событий
        self.events_logger = logging.getLogger(f"model_events_{self.experiment_name}")
        self.events_logger.setLevel(logging.INFO)
        
        # Обработчик для файла событий
        events_handler = logging.FileHandler(self.events_file, encoding='utf-8')
        events_handler.setFormatter(JSONFormatter())
        self.events_logger.addHandler(events_handler)
        
        self.events_logger.info(f"Инициализирован эксперимент: {self.experiment_name}")
    
    def log_config(self, config: Dict[str, Any]):
        """Сохранение конфигурации эксперимента"""
        config_data = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': config
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        self.events_logger.info("Конфигурация сохранена", extra={'extra_data': {'config_keys': list(config.keys())}})
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float], phase: str = "train"):
        """
        Логирование метрик
        
        Args:
            epoch: Номер эпохи
            metrics: Словарь с метриками
            phase: Фаза (train, val, test)
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'experiment': self.experiment_name,
            'epoch': epoch,
            'phase': phase,
            'metrics': metrics
        }
        
        # Добавление в файл метрик
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # Логирование в события
        self.events_logger.info(f"Метрики {phase} эпохи {epoch}", 
                               extra={'extra_data': metrics})
    
    def log_model_checkpoint(self, epoch: int, model_path: str, metrics: Dict[str, float]):
        """Логирование сохранения checkpoint модели"""
        self.events_logger.info(f"Сохранен checkpoint модели", 
                               extra={'extra_data': {
                                   'epoch': epoch,
                                   'model_path': model_path,
                                   'metrics': metrics
                               }})
    
    def log_training_start(self, total_epochs: int, model_params: int):
        """Логирование начала обучения"""
        self.events_logger.info("Начато обучение модели", 
                               extra={'extra_data': {
                                   'total_epochs': total_epochs,
                                   'model_parameters': model_params
                               }})
    
    def log_training_end(self, total_time: float, best_metrics: Dict[str, float]):
        """Логирование окончания обучения"""
        self.events_logger.info("Обучение завершено", 
                               extra={'extra_data': {
                                   'total_time_seconds': total_time,
                                   'best_metrics': best_metrics
                               }})
    
    def log_error(self, error: Exception, context: str = ""):
        """Логирование ошибок"""
        self.events_logger.error(f"Ошибка: {context}", 
                                extra={'extra_data': {
                                    'error_type': type(error).__name__,
                                    'error_message': str(error),
                                    'traceback': traceback.format_exc()
                                }}, exc_info=True)

class ProjectLogger:
    """Главный логгер проекта"""
    
    def __init__(self, 
                 log_dir: Path,
                 project_name: str = "video_object_detection",
                 log_level: str = "INFO",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = True):
        """
        Инициализация логгера проекта
        
        Args:
            log_dir: Директория для логов
            project_name: Имя проекта
            log_level: Уровень логирования
            enable_console: Включить консольный вывод
            enable_file: Включить файловый вывод
            enable_json: Включить JSON формат
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.project_name = project_name
        self.log_level = getattr(logging, log_level.upper())
        
        # Создание основного логгера
        self.logger = logging.getLogger(project_name)
        self.logger.setLevel(self.log_level)
        
        # Очистка существующих обработчиков
        self.logger.handlers.clear()
        
        # Настройка обработчиков
        if enable_console:
            self._setup_console_handler()
        
        if enable_file:
            self._setup_file_handler()
        
        if enable_json:
            self._setup_json_handler()
        
        # Предотвращение дублирования сообщений
        self.logger.propagate = False
        
        self.logger.info(f"Инициализирован логгер проекта: {project_name}")
    
    def _setup_console_handler(self):
        """Настройка консольного обработчика"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        # Цветной форматтер для консоли
        console_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        console_formatter = ColoredFormatter(console_format, datefmt='%H:%M:%S')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Настройка файлового обработчика"""
        log_file = self.log_dir / f"{self.project_name}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.log_level)
        
        # Обычный форматтер для файла
        file_format = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(filename)s:%(lineno)d | %(message)s'
        file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(file_handler)
    
    def _setup_json_handler(self):
        """Настройка JSON обработчика"""
        json_file = self.log_dir / f"{self.project_name}.jsonl"
        json_handler = logging.FileHandler(json_file, encoding='utf-8')
        json_handler.setLevel(self.log_level)
        
        # JSON форматтер
        json_formatter = JSONFormatter()
        json_handler.setFormatter(json_formatter)
        
        self.logger.addHandler(json_handler)
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """Получение логгера для конкретного модуля"""
        if name:
            return logging.getLogger(f"{self.project_name}.{name}")
        return self.logger
    
    def create_model_logger(self, experiment_name: str = None) -> ModelLogger:
        """Создание специализированного логгера для модели"""
        return ModelLogger(self.log_dir, experiment_name)

# Глобальная переменная для хранения логгера проекта
_project_logger = None

def setup_logging(log_dir: Path = None,
                  project_name: str = "video_object_detection",
                  log_level: str = "INFO",
                  enable_console: bool = True,
                  enable_file: bool = True,
                  enable_json: bool = True) -> ProjectLogger:
    """
    Настройка системы логирования для всего проекта
    
    Args:
        log_dir: Директория для логов
        project_name: Имя проекта
        log_level: Уровень логирования
        enable_console: Включить консольный вывод
        enable_file: Включить файловый вывод
        enable_json: Включить JSON формат
        
    Returns:
        ProjectLogger: Настроенный логгер проекта
    """
    global _project_logger
    
    if log_dir is None:
        from config.config import config
        log_dir = config.paths.logs_dir
    
    _project_logger = ProjectLogger(
        log_dir=log_dir,
        project_name=project_name,
        log_level=log_level,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_json=enable_json
    )
    
    return _project_logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Получение логгера для модуля
    
    Args:
        name: Имя модуля
        
    Returns:
        logging.Logger: Логгер для модуля
    """
    global _project_logger
    
    if _project_logger is None:
        _project_logger = setup_logging()
    
    return _project_logger.get_logger(name)

def get_model_logger(experiment_name: str = None) -> ModelLogger:
    """
    Получение логгера для модели
    
    Args:
        experiment_name: Имя эксперимента
        
    Returns:
        ModelLogger: Логгер для модели
    """
    global _project_logger
    
    if _project_logger is None:
        _project_logger = setup_logging()
    
    return _project_logger.create_model_logger(experiment_name)

# Декораторы для логирования
def log_execution_time(logger: logging.Logger = None):
    """Декоратор для логирования времени выполнения функции"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            if logger is None:
                func_logger = get_logger(func.__module__)
            else:
                func_logger = logger
            
            start_time = time.time()
            func_logger.info(f"Начало выполнения {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                func_logger.info(f"Завершено выполнение {func.__name__} за {execution_time:.2f}с")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                func_logger.error(f"Ошибка в {func.__name__} после {execution_time:.2f}с: {e}")
                raise
        
        return wrapper
    return decorator

def log_function_call(logger: logging.Logger = None):
    """Декоратор для логирования вызовов функций с параметрами"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if logger is None:
                func_logger = get_logger(func.__module__)
            else:
                func_logger = logger
            
            # Логирование вызова (без sensitive данных)
            func_logger.debug(f"Вызов {func.__name__} с {len(args)} аргументами и {len(kwargs)} именованными параметрами")
            
            try:
                result = func(*args, **kwargs)
                func_logger.debug(f"Успешное выполнение {func.__name__}")
                return result
            except Exception as e:
                func_logger.error(f"Ошибка в {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator