"""
Основной конфигурационный файл для проекта детекции объектов
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Базовые пути проекта
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

@dataclass
class PathConfig:
    """Конфигурация путей к файлам и директориям"""
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = DATA_DIR / "raw"
    processed_data_dir: Path = DATA_DIR / "processed"
    annotations_dir: Path = DATA_DIR / "annotations"
    datasets_dir: Path = DATA_DIR / "datasets"
    
    models_dir: Path = MODELS_DIR
    pretrained_models_dir: Path = MODELS_DIR / "pretrained"
    trained_models_dir: Path = MODELS_DIR / "trained"
    
    outputs_dir: Path = OUTPUTS_DIR
    logs_dir: Path = OUTPUTS_DIR / "logs"
    metrics_dir: Path = OUTPUTS_DIR / "metrics"
    visualizations_dir: Path = OUTPUTS_DIR / "visualizations"
    reports_dir: Path = OUTPUTS_DIR / "reports"
    
    def create_directories(self):
        """Создание всех необходимых директорий"""
        directories = [
            self.raw_data_dir, self.processed_data_dir, 
            self.annotations_dir, self.datasets_dir,
            self.pretrained_models_dir, self.trained_models_dir,
            self.logs_dir, self.metrics_dir, 
            self.visualizations_dir, self.reports_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

@dataclass
class VideoProcessingConfig:
    """Конфигурация для обработки видео"""
    # Параметры извлечения кадров
    frame_extraction_rate: int = 30  # Каждый N-й кадр
    target_resolution: tuple = (640, 640)  # Целевое разрешение
    min_frame_quality: float = 0.7  # Минимальное качество кадра
    
    # Форматы файлов
    supported_video_formats: List[str] = None
    output_image_format: str = "jpg"
    
    def __post_init__(self):
        if self.supported_video_formats is None:
            self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

@dataclass
class AnnotationConfig:
    """Конфигурация для автоматической аннотации"""
    # Предобученная модель для аннотации
    pretrained_model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.1
    iou_threshold: float = 0.45
    
    # Классы для детекции (можно настроить под конкретную задачу)
    target_classes: Optional[List[str]] = None
    
    # Параметры улучшения аннотаций
    enable_auto_refinement: bool = True
    refinement_confidence_threshold: float = 0.5
    
    # Валидация аннотаций
    min_bbox_area: int = 100  # Минимальная площадь bbox
    max_bbox_ratio: float = 0.8  # Максимальное отношение bbox к изображению
    
    def __post_init__(self):
        if self.target_classes is None:
            # COCO классы, которые могут встречаться в ресторанной сцене
            self.target_classes = [
                "person", "chair", "dining table", "cup", "fork", "knife", 
                "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                "bottle", "wine glass", "cell phone", "laptop", "book"
            ]

@dataclass
class DatasetConfig:
    """Конфигурация для создания датасета"""
    # Разделение данных
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    
    # Аугментация
    enable_augmentation: bool = True
    augmentation_factor: int = 3  # Количество аугментированных версий
    
    # Валидация датасета
    min_images_per_class: int = 10
    max_images_per_class: int = 1000

@dataclass
class ModelConfig:
    """Конфигурация модели YOLOv11"""
    # Архитектура модели
    model_size: str = "n"  # n, s, m, l, x
    input_size: int = 640
    
    # Обучение
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    patience: int = 10  # Early stopping
    
    # Оптимизатор
    optimizer: str = "AdamW"
    weight_decay: float = 0.0005
    momentum: float = 0.937
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    
    # Регуляризация
    dropout: float = 0.0
    label_smoothing: float = 0.0
    
    # Аугментация при обучении
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0

# @dataclass
# class TrainingConfig:
#     """Конфигурация процесса обучения"""
#     # Устройство
#     device: str = "auto"  # auto, cpu, cuda, mps
#     workers: int = 4
    
#     # Сохранение модели
#     save_period: int = 10  # Сохранять каждые N эпох
#     save_best_only: bool = True
    
#     # Валидация
#     val_period: int = 1  # Валидация каждые N эпох
    
#     # Логирование
#     log_metrics: bool = True
#     log_images: bool = True
#     log_period: int = 10  # Логировать каждые N батчей
    
#     # Checkpoint
#     resume_training: bool = False
#     checkpoint_path: Optional[str] = None

@dataclass
class TrainingConfig:
    """Конфигурация процесса обучения"""
    # Устройство
    device: str = "auto"  # auto, cpu, cuda, mps
    workers: int = 4
    
    # Сохранение модели
    save_period: int = 10  # Сохранять каждые N эпох
    save_best_only: bool = True
    
    # Валидация
    val_period: int = 1  # Валидация каждые N эпох
    
    # Early stopping
    patience: int = 10  # Early stopping patience
    
    # Логирование
    log_metrics: bool = True
    log_images: bool = True
    log_period: int = 10  # Логировать каждые N батчей
    
    # Checkpoint
    resume_training: bool = False
    checkpoint_path: Optional[str] = None

@dataclass
class EvaluationConfig:
    """Конфигурация для оценки модели"""
    # Метрики
    iou_thresholds: List[float] = None
    confidence_thresholds: List[float] = None
    
    # Визуализация
    max_detection_images: int = 100
    max_predictions_per_image: int = 10
    
    # Анализ ошибок
    enable_error_analysis: bool = True
    save_confusion_matrix: bool = True
    save_pr_curves: bool = True
    
    def __post_init__(self):
        if self.iou_thresholds is None:
            self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        if self.confidence_thresholds is None:
            self.confidence_thresholds = [i/100 for i in range(5, 100, 5)]

@dataclass
class InferenceConfig:
    """Конфигурация для инференса"""
    # Параметры детекции
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    
    # Постобработка
    agnostic_nms: bool = False
    multi_label: bool = False
    
    # Визуализация
    line_thickness: int = 3
    font_size: float = 0.5
    show_labels: bool = True
    show_confidence: bool = True
    
    # Сохранение результатов
    save_txt: bool = True
    save_conf: bool = True
    save_crop: bool = False

class Config:
    """Главный класс конфигурации"""
    
    def __init__(self):
        self.paths = PathConfig()
        self.video_processing = VideoProcessingConfig()
        self.annotation = AnnotationConfig()
        self.dataset = DatasetConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.inference = InferenceConfig()
        
        # Создание директорий при инициализации
        self.paths.create_directories()
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Обновление конфигурации из словаря"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование конфигурации в словарь"""
        return {
            'paths': self.paths.__dict__,
            'video_processing': self.video_processing.__dict__,
            'annotation': self.annotation.__dict__,
            'dataset': self.dataset.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'inference': self.inference.__dict__,
        }

# Глобальная конфигурация
config = Config()