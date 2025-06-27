"""
Конфигурация проекта для детекции объектов в ресторане с GroundingDINO
Обновлено для использования специализированных классов еды и посуды
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass 
class VideoProcessingConfig:
    """Конфигурация обработки видео"""
    # Извлечение кадров
    fps_extraction: float = 2.0
    max_frames_per_video: int = 1000
    
    # Обработка изображений
    resize_frames: bool = True
    target_size: tuple = (640, 640)
    
    # Поддерживаемые форматы
    supported_formats: List[str] = field(default_factory=lambda: 
        ['.mp4', '.avi', '.mov', '.mkv', '.wmv'])


@dataclass
class AnnotationConfig:
    """Конфигурация для автоматической аннотации с GroundingDINO"""
    # GroundingDINO модель
    groundingdino_checkpoint: str = "groundingdino_swinb_cogcoor.pth"
    groundingdino_config_paths: List[str] = field(default_factory=lambda: [
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "groundingdino_config.py"
    ])
    
    # Пороги детекции
    confidence_threshold: float = 0.25
    text_threshold: float = 0.25
    box_threshold: float = 0.25
    
    # Специализированные классы для ресторанной среды
    target_classes: Optional[List[str]] = None
    detection_prompt: str = "chicken . meat . salad . soup . cup . plate . bowl . spoon . fork . knife ."
    
    # Параметры улучшения аннотаций
    enable_auto_refinement: bool = True
    refinement_confidence_threshold: float = 0.5
    
    # Валидация аннотаций
    min_bbox_area: float = 0.01  # Минимальная площадь bbox (нормализованная)
    max_bbox_area: float = 0.9   # Максимальная площадь bbox
    min_bbox_side: float = 0.05  # Минимальная сторона bbox
    aspect_ratio_range: tuple = (0.1, 10.0)  # Диапазон соотношения сторон
    
    def __post_init__(self):
        if self.target_classes is None:
            # Специализированные классы еды и посуды для GroundingDINO
            self.target_classes = [
                "chicken",     # Курица
                "meat",        # Мясо
                "salad",       # Салат  
                "soup",        # Суп
                "cup",         # Чашка
                "plate",       # Тарелка
                "bowl",        # Миска
                "spoon",       # Ложка
                "fork",        # Вилка
                "knife"        # Нож
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
    
    # Loss weights
    box_loss_weight: float = 7.5
    cls_loss_weight: float = 0.5
    dfl_loss_weight: float = 1.5


@dataclass
class GroundingDINOConfig:
    """Специальная конфигурация для GroundingDINO"""
    # Пути к модели
    checkpoint_path: str = "groundingdino_swinb_cogcoor.pth"
    config_paths: List[str] = field(default_factory=lambda: [
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "groundingdino_config.py"
    ])
    
    # Параметры детекции
    detection_threshold: float = 0.25
    text_threshold: float = 0.25
    box_threshold: float = 0.25
    
    # Промпты для детекции
    main_prompt: str = "chicken . meat . salad . soup . cup . plate . bowl . spoon . fork . knife ."
    
    # Маппинг промптов на классы
    class_prompts: Dict[str, List[str]] = field(default_factory=lambda: {
        "chicken": ["chicken", "курица", "птица"],
        "meat": ["meat", "мясо", "говядина", "свинина"],
        "salad": ["salad", "салат", "зелень", "овощи"],
        "soup": ["soup", "суп", "бульон"],
        "cup": ["cup", "чашка", "кружка"],
        "plate": ["plate", "тарелка", "блюдо"],
        "bowl": ["bowl", "миска", "чаша"],
        "spoon": ["spoon", "ложка"],
        "fork": ["fork", "вилка"],
        "knife": ["knife", "нож"]
    })
    
    # Настройки устройства
    device: str = "auto"  # auto, cuda, cpu
    use_half_precision: bool = True


@dataclass  
class QualityControlConfig:
    """Конфигурация контроля качества аннотаций"""
    # Фильтрация детекций
    min_confidence: float = 0.15
    min_detection_size: float = 0.01  # Минимальный размер детекции
    max_detection_size: float = 0.8   # Максимальный размер детекции
    aspect_ratio_range: tuple = (0.1, 10.0)  # Диапазон соотношения сторон
    
    # Удаление дубликатов
    duplicate_removal_enabled: bool = True
    iou_threshold_duplicates: float = 0.6
    
    # TTA (Test Time Augmentation)
    enable_tta: bool = False
    tta_scales: List[float] = field(default_factory=lambda: [1.0, 1.1, 0.9])
    tta_flips: List[bool] = field(default_factory=lambda: [False, True])


@dataclass
class PipelineConfig:
    """Основная конфигурация пайплайна"""
    # Компоненты конфигурации
    video_processing: VideoProcessingConfig = field(default_factory=VideoProcessingConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    groundingdino: GroundingDINOConfig = field(default_factory=GroundingDINOConfig)
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)
    
    # Общие настройки
    project_name: str = "restaurant_object_detection"
    version: str = "2.0_groundingdino"
    random_seed: int = 42
    
    # Пути
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    
    # Производительность
    use_gpu: bool = True
    num_workers: int = 4
    batch_size: int = 8
    memory_limit_gb: int = 8
    
    # Отладка и логирование
    debug_mode: bool = False
    log_level: str = "INFO"
    save_intermediate_results: bool = True
    cleanup_temp_files: bool = True


# Константы для ресторанных классов (обновлено для GroundingDINO)
RESTAURANT_CLASSES = [
    "chicken",     # Курица
    "meat",        # Мясо
    "salad",       # Салат
    "soup",        # Суп
    "cup",         # Чашка
    "plate",       # Тарелка
    "bowl",        # Миска
    "spoon",       # Ложка
    "fork",        # Вилка
    "knife"        # Нож
]

# Цветовая схема для визуализации классов
CLASS_COLORS = {
    "chicken": (255, 165, 0),    # Оранжевый
    "meat": (139, 69, 19),       # Коричневый
    "salad": (0, 128, 0),        # Зеленый
    "soup": (255, 215, 0),       # Золотой
    "cup": (70, 130, 180),       # Стальной голубой
    "plate": (220, 220, 220),    # Светло-серый
    "bowl": (255, 192, 203),     # Розовый
    "spoon": (192, 192, 192),    # Серебряный
    "fork": (169, 169, 169),     # Темно-серый
    "knife": (105, 105, 105)     # Тускло-серый
}

# Маппинг классов для совместимости с COCO (если требуется)
COCO_TO_RESTAURANT_MAPPING = {
    62: "chair",           # COCO chair -> не используется в новой схеме
    67: "dining_table",    # COCO dining table -> не используется
    47: "cup",             # COCO cup -> cup
    51: "bowl",            # COCO bowl -> bowl  
    44: "bottle",          # COCO bottle -> не используется
    46: "wine_glass",      # COCO wine glass -> не используется
    48: "fork",            # COCO fork -> fork
    49: "knife",           # COCO knife -> knife
    50: "spoon"            # COCO spoon -> spoon
}


def get_default_config() -> PipelineConfig:
    """Получение конфигурации по умолчанию"""
    return PipelineConfig()


def load_config_from_file(config_path: Path) -> PipelineConfig:
    """Загрузка конфигурации из файла"""
    import json
    
    config = get_default_config()
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # Обновление конфигурации из файла
            # Реализация зависит от структуры JSON файла
            
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
    
    return config


def save_config_to_file(config: PipelineConfig, config_path: Path):
    """Сохранение конфигурации в файл"""
    import json
    from dataclasses import asdict
    
    try:
        config_dict = asdict(config)
        
        # Конвертация Path объектов в строки
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        config_dict = convert_paths(config_dict)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Ошибка сохранения конфигурации: {e}")


# Валидация конфигурации
def validate_config(config: PipelineConfig) -> List[str]:
    """Валидация конфигурации и возврат списка ошибок"""
    errors = []
    
    # Проверка GroundingDINO файлов
    if not Path(config.groundingdino.checkpoint_path).exists():
        errors.append(f"Файл модели GroundingDINO не найден: {config.groundingdino.checkpoint_path}")
    
    # Проверка соотношения splits
    total_split = config.dataset.train_split + config.dataset.val_split + config.dataset.test_split
    if abs(total_split - 1.0) > 0.001:
        errors.append(f"Сумма train/val/test splits должна быть 1.0, получено: {total_split}")
    
    # Проверка пороговых значений
    if not (0.0 <= config.annotation.confidence_threshold <= 1.0):
        errors.append("confidence_threshold должен быть между 0.0 и 1.0")
    
    if not (0.0 <= config.quality_control.min_confidence <= 1.0):
        errors.append("min_confidence должен быть между 0.0 и 1.0")
    
    # Проверка классов
    if not config.annotation.target_classes:
        errors.append("target_classes не может быть пустым")
    
    return errors


# Создание глобального объекта конфигурации для совместимости
config = get_default_config()

# Настройка конфигурации для инференса (добавлено для совместимости)
class InferenceConfig:
    """Конфигурация для инференса (для совместимости со старым кодом)"""
    def __init__(self):
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        self.device = "auto"
        self.batch_size = 1
        self.save_visualizations = True

# Добавление к глобальному config объекту
config.inference = InferenceConfig()