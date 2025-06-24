"""
Модуль для управления моделями YOLOv11
Включает загрузку, сохранение, версионирование и оптимизацию моделей
"""
import torch
import shutil
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from ultralytics import YOLO
import yaml
import numpy as np
from src.utils.logger import get_logger, log_execution_time
from src.utils.device_manager import get_device_manager
from config.config import config

class ModelManager:
    """Класс для управления жизненным циклом моделей"""
    
    def __init__(self, models_dir: Path = None):
        """
        Инициализация менеджера моделей
        
        Args:
            models_dir: Директория для хранения моделей
        """
        self.logger = get_logger(__name__)
        self.device_manager = get_device_manager()
        
        self.models_dir = models_dir or config.paths.models_dir
        self.pretrained_dir = self.models_dir / "pretrained"
        self.trained_dir = self.models_dir / "trained"
        self.exports_dir = self.models_dir / "exported"
        
        # Создание директорий
        for dir_path in [self.pretrained_dir, self.trained_dir, self.exports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Реестр моделей
        self.registry_file = self.models_dir / "model_registry.json"
        self.model_registry = self._load_registry()
        
        self.logger.info(f"Инициализирован ModelManager в: {self.models_dir}")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Загрузка реестра моделей"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Ошибка при загрузке реестра: {e}")
        
        return {
            "models": {},
            "last_updated": datetime.now().isoformat(),
            "version": "1.0"
        }
    
    def _save_registry(self):
        """Сохранение реестра моделей"""
        self.model_registry["last_updated"] = datetime.now().isoformat()
        
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.model_registry, f, ensure_ascii=False, indent=2, default=str)
    
    def _calculate_model_hash(self, model_path: Path) -> str:
        """Вычисление хэша модели для контроля версий"""
        hash_sha256 = hashlib.sha256()
        
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def register_model(self, 
                      model_path: Path, 
                      model_name: str,
                      model_type: str = "trained",
                      description: str = "",
                      metrics: Optional[Dict[str, float]] = None,
                      training_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Регистрация модели в реестре
        
        Args:
            model_path: Путь к модели
            model_name: Имя модели
            model_type: Тип модели (pretrained, trained, exported)
            description: Описание модели
            metrics: Метрики модели
            training_config: Конфигурация обучения
            
        Returns:
            ID зарегистрированной модели
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        # Генерация уникального ID
        model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Вычисление хэша
        model_hash = self._calculate_model_hash(model_path)
        
        # Получение информации о модели
        model_info = self._extract_model_info(model_path)
        
        # Запись в реестр
        self.model_registry["models"][model_id] = {
            "name": model_name,
            "type": model_type,
            "path": str(model_path),
            "description": description,
            "hash": model_hash,
            "size_mb": model_path.stat().st_size / (1024 * 1024),
            "created_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "training_config": training_config or {},
            "model_info": model_info,
            "device_compatibility": self._check_device_compatibility(model_path)
        }
        
        self._save_registry()
        
        self.logger.info(f"Модель зарегистрирована: {model_id}")
        return model_id
    
    def _extract_model_info(self, model_path: Path) -> Dict[str, Any]:
        """Извлечение информации о модели"""
        try:
            # Загрузка модели для анализа
            model = YOLO(str(model_path))
            
            info = {
                "architecture": "YOLOv8",  # YOLO архитектура
                "input_size": getattr(model.model, 'imgsz', 640),
                "classes": len(model.names) if hasattr(model, 'names') else 0,
                "class_names": list(model.names.values()) if hasattr(model, 'names') else [],
                "parameters": sum(p.numel() for p in model.model.parameters()) if hasattr(model, 'model') else 0
            }
            
            return info
            
        except Exception as e:
            self.logger.warning(f"Не удалось извлечь информацию о модели: {e}")
            return {}
    
    def _check_device_compatibility(self, model_path: Path) -> Dict[str, bool]:
        """Проверка совместимости с устройствами"""
        compatibility = {
            "cpu": True,  # Все модели работают на CPU
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }
        
        try:
            # Попытка загрузить модель на текущем устройстве
            model = YOLO(str(model_path))
            model.to(self.device_manager.get_device())
            compatibility["current_device"] = True
        except Exception:
            compatibility["current_device"] = False
        
        return compatibility
    
    def load_model(self, model_id: str) -> YOLO:
        """
        Загрузка модели по ID
        
        Args:
            model_id: ID модели в реестре
            
        Returns:
            Загруженная модель YOLO
        """
        if model_id not in self.model_registry["models"]:
            raise ValueError(f"Модель {model_id} не найдена в реестре")
        
        model_info = self.model_registry["models"][model_id]
        model_path = Path(model_info["path"])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        # Проверка хэша для целостности
        current_hash = self._calculate_model_hash(model_path)
        if current_hash != model_info["hash"]:
            self.logger.warning(f"Хэш модели {model_id} не совпадает! Возможно файл был изменен.")
        
        # Загрузка модели
        model = YOLO(str(model_path))
        model.to(self.device_manager.get_device())
        
        self.logger.info(f"Модель {model_id} загружена на {self.device_manager.get_device()}")
        return model
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получение списка моделей
        
        Args:
            model_type: Фильтр по типу модели
            
        Returns:
            Список информации о моделях
        """
        models = []
        
        for model_id, model_info in self.model_registry["models"].items():
            if model_type is None or model_info["type"] == model_type:
                model_summary = {
                    "id": model_id,
                    "name": model_info["name"],
                    "type": model_info["type"],
                    "description": model_info["description"],
                    "size_mb": model_info["size_mb"],
                    "created_at": model_info["created_at"],
                    "classes": model_info.get("model_info", {}).get("classes", 0),
                    "best_map": model_info.get("metrics", {}).get("map50", 0.0)
                }
                models.append(model_summary)
        
        # Сортировка по дате создания
        models.sort(key=lambda x: x["created_at"], reverse=True)
        return models
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Получение детальной информации о модели"""
        if model_id not in self.model_registry["models"]:
            raise ValueError(f"Модель {model_id} не найдена")
        
        return self.model_registry["models"][model_id].copy()
    
    def delete_model(self, model_id: str, delete_file: bool = False) -> bool:
        """
        Удаление модели из реестра
        
        Args:
            model_id: ID модели
            delete_file: Удалить файл модели
            
        Returns:
            True, если удаление успешно
        """
        if model_id not in self.model_registry["models"]:
            self.logger.warning(f"Модель {model_id} не найдена в реестре")
            return False
        
        model_info = self.model_registry["models"][model_id]
        
        # Удаление файла, если требуется
        if delete_file:
            model_path = Path(model_info["path"])
            if model_path.exists():
                model_path.unlink()
                self.logger.info(f"Файл модели удален: {model_path}")
        
        # Удаление из реестра
        del self.model_registry["models"][model_id]
        self._save_registry()
        
        self.logger.info(f"Модель {model_id} удалена из реестра")
        return True
    
    @log_execution_time()
    def export_model(self, 
                    model_id: str, 
                    format: str = "onnx",
                    optimize: bool = True) -> Path:
        """
        Экспорт модели в различные форматы
        
        Args:
            model_id: ID модели
            format: Формат экспорта
            optimize: Оптимизировать модель
            
        Returns:
            Путь к экспортированной модели
        """
        # Загрузка модели
        model = self.load_model(model_id)
        model_info = self.get_model_info(model_id)
        
        # Экспорт
        self.logger.info(f"Экспорт модели {model_id} в формат {format}...")
        
        exported_path = model.export(
            format=format,
            optimize=optimize,
            half=False,  # Для совместимости
            dynamic=False
        )
        
        # Перемещение в директорию экспортов
        exported_file = Path(exported_path)
        final_path = self.exports_dir / f"{model_info['name']}_{format}.{exported_file.suffix}"
        
        if exported_file != final_path:
            shutil.move(str(exported_file), str(final_path))
        
        # Регистрация экспортированной модели
        export_id = self.register_model(
            model_path=final_path,
            model_name=f"{model_info['name']}_{format}",
            model_type="exported",
            description=f"Экспорт модели {model_id} в формат {format}",
            metrics=model_info.get("metrics"),
            training_config={"source_model": model_id, "export_format": format}
        )
        
        self.logger.info(f"Модель экспортирована: {final_path} (ID: {export_id})")
        return final_path
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        Сравнение нескольких моделей по метрикам
        
        Args:
            model_ids: Список ID моделей для сравнения
            
        Returns:
            Результаты сравнения
        """
        comparison = {
            "models": {},
            "summary": {},
            "best_by_metric": {}
        }
        
        metrics_keys = set()
        
        # Сбор информации о моделях
        for model_id in model_ids:
            if model_id in self.model_registry["models"]:
                model_info = self.model_registry["models"][model_id]
                comparison["models"][model_id] = {
                    "name": model_info["name"],
                    "metrics": model_info.get("metrics", {}),
                    "size_mb": model_info["size_mb"],
                    "created_at": model_info["created_at"]
                }
                metrics_keys.update(model_info.get("metrics", {}).keys())
        
        # Анализ метрик
        for metric in metrics_keys:
            values = []
            best_model = None
            best_value = -float('inf')
            
            for model_id in comparison["models"]:
                value = comparison["models"][model_id]["metrics"].get(metric, 0)
                values.append(value)
                
                if value > best_value:
                    best_value = value
                    best_model = model_id
            
            comparison["summary"][metric] = {
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "mean": sum(values) / len(values) if values else 0,
                "std": np.std(values) if values else 0
            }
            
            comparison["best_by_metric"][metric] = {
                "model_id": best_model,
                "value": best_value
            }
        
        return comparison
    
    def backup_models(self, backup_dir: Path) -> List[str]:
        """
        Создание резервной копии всех моделей
        
        Args:
            backup_dir: Директория для бэкапа
            
        Returns:
            Список скопированных моделей
        """
        backup_dir.mkdir(parents=True, exist_ok=True)
        backed_up = []
        
        # Копирование реестра
        backup_registry = backup_dir / "model_registry.json"
        shutil.copy2(self.registry_file, backup_registry)
        
        # Копирование моделей
        for model_id, model_info in self.model_registry["models"].items():
            model_path = Path(model_info["path"])
            
            if model_path.exists():
                backup_path = backup_dir / model_path.name
                shutil.copy2(model_path, backup_path)
                backed_up.append(model_id)
        
        self.logger.info(f"Создан бэкап {len(backed_up)} моделей в: {backup_dir}")
        return backed_up
    
    def restore_models(self, backup_dir: Path) -> int:
        """
        Восстановление моделей из бэкапа
        
        Args:
            backup_dir: Директория с бэкапом
            
        Returns:
            Количество восстановленных моделей
        """
        backup_registry = backup_dir / "model_registry.json"
        
        if not backup_registry.exists():
            raise FileNotFoundError(f"Реестр бэкапа не найден: {backup_registry}")
        
        # Загрузка реестра бэкапа
        with open(backup_registry, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        restored_count = 0
        
        # Восстановление моделей
        for model_id, model_info in backup_data["models"].items():
            backup_model_path = backup_dir / Path(model_info["path"]).name
            
            if backup_model_path.exists():
                # Восстановление в оригинальную директорию
                original_path = Path(model_info["path"])
                original_path.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(backup_model_path, original_path)
                
                # Обновление реестра
                self.model_registry["models"][model_id] = model_info
                restored_count += 1
        
        self._save_registry()
        
        self.logger.info(f"Восстановлено {restored_count} моделей из бэкапа")
        return restored_count
    
    def cleanup_old_models(self, keep_last_n: int = 5) -> List[str]:
        """
        Очистка старых моделей
        
        Args:
            keep_last_n: Количество последних моделей для сохранения
            
        Returns:
            Список удаленных моделей
        """
        models_by_name = {}
        
        # Группировка по именам
        for model_id, model_info in self.model_registry["models"].items():
            name = model_info["name"]
            if name not in models_by_name:
                models_by_name[name] = []
            models_by_name[name].append((model_id, model_info["created_at"]))
        
        deleted_models = []
        
        # Удаление старых версий
        for name, models in models_by_name.items():
            # Сортировка по дате создания
            models.sort(key=lambda x: x[1], reverse=True)
            
            # Удаление старых моделей
            for model_id, _ in models[keep_last_n:]:
                if self.delete_model(model_id, delete_file=True):
                    deleted_models.append(model_id)
        
        self.logger.info(f"Удалено {len(deleted_models)} старых моделей")
        return deleted_models