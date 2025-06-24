"""
API для детекции объектов с использованием YOLOv11
Предоставляет REST-like интерфейс для различных типов инференса
"""
import base64
import io
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import cv2
from PIL import Image

from src.api.base_api import (
    CachedAPI, BatchAPI, ValidationError, ProcessingError, 
    ResourceNotFoundError, APIResponse, APIStatus
)
from src.models.inference import YOLOInference
from src.models.model_manager import ModelManager
from src.utils.logger import get_logger

class DetectionAPI(CachedAPI, BatchAPI):
    """API для детекции объектов"""
    
    def __init__(self, 
                 model_path: Optional[Path] = None,
                 model_manager: Optional[ModelManager] = None,
                 cache_ttl: int = 1800,  # 30 минут
                 max_batch_size: int = 50):
        """
        Инициализация API детекции
        
        Args:
            model_path: Путь к модели (если не используется model_manager)
            model_manager: Менеджер моделей
            cache_ttl: Время жизни кэша
            max_batch_size: Максимальный размер пакета
        """
        super().__init__(
            name="DetectionAPI",
            cache_ttl=cache_ttl,
            max_cache_size=200
        )
        
        self.model_manager = model_manager
        self.inference_engine = None
        self.current_model_id = None
        
        # Инициализация модели
        if model_path:
            self._load_model_from_path(model_path)
        elif model_manager:
            # Загрузка последней обученной модели
            trained_models = model_manager.list_models("trained")
            if trained_models:
                self._load_model_by_id(trained_models[0]["id"])
        
        if not self.inference_engine:
            raise ValueError("Не удалось инициализировать модель")
        
        self.max_batch_size = max_batch_size
        
        self.logger.info("DetectionAPI инициализирован")
    
    def _load_model_from_path(self, model_path: Path):
        """Загрузка модели по пути"""
        try:
            self.inference_engine = YOLOInference(model_path)
            self.current_model_id = f"direct_{model_path.name}"
            self.logger.info(f"Модель загружена из: {model_path}")
        except Exception as e:
            raise ProcessingError(f"Ошибка загрузки модели: {e}", "model_loading")
    
    def _load_model_by_id(self, model_id: str):
        """Загрузка модели по ID через менеджер"""
        try:
            if not self.model_manager:
                raise ProcessingError("ModelManager не инициализирован", "model_loading")
            
            model = self.model_manager.load_model(model_id)
            model_info = self.model_manager.get_model_info(model_id)
            model_path = Path(model_info["path"])
            
            self.inference_engine = YOLOInference(model_path)
            self.current_model_id = model_id
            self.logger.info(f"Модель {model_id} загружена")
        except Exception as e:
            raise ProcessingError(f"Ошибка загрузки модели {model_id}: {e}", "model_loading")
    
    def _custom_validation(self, request_data: Dict[str, Any]) -> None:
        """Кастомная валидация для запросов детекции"""
        action = request_data.get("action")
        
        if not action:
            raise ValidationError("Поле 'action' обязательно", "action", None)
        
        valid_actions = [
            "detect_image", "detect_batch", "detect_video", 
            "detect_base64", "list_models", "switch_model",
            "get_model_info", "health_check"
        ]
        
        if action not in valid_actions:
            raise ValidationError(f"Недопустимое действие. Доступны: {valid_actions}", "action", action)
        
        # Специфическая валидация для каждого действия
        if action == "detect_image":
            self._validate_image_detection_request(request_data)
        elif action == "detect_batch":
            self._validate_batch_detection_request(request_data)
        elif action == "detect_video":
            self._validate_video_detection_request(request_data)
        elif action == "detect_base64":
            self._validate_base64_detection_request(request_data)
        elif action == "switch_model":
            self._validate_model_switch_request(request_data)
    
    def _validate_image_detection_request(self, request_data: Dict[str, Any]):
        """Валидация запроса детекции изображения"""
        image_path = request_data.get("image_path")
        if not image_path:
            raise ValidationError("Поле 'image_path' обязательно", "image_path", None)
        
        if not Path(image_path).exists():
            raise ValidationError("Файл изображения не найден", "image_path", image_path)
    
    def _validate_batch_detection_request(self, request_data: Dict[str, Any]):
        """Валидация запроса пакетной детекции"""
        image_paths = request_data.get("image_paths")
        if not image_paths:
            raise ValidationError("Поле 'image_paths' обязательно", "image_paths", None)
        
        if not isinstance(image_paths, list):
            raise ValidationError("Поле 'image_paths' должно быть списком", "image_paths", type(image_paths))
        
        if len(image_paths) > self.max_batch_size:
            raise ValidationError(
                f"Слишком много изображений (максимум {self.max_batch_size})", 
                "image_paths", len(image_paths)
            )
        
        for i, path in enumerate(image_paths):
            if not Path(path).exists():
                raise ValidationError(f"Изображение {i} не найдено", f"image_paths[{i}]", path)
    
    def _validate_video_detection_request(self, request_data: Dict[str, Any]):
        """Валидация запроса детекции видео"""
        video_path = request_data.get("video_path")
        if not video_path:
            raise ValidationError("Поле 'video_path' обязательно", "video_path", None)
        
        if not Path(video_path).exists():
            raise ValidationError("Файл видео не найден", "video_path", video_path)
    
    def _validate_base64_detection_request(self, request_data: Dict[str, Any]):
        """Валидация запроса детекции base64 изображения"""
        image_data = request_data.get("image_base64")
        if not image_data:
            raise ValidationError("Поле 'image_base64' обязательно", "image_base64", None)
        
        try:
            # Попытка декодировать base64
            base64.b64decode(image_data)
        except Exception:
            raise ValidationError("Некорректные данные base64", "image_base64", "invalid")
    
    def _validate_model_switch_request(self, request_data: Dict[str, Any]):
        """Валидация запроса смены модели"""
        model_id = request_data.get("model_id")
        if not model_id:
            raise ValidationError("Поле 'model_id' обязательно", "model_id", None)
        
        if self.model_manager and model_id not in self.model_manager.model_registry["models"]:
            raise ResourceNotFoundError("model", model_id)
    
    def _process_request(self, request_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Обработка запроса детекции"""
        action = request_data["action"]
        
        # Маршрутизация по действиям
        if action == "detect_image":
            return self._handle_image_detection(request_data, request_id)
        elif action == "detect_batch":
            return self._handle_batch_detection(request_data, request_id)
        elif action == "detect_video":
            return self._handle_video_detection(request_data, request_id)
        elif action == "detect_base64":
            return self._handle_base64_detection(request_data, request_id)
        elif action == "list_models":
            return self._handle_list_models(request_data, request_id)
        elif action == "switch_model":
            return self._handle_model_switch(request_data, request_id)
        elif action == "get_model_info":
            return self._handle_get_model_info(request_data, request_id)
        elif action == "health_check":
            return self._handle_health_check(request_data, request_id)
        else:
            raise ProcessingError(f"Неподдерживаемое действие: {action}", "routing")
    
    def _handle_image_detection(self, request_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Обработка детекции одного изображения"""
        image_path = Path(request_data["image_path"])
        
        # Опциональные параметры
        confidence_threshold = request_data.get("confidence_threshold")
        iou_threshold = request_data.get("iou_threshold")
        save_visualization = request_data.get("save_visualization", False)
        
        # Временное изменение порогов, если указаны
        original_confidence = None
        original_iou = None
        
        if confidence_threshold is not None:
            original_confidence = self.inference_engine.confidence_threshold
            self.inference_engine.confidence_threshold = confidence_threshold
        
        if iou_threshold is not None:
            original_iou = self.inference_engine.iou_threshold
            self.inference_engine.iou_threshold = iou_threshold
        
        try:
            # Выполнение детекции
            result = self.inference_engine.predict_image(image_path)
            
            # Подготовка ответа
            response_data = {
                "request_id": request_id,
                "image_path": str(image_path),
                "model_id": self.current_model_id,
                "detections": [det.to_dict() for det in result.detections],
                "inference_time": result.inference_time,
                "image_size": result.image_size,
                "detection_count": len(result.detections)
            }
            
            # Сохранение визуализации, если запрошено
            if save_visualization and result.detections:
                vis_path = self._save_visualization(image_path, result.detections, request_id)
                response_data["visualization_path"] = str(vis_path)
            
            return response_data
            
        finally:
            # Восстановление оригинальных порогов
            if original_confidence is not None:
                self.inference_engine.confidence_threshold = original_confidence
            if original_iou is not None:
                self.inference_engine.iou_threshold = original_iou
    
    def _handle_batch_detection(self, request_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Обработка пакетной детекции"""
        image_paths = [Path(p) for p in request_data["image_paths"]]
        
        # Выполнение пакетной детекции
        results = self.inference_engine.predict_batch(image_paths)
        
        # Подготовка ответа
        batch_results = []
        total_detections = 0
        total_inference_time = 0
        
        for result in results:
            result_dict = {
                "image_path": result.image_path,
                "detections": [det.to_dict() for det in result.detections],
                "inference_time": result.inference_time,
                "detection_count": len(result.detections)
            }
            batch_results.append(result_dict)
            total_detections += len(result.detections)
            total_inference_time += result.inference_time
        
        return {
            "request_id": request_id,
            "model_id": self.current_model_id,
            "batch_size": len(image_paths),
            "results": batch_results,
            "summary": {
                "total_detections": total_detections,
                "total_inference_time": total_inference_time,
                "average_inference_time": total_inference_time / len(results) if results else 0,
                "average_detections_per_image": total_detections / len(results) if results else 0
            }
        }
    
    def _handle_video_detection(self, request_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Обработка детекции видео"""
        video_path = Path(request_data["video_path"])
        
        # Опциональные параметры
        output_path = request_data.get("output_path")
        frame_skip = request_data.get("frame_skip", 1)
        save_frames = request_data.get("save_frames", False)
        
        # Выполнение детекции видео
        results = self.inference_engine.predict_video(
            video_path=video_path,
            output_path=output_path,
            save_frames=save_frames,
            frame_skip=frame_skip
        )
        
        # Добавление метаданных запроса
        results["request_id"] = request_id
        results["model_id"] = self.current_model_id
        
        return results
    
    def _handle_base64_detection(self, request_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Обработка детекции base64 изображения"""
        image_base64 = request_data["image_base64"]
        
        try:
            # Декодирование base64
            image_data = base64.b64decode(image_base64)
            
            # Преобразование в изображение OpenCV
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ProcessingError("Не удалось декодировать изображение", "image_decoding")
            
            # Выполнение детекции
            result = self.inference_engine.predict_image(image)
            
            # Подготовка ответа
            response_data = {
                "request_id": request_id,
                "model_id": self.current_model_id,
                "detections": [det.to_dict() for det in result.detections],
                "inference_time": result.inference_time,
                "image_size": result.image_size,
                "detection_count": len(result.detections)
            }
            
            # Опционально возвращаем визуализацию в base64
            if request_data.get("return_visualization", False) and result.detections:
                vis_image = self.inference_engine.visualizer.draw_detections(image, result.detections)
                _, buffer = cv2.imencode('.jpg', vis_image)
                vis_base64 = base64.b64encode(buffer).decode('utf-8')
                response_data["visualization_base64"] = vis_base64
            
            return response_data
            
        except Exception as e:
            raise ProcessingError(f"Ошибка обработки base64 изображения: {e}", "base64_processing")
    
    def _handle_list_models(self, request_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Получение списка доступных моделей"""
        if not self.model_manager:
            return {
                "request_id": request_id,
                "current_model": self.current_model_id,
                "available_models": [],
                "message": "ModelManager не инициализирован"
            }
        
        # Получение списка моделей
        all_models = self.model_manager.list_models()
        
        return {
            "request_id": request_id,
            "current_model": self.current_model_id,
            "available_models": all_models,
            "total_models": len(all_models)
        }
    
    def _handle_model_switch(self, request_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Переключение модели"""
        model_id = request_data["model_id"]
        
        try:
            # Загрузка новой модели
            previous_model = self.current_model_id
            self._load_model_by_id(model_id)
            
            return {
                "request_id": request_id,
                "previous_model": previous_model,
                "current_model": self.current_model_id,
                "message": f"Модель переключена на {model_id}"
            }
            
        except Exception as e:
            raise ProcessingError(f"Ошибка переключения модели: {e}", "model_switch")
    
    def _handle_get_model_info(self, request_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Получение информации о текущей модели"""
        model_info = {
            "request_id": request_id,
            "current_model_id": self.current_model_id,
            "inference_config": {
                "confidence_threshold": self.inference_engine.confidence_threshold,
                "iou_threshold": self.inference_engine.iou_threshold,
                "device": str(self.inference_engine.device_manager.get_device())
            }
        }
        
        # Дополнительная информация из model_manager
        if self.model_manager and self.current_model_id in self.model_manager.model_registry["models"]:
            registry_info = self.model_manager.get_model_info(self.current_model_id)
            model_info["model_details"] = registry_info
        
        # Статистика производительности
        perf_stats = self.inference_engine.get_performance_stats()
        model_info["performance_stats"] = perf_stats
        
        return model_info
    
    def _handle_health_check(self, request_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Проверка состояния API"""
        health_status = self.get_health_status()
        
        # Дополнительные проверки для детекции
        model_status = "healthy" if self.inference_engine else "error"
        device_status = "healthy"
        
        try:
            # Проверка доступности устройства
            device = self.inference_engine.device_manager.get_device()
            memory_info = self.inference_engine.device_manager.get_memory_usage()
            
            if device.type == "cuda" and memory_info.get("usage_percent", 0) > 90:
                device_status = "warning"
                
        except Exception:
            device_status = "error"
        
        health_status.update({
            "request_id": request_id,
            "model_status": model_status,
            "device_status": device_status,
            "current_model": self.current_model_id,
            "cache_stats": self.get_cache_stats()
        })
        
        return health_status
    
    def _save_visualization(self, image_path: Path, detections: List, request_id: str) -> Path:
        """Сохранение визуализации детекций"""
        try:
            # Загрузка оригинального изображения
            image = cv2.imread(str(image_path))
            
            # Создание визуализации
            vis_image = self.inference_engine.visualizer.draw_detections(image, detections)
            
            # Сохранение
            vis_filename = f"detection_vis_{request_id}_{image_path.stem}.jpg"
            vis_path = Path("outputs/visualizations") / vis_filename
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(vis_path), vis_image)
            
            return vis_path
            
        except Exception as e:
            self.logger.warning(f"Ошибка сохранения визуализации: {e}")
            raise ProcessingError(f"Ошибка сохранения визуализации: {e}", "visualization")
    
    # Удобные методы для прямого использования (без REST структуры)
    
    def detect_image_direct(self, 
                          image_path: Union[str, Path],
                          confidence_threshold: Optional[float] = None,
                          iou_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Прямая детекция изображения без REST обертки
        
        Args:
            image_path: Путь к изображению
            confidence_threshold: Порог уверенности
            iou_threshold: Порог IoU
            
        Returns:
            Результаты детекции
        """
        request_data = {
            "action": "detect_image",
            "image_path": str(image_path)
        }
        
        if confidence_threshold is not None:
            request_data["confidence_threshold"] = confidence_threshold
        if iou_threshold is not None:
            request_data["iou_threshold"] = iou_threshold
        
        response = self.execute(request_data)
        
        if response.status == APIStatus.SUCCESS:
            return response.data
        else:
            raise ProcessingError(response.message, response.error_code)
    
    def detect_batch_direct(self, 
                          image_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Прямая пакетная детекция без REST обертки
        
        Args:
            image_paths: Список путей к изображениям
            
        Returns:
            Результаты пакетной детекции
        """
        request_data = {
            "action": "detect_batch",
            "image_paths": [str(p) for p in image_paths]
        }
        
        response = self.execute(request_data)
        
        if response.status == APIStatus.SUCCESS:
            return response.data
        else:
            raise ProcessingError(response.message, response.error_code)
    
    def detect_base64_direct(self, 
                           image_base64: str,
                           return_visualization: bool = False) -> Dict[str, Any]:
        """
        Прямая детекция base64 изображения без REST обертки
        
        Args:
            image_base64: Изображение в формате base64
            return_visualization: Возвращать визуализацию
            
        Returns:
            Результаты детекции
        """
        request_data = {
            "action": "detect_base64",
            "image_base64": image_base64,
            "return_visualization": return_visualization
        }
        
        response = self.execute(request_data)
        
        if response.status == APIStatus.SUCCESS:
            return response.data
        else:
            raise ProcessingError(response.message, response.error_code)

class FastAPIWrapper:
    """Обертка для интеграции с FastAPI"""
    
    def __init__(self, detection_api: DetectionAPI):
        """
        Инициализация FastAPI обертки
        
        Args:
            detection_api: Экземпляр DetectionAPI
        """
        self.detection_api = detection_api
        self.logger = get_logger(__name__)
    
    def create_fastapi_app(self):
        """Создание FastAPI приложения"""
        try:
            from fastapi import FastAPI, HTTPException, UploadFile, File
            from fastapi.responses import JSONResponse
            from pydantic import BaseModel
            from typing import List as TypingList
        except ImportError:
            raise ImportError("Для FastAPI обертки требуется: pip install fastapi python-multipart")
        
        app = FastAPI(
            title="YOLOv11 Object Detection API",
            description="API для детекции объектов с использованием YOLOv11",
            version="1.0.0"
        )
        
        # Модели данных для Pydantic
        class DetectionRequest(BaseModel):
            image_path: str
            confidence_threshold: Optional[float] = None
            iou_threshold: Optional[float] = None
            save_visualization: bool = False
        
        class BatchDetectionRequest(BaseModel):
            image_paths: TypingList[str]
            confidence_threshold: Optional[float] = None
            iou_threshold: Optional[float] = None
        
        class Base64DetectionRequest(BaseModel):
            image_base64: str
            return_visualization: bool = False
        
        class ModelSwitchRequest(BaseModel):
            model_id: str
        
        # Эндпоинты
        
        @app.get("/health")
        async def health_check():
            """Проверка состояния API"""
            try:
                response = self.detection_api.execute({"action": "health_check"})
                return JSONResponse(content=response.to_dict())
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/detect/image")
        async def detect_image(request: DetectionRequest):
            """Детекция объектов на изображении"""
            try:
                request_data = {
                    "action": "detect_image",
                    **request.dict()
                }
                response = self.detection_api.execute(request_data)
                return JSONResponse(content=response.to_dict())
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/detect/batch")
        async def detect_batch(request: BatchDetectionRequest):
            """Пакетная детекция объектов"""
            try:
                request_data = {
                    "action": "detect_batch",
                    **request.dict()
                }
                response = self.detection_api.execute(request_data)
                return JSONResponse(content=response.to_dict())
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/detect/base64")
        async def detect_base64(request: Base64DetectionRequest):
            """Детекция объектов на base64 изображении"""
            try:
                request_data = {
                    "action": "detect_base64",
                    **request.dict()
                }
                response = self.detection_api.execute(request_data)
                return JSONResponse(content=response.to_dict())
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/detect/upload")
        async def detect_upload(file: UploadFile = File(...)):
            """Детекция объектов на загруженном файле"""
            try:
                # Чтение файла
                contents = await file.read()
                
                # Конвертация в base64
                image_base64 = base64.b64encode(contents).decode('utf-8')
                
                request_data = {
                    "action": "detect_base64",
                    "image_base64": image_base64,
                    "return_visualization": False
                }
                
                response = self.detection_api.execute(request_data)
                return JSONResponse(content=response.to_dict())
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/models")
        async def list_models():
            """Получение списка доступных моделей"""
            try:
                response = self.detection_api.execute({"action": "list_models"})
                return JSONResponse(content=response.to_dict())
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/models/switch")
        async def switch_model(request: ModelSwitchRequest):
            """Переключение модели"""
            try:
                request_data = {
                    "action": "switch_model",
                    **request.dict()
                }
                response = self.detection_api.execute(request_data)
                return JSONResponse(content=response.to_dict())
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/models/current")
        async def get_current_model():
            """Получение информации о текущей модели"""
            try:
                response = self.detection_api.execute({"action": "get_model_info"})
                return JSONResponse(content=response.to_dict())
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/stats")
        async def get_stats():
            """Получение статистики API"""
            try:
                health_status = self.detection_api.get_health_status()
                cache_stats = self.detection_api.get_cache_stats()
                
                stats = {
                    **health_status,
                    "cache": cache_stats
                }
                
                return JSONResponse(content=stats)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        return app

# Фабричные функции для создания API

def create_detection_api(model_path: Optional[Path] = None,
                        model_manager: Optional[ModelManager] = None) -> DetectionAPI:
    """
    Фабричная функция для создания DetectionAPI
    
    Args:
        model_path: Путь к модели
        model_manager: Менеджер моделей
        
    Returns:
        Настроенный DetectionAPI
    """
    return DetectionAPI(model_path=model_path, model_manager=model_manager)

def create_fastapi_app(detection_api: DetectionAPI):
    """
    Фабричная функция для создания FastAPI приложения
    
    Args:
        detection_api: Экземпляр DetectionAPI
        
    Returns:
        FastAPI приложение
    """
    wrapper = FastAPIWrapper(detection_api)
    return wrapper.create_fastapi_app()

# Пример использования
if __name__ == "__main__":
    # Пример создания и использования API
    
    # Создание модель-менеджера
    model_manager = ModelManager()
    
    # Создание DetectionAPI
    api = create_detection_api(model_manager=model_manager)
    
    # Пример использования
    try:
        # Детекция изображения
        result = api.detect_image_direct("path/to/image.jpg")
        print("Детекции:", result["detection_count"])
        
        # Пакетная детекция
        batch_result = api.detect_batch_direct(["image1.jpg", "image2.jpg"])
        print("Пакетный результат:", batch_result["summary"])
        
    except Exception as e:
        print(f"Ошибка: {e}")
    
    # Создание FastAPI приложения
    # app = create_fastapi_app(api)
    # 
    # # Запуск сервера
    # if __name__ == "__main__":
    #     import uvicorn
    #     uvicorn.run(app, host="0.0.0.0", port=8000)