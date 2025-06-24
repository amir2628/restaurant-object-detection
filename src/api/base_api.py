"""
Базовые классы для API интерфейсов
Обеспечивают единообразную структуру и обработку ошибок
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import traceback

from src.utils.logger import get_logger

class APIStatus(Enum):
    """Статусы ответов API"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PROCESSING = "processing"

@dataclass
class APIResponse:
    """Стандартный ответ API"""
    status: APIStatus
    message: str
    data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    timestamp: str = None
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        result = asdict(self)
        result['status'] = self.status.value
        return result
    
    def to_json(self) -> str:
        """Преобразование в JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

@dataclass
class APIRequest:
    """Базовый запрос API"""
    request_id: str
    timestamp: str = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class BaseAPIError(Exception):
    """Базовое исключение для API"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}

class ValidationError(BaseAPIError):
    """Ошибка валидации входных данных"""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field
        self.value = value
        self.details = {"field": field, "value": str(value) if value is not None else None}

class ProcessingError(BaseAPIError):
    """Ошибка обработки"""
    
    def __init__(self, message: str, operation: str = None):
        super().__init__(message, "PROCESSING_ERROR")
        self.operation = operation
        self.details = {"operation": operation}

class ResourceNotFoundError(BaseAPIError):
    """Ошибка - ресурс не найден"""
    
    def __init__(self, resource_type: str, resource_id: str):
        message = f"{resource_type} с ID '{resource_id}' не найден"
        super().__init__(message, "RESOURCE_NOT_FOUND")
        self.details = {"resource_type": resource_type, "resource_id": resource_id}

class BaseAPI(ABC):
    """Базовый класс для всех API"""
    
    def __init__(self, name: str = None):
        """
        Инициализация базового API
        
        Args:
            name: Имя API сервиса
        """
        self.name = name or self.__class__.__name__
        self.logger = get_logger(f"api.{self.name.lower()}")
        self.request_count = 0
        self.start_time = time.time()
        
        # Статистика
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "uptime_seconds": 0.0
        }
        
        self.logger.info(f"Инициализирован {self.name} API")
    
    def _generate_request_id(self) -> str:
        """Генерация уникального ID запроса"""
        self.request_count += 1
        timestamp = int(time.time() * 1000)
        return f"{self.name.lower()}_{timestamp}_{self.request_count:06d}"
    
    def _validate_request(self, request_data: Dict[str, Any]) -> None:
        """
        Валидация входного запроса
        
        Args:
            request_data: Данные запроса
            
        Raises:
            ValidationError: При ошибке валидации
        """
        if not isinstance(request_data, dict):
            raise ValidationError("Запрос должен быть словарем", "request_data", type(request_data))
        
        # Дополнительная валидация в подклассах
        self._custom_validation(request_data)
    
    def _custom_validation(self, request_data: Dict[str, Any]) -> None:
        """Кастомная валидация для конкретного API (переопределяется в подклассах)"""
        pass
    
    def _handle_error(self, error: Exception, request_id: str) -> APIResponse:
        """
        Обработка ошибок
        
        Args:
            error: Исключение
            request_id: ID запроса
            
        Returns:
            Ответ с ошибкой
        """
        self.stats["failed_requests"] += 1
        
        if isinstance(error, BaseAPIError):
            self.logger.error(f"API ошибка в запросе {request_id}: {error.message}")
            return APIResponse(
                status=APIStatus.ERROR,
                message=error.message,
                error_code=error.error_code,
                data={"request_id": request_id, "details": error.details}
            )
        else:
            # Неожиданная ошибка
            error_message = f"Внутренняя ошибка сервера: {str(error)}"
            self.logger.error(f"Неожиданная ошибка в запросе {request_id}: {error}")
            self.logger.error(traceback.format_exc())
            
            return APIResponse(
                status=APIStatus.ERROR,
                message="Внутренняя ошибка сервера",
                error_code="INTERNAL_ERROR",
                data={"request_id": request_id, "error_details": str(error)}
            )
    
    def _process_request(self, request_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """
        Обработка запроса (абстрактный метод)
        
        Args:
            request_data: Данные запроса
            request_id: ID запроса
            
        Returns:
            Результат обработки
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")
    
    def execute(self, request_data: Dict[str, Any], request_id: str = None) -> APIResponse:
        """
        Выполнение запроса с полной обработкой ошибок
        
        Args:
            request_data: Данные запроса
            request_id: ID запроса (генерируется автоматически, если не указан)
            
        Returns:
            Результат выполнения запроса
        """
        start_time = time.time()
        
        if request_id is None:
            request_id = self._generate_request_id()
        
        self.stats["total_requests"] += 1
        
        try:
            # Логирование запроса
            self.logger.info(f"Обработка запроса {request_id}")
            self.logger.debug(f"Данные запроса: {request_data}")
            
            # Валидация
            self._validate_request(request_data)
            
            # Обработка
            result_data = self._process_request(request_data, request_id)
            
            # Успешный ответ
            processing_time = time.time() - start_time
            self.stats["successful_requests"] += 1
            self._update_processing_time(processing_time)
            
            response = APIResponse(
                status=APIStatus.SUCCESS,
                message="Запрос выполнен успешно",
                data=result_data,
                processing_time=processing_time
            )
            
            self.logger.info(f"Запрос {request_id} выполнен за {processing_time:.3f}с")
            return response
            
        except Exception as error:
            processing_time = time.time() - start_time
            response = self._handle_error(error, request_id)
            response.processing_time = processing_time
            return response
    
    def _update_processing_time(self, processing_time: float):
        """Обновление статистики времени обработки"""
        total_successful = self.stats["successful_requests"]
        current_avg = self.stats["average_processing_time"]
        
        # Вычисление нового среднего
        new_avg = (current_avg * (total_successful - 1) + processing_time) / total_successful
        self.stats["average_processing_time"] = new_avg
    
    def get_health_status(self) -> Dict[str, Any]:
        """Получение статуса здоровья API"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        self.stats["uptime_seconds"] = uptime
        
        # Вычисление success rate
        total_requests = self.stats["total_requests"]
        success_rate = (self.stats["successful_requests"] / total_requests * 100) if total_requests > 0 else 0
        
        health_status = {
            "service_name": self.name,
            "status": "healthy" if success_rate > 95 else "degraded" if success_rate > 80 else "unhealthy",
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_uptime(uptime),
            "statistics": {
                **self.stats,
                "success_rate_percent": round(success_rate, 2),
                "requests_per_minute": round(total_requests / (uptime / 60), 2) if uptime > 0 else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return health_status
    
    def _format_uptime(self, uptime_seconds: float) -> str:
        """Форматирование времени работы"""
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {seconds}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def reset_stats(self):
        """Сброс статистики"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "uptime_seconds": 0.0
        }
        self.start_time = time.time()
        self.request_count = 0
        
        self.logger.info("Статистика API сброшена")

class AsyncBaseAPI(BaseAPI):
    """Базовый класс для асинхронных API"""
    
    def __init__(self, name: str = None, max_concurrent_requests: int = 10):
        """
        Инициализация асинхронного API
        
        Args:
            name: Имя API сервиса
            max_concurrent_requests: Максимальное количество одновременных запросов
        """
        super().__init__(name)
        self.max_concurrent_requests = max_concurrent_requests
        self.active_requests = 0
        self.queue_size = 0
    
    async def execute_async(self, request_data: Dict[str, Any], request_id: str = None) -> APIResponse:
        """
        Асинхронное выполнение запроса
        
        Args:
            request_data: Данные запроса
            request_id: ID запроса
            
        Returns:
            Результат выполнения запроса
        """
        # Проверка лимита одновременных запросов
        if self.active_requests >= self.max_concurrent_requests:
            return APIResponse(
                status=APIStatus.ERROR,
                message="Превышен лимит одновременных запросов",
                error_code="TOO_MANY_REQUESTS"
            )
        
        self.active_requests += 1
        
        try:
            # Используем синхронную версию execute
            response = self.execute(request_data, request_id)
            return response
        finally:
            self.active_requests -= 1
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Получение статуса очереди запросов"""
        return {
            "active_requests": self.active_requests,
            "max_concurrent_requests": self.max_concurrent_requests,
            "queue_utilization_percent": (self.active_requests / self.max_concurrent_requests) * 100,
            "can_accept_requests": self.active_requests < self.max_concurrent_requests
        }

class BatchAPI(BaseAPI):
    """Базовый класс для пакетной обработки"""
    
    def __init__(self, name: str = None, max_batch_size: int = 100):
        """
        Инициализация пакетного API
        
        Args:
            name: Имя API сервиса
            max_batch_size: Максимальный размер пакета
        """
        super().__init__(name)
        self.max_batch_size = max_batch_size
    
    def execute_batch(self, batch_requests: List[Dict[str, Any]]) -> List[APIResponse]:
        """
        Выполнение пакета запросов
        
        Args:
            batch_requests: Список запросов для обработки
            
        Returns:
            Список ответов
        """
        if len(batch_requests) > self.max_batch_size:
            error_response = APIResponse(
                status=APIStatus.ERROR,
                message=f"Размер пакета превышает максимальный ({self.max_batch_size})",
                error_code="BATCH_TOO_LARGE"
            )
            return [error_response]
        
        responses = []
        batch_id = self._generate_request_id()
        
        self.logger.info(f"Обработка пакета {batch_id} с {len(batch_requests)} запросами")
        
        for i, request_data in enumerate(batch_requests):
            request_id = f"{batch_id}_item_{i:04d}"
            response = self.execute(request_data, request_id)
            responses.append(response)
        
        # Статистика пакета
        successful = sum(1 for r in responses if r.status == APIStatus.SUCCESS)
        failed = len(responses) - successful
        
        self.logger.info(f"Пакет {batch_id} обработан: {successful} успешно, {failed} с ошибками")
        
        return responses

class CachedAPI(BaseAPI):
    """Базовый класс с поддержкой кэширования"""
    
    def __init__(self, name: str = None, cache_ttl: int = 3600, max_cache_size: int = 1000):
        """
        Инициализация API с кэшированием
        
        Args:
            name: Имя API сервиса
            cache_ttl: Время жизни кэша в секундах
            max_cache_size: Максимальный размер кэша
        """
        super().__init__(name)
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Генерация ключа кэша из данных запроса"""
        import hashlib
        
        # Сортируем ключи для консистентности
        sorted_data = json.dumps(request_data, sort_keys=True, default=str)
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Проверка валидности записи кэша"""
        return time.time() - cache_entry["timestamp"] < self.cache_ttl
    
    def _cleanup_cache(self):
        """Очистка устаревших записей кэша"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry["timestamp"] >= self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        # Ограничение размера кэша
        if len(self.cache) > self.max_cache_size:
            # Удаляем самые старые записи
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1]["timestamp"])
            items_to_remove = len(self.cache) - self.max_cache_size
            
            for i in range(items_to_remove):
                del self.cache[sorted_items[i][0]]
    
    def execute(self, request_data: Dict[str, Any], request_id: str = None) -> APIResponse:
        """Выполнение запроса с кэшированием"""
        cache_key = self._generate_cache_key(request_data)
        
        # Проверка кэша
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            self.cache_hits += 1
            cached_response = self.cache[cache_key]["response"]
            
            # Обновляем timestamp и request_id
            cached_response.timestamp = datetime.now().isoformat()
            if "request_id" in cached_response.data:
                cached_response.data["request_id"] = request_id or self._generate_request_id()
            
            self.logger.debug(f"Возвращен кэшированный ответ для ключа {cache_key[:8]}...")
            return cached_response
        
        # Выполнение запроса
        self.cache_misses += 1
        response = super().execute(request_data, request_id)
        
        # Кэширование успешного ответа
        if response.status == APIStatus.SUCCESS:
            self.cache[cache_key] = {
                "response": response,
                "timestamp": time.time()
            }
            
            # Очистка кэша при необходимости
            self._cleanup_cache()
        
        return response
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_ttl_seconds": self.cache_ttl
        }
    
    def clear_cache(self):
        """Очистка всего кэша"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("Кэш очищен")

class RateLimitedAPI(BaseAPI):
    """Базовый класс с ограничением частоты запросов"""
    
    def __init__(self, name: str = None, requests_per_minute: int = 60):
        """
        Инициализация API с ограничением частоты
        
        Args:
            name: Имя API сервиса
            requests_per_minute: Максимальное количество запросов в минуту
        """
        super().__init__(name)
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = []
    
    def _check_rate_limit(self) -> bool:
        """Проверка ограничения частоты запросов"""
        current_time = time.time()
        
        # Удаляем старые записи (старше минуты)
        self.request_timestamps = [
            timestamp for timestamp in self.request_timestamps
            if current_time - timestamp < 60
        ]
        
        # Проверяем лимит
        return len(self.request_timestamps) < self.requests_per_minute
    
    def execute(self, request_data: Dict[str, Any], request_id: str = None) -> APIResponse:
        """Выполнение запроса с проверкой rate limit"""
        if not self._check_rate_limit():
            return APIResponse(
                status=APIStatus.ERROR,
                message=f"Превышен лимит запросов ({self.requests_per_minute}/мин)",
                error_code="RATE_LIMIT_EXCEEDED",
                data={"requests_per_minute": self.requests_per_minute}
            )
        
        # Записываем время запроса
        self.request_timestamps.append(time.time())
        
        return super().execute(request_data, request_id)
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Получение статуса ограничения частоты"""
        current_time = time.time()
        
        # Подсчет запросов за последнюю минуту
        recent_requests = sum(
            1 for timestamp in self.request_timestamps
            if current_time - timestamp < 60
        )
        
        return {
            "requests_per_minute_limit": self.requests_per_minute,
            "requests_in_last_minute": recent_requests,
            "remaining_requests": max(0, self.requests_per_minute - recent_requests),
            "reset_time": current_time + 60 - min(
                current_time - timestamp for timestamp in self.request_timestamps
            ) if self.request_timestamps else current_time
        }