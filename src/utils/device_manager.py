"""
Менеджер устройств для автоматического выбора GPU/CPU
Обеспечивает graceful fallback с GPU на CPU
"""
import torch
import logging
import platform
from typing import Optional, List, Dict, Any
import subprocess
import psutil

class DeviceManager:
    """Класс для управления устройствами и автоматического выбора GPU/CPU"""
    
    def __init__(self, preferred_device: str = "auto"):
        """
        Инициализация менеджера устройств
        
        Args:
            preferred_device: Предпочитаемое устройство ('auto', 'cpu', 'cuda', 'mps')
        """
        self.logger = logging.getLogger(__name__)
        self.preferred_device = preferred_device
        self.device_info = self._get_device_info()
        self.selected_device = self._select_device()
        
        self.logger.info(f"Выбрано устройство: {self.selected_device}")
        self._print_device_info()
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Получение информации о доступных устройствах"""
        info = {
            'system': platform.system(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            'cuda_devices': []
        }
        
        # Информация о CUDA устройствах
        if info['cuda_available']:
            for i in range(info['cuda_device_count']):
                device_props = torch.cuda.get_device_properties(i)
                info['cuda_devices'].append({
                    'index': i,
                    'name': device_props.name,
                    'memory_gb': round(device_props.total_memory / (1024**3), 2),
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                })
        
        return info
    
    def _select_device(self) -> torch.device:
        """Выбор оптимального устройства"""
        if self.preferred_device.lower() == "cpu":
            return torch.device("cpu")
        
        elif self.preferred_device.lower() == "cuda":
            if self.device_info['cuda_available']:
                return torch.device("cuda")
            else:
                self.logger.warning("CUDA не доступна, переключение на CPU")
                return torch.device("cpu")
        
        elif self.preferred_device.lower() == "mps":
            if self.device_info['mps_available']:
                return torch.device("mps")
            else:
                self.logger.warning("MPS не доступна, переключение на CPU")
                return torch.device("cpu")
        
        else:  # auto
            return self._auto_select_device()
    
    def _auto_select_device(self) -> torch.device:
        """Автоматический выбор лучшего доступного устройства"""
        # Приоритет: CUDA > MPS > CPU
        if self.device_info['cuda_available']:
            # Выбор лучшей CUDA карты по памяти
            best_gpu = max(self.device_info['cuda_devices'], 
                          key=lambda x: x['memory_gb'])
            self.logger.info(f"Автоматически выбрана CUDA карта: {best_gpu['name']}")
            return torch.device(f"cuda:{best_gpu['index']}")
        
        elif self.device_info['mps_available']:
            self.logger.info("Автоматически выбрана MPS (Apple Silicon)")
            return torch.device("mps")
        
        else:
            self.logger.info("Автоматически выбран CPU")
            return torch.device("cpu")
    
    def _print_device_info(self):
        """Вывод информации о системе и выбранном устройстве"""
        self.logger.info("=== ИНФОРМАЦИЯ О СИСТЕМЕ ===")
        self.logger.info(f"Операционная система: {self.device_info['system']}")
        self.logger.info(f"CPU ядер: {self.device_info['cpu_count']}")
        self.logger.info(f"Оперативная память: {self.device_info['memory_gb']} GB")
        
        if self.device_info['cuda_available']:
            self.logger.info(f"CUDA устройств: {self.device_info['cuda_device_count']}")
            for gpu in self.device_info['cuda_devices']:
                self.logger.info(f"  GPU {gpu['index']}: {gpu['name']} "
                               f"({gpu['memory_gb']} GB, CC {gpu['compute_capability']})")
        else:
            self.logger.info("CUDA: недоступна")
        
        if self.device_info['mps_available']:
            self.logger.info("MPS: доступна")
        else:
            self.logger.info("MPS: недоступна")
        
        self.logger.info(f"Выбранное устройство: {self.selected_device}")
        self.logger.info("=" * 40)
    
    def get_device(self) -> torch.device:
        """Получение выбранного устройства"""
        return self.selected_device
    
    def get_batch_size_recommendation(self, base_batch_size: int = 16) -> int:
        """
        Рекомендация размера батча на основе доступной памяти
        
        Args:
            base_batch_size: Базовый размер батча
            
        Returns:
            Рекомендуемый размер батча
        """
        if self.selected_device.type == "cuda":
            gpu_memory = self.device_info['cuda_devices'][self.selected_device.index]['memory_gb']
            
            # Эвристические правила для размера батча
            if gpu_memory >= 24:  # High-end GPU
                recommended_batch_size = base_batch_size * 4
            elif gpu_memory >= 12:  # Mid-range GPU
                recommended_batch_size = base_batch_size * 2
            elif gpu_memory >= 8:  # Entry-level GPU
                recommended_batch_size = base_batch_size
            elif gpu_memory >= 4:  # Low memory GPU
                recommended_batch_size = max(base_batch_size // 2, 1)
            else:  # Very low memory
                recommended_batch_size = max(base_batch_size // 4, 1)
            
            self.logger.info(f"Рекомендуемый размер батча для GPU {gpu_memory}GB: {recommended_batch_size}")
            return recommended_batch_size
        
        elif self.selected_device.type == "mps":
            # MPS обычно имеет ограниченную память
            recommended_batch_size = max(base_batch_size // 2, 1)
            self.logger.info(f"Рекомендуемый размер батча для MPS: {recommended_batch_size}")
            return recommended_batch_size
        
        else:  # CPU
            # CPU обучение обычно медленнее, используем меньший батч
            cpu_memory = self.device_info['memory_gb']
            if cpu_memory >= 32:
                recommended_batch_size = base_batch_size
            elif cpu_memory >= 16:
                recommended_batch_size = max(base_batch_size // 2, 1)
            else:
                recommended_batch_size = max(base_batch_size // 4, 1)
            
            self.logger.info(f"Рекомендуемый размер батча для CPU {cpu_memory}GB RAM: {recommended_batch_size}")
            return recommended_batch_size
    
    def get_worker_recommendation(self, base_workers: int = 4) -> int:
        """
        Рекомендация количества воркеров для DataLoader
        
        Args:
            base_workers: Базовое количество воркеров
            
        Returns:
            Рекомендуемое количество воркеров
        """
        if self.selected_device.type == "cuda":
            # Для GPU используем больше воркеров
            recommended_workers = min(base_workers * 2, self.device_info['cpu_count'])
        else:
            # Для CPU используем меньше воркеров, чтобы оставить ядра для вычислений
            recommended_workers = min(base_workers, max(self.device_info['cpu_count'] // 2, 1))
        
        self.logger.info(f"Рекомендуемое количество воркеров: {recommended_workers}")
        return recommended_workers
    
    def optimize_memory(self):
        """Оптимизация использования памяти"""
        if self.selected_device.type == "cuda":
            # Очистка кэша CUDA
            torch.cuda.empty_cache()
            
            # Настройка параметров CUDA для оптимизации памяти
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            self.logger.info("Выполнена оптимизация памяти CUDA")
        
        elif self.selected_device.type == "mps":
            # Очистка кэша MPS
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            self.logger.info("Выполнена оптимизация памяти MPS")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Получение информации об использовании памяти
        
        Returns:
            Словарь с информацией о памяти
        """
        memory_info = {}
        
        if self.selected_device.type == "cuda":
            memory_info['allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            memory_info['reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            memory_info['total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_info['usage_percent'] = (memory_info['allocated_gb'] / memory_info['total_gb']) * 100
        
        elif self.selected_device.type == "mps":
            if hasattr(torch.mps, 'current_allocated_memory'):
                memory_info['allocated_gb'] = torch.mps.current_allocated_memory() / (1024**3)
            else:
                memory_info['allocated_gb'] = 0.0
            memory_info['total_gb'] = self.device_info['memory_gb']  # Системная память
            memory_info['usage_percent'] = (memory_info['allocated_gb'] / memory_info['total_gb']) * 100
        
        else:  # CPU
            memory = psutil.virtual_memory()
            memory_info['allocated_gb'] = (memory.total - memory.available) / (1024**3)
            memory_info['total_gb'] = memory.total / (1024**3)
            memory_info['usage_percent'] = memory.percent
        
        return memory_info
    
    def monitor_memory(self):
        """Мониторинг использования памяти"""
        memory_info = self.get_memory_usage()
        
        if 'allocated_gb' in memory_info:
            self.logger.info(f"Память: {memory_info['allocated_gb']:.2f}GB / "
                           f"{memory_info['total_gb']:.2f}GB "
                           f"({memory_info['usage_percent']:.1f}%)")
        
        # Предупреждение при высоком использовании памяти
        if memory_info.get('usage_percent', 0) > 90:
            self.logger.warning("Высокое использование памяти! Рекомендуется уменьшить размер батча.")
    
    def to_device(self, tensor_or_model, non_blocking: bool = True):
        """
        Перемещение тензора или модели на выбранное устройство
        
        Args:
            tensor_or_model: Тензор или модель для перемещения
            non_blocking: Неблокирующая передача (только для CUDA)
            
        Returns:
            Тензор или модель на целевом устройстве
        """
        try:
            if self.selected_device.type == "cuda" and hasattr(tensor_or_model, 'cuda'):
                return tensor_or_model.cuda(non_blocking=non_blocking)
            else:
                return tensor_or_model.to(self.selected_device, non_blocking=False)
        except RuntimeError as e:
            self.logger.error(f"Ошибка при перемещении на устройство: {e}")
            # Fallback на CPU
            self.logger.warning("Переключение на CPU из-за ошибки")
            self.selected_device = torch.device("cpu")
            return tensor_or_model.to(self.selected_device)
    
    def get_device_capabilities(self) -> Dict[str, Any]:
        """Получение возможностей устройства"""
        capabilities = {
            'device_type': self.selected_device.type,
            'supports_mixed_precision': False,
            'supports_gradient_checkpointing': True,
            'memory_gb': 0.0
        }
        
        if self.selected_device.type == "cuda":
            # Проверка поддержки смешанной точности
            if torch.cuda.get_device_capability()[0] >= 7:  # Volta и новее
                capabilities['supports_mixed_precision'] = True
            
            capabilities['memory_gb'] = self.device_info['cuda_devices'][0]['memory_gb']
            capabilities['compute_capability'] = self.device_info['cuda_devices'][0]['compute_capability']
        
        elif self.selected_device.type == "mps":
            # MPS поддерживает смешанную точность на некоторых операциях
            capabilities['supports_mixed_precision'] = True
            capabilities['memory_gb'] = self.device_info['memory_gb']
        
        else:  # CPU
            capabilities['memory_gb'] = self.device_info['memory_gb']
        
        return capabilities
    
    def __str__(self) -> str:
        """Строковое представление менеджера устройств"""
        return f"DeviceManager(device={self.selected_device}, " \
               f"cuda_available={self.device_info['cuda_available']}, " \
               f"mps_available={self.device_info['mps_available']})"
    
    def __repr__(self) -> str:
        return self.__str__()


def create_device_manager(preferred_device: str = "auto") -> DeviceManager:
    """
    Фабричная функция для создания менеджера устройств
    
    Args:
        preferred_device: Предпочитаемое устройство
        
    Returns:
        Экземпляр DeviceManager
    """
    return DeviceManager(preferred_device)


# Глобальный экземпляр менеджера устройств
device_manager = None

def get_device_manager(preferred_device: str = "auto") -> DeviceManager:
    """
    Получение глобального экземпляра менеджера устройств (Singleton pattern)
    
    Args:
        preferred_device: Предпочитаемое устройство
        
    Returns:
        Глобальный экземпляр DeviceManager
    """
    global device_manager
    if device_manager is None:
        device_manager = DeviceManager(preferred_device)
    return device_manager