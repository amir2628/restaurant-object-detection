"""
Модуль для обработки видео и извлечения кадров
Поддерживает различные форматы видео и обеспечивает качественное извлечение кадров
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Generator
import logging
from tqdm import tqdm
import hashlib
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import time

from src.utils.logger import get_logger, log_execution_time
from config.config import config

@dataclass
class FrameInfo:
    """Информация о кадре"""
    frame_number: int
    timestamp: float
    file_path: Path
    width: int
    height: int
    quality_score: float
    blur_score: float
    brightness: float
    contrast: float

class QualityAssessment:
    """Класс для оценки качества кадров"""
    
    @staticmethod
    def calculate_blur_score(image: np.ndarray) -> float:
        """
        Вычисление показателя размытости изображения (Laplacian variance)
        
        Args:
            image: Изображение в формате BGR
            
        Returns:
            Показатель размытости (выше = четче)
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return 0.0
    
    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """Вычисление средней яркости изображения"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return np.mean(gray)
        except:
            return 0.0
    
    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """Вычисление контрастности изображения (стандартное отклонение)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return np.std(gray)
        except:
            return 0.0
    
    @staticmethod
    def calculate_overall_quality(image: np.ndarray) -> float:
        """
        Вычисление общего показателя качества изображения
        
        Args:
            image: Изображение в формате BGR
            
        Returns:
            Показатель качества от 0 до 1
        """
        try:
            blur_score = QualityAssessment.calculate_blur_score(image)
            brightness = QualityAssessment.calculate_brightness(image)
            contrast = QualityAssessment.calculate_contrast(image)
            
            # Нормализация показателей
            blur_normalized = min(blur_score / 1000, 1.0)  # Нормализация размытости
            brightness_normalized = 1.0 - abs(brightness - 128) / 128  # Оптимальная яркость около 128
            contrast_normalized = min(contrast / 64, 1.0)  # Нормализация контрастности
            
            # Взвешенная сумма показателей
            quality = (0.5 * blur_normalized + 0.3 * brightness_normalized + 0.2 * contrast_normalized)
            return quality
        except:
            return 0.0

class VideoProcessor:
    """Класс для обработки видео и извлечения кадров"""
    
    def __init__(self, 
                 extraction_rate: int = None,
                 target_resolution: Tuple[int, int] = None,
                 min_quality: float = None,
                 output_format: str = None):
        """
        Инициализация процессора видео
        
        Args:
            extraction_rate: Частота извлечения кадров (каждый N-й кадр)
            target_resolution: Целевое разрешение (width, height)
            min_quality: Минимальное качество кадра (0-1)
            output_format: Формат выходных изображений
        """
        self.logger = get_logger(__name__)
        
        # Использование конфигурации по умолчанию или переданных параметров
        self.extraction_rate = extraction_rate or config.video_processing.frame_extraction_rate
        self.target_resolution = target_resolution or config.video_processing.target_resolution
        self.min_quality = min_quality or config.video_processing.min_frame_quality
        self.output_format = output_format or config.video_processing.output_image_format
        
        # Поддерживаемые форматы видео
        self.supported_formats = config.video_processing.supported_video_formats
        
        self.logger.info(f"Инициализирован VideoProcessor с параметрами:")
        self.logger.info(f"  - Частота извлечения: каждый {self.extraction_rate}-й кадр")
        self.logger.info(f"  - Целевое разрешение: {self.target_resolution}")
        self.logger.info(f"  - Минимальное качество: {self.min_quality}")
        self.logger.info(f"  - Формат вывода: {self.output_format}")
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Получение информации о видео файле
        
        Args:
            video_path: Путь к видео файлу
            
        Returns:
            Словарь с информацией о видео
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Видео файл не найден: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        try:
            # Получение основных параметров
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Дополнительная информация
            codec = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_name = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)]) if codec > 0 else "unknown"
            
            video_info = {
                'file_path': str(video_path),
                'file_size_mb': video_path.stat().st_size / (1024 * 1024),
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration_seconds': duration,
                'codec': codec_name,
                'estimated_extracted_frames': max(1, total_frames // self.extraction_rate)
            }
            
            self.logger.info(f"Информация о видео {video_path.name}:")
            for key, value in video_info.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.2f}")
                else:
                    self.logger.info(f"  {key}: {value}")
            
            return video_info
            
        finally:
            cap.release()
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Изменение размера кадра с сохранением пропорций
        
        Args:
            frame: Исходный кадр
            
        Returns:
            Кадр с измененным размером
        """
        height, width = frame.shape[:2]
        target_width, target_height = self.target_resolution
        
        # Вычисление коэффициента масштабирования
        scale = min(target_width / width, target_height / height)
        
        # Новые размеры
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Изменение размера
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Добавление padding для получения точного целевого размера
        if new_width != target_width or new_height != target_height:
            # Создание черного фона
            padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Вычисление позиции для центрирования
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            # Размещение изображения по центру
            padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
            return padded
        
        return resized
    
    def _generate_frame_filename(self, video_name: str, frame_number: int, timestamp: float) -> str:
        """
        Генерация имени файла для кадра
        
        Args:
            video_name: Имя видео файла
            frame_number: Номер кадра
            timestamp: Временная метка
            
        Returns:
            Имя файла кадра
        """
        # Удаление расширения из имени видео
        video_base = Path(video_name).stem
        
        # Создание уникального хэша для предотвращения коллизий
        hash_input = f"{video_name}_{frame_number}_{timestamp:.3f}"
        frame_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        return f"{video_base}_frame_{frame_number:06d}_{frame_hash}.{self.output_format}"
    
    @log_execution_time()
    def extract_frames(self, video_path: Path, output_dir: Path) -> List[FrameInfo]:
        """
        Извлечение кадров из видео
        
        Args:
            video_path: Путь к видео файлу
            output_dir: Директория для сохранения кадров
            
        Returns:
            Список информации об извлеченных кадрах
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Видео файл не найден: {video_path}")
        
        # Проверка формата видео
        if video_path.suffix.lower() not in self.supported_formats:
            self.logger.warning(f"Возможно неподдерживаемый формат видео: {video_path.suffix}")
        
        # Создание выходной директории для данного видео
        video_output_dir = output_dir / video_path.stem
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Получение информации о видео
        try:
            video_info = self.get_video_info(video_path)
            total_frames = video_info['total_frames']
            fps = video_info['fps']
        except Exception as e:
            self.logger.error(f"Ошибка получения информации о видео {video_path}: {e}")
            return []
        
        # Инициализация захвата видео
        cap = cv2.VideoCapture(str(video_path))
        extracted_frames = []
        
        if not cap.isOpened():
            self.logger.error(f"Не удалось открыть видео: {video_path}")
            return []
        
        try:
            frame_count = 0
            extracted_count = 0
            
            # Прогресс бар
            estimated_frames = max(1, total_frames // self.extraction_rate)
            with tqdm(total=estimated_frames, 
                     desc=f"Извлечение кадров из {video_path.name}") as pbar:
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Извлечение каждого N-го кадра
                    if frame_count % self.extraction_rate == 0:
                        try:
                            # Вычисление временной метки
                            timestamp = frame_count / fps if fps > 0 else frame_count
                            
                            # Изменение размера кадра
                            resized_frame = self._resize_frame(frame)
                            
                            # Оценка качества кадра
                            quality_score = QualityAssessment.calculate_overall_quality(resized_frame)
                            
                            # Фильтрация по качеству
                            if quality_score >= self.min_quality:
                                # Генерация имени файла
                                filename = self._generate_frame_filename(
                                    video_path.name, frame_count, timestamp
                                )
                                frame_path = video_output_dir / filename
                                
                                # Сохранение кадра
                                success = cv2.imwrite(str(frame_path), resized_frame)
                                
                                if success:
                                    # Сбор информации о кадре
                                    frame_info = FrameInfo(
                                        frame_number=frame_count,
                                        timestamp=timestamp,
                                        file_path=frame_path,
                                        width=self.target_resolution[0],
                                        height=self.target_resolution[1],
                                        quality_score=quality_score,
                                        blur_score=QualityAssessment.calculate_blur_score(resized_frame),
                                        brightness=QualityAssessment.calculate_brightness(resized_frame),
                                        contrast=QualityAssessment.calculate_contrast(resized_frame)
                                    )
                                    
                                    extracted_frames.append(frame_info)
                                    extracted_count += 1
                                else:
                                    self.logger.warning(f"Не удалось сохранить кадр: {frame_path}")
                            
                            pbar.update(1)
                        except Exception as e:
                            self.logger.warning(f"Ошибка обработки кадра {frame_count}: {e}")
                    
                    frame_count += 1
                
        except Exception as e:
            self.logger.error(f"Ошибка при извлечении кадров из {video_path}: {e}")
        finally:
            cap.release()
        
        self.logger.info(f"Извлечено {extracted_count} кадров из {total_frames} "
                        f"(коэффициент: {extracted_count/total_frames*100:.1f}%)")
        
        return extracted_frames
    
    def process_videos_batch(self, 
                           video_paths: List[Path], 
                           output_base_dir: Path,
                           max_workers: int = None) -> Dict[str, List[FrameInfo]]:
        """
        Пакетная обработка нескольких видео
        
        Args:
            video_paths: Список путей к видео файлам
            output_base_dir: Базовая директория для вывода
            max_workers: Максимальное количество воркеров
            
        Returns:
            Словарь с результатами обработки каждого видео
        """
        if max_workers is None:
            max_workers = min(len(video_paths), mp.cpu_count() // 2)  # Ограничиваем для экономии памяти
        
        self.logger.info(f"Начинается пакетная обработка {len(video_paths)} видео "
                        f"с использованием {max_workers} воркеров")
        
        results = {}
        
        # Последовательная обработка для большей стабильности
        for video_path in tqdm(video_paths, desc="Обработка видео"):
            try:
                frames_info = self.extract_frames(video_path, output_base_dir)
                results[str(video_path)] = frames_info
                self.logger.info(f"Успешно обработано: {video_path.name} ({len(frames_info)} кадров)")
            except Exception as e:
                self.logger.error(f"Ошибка при обработке {video_path.name}: {e}")
                results[str(video_path)] = []
        
        # Сохранение общей статистики
        self._save_processing_statistics(results, output_base_dir)
        
        return results
    
    def _save_processing_statistics(self, 
                                  results: Dict[str, List[FrameInfo]], 
                                  output_dir: Path):
        """Сохранение статистики обработки"""
        from datetime import datetime
        
        stats = {
            'processing_timestamp': datetime.now().isoformat(),
            'total_videos': len(results),
            'total_frames_extracted': sum(len(frames) for frames in results.values()),
            'videos_statistics': {}
        }
        
        for video_path, frames in results.items():
            video_name = Path(video_path).name
            if frames:
                avg_quality = np.mean([f.quality_score for f in frames])
                avg_blur = np.mean([f.blur_score for f in frames])
                avg_brightness = np.mean([f.brightness for f in frames])
                avg_contrast = np.mean([f.contrast for f in frames])
            else:
                avg_quality = avg_blur = avg_brightness = avg_contrast = 0
            
            stats['videos_statistics'][video_name] = {
                'frames_extracted': len(frames),
                'average_quality': avg_quality,
                'average_blur_score': avg_blur,
                'average_brightness': avg_brightness,
                'average_contrast': avg_contrast
            }
        
        # Сохранение в JSON файл
        stats_file = output_dir / 'processing_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Статистика обработки сохранена в: {stats_file}")
    
    def extract_frames_from_directory(self, 
                                    input_dir: Path, 
                                    output_dir: Path,
                                    recursive: bool = True) -> Dict[str, List[FrameInfo]]:
        """
        Извлечение кадров из всех видео в директории
        
        Args:
            input_dir: Входная директория с видео
            output_dir: Выходная директория для кадров
            recursive: Рекурсивный поиск видео файлов
            
        Returns:
            Результаты обработки всех видео
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Входная директория не найдена: {input_dir}")
        
        # Поиск видео файлов
        video_files = []
        
        # Основные видео форматы
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        
        if recursive:
            for ext in video_extensions:
                video_files.extend(input_dir.rglob(f"*{ext}"))
                video_files.extend(input_dir.rglob(f"*{ext.upper()}"))
        else:
            for ext in video_extensions:
                video_files.extend(input_dir.glob(f"*{ext}"))
                video_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        if not video_files:
            self.logger.warning(f"Видео файлы не найдены в директории: {input_dir}")
            return {}
        
        self.logger.info(f"Найдено {len(video_files)} видео файлов")
        
        # Пакетная обработка
        return self.process_videos_batch(video_files, output_dir)
    
    def get_frame_statistics(self, frames_info: List[FrameInfo]) -> Dict[str, Any]:
        """
        Получение статистики по извлеченным кадрам
        
        Args:
            frames_info: Список информации о кадрах
            
        Returns:
            Словарь со статистикой
        """
        if not frames_info:
            return {'total_frames': 0}
        
        qualities = [f.quality_score for f in frames_info]
        blur_scores = [f.blur_score for f in frames_info]
        brightnesses = [f.brightness for f in frames_info]
        contrasts = [f.contrast for f in frames_info]
        
        stats = {
            'total_frames': len(frames_info),
            'quality_stats': {
                'mean': np.mean(qualities),
                'std': np.std(qualities),
                'min': np.min(qualities),
                'max': np.max(qualities),
                'median': np.median(qualities)
            },
            'blur_stats': {
                'mean': np.mean(blur_scores),
                'std': np.std(blur_scores),
                'min': np.min(blur_scores),
                'max': np.max(blur_scores)
            },
            'brightness_stats': {
                'mean': np.mean(brightnesses),
                'std': np.std(brightnesses)
            },
            'contrast_stats': {
                'mean': np.mean(contrasts),
                'std': np.std(contrasts)
            },
            'duration_coverage': {
                'first_frame_time': frames_info[0].timestamp,
                'last_frame_time': frames_info[-1].timestamp,
                'total_duration': frames_info[-1].timestamp - frames_info[0].timestamp
            }
        }
        
        return stats


# Дополнительные утилиты

def create_frames_dataframe(frames_info: List[FrameInfo]):
    """
    Создание DataFrame из информации о кадрах для анализа
    
    Args:
        frames_info: Список информации о кадрах
        
    Returns:
        DataFrame с данными о кадрах
    """
    try:
        import pandas as pd
        
        data = []
        for frame in frames_info:
            data.append({
                'frame_number': frame.frame_number,
                'timestamp': frame.timestamp,
                'file_path': str(frame.file_path),
                'width': frame.width,
                'height': frame.height,
                'quality_score': frame.quality_score,
                'blur_score': frame.blur_score,
                'brightness': frame.brightness,
                'contrast': frame.contrast
            })
        
        return pd.DataFrame(data)
    except ImportError:
        # Если pandas не установлен, возвращаем обычный словарь
        return [frame.__dict__ for frame in frames_info]

def filter_frames_by_quality(frames_info: List[FrameInfo], 
                           min_quality: float = 0.5,
                           min_blur: float = None) -> List[FrameInfo]:
    """
    Фильтрация кадров по качеству
    
    Args:
        frames_info: Список информации о кадрах
        min_quality: Минимальное общее качество
        min_blur: Минимальный показатель резкости
        
    Returns:
        Отфильтрованный список кадров
    """
    filtered = []
    
    for frame in frames_info:
        if frame.quality_score >= min_quality:
            if min_blur is None or frame.blur_score >= min_blur:
                filtered.append(frame)
    
    return filtered

def extract_frames_with_intervals(video_path: Path, 
                                output_dir: Path,
                                interval_seconds: float = 1.0) -> List[FrameInfo]:
    """
    Извлечение кадров с заданным временным интервалом
    
    Args:
        video_path: Путь к видео
        output_dir: Выходная директория
        interval_seconds: Интервал между кадрами в секундах
        
    Returns:
        Список извлеченных кадров
    """
    processor = VideoProcessor()
    
    # Получение информации о видео
    video_info = processor.get_video_info(video_path)
    fps = video_info['fps']
    
    # Вычисление частоты извлечения на основе интервала
    extraction_rate = max(1, int(fps * interval_seconds))
    
    # Создание процессора с новой частотой
    custom_processor = VideoProcessor(extraction_rate=extraction_rate)
    
    return custom_processor.extract_frames(video_path, output_dir)