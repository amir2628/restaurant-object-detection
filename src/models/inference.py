"""
Модуль для инференса обученной модели YOLOv11
Поддерживает обработку изображений, видео и потоков в реальном времени
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import time
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm

from src.utils.logger import get_logger, log_execution_time
from src.utils.device_manager import get_device_manager
from src.utils.visualization import BBoxVisualizer
from config.config import config

@dataclass
class Detection:
    """Результат детекции"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    bbox_normalized: Tuple[float, float, float, float]  # нормализованные координаты
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return asdict(self)

@dataclass
class InferenceResult:
    """Результат инференса для изображения"""
    image_path: str
    image_size: Tuple[int, int]  # width, height
    detections: List[Detection]
    inference_time: float
    preprocessing_time: float
    postprocessing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        result = asdict(self)
        result['detections'] = [det.to_dict() for det in self.detections]
        return result

class YOLOInference:
    """Класс для инференса модели YOLO"""
    
    def __init__(self, 
                 model_path: Path,
                 confidence_threshold: float = None,
                 iou_threshold: float = None,
                 device: str = "auto"):
        """
        Инициализация модуля инференса
        
        Args:
            model_path: Путь к обученной модели
            confidence_threshold: Порог уверенности
            iou_threshold: Порог IoU для NMS
            device: Устройство для вычислений
        """
        self.logger = get_logger(__name__)
        self.device_manager = get_device_manager(device)
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        # Параметры инференса
        self.confidence_threshold = confidence_threshold or config.inference.confidence_threshold
        self.iou_threshold = iou_threshold or config.inference.iou_threshold
        
        # Загрузка модели
        self.model = self._load_model()
        
        # Инициализация визуализатора
        self.visualizer = BBoxVisualizer()
        
        # Статистика
        self.inference_stats = {
            'total_inferences': 0,
            'total_detections': 0,
            'average_inference_time': 0.0,
            'images_processed': 0,
            'videos_processed': 0
        }
        
        self.logger.info(f"Инициализирован YOLOInference:")
        self.logger.info(f"  - Модель: {self.model_path}")
        self.logger.info(f"  - Устройство: {self.device_manager.get_device()}")
        self.logger.info(f"  - Порог уверенности: {self.confidence_threshold}")
        self.logger.info(f"  - Порог IoU: {self.iou_threshold}")
    
    def _load_model(self) -> YOLO:
        """Загрузка обученной модели"""
        try:
            self.logger.info(f"Загрузка модели {self.model_path}...")
            
            model = YOLO(str(self.model_path))
            
            # Перемещение на выбранное устройство
            model.to(self.device_manager.get_device())
            
            # Информация о модели
            self.logger.info(f"Модель загружена успешно")
            self.logger.info(f"Классов в модели: {len(model.names)}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Предобработка изображения
        
        Args:
            image: Исходное изображение
            
        Returns:
            Кортеж (обработанное изображение, время обработки)
        """
        start_time = time.time()
        
        # Здесь может быть дополнительная предобработка
        # YOLO автоматически выполняет resize и нормализацию
        processed_image = image.copy()
        
        processing_time = time.time() - start_time
        return processed_image, processing_time
    
    def _postprocess_results(self, results, image_shape: Tuple[int, int]) -> Tuple[List[Detection], float]:
        """
        Постобработка результатов модели
        
        Args:
            results: Результаты модели YOLO
            image_shape: Размер изображения (height, width)
            
        Returns:
            Кортеж (список детекций, время обработки)
        """
        start_time = time.time()
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                height, width = image_shape[:2]
                
                for box in result.boxes:
                    # Извлечение данных
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Фильтрация по уверенности
                    if confidence >= self.confidence_threshold:
                        class_name = self.model.names[class_id]
                        
                        # Нормализованные координаты
                        bbox_normalized = (
                            x1 / width,
                            y1 / height,
                            x2 / width,
                            y2 / height
                        )
                        
                        detection = Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            bbox_normalized=bbox_normalized
                        )
                        
                        detections.append(detection)
        
        processing_time = time.time() - start_time
        return detections, processing_time
    
    @log_execution_time()
    def predict_image(self, image_path: Union[str, Path, np.ndarray]) -> InferenceResult:
        """
        Предсказание для одного изображения
        
        Args:
            image_path: Путь к изображению или массив изображения
            
        Returns:
            Результат инференса
        """
        start_total_time = time.time()
        
        # Загрузка изображения
        if isinstance(image_path, (str, Path)):
            image_path_str = str(image_path)
            image = cv2.imread(image_path_str)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        else:
            image_path_str = "array_input"
            image = image_path
        
        height, width = image.shape[:2]
        
        # Предобработка
        processed_image, preprocess_time = self._preprocess_image(image)
        
        # Инференс
        inference_start = time.time()
        results = self.model.predict(
            source=processed_image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        inference_time = time.time() - inference_start
        
        # Постобработка
        detections, postprocess_time = self._postprocess_results(results, image.shape)
        
        # Обновление статистики
        self.inference_stats['total_inferences'] += 1
        self.inference_stats['total_detections'] += len(detections)
        self.inference_stats['images_processed'] += 1
        
        # Обновление среднего времени инференса
        total_time = self.inference_stats['total_inferences']
        self.inference_stats['average_inference_time'] = (
            (self.inference_stats['average_inference_time'] * (total_time - 1) + inference_time) / total_time
        )
        
        result = InferenceResult(
            image_path=image_path_str,
            image_size=(width, height),
            detections=detections,
            inference_time=inference_time,
            preprocessing_time=preprocess_time,
            postprocessing_time=postprocess_time
        )
        
        return result
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[InferenceResult]:
        """
        Пакетное предсказание для нескольких изображений
        
        Args:
            image_paths: Список путей к изображениям
            
        Returns:
            Список результатов инференса
        """
        self.logger.info(f"Начинается пакетный инференс для {len(image_paths)} изображений")
        
        results = []
        
        for image_path in tqdm(image_paths, desc="Обработка изображений"):
            try:
                result = self.predict_image(image_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Ошибка при обработке {image_path}: {e}")
                continue
        
        self.logger.info(f"Обработано {len(results)}/{len(image_paths)} изображений")
        return results
    
    @log_execution_time()
    def predict_video(self, video_path: Union[str, Path], 
                     output_path: Optional[Union[str, Path]] = None,
                     save_frames: bool = False,
                     frame_skip: int = 1) -> Dict[str, Any]:
        """
        Предсказание для видео
        
        Args:
            video_path: Путь к видео файлу
            output_path: Путь для сохранения результата
            save_frames: Сохранять обработанные кадры
            frame_skip: Обрабатывать каждый N-й кадр
            
        Returns:
            Результаты обработки видео
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Видео файл не найден: {video_path}")
        
        self.logger.info(f"Начинается обработка видео: {video_path}")
        
        # Открытие видео
        cap = cv2.VideoCapture(str(video_path))
        
        # Получение параметров видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Подготовка записи видео
        video_writer = None
        if output_path:
            output_path = Path(output_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height)
            )
        
        # Результаты
        frame_results = []
        frame_count = 0
        processed_frames = 0
        
        try:
            with tqdm(total=total_frames, desc="Обработка кадров") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Обработка каждого N-го кадра
                    if frame_count % frame_skip == 0:
                        # Инференс
                        result = self.predict_image(frame)
                        frame_results.append({
                            'frame_number': frame_count,
                            'timestamp': frame_count / fps,
                            'detections': [det.to_dict() for det in result.detections],
                            'inference_time': result.inference_time
                        })
                        
                        # Визуализация результатов
                        if result.detections:
                            visualized_frame = self.visualizer.draw_detections(
                                frame, result.detections
                            )
                        else:
                            visualized_frame = frame
                        
                        processed_frames += 1
                    else:
                        visualized_frame = frame
                    
                    # Запись в выходное видео
                    if video_writer:
                        video_writer.write(visualized_frame)
                    
                    # Сохранение кадра
                    if save_frames and frame_count % frame_skip == 0:
                        frame_dir = output_path.parent / f"{output_path.stem}_frames"
                        frame_dir.mkdir(exist_ok=True)
                        frame_filename = frame_dir / f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(str(frame_filename), visualized_frame)
                    
                    frame_count += 1
                    pbar.update(1)
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
        
        # Обновление статистики
        self.inference_stats['videos_processed'] += 1
        
        # Сводка результатов
        total_detections = sum(len(fr['detections']) for fr in frame_results)
        avg_inference_time = np.mean([fr['inference_time'] for fr in frame_results]) if frame_results else 0
        
        video_results = {
            'video_path': str(video_path),
            'output_path': str(output_path) if output_path else None,
            'video_info': {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames,
                'processed_frames': processed_frames
            },
            'detection_summary': {
                'total_detections': total_detections,
                'average_detections_per_frame': total_detections / processed_frames if processed_frames > 0 else 0,
                'average_inference_time': avg_inference_time
            },
            'frame_results': frame_results
        }
        
        self.logger.info(f"Обработка видео завершена:")
        self.logger.info(f"  - Обработано кадров: {processed_frames}/{total_frames}")
        self.logger.info(f"  - Общее количество детекций: {total_detections}")
        self.logger.info(f"  - Среднее время инференса: {avg_inference_time:.3f}с")
        
        return video_results
    
    def predict_realtime(self, camera_id: int = 0, 
                        display_results: bool = True,
                        save_video: bool = False,
                        output_path: Optional[str] = None) -> None:
        """
        Инференс в реальном времени с камеры
        
        Args:
            camera_id: ID камеры
            display_results: Отображать результаты
            save_video: Сохранять видео
            output_path: Путь для сохранения видео
        """
        self.logger.info(f"Запуск инференса в реальном времени с камеры {camera_id}")
        
        # Инициализация камеры
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру {camera_id}")
        
        # Получение параметров
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Подготовка записи
        video_writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.logger.info("Нажмите 'q' для выхода")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Инференс
                result = self.predict_image(frame)
                
                # Визуализация
                if result.detections:
                    visualized_frame = self.visualizer.draw_detections(
                        frame, result.detections
                    )
                else:
                    visualized_frame = frame
                
                # Добавление информации о производительности
                cv2.putText(
                    visualized_frame,
                    f"Inference: {result.inference_time:.3f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.putText(
                    visualized_frame,
                    f"Detections: {len(result.detections)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Отображение
                if display_results:
                    cv2.imshow('YOLO Real-time Detection', visualized_frame)
                
                # Сохранение
                if video_writer:
                    video_writer.write(visualized_frame)
                
                # Выход по нажатию 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if display_results:
                cv2.destroyAllWindows()
        
        self.logger.info("Инференс в реальном времени завершен")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Получение статистики производительности"""
        return self.inference_stats.copy()
    
    def save_results(self, results: Union[InferenceResult, List[InferenceResult], Dict], 
                    output_path: Union[str, Path], format: str = "json") -> None:
        """
        Сохранение результатов инференса
        
        Args:
            results: Результаты для сохранения
            output_path: Путь для сохранения
            format: Формат сохранения ('json', 'txt')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            if isinstance(results, InferenceResult):
                data = results.to_dict()
            elif isinstance(results, list):
                data = [r.to_dict() if isinstance(r, InferenceResult) else r for r in results]
            else:
                data = results
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        elif format.lower() == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                if isinstance(results, InferenceResult):
                    f.write(f"Image: {results.image_path}\n")
                    f.write(f"Detections: {len(results.detections)}\n")
                    for det in results.detections:
                        f.write(f"  {det.class_name}: {det.confidence:.3f}\n")
                elif isinstance(results, list):
                    for result in results:
                        if isinstance(result, InferenceResult):
                            f.write(f"Image: {result.image_path}\n")
                            f.write(f"Detections: {len(result.detections)}\n")
                            for det in result.detections:
                                f.write(f"  {det.class_name}: {det.confidence:.3f}\n")
                            f.write("\n")
        
        self.logger.info(f"Результаты сохранены в: {output_path}")

# Утилиты для инференса
def benchmark_model(model_path: Path, 
                   test_images: List[Path], 
                   device: str = "auto") -> Dict[str, Any]:
    """
    Бенчмарк модели на тестовых изображениях
    
    Args:
        model_path: Путь к модели
        test_images: Список тестовых изображений
        device: Устройство для тестирования
        
    Returns:
        Результаты бенчмарка
    """
    logger = get_logger(__name__)
    
    inference = YOLOInference(model_path, device=device)
    
    logger.info(f"Запуск бенчмарка на {len(test_images)} изображениях")
    
    start_time = time.time()
    results = inference.predict_batch(test_images)
    total_time = time.time() - start_time
    
    # Анализ результатов
    inference_times = [r.inference_time for r in results]
    total_detections = sum(len(r.detections) for r in results)
    
    benchmark_results = {
        'total_images': len(test_images),
        'successful_predictions': len(results),
        'total_time': total_time,
        'average_time_per_image': total_time / len(results) if results else 0,
        'total_detections': total_detections,
        'average_detections_per_image': total_detections / len(results) if results else 0,
        'inference_time_stats': {
            'min': min(inference_times) if inference_times else 0,
            'max': max(inference_times) if inference_times else 0,
            'mean': np.mean(inference_times) if inference_times else 0,
            'std': np.std(inference_times) if inference_times else 0
        },
        'fps': len(results) / total_time if total_time > 0 else 0
    }
    
    logger.info(f"Бенчмарк завершен:")
    logger.info(f"  - FPS: {benchmark_results['fps']:.2f}")
    logger.info(f"  - Среднее время инференса: {benchmark_results['inference_time_stats']['mean']:.3f}с")
    logger.info(f"  - Общее количество детекций: {total_detections}")
    
    return benchmark_results