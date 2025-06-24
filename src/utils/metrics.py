"""
Модуль для вычисления и анализа метрик детекции объектов
Включает mAP, Precision, Recall, F1-Score и другие метрики для YOLOv11
"""
import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import json
from collections import defaultdict
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

@dataclass
class DetectionResult:
    """Результат детекции для одного объекта"""
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    image_id: str

@dataclass
class GroundTruth:
    """Истинная аннотация объекта"""
    class_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    image_id: str
    difficult: bool = False

class BBoxUtils:
    """Утилиты для работы с ограничивающими прямоугольниками"""
    
    @staticmethod
    def calculate_iou(bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> float:
        """
        Вычисление IoU (Intersection over Union) между двумя bbox
        
        Args:
            bbox1: Первый bbox (x1, y1, x2, y2)
            bbox2: Второй bbox (x1, y1, x2, y2)
            
        Returns:
            Значение IoU от 0 до 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Координаты пересечения
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Проверка на пересечение
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        # Площадь пересечения
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Площади bbox'ов
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Площадь объединения
        union = area1 + area2 - intersection
        
        # IoU
        iou = intersection / union if union > 0 else 0.0
        return iou
    
    @staticmethod
    def calculate_area(bbox: Tuple[float, float, float, float]) -> float:
        """Вычисление площади bbox"""
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)
    
    @staticmethod
    def bbox_to_center_format(bbox: Tuple[float, float, float, float], 
                             img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Преобразование bbox в центральный формат (YOLO)"""
        x1, y1, x2, y2 = bbox
        
        center_x = ((x1 + x2) / 2) / img_width
        center_y = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        return center_x, center_y, width, height

class MetricsCalculator:
    """Калькулятор метрик для детекции объектов"""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Инициализация калькулятора метрик
        
        Args:
            num_classes: Количество классов
            class_names: Имена классов
        """
        self.logger = get_logger(__name__)
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        
        # Хранение результатов
        self.detections: List[DetectionResult] = []
        self.ground_truths: List[GroundTruth] = []
        
        self.logger.info(f"Инициализирован MetricsCalculator для {num_classes} классов")
    
    def add_detections(self, detections: List[DetectionResult]):
        """Добавление результатов детекции"""
        self.detections.extend(detections)
    
    def add_ground_truths(self, ground_truths: List[GroundTruth]):
        """Добавление истинных аннотаций"""
        self.ground_truths.extend(ground_truths)
    
    def clear(self):
        """Очистка накопленных данных"""
        self.detections.clear()
        self.ground_truths.clear()
    
    def calculate_ap_per_class(self, 
                              class_id: int, 
                              iou_threshold: float = 0.5) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Вычисление Average Precision для одного класса
        
        Args:
            class_id: ID класса
            iou_threshold: Порог IoU
            
        Returns:
            Кортеж (AP, precision_curve, recall_curve)
        """
        # Фильтрация по классу
        class_detections = [d for d in self.detections if d.class_id == class_id]
        class_gts = [gt for gt in self.ground_truths if gt.class_id == class_id]
        
        if len(class_gts) == 0:
            return 0.0, np.array([]), np.array([])
        
        # Сортировка детекций по уверенности
        class_detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Группировка GT по изображениям
        gt_by_image = defaultdict(list)
        for gt in class_gts:
            gt_by_image[gt.image_id].append(gt)
        
        # Массивы для precision/recall
        tp = np.zeros(len(class_detections))
        fp = np.zeros(len(class_detections))
        
        # Отслеживание использованных GT
        matched_gts = set()
        
        # Проход по детекциям
        for i, detection in enumerate(class_detections):
            image_gts = gt_by_image.get(detection.image_id, [])
            
            max_iou = 0.0
            best_gt_idx = -1
            
            # Поиск лучшего соответствия
            for j, gt in enumerate(image_gts):
                iou = BBoxUtils.calculate_iou(detection.bbox, gt.bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = j
            
            # Определение TP/FP
            gt_key = f"{detection.image_id}_{best_gt_idx}"
            
            if max_iou >= iou_threshold and gt_key not in matched_gts:
                tp[i] = 1
                matched_gts.add(gt_key)
            else:
                fp[i] = 1
        
        # Накопительные суммы
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Precision и Recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / len(class_gts)
        
        # Вычисление AP (интеграл под PR-кривой)
        ap = self._calculate_ap_from_pr_curve(precision, recall)
        
        return ap, precision, recall
    
    def _calculate_ap_from_pr_curve(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """Вычисление AP из precision-recall кривой (метод VOC2010)"""
        # Добавление граничных точек
        mpre = np.concatenate(([0], precision, [0]))
        mrec = np.concatenate(([0], recall, [1]))
        
        # Сглаживание precision
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        # Поиск точек изменения recall
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # Интегрирование
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        return ap
    
    def calculate_map(self, 
                     iou_thresholds: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Вычисление mean Average Precision
        
        Args:
            iou_thresholds: Список порогов IoU для усреднения
            
        Returns:
            Словарь с mAP метриками
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5]  # mAP@0.5
        
        results = {}
        
        # Вычисление для каждого порога IoU
        for iou_thresh in iou_thresholds:
            aps = []
            
            for class_id in range(self.num_classes):
                ap, _, _ = self.calculate_ap_per_class(class_id, iou_thresh)
                aps.append(ap)
            
            map_value = np.mean(aps)
            results[f"mAP@{iou_thresh:.2f}"] = map_value