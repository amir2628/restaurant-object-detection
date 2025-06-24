"""
Модуль для визуализации результатов детекции и метрик обучения
Поддерживает различные типы графиков и интерактивные визуализации
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import pandas as pd
from dataclasses import dataclass
import colorsys
import random

from src.utils.logger import get_logger
from config.config import config

@dataclass
class Detection:
    """Результат детекции для визуализации"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    bbox_normalized: Optional[Tuple[float, float, float, float]] = None

class BBoxVisualizer:
    """Класс для визуализации ограничивающих прямоугольников"""
    
    def __init__(self, 
                 class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
                 font_scale: float = 0.6,
                 thickness: int = 2,
                 show_confidence: bool = True):
        """
        Инициализация визуализатора
        
        Args:
            class_colors: Словарь цветов для классов
            font_scale: Размер шрифта
            thickness: Толщина линий
            show_confidence: Показывать уверенность
        """
        self.logger = get_logger(__name__)
        self.font_scale = font_scale
        self.thickness = thickness
        self.show_confidence = show_confidence
        
        # Инициализация цветов
        self.class_colors = class_colors or {}
        self._color_generator = self._generate_colors()
        
        self.logger.info("Инициализирован BBoxVisualizer")
    
    def _generate_colors(self):
        """Генератор уникальных цветов для классов"""
        used_colors = set(self.class_colors.values())
        
        while True:
            # Генерация случайного цвета с хорошей контрастностью
            hue = random.random()
            saturation = 0.7 + random.random() * 0.3  # 0.7-1.0
            value = 0.8 + random.random() * 0.2       # 0.8-1.0
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            color = tuple(int(c * 255) for c in rgb)
            
            if color not in used_colors:
                used_colors.add(color)
                yield color
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Получение цвета для класса
        
        Args:
            class_name: Имя класса
            
        Returns:
            Цвет в формате BGR
        """
        if class_name not in self.class_colors:
            color_rgb = next(self._color_generator)
            # Конвертация RGB в BGR для OpenCV
            self.class_colors[class_name] = (color_rgb[2], color_rgb[1], color_rgb[0])
        
        return self.class_colors[class_name]
    
    def draw_detections(self, 
                       image: np.ndarray, 
                       detections: List[Detection],
                       class_filter: Optional[List[str]] = None) -> np.ndarray:
        """
        Рисование детекций на изображении
        
        Args:
            image: Исходное изображение
            detections: Список детекций
            class_filter: Фильтр классов для отображения
            
        Returns:
            Изображение с нанесенными детекциями
        """
        result_image = image.copy()
        
        for detection in detections:
            # Фильтрация по классам
            if class_filter and detection.class_name not in class_filter:
                continue
            
            # Получение координат
            x1, y1, x2, y2 = detection.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Получение цвета класса
            color = self.get_class_color(detection.class_name)
            
            # Рисование прямоугольника
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, self.thickness)
            
            # Подготовка текста
            if self.show_confidence:
                label = f"{detection.class_name}: {detection.confidence:.2f}"
            else:
                label = detection.class_name
            
            # Размер текста для подложки
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness
            )
            
            # Координаты для текста
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > label_height else y1 + label_height + 10
            
            # Рисование подложки для текста
            cv2.rectangle(
                result_image,
                (text_x, text_y - label_height - baseline),
                (text_x + label_width, text_y + baseline),
                color,
                -1
            )
            
            # Рисование текста
            cv2.putText(
                result_image,
                label,
                (text_x, text_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),  # Белый текст
                self.thickness
            )
        
        return result_image
    
    def create_detection_summary_image(self, 
                                     detections: List[Detection],
                                     image_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """
        Создание сводного изображения с информацией о детекциях
        
        Args:
            detections: Список детекций
            image_size: Размер итогового изображения
            
        Returns:
            Сводное изображение
        """
        width, height = image_size
        summary_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Подсчет классов
        class_counts = {}
        for detection in detections:
            class_counts[detection.class_name] = class_counts.get(detection.class_name, 0) + 1
        
        # Заголовок
        title = f"Detection Summary: {len(detections)} objects detected"
        cv2.putText(summary_image, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Информация по классам
        y_offset = 80
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            color = self.get_class_color(class_name)
            text = f"{class_name}: {count}"
            
            # Цветной квадрат
            cv2.rectangle(summary_image, (20, y_offset - 15), (40, y_offset - 5), color, -1)
            
            # Текст
            cv2.putText(summary_image, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            y_offset += 30
        
        return summary_image

class MetricsVisualizer:
    """Класс для визуализации метрик обучения"""
    
    def __init__(self):
        """Инициализация визуализатора метрик"""
        self.logger = get_logger(__name__)
        
        # Настройка стиля matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.logger.info("Инициализирован MetricsVisualizer")
    
    def plot_training_history(self, 
                             metrics_data: List[Dict[str, Any]], 
                             output_path: Optional[Path] = None,
                             show_plot: bool = False) -> None:
        """
        Построение графиков истории обучения
        
        Args:
            metrics_data: Данные метрик по эпохам
            output_path: Путь для сохранения графика
            show_plot: Показать график
        """
        if not metrics_data:
            self.logger.warning("Нет данных для построения графика")
            return
        
        # Преобразование в DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Создание подграфиков
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics History', fontsize=16)
        
        # График потерь
        if 'train_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
        if 'val_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # График mAP
        if 'map50' in df.columns:
            axes[0, 1].plot(df['epoch'], df['map50'], label='mAP@0.5', color='green')
        if 'map50_95' in df.columns:
            axes[0, 1].plot(df['epoch'], df['map50_95'], label='mAP@0.5:0.95', color='orange')
        axes[0, 1].set_title('Mean Average Precision')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # График Precision/Recall
        if 'precision' in df.columns:
            axes[1, 0].plot(df['epoch'], df['precision'], label='Precision', color='purple')
        if 'recall' in df.columns:
            axes[1, 0].plot(df['epoch'], df['recall'], label='Recall', color='brown')
        if 'f1_score' in df.columns:
            axes[1, 0].plot(df['epoch'], df['f1_score'], label='F1-Score', color='pink')
        axes[1, 0].set_title('Precision, Recall, F1-Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # График Learning Rate
        if 'learning_rate' in df.columns:
            axes[1, 1].plot(df['epoch'], df['learning_rate'], label='Learning Rate', color='red')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Сохранение
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"График сохранен в: {output_path}")
        
        # Отображение
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_class_distribution(self, 
                               class_counts: Dict[str, int],
                               output_path: Optional[Path] = None,
                               show_plot: bool = False) -> None:
        """
        Построение распределения классов
        
        Args:
            class_counts: Счетчик объектов по классам
            output_path: Путь для сохранения
            show_plot: Показать график
        """
        if not class_counts:
            self.logger.warning("Нет данных о классах")
            return
        
        # Подготовка данных
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # Создание графика
        plt.figure(figsize=(12, 8))
        
        # Горизонтальная гистограмма
        bars = plt.barh(classes, counts, color=plt.cm.Set3(np.linspace(0, 1, len(classes))))
        
        # Добавление значений на бары
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{counts[i]}', ha='left', va='center')
        
        plt.title('Class Distribution', fontsize=16)
        plt.xlabel('Number of Objects')
        plt.ylabel('Classes')
        plt.grid(axis='x', alpha=0.3)
        
        # Настройка макета
        plt.tight_layout()
        
        # Сохранение
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"График распределения классов сохранен в: {output_path}")
        
        # Отображение
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def create_interactive_metrics_dashboard(self, 
                                           metrics_data: List[Dict[str, Any]],
                                           output_path: Optional[Path] = None) -> str:
        """
        Создание интерактивного дашборда метрик
        
        Args:
            metrics_data: Данные метрик
            output_path: Путь для сохранения HTML
            
        Returns:
            HTML код дашборда
        """
        if not metrics_data:
            self.logger.warning("Нет данных для дашборда")
            return ""
        
        df = pd.DataFrame(metrics_data)
        
        # Создание подграфиков
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'mAP', 'Precision/Recall/F1', 'Learning Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # График потерь
        if 'train_loss' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['train_loss'], name='Train Loss', line_color='blue'),
                row=1, col=1
            )
        if 'val_loss' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['val_loss'], name='Val Loss', line_color='red'),
                row=1, col=1
            )
        
        # График mAP
        if 'map50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['map50'], name='mAP@0.5', line_color='green'),
                row=1, col=2
            )
        if 'map50_95' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['map50_95'], name='mAP@0.5:0.95', line_color='orange'),
                row=1, col=2
            )
        
        # График Precision/Recall/F1
        if 'precision' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['precision'], name='Precision', line_color='purple'),
                row=2, col=1
            )
        if 'recall' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['recall'], name='Recall', line_color='brown'),
                row=2, col=1
            )
        if 'f1_score' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['f1_score'], name='F1-Score', line_color='pink'),
                row=2, col=1
            )
        
        # График Learning Rate
        if 'learning_rate' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['learning_rate'], name='Learning Rate', 
                          line_color='red', yaxis='y2'),
                row=2, col=2
            )
            fig.update_yaxes(type="log", row=2, col=2)
        
        # Обновление макета
        fig.update_layout(
            title_text="Training Metrics Dashboard",
            showlegend=True,
            height=800
        )
        
        # Конвертация в HTML
        html_str = fig.to_html(include_plotlyjs=True)
        
        # Сохранение
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_str)
            self.logger.info(f"Интерактивный дашборд сохранен в: {output_path}")
        
        return html_str
    
    def plot_confusion_matrix(self, 
                             confusion_matrix: np.ndarray,
                             class_names: List[str],
                             output_path: Optional[Path] = None,
                             show_plot: bool = False) -> None:
        """
        Построение матрицы ошибок
        
        Args:
            confusion_matrix: Матрица ошибок
            class_names: Имена классов
            output_path: Путь для сохранения
            show_plot: Показать график
        """
        plt.figure(figsize=(10, 8))
        
        # Нормализация матрицы
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Построение heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title('Confusion Matrix (Normalized)', fontsize=16)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Сохранение
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Матрица ошибок сохранена в: {output_path}")
        
        # Отображение
        if show_plot:
            plt.show()
        else:
            plt.close()

class ReportGenerator:
    """Генератор отчетов с визуализациями"""
    
    def __init__(self):
        """Инициализация генератора отчетов"""
        self.logger = get_logger(__name__)
        self.bbox_visualizer = BBoxVisualizer()
        self.metrics_visualizer = MetricsVisualizer()
    
    def generate_training_report(self, 
                               training_results: Dict[str, Any],
                               output_dir: Path) -> Path:
        """
        Генерация отчета об обучении
        
        Args:
            training_results: Результаты обучения
            output_dir: Директория для сохранения отчета
            
        Returns:
            Путь к созданному отчету
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Создание графиков
        if 'metrics_history' in training_results:
            # График метрик
            metrics_plot_path = output_dir / "training_metrics.png"
            self.metrics_visualizer.plot_training_history(
                training_results['metrics_history'],
                output_path=metrics_plot_path
            )
            
            # Интерактивный дашборд
            dashboard_path = output_dir / "interactive_dashboard.html"
            self.metrics_visualizer.create_interactive_metrics_dashboard(
                training_results['metrics_history'],
                output_path=dashboard_path
            )
        
        # HTML отчет
        report_path = output_dir / "training_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLOv11 Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .image {{ text-align: center; margin: 20px 0; }}
                .config {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>YOLOv11 Training Report</h1>
                <p>Experiment: {training_results.get('experiment_name', 'Unknown')}</p>
                <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Training Summary</h2>
                <table class="metrics-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Training Time</td><td>{training_results.get('training_time_hours', 0):.2f} hours</td></tr>
                    <tr><td>Best mAP@0.5</td><td>{training_results.get('best_metrics', {}).get('map50', 0):.4f}</td></tr>
                    <tr><td>Best mAP@0.5:0.95</td><td>{training_results.get('best_metrics', {}).get('map50_95', 0):.4f}</td></tr>
                    <tr><td>Best Precision</td><td>{training_results.get('best_metrics', {}).get('precision', 0):.4f}</td></tr>
                    <tr><td>Best Recall</td><td>{training_results.get('best_metrics', {}).get('recall', 0):.4f}</td></tr>
                    <tr><td>Best F1-Score</td><td>{training_results.get('best_metrics', {}).get('f1_score', 0):.4f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Training Metrics</h2>
                <div class="image">
                    <img src="training_metrics.png" alt="Training Metrics" style="max-width: 100%;">
                </div>
                <p><a href="interactive_dashboard.html">View Interactive Dashboard</a></p>
            </div>
            
            <div class="section">
                <h2>Model Configuration</h2>
                <div class="config">
                    <pre>{json.dumps(training_results.get('config', {}), indent=2)}</pre>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Отчет об обучении создан: {report_path}")
        return report_path
    
    def generate_inference_report(self, 
                                inference_results: List[Dict[str, Any]],
                                output_dir: Path) -> Path:
        """
        Генерация отчета об инференсе
        
        Args:
            inference_results: Результаты инференса
            output_dir: Директория для сохранения
            
        Returns:
            Путь к отчету
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Анализ результатов
        total_images = len(inference_results)
        total_detections = sum(len(r.get('detections', [])) for r in inference_results)
        avg_inference_time = np.mean([r.get('inference_time', 0) for r in inference_results])
        
        # Подсчет классов
        class_counts = {}
        for result in inference_results:
            for detection in result.get('detections', []):
                class_name = detection.get('class_name', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # График распределения классов
        if class_counts:
            class_dist_path = output_dir / "class_distribution.png"
            self.metrics_visualizer.plot_class_distribution(
                class_counts,
                output_path=class_dist_path
            )
        
        # HTML отчет
        report_path = output_dir / "inference_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLOv11 Inference Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .stats-table {{ border-collapse: collapse; width: 100%; }}
                .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .stats-table th {{ background-color: #f2f2f2; }}
                .image {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>YOLOv11 Inference Report</h1>
                <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Inference Summary</h2>
                <table class="stats-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Images</td><td>{total_images}</td></tr>
                    <tr><td>Total Detections</td><td>{total_detections}</td></tr>
                    <tr><td>Average Detections per Image</td><td>{total_detections/total_images if total_images > 0 else 0:.2f}</td></tr>
                    <tr><td>Average Inference Time</td><td>{avg_inference_time:.3f} seconds</td></tr>
                    <tr><td>Estimated FPS</td><td>{1/avg_inference_time if avg_inference_time > 0 else 0:.1f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Class Distribution</h2>
                <div class="image">
                    <img src="class_distribution.png" alt="Class Distribution" style="max-width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <h2>Detection Details</h2>
                <table class="stats-table">
                    <tr><th>Class</th><th>Count</th><th>Percentage</th></tr>
        """
        
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_detections) * 100 if total_detections > 0 else 0
            html_content += f"<tr><td>{class_name}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Отчет об инференсе создан: {report_path}")
        return report_path