"""
Скрипт для генерации различных отчетов:
- Отчеты о данных и их качестве
- Отчеты об обучении модели
- Сравнительные отчеты
- Отчеты о производительности
"""
import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import yaml

# Добавление корневой директории в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger
from src.utils.visualization import ReportGenerator, MetricsVisualizer
from src.models.model_manager import ModelManager
from src.data.annotator import AnnotationValidator
from config.config import config

class ComprehensiveReportGenerator:
    """Генератор комплексных отчетов"""
    
    def __init__(self, output_dir: Path):
        """
        Инициализация генератора отчетов
        
        Args:
            output_dir: Директория для сохранения отчетов
        """
        self.logger = get_logger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализация компонентов
        self.report_generator = ReportGenerator()
        self.metrics_visualizer = MetricsVisualizer()
        self.model_manager = ModelManager()
        self.annotation_validator = AnnotationValidator()
        
        # Настройка стилей для графиков
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.logger.info(f"Генератор отчетов инициализирован. Выходная директория: {output_dir}")
    
    def generate_data_quality_report(self, 
                                   dataset_dir: Path,
                                   annotations_dir: Optional[Path] = None) -> Path:
        """
        Генерация отчета о качестве данных
        
        Args:
            dataset_dir: Директория с датасетом
            annotations_dir: Директория с аннотациями (если отличается)
            
        Returns:
            Путь к созданному отчету
        """
        self.logger.info("Генерация отчета о качестве данных...")
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'dataset_directory': str(dataset_dir),
            'analysis_results': {}
        }
        
        # Анализ структуры датасета
        dataset_structure = self._analyze_dataset_structure(dataset_dir)
        report_data['analysis_results']['dataset_structure'] = dataset_structure
        
        # Анализ аннотаций
        if annotations_dir or (dataset_dir / 'train' / 'labels').exists():
            ann_dir = annotations_dir or dataset_dir / 'train' / 'labels'
            validation_report = self.annotation_validator.validate_annotation_directory(
                ann_dir, dataset_dir / 'train' / 'images'
            )
            report_data['analysis_results']['annotation_quality'] = validation_report
        
        # Анализ изображений
        images_analysis = self._analyze_images_quality(dataset_dir)
        report_data['analysis_results']['images_analysis'] = images_analysis
        
        # Генерация HTML отчета
        html_report = self._create_data_quality_html_report(report_data)
        
        # Сохранение отчета
        report_path = self.output_dir / 'data_quality_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # Сохранение JSON данных
        json_path = self.output_dir / 'data_quality_report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Отчет о качестве данных создан: {report_path}")
        return report_path
    
    def _analyze_dataset_structure(self, dataset_dir: Path) -> Dict[str, Any]:
        """Анализ структуры датасета"""
        structure = {
            'splits': {},
            'total_images': 0,
            'class_distribution': {},
            'issues': []
        }
        
        # Анализ каждого split'а
        for split in ['train', 'val', 'test']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                images_dir = split_dir / 'images'
                labels_dir = split_dir / 'labels'
                
                # Подсчет файлов
                image_count = 0
                label_count = 0
                
                if images_dir.exists():
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image_count += len(list(images_dir.glob(f"*{ext}")))
                
                if labels_dir.exists():
                    label_count = len(list(labels_dir.glob("*.txt")))
                
                structure['splits'][split] = {
                    'images': image_count,
                    'labels': label_count,
                    'has_mismatch': image_count != label_count
                }
                
                structure['total_images'] += image_count
                
                # Проверка на проблемы
                if image_count != label_count:
                    structure['issues'].append(
                        f"Несоответствие количества изображений и аннотаций в {split}: "
                        f"{image_count} изображений, {label_count} аннотаций"
                    )
        
        return structure
    
    def _analyze_images_quality(self, dataset_dir: Path) -> Dict[str, Any]:
        """Анализ качества изображений"""
        import cv2
        
        analysis = {
            'total_analyzed': 0,
            'corrupted_images': 0,
            'resolution_stats': {
                'widths': [],
                'heights': [],
                'aspects': []
            },
            'file_size_stats': {
                'sizes_mb': [],
                'average_size_mb': 0
            },
            'format_distribution': {}
        }
        
        # Анализ изображений из всех splits
        for split in ['train', 'val', 'test']:
            images_dir = dataset_dir / split / 'images'
            if not images_dir.exists():
                continue
            
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(images_dir.glob(f"*{ext}"))
            
            # Ограничиваем анализ для производительности
            sample_size = min(len(image_files), 100)
            sample_files = np.random.choice(image_files, sample_size, replace=False) if sample_size > 0 else []
            
            for img_path in sample_files:
                try:
                    # Анализ файла
                    file_size_mb = img_path.stat().st_size / (1024 * 1024)
                    analysis['file_size_stats']['sizes_mb'].append(file_size_mb)
                    
                    # Формат файла
                    ext = img_path.suffix.lower()
                    analysis['format_distribution'][ext] = analysis['format_distribution'].get(ext, 0) + 1
                    
                    # Загрузка изображения для анализа
                    image = cv2.imread(str(img_path))
                    if image is not None:
                        h, w = image.shape[:2]
                        analysis['resolution_stats']['widths'].append(w)
                        analysis['resolution_stats']['heights'].append(h)
                        analysis['resolution_stats']['aspects'].append(w / h)
                        analysis['total_analyzed'] += 1
                    else:
                        analysis['corrupted_images'] += 1
                        
                except Exception as e:
                    self.logger.warning(f"Ошибка анализа {img_path}: {e}")
                    analysis['corrupted_images'] += 1
        
        # Вычисление статистики
        if analysis['file_size_stats']['sizes_mb']:
            analysis['file_size_stats']['average_size_mb'] = np.mean(
                analysis['file_size_stats']['sizes_mb']
            )
        
        return analysis
    
    def _create_data_quality_html_report(self, report_data: Dict[str, Any]) -> str:
        """Создание HTML отчета о качестве данных"""
        structure = report_data['analysis_results']['dataset_structure']
        images_analysis = report_data['analysis_results']['images_analysis']
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Отчет о качестве данных</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #e0e0e0; padding-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .section h2 {{ color: #2c3e50; border-left: 4px solid #3498db; padding-left: 15px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
                .stat-card {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
                .stat-number {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
                .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
                .table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .table th {{ background-color: #3498db; color: white; font-weight: bold; }}
                .table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .issue {{ background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; padding: 10px; margin: 5px 0; }}
                .issue.error {{ background-color: #f8d7da; border-color: #f5c6cb; }}
                .good {{ color: #27ae60; font-weight: bold; }}
                .warning {{ color: #f39c12; font-weight: bold; }}
                .error {{ color: #e74c3c; font-weight: bold; }}
                .progress-bar {{ width: 100%; height: 20px; background-color: #ecf0f1; border-radius: 10px; overflow: hidden; }}
                .progress-fill {{ height: 100%; background-color: #3498db; transition: width 0.3s ease; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Отчет о качестве данных</h1>
                    <p>Сгенерирован: {report_data['timestamp']}</p>
                    <p>Датасет: {report_data['dataset_directory']}</p>
                </div>

                <div class="section">
                    <h2>🏗️ Структура датасета</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number">{structure['total_images']}</div>
                            <div class="stat-label">Общее количество изображений</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{len(structure['splits'])}</div>
                            <div class="stat-label">Разделений (splits)</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{len(structure['issues'])}</div>
                            <div class="stat-label">Обнаруженных проблем</div>
                        </div>
                    </div>
                    
                    <table class="table">
                        <tr><th>Разделение</th><th>Изображения</th><th>Аннотации</th><th>Статус</th></tr>
        """
        
        for split, data in structure['splits'].items():
            status = "✅ OK" if not data['has_mismatch'] else "⚠️ Несоответствие"
            status_class = "good" if not data['has_mismatch'] else "warning"
            
            html_content += f"""
                        <tr>
                            <td>{split}</td>
                            <td>{data['images']}</td>
                            <td>{data['labels']}</td>
                            <td class="{status_class}">{status}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>

                <div class="section">
                    <h2>🖼️ Анализ изображений</h2>
                    <div class="stats-grid">
        """
        
        if images_analysis['total_analyzed'] > 0:
            avg_width = np.mean(images_analysis['resolution_stats']['widths']) if images_analysis['resolution_stats']['widths'] else 0
            avg_height = np.mean(images_analysis['resolution_stats']['heights']) if images_analysis['resolution_stats']['heights'] else 0
            
            html_content += f"""
                        <div class="stat-card">
                            <div class="stat-number">{images_analysis['total_analyzed']}</div>
                            <div class="stat-label">Проанализировано изображений</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{avg_width:.0f}x{avg_height:.0f}</div>
                            <div class="stat-label">Среднее разрешение</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{images_analysis['file_size_stats']['average_size_mb']:.2f} MB</div>
                            <div class="stat-label">Средний размер файла</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{images_analysis['corrupted_images']}</div>
                            <div class="stat-label">Поврежденных файлов</div>
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
        """
        
        # Добавление информации об аннотациях, если доступна
        if 'annotation_quality' in report_data['analysis_results']:
            ann_quality = report_data['analysis_results']['annotation_quality']
            validity_rate = (ann_quality['valid_files'] / ann_quality['total_files'] * 100) if ann_quality['total_files'] > 0 else 0
            
            html_content += f"""
                <div class="section">
                    <h2>📝 Качество аннотаций</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number">{ann_quality['total_files']}</div>
                            <div class="stat-label">Файлов аннотаций</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{validity_rate:.1f}%</div>
                            <div class="stat-label">Валидность аннотаций</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{ann_quality['summary_statistics']['total_objects']}</div>
                            <div class="stat-label">Общее количество объектов</div>
                        </div>
                    </div>
                    
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {validity_rate}%"></div>
                    </div>
                    <p style="text-align: center; margin-top: 10px;">Процент валидных аннотаций</p>
                </div>
            """
        
        # Добавление проблем и рекомендаций
        html_content += """
                <div class="section">
                    <h2>⚠️ Обнаруженные проблемы</h2>
        """
        
        if structure['issues']:
            for issue in structure['issues']:
                html_content += f'<div class="issue">{issue}</div>'
        else:
            html_content += '<div class="issue good">✅ Критических проблем не обнаружено</div>'
        
        html_content += """
                </div>
                
                <div class="section">
                    <h2>💡 Рекомендации</h2>
                    <ul>
        """
        
        # Генерация рекомендаций на основе анализа
        recommendations = self._generate_data_recommendations(report_data)
        for rec in recommendations:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_data_recommendations(self, report_data: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций на основе анализа данных"""
        recommendations = []
        
        structure = report_data['analysis_results']['dataset_structure']
        images_analysis = report_data['analysis_results']['images_analysis']
        
        # Рекомендации по структуре
        if structure['issues']:
            recommendations.append("Исправьте несоответствия между количеством изображений и аннотаций")
        
        if structure['total_images'] < 1000:
            recommendations.append("Рассмотрите возможность увеличения размера датасета для лучшего качества модели")
        
        # Рекомендации по изображениям
        if images_analysis['corrupted_images'] > 0:
            recommendations.append(f"Удалите или восстановите {images_analysis['corrupted_images']} поврежденных изображений")
        
        if images_analysis['resolution_stats']['widths']:
            min_width = min(images_analysis['resolution_stats']['widths'])
            max_width = max(images_analysis['resolution_stats']['widths'])
            if max_width / min_width > 5:
                recommendations.append("Большой разброс в разрешениях изображений. Рассмотрите нормализацию размеров")
        
        # Рекомендации по аннотациям
        if 'annotation_quality' in report_data['analysis_results']:
            ann_quality = report_data['analysis_results']['annotation_quality']
            validity_rate = (ann_quality['valid_files'] / ann_quality['total_files']) if ann_quality['total_files'] > 0 else 0
            
            if validity_rate < 0.95:
                recommendations.append("Низкая валидность аннотаций. Проверьте и исправьте ошибки в разметке")
            
            if ann_quality['missing_images'] > 0:
                recommendations.append(f"Найдено {ann_quality['missing_images']} изображений без аннотаций")
        
        if not recommendations:
            recommendations.append("Качество данных хорошее. Датасет готов для обучения")
        
        return recommendations
    
    def generate_training_report(self, experiment_dir: Path) -> Path:
        """
        Генерация отчета об обучении модели
        
        Args:
            experiment_dir: Директория с результатами эксперимента
            
        Returns:
            Путь к созданному отчету
        """
        self.logger.info(f"Генерация отчета об обучении для: {experiment_dir}")
        
        # Загрузка данных об обучении
        training_results_file = experiment_dir / 'training_results.json'
        if not training_results_file.exists():
            raise FileNotFoundError(f"Результаты обучения не найдены: {training_results_file}")
        
        with open(training_results_file, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        # Использование встроенного генератора отчетов
        report_path = self.report_generator.generate_training_report(training_data, self.output_dir)
        
        self.logger.info(f"Отчет об обучении создан: {report_path}")
        return report_path
    
    def generate_model_comparison_report(self, model_ids: List[str]) -> Path:
        """
        Генерация сравнительного отчета моделей
        
        Args:
            model_ids: Список ID моделей для сравнения
            
        Returns:
            Путь к созданному отчету
        """
        self.logger.info(f"Генерация сравнительного отчета для {len(model_ids)} моделей")
        
        # Получение данных о моделях
        models_data = {}
        for model_id in model_ids:
            try:
                model_info = self.model_manager.get_model_info(model_id)
                models_data[model_id] = model_info
            except Exception as e:
                self.logger.warning(f"Не удалось получить информацию о модели {model_id}: {e}")
        
        if not models_data:
            raise ValueError("Не удалось загрузить данные ни об одной модели")
        
        # Сравнение моделей
        comparison_results = self.model_manager.compare_models(list(models_data.keys()))
        
        # Генерация HTML отчета
        html_report = self._create_model_comparison_html_report(comparison_results, models_data)
        
        # Сохранение отчета
        report_path = self.output_dir / 'model_comparison_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # Сохранение JSON данных
        json_path = self.output_dir / 'model_comparison_data.json'
        comparison_data = {
            'models_data': models_data,
            'comparison_results': comparison_results,
            'timestamp': datetime.now().isoformat()
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Сравнительный отчет моделей создан: {report_path}")
        return report_path
    
    def _create_model_comparison_html_report(self, comparison_results: Dict[str, Any], 
                                           models_data: Dict[str, Any]) -> str:
        """Создание HTML отчета сравнения моделей"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Сравнительный отчет моделей</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #e0e0e0; padding-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .section h2 {{ color: #2c3e50; border-left: 4px solid #e74c3c; padding-left: 15px; }}
                .models-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .model-card {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-top: 4px solid #e74c3c; }}
                .model-name {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
                .metric-item {{ display: flex; justify-content: space-between; margin: 5px 0; }}
                .metric-label {{ color: #7f8c8d; }}
                .metric-value {{ font-weight: bold; color: #2c3e50; }}
                .best-metric {{ background-color: #d4edda; padding: 2px 8px; border-radius: 4px; color: #155724; }}
                .comparison-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                .comparison-table th, .comparison-table td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
                .comparison-table th {{ background-color: #e74c3c; color: white; font-weight: bold; }}
                .comparison-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .winner {{ background-color: #d4edda; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏆 Сравнительный отчет моделей</h1>
                    <p>Сгенерирован: {datetime.now().isoformat()}</p>
                    <p>Сравнивается моделей: {len(models_data)}</p>
                </div>

                <div class="section">
                    <h2>📋 Обзор моделей</h2>
                    <div class="models-grid">
        """
        
        # Карточки моделей
        for model_id, model_data in models_data.items():
            metrics = model_data.get('metrics', {})
            
            html_content += f"""
                        <div class="model-card">
                            <div class="model-name">🔬 {model_data.get('name', model_id)}</div>
                            <div class="metric-item">
                                <span class="metric-label">Размер:</span>
                                <span class="metric-value">{model_data.get('size_mb', 0):.1f} MB</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Создан:</span>
                                <span class="metric-value">{model_data.get('created_at', 'Неизвестно')[:10]}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">mAP@0.5:</span>
                                <span class="metric-value">{metrics.get('map50', 0):.4f}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Precision:</span>
                                <span class="metric-value">{metrics.get('precision', 0):.4f}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Recall:</span>
                                <span class="metric-value">{metrics.get('recall', 0):.4f}</span>
                            </div>
                        </div>
            """
        
        html_content += """
                    </div>
                </div>

                <div class="section">
                    <h2>📊 Сравнение метрик</h2>
                    <table class="comparison-table">
                        <tr>
                            <th>Модель</th>
                            <th>mAP@0.5</th>
                            <th>mAP@0.5:0.95</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>Размер (MB)</th>
                        </tr>
        """
        
        # Поиск лучших значений для выделения
        best_values = {}
        for metric in ['map50', 'map50_95', 'precision', 'recall']:
            best_values[metric] = max(
                models_data[mid].get('metrics', {}).get(metric, 0) 
                for mid in models_data.keys()
            )
        
        # Строки таблицы
        for model_id, model_data in models_data.items():
            metrics = model_data.get('metrics', {})
            
            # Определение лучших значений
            map50_class = "winner" if metrics.get('map50', 0) == best_values.get('map50', 0) else ""
            precision_class = "winner" if metrics.get('precision', 0) == best_values.get('precision', 0) else ""
            recall_class = "winner" if metrics.get('recall', 0) == best_values.get('recall', 0) else ""
            
            html_content += f"""
                        <tr>
                            <td>{model_data.get('name', model_id)}</td>
                            <td class="{map50_class}">{metrics.get('map50', 0):.4f}</td>
                            <td>{metrics.get('map50_95', 0):.4f}</td>
                            <td class="{precision_class}">{metrics.get('precision', 0):.4f}</td>
                            <td class="{recall_class}">{metrics.get('recall', 0):.4f}</td>
                            <td>{model_data.get('size_mb', 0):.1f}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>

                <div class="section">
                    <h2>🏅 Лучшие по категориям</h2>
        """
        
        # Лучшие модели по категориям
        if 'best_by_metric' in comparison_results:
            for metric, best_info in comparison_results['best_by_metric'].items():
                model_name = models_data.get(best_info['model_id'], {}).get('name', best_info['model_id'])
                html_content += f"""
                    <div class="metric-item">
                        <span class="metric-label">Лучший {metric}:</span>
                        <span class="best-metric">{model_name} ({best_info['value']:.4f})</span>
                    </div>
                """
        
        html_content += """
                </div>

                <div class="section">
                    <h2>💡 Рекомендации</h2>
                    <ul>
        """
        
        # Генерация рекомендаций
        recommendations = self._generate_model_recommendations(comparison_results, models_data)
        for rec in recommendations:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_model_recommendations(self, comparison_results: Dict[str, Any], 
                                      models_data: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по моделям"""
        recommendations = []
        
        if not models_data:
            return ["Нет данных для анализа"]
        
        # Анализ лучшей модели
        if 'best_by_metric' in comparison_results and 'map50' in comparison_results['best_by_metric']:
            best_model_id = comparison_results['best_by_metric']['map50']['model_id']
            best_model_name = models_data.get(best_model_id, {}).get('name', best_model_id)
            best_map = comparison_results['best_by_metric']['map50']['value']
            
            recommendations.append(f"Рекомендуется использовать модель '{best_model_name}' с лучшим mAP@0.5: {best_map:.4f}")
        
        # Анализ размеров моделей
        sizes = [data.get('size_mb', 0) for data in models_data.values()]
        if sizes:
            min_size = min(sizes)
            max_size = max(sizes)
            if max_size / min_size > 3:
                recommendations.append("Большой разброс в размерах моделей. Учитывайте требования к производительности")
        
        # Анализ метрик
        if 'summary' in comparison_results:
            for metric, stats in comparison_results['summary'].items():
                if stats.get('std', 0) > 0.1:
                    recommendations.append(f"Высокая вариативность в метрике {metric}. Рассмотрите дополнительную настройку")
        
        return recommendations
    
    def generate_performance_report(self, 
                                  model_path: Path,
                                  test_images_dir: Path,
                                  benchmark_results: Optional[Dict[str, Any]] = None) -> Path:
        """
        Генерация отчета о производительности модели
        
        Args:
            model_path: Путь к модели
            test_images_dir: Директория с тестовыми изображениями
            benchmark_results: Результаты бенчмарка (если есть)
            
        Returns:
            Путь к созданному отчету
        """
        self.logger.info("Генерация отчета о производительности...")
        
        # Если результаты бенчмарка не предоставлены, запускаем собственный
        if benchmark_results is None:
            from src.models.inference import benchmark_model
            
            # Получение списка тестовых изображений
            test_images = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                test_images.extend(test_images_dir.glob(f"*{ext}"))
            
            if not test_images:
                raise ValueError(f"Тестовые изображения не найдены в: {test_images_dir}")
            
            # Ограничиваем количество для быстроты
            test_images = test_images[:50]
            
            benchmark_results = benchmark_model(model_path, test_images)
        
        # Генерация HTML отчета
        html_report = self._create_performance_html_report(model_path, benchmark_results)
        
        # Сохранение отчета
        report_path = self.output_dir / 'performance_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # Сохранение JSON данных
        json_path = self.output_dir / 'performance_data.json'
        performance_data = {
            'model_path': str(model_path),
            'benchmark_results': benchmark_results,
            'timestamp': datetime.now().isoformat()
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(performance_data, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Отчет о производительности создан: {report_path}")
        return report_path
    
    def _create_performance_html_report(self, model_path: Path, 
                                      benchmark_results: Dict[str, Any]) -> str:
        """Создание HTML отчета о производительности"""
        fps = benchmark_results.get('fps', 0)
        avg_time = benchmark_results.get('average_time_per_image', 0)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Отчет о производительности модели</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #e0e0e0; padding-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .section h2 {{ color: #2c3e50; border-left: 4px solid #27ae60; padding-left: 15px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
                .metric-card {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-top: 4px solid #27ae60; }}
                .metric-number {{ font-size: 2.5em; font-weight: bold; color: #27ae60; }}
                .metric-label {{ color: #7f8c8d; margin-top: 10px; font-size: 0.9em; }}
                .performance-indicator {{ font-size: 1.2em; margin-top: 10px; }}
                .excellent {{ color: #27ae60; }}
                .good {{ color: #f39c12; }}
                .poor {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚡ Отчет о производительности модели</h1>
                    <p>Модель: {model_path.name}</p>
                    <p>Сгенерирован: {datetime.now().isoformat()}</p>
                </div>

                <div class="section">
                    <h2>🚀 Основные метрики производительности</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-number">{fps:.1f}</div>
                            <div class="metric-label">FPS<br>(кадров в секунду)</div>
                            <div class="performance-indicator {'excellent' if fps > 30 else 'good' if fps > 15 else 'poor'}">
                                {'Отлично' if fps > 30 else 'Хорошо' if fps > 15 else 'Требует оптимизации'}
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-number">{avg_time*1000:.0f}</div>
                            <div class="metric-label">мс<br>(время на изображение)</div>
                            <div class="performance-indicator {'excellent' if avg_time < 0.05 else 'good' if avg_time < 0.1 else 'poor'}">
                                {'Быстро' if avg_time < 0.05 else 'Средне' if avg_time < 0.1 else 'Медленно'}
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-number">{benchmark_results.get('total_images', 0)}</div>
                            <div class="metric-label">Изображений<br>протестировано</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-number">{benchmark_results.get('total_detections', 0)}</div>
                            <div class="metric-label">Общее количество<br>детекций</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-number">{benchmark_results.get('average_detections_per_image', 0):.1f}</div>
                            <div class="metric-label">Среднее детекций<br>на изображение</div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>📈 Статистика времени инференса</h2>
        """
        
        inference_stats = benchmark_results.get('inference_time_stats', {})
        if inference_stats:
            html_content += f"""
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-number">{inference_stats.get('min', 0)*1000:.1f}</div>
                            <div class="metric-label">мс<br>Минимальное время</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-number">{inference_stats.get('max', 0)*1000:.1f}</div>
                            <div class="metric-label">мс<br>Максимальное время</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-number">{inference_stats.get('mean', 0)*1000:.1f}</div>
                            <div class="metric-label">мс<br>Среднее время</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-number">{inference_stats.get('std', 0)*1000:.1f}</div>
                            <div class="metric-label">мс<br>Стандартное отклонение</div>
                        </div>
                    </div>
            """
        
        html_content += """
                </div>

                <div class="section">
                    <h2>💡 Рекомендации по оптимизации</h2>
                    <ul>
        """
        
        # Генерация рекомендаций по производительности
        perf_recommendations = self._generate_performance_recommendations(benchmark_results)
        for rec in perf_recommendations:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_performance_recommendations(self, benchmark_results: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по производительности"""
        recommendations = []
        
        fps = benchmark_results.get('fps', 0)
        avg_time = benchmark_results.get('average_time_per_image', 0)
        
        if fps < 15:
            recommendations.append("Низкая производительность (< 15 FPS). Рассмотрите оптимизацию модели или использование GPU")
        elif fps < 30:
            recommendations.append("Средняя производительность. Для real-time приложений рекомендуется оптимизация")
        else:
            recommendations.append("Отличная производительность! Модель подходит для real-time обработки")
        
        if avg_time > 0.1:
            recommendations.append("Время инференса > 100мс. Рассмотрите использование более легкой модели (YOLOv8n)")
        
        inference_stats = benchmark_results.get('inference_time_stats', {})
        if inference_stats.get('std', 0) > inference_stats.get('mean', 0) * 0.5:
            recommendations.append("Высокая вариативность времени инференса. Проверьте стабильность системы")
        
        return recommendations
    
    def generate_comprehensive_report(self, 
                                    dataset_dir: Path,
                                    experiment_dirs: List[Path],
                                    model_ids: Optional[List[str]] = None) -> Path:
        """
        Генерация комплексного отчета по всему проекту
        
        Args:
            dataset_dir: Директория с датасетом
            experiment_dirs: Список директорий экспериментов
            model_ids: Список ID моделей для сравнения
            
        Returns:
            Путь к созданному отчету
        """
        self.logger.info("Генерация комплексного отчета...")
        
        # Создание отдельной директории для комплексного отчета
        comprehensive_dir = self.output_dir / "comprehensive_report"
        comprehensive_dir.mkdir(exist_ok=True)
        
        report_sections = {}
        
        # 1. Отчет о качестве данных
        try:
            data_report = self.generate_data_quality_report(dataset_dir)
            report_sections['data_quality'] = data_report
        except Exception as e:
            self.logger.warning(f"Ошибка генерации отчета о данных: {e}")
        
        # 2. Отчеты об обучении
        training_reports = []
        for exp_dir in experiment_dirs:
            try:
                # Временно изменяем выходную директорию
                original_output = self.output_dir
                self.output_dir = comprehensive_dir
                
                training_report = self.generate_training_report(exp_dir)
                training_reports.append(training_report)
                
                self.output_dir = original_output
            except Exception as e:
                self.logger.warning(f"Ошибка генерации отчета обучения для {exp_dir}: {e}")
        
        report_sections['training_reports'] = training_reports
        
        # 3. Сравнение моделей
        if model_ids:
            try:
                original_output = self.output_dir
                self.output_dir = comprehensive_dir
                
                comparison_report = self.generate_model_comparison_report(model_ids)
                report_sections['model_comparison'] = comparison_report
                
                self.output_dir = original_output
            except Exception as e:
                self.logger.warning(f"Ошибка сравнения моделей: {e}")
        
        # 4. Создание главного индексного файла
        index_html = self._create_comprehensive_index(report_sections, comprehensive_dir)
        
        index_path = comprehensive_dir / "index.html"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_html)
        
        self.logger.info(f"Комплексный отчет создан: {index_path}")
        return index_path
    
    def _create_comprehensive_index(self, report_sections: Dict[str, Any], 
                                  output_dir: Path) -> str:
        """Создание главной страницы комплексного отчета"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Комплексный отчет проекта YOLOv11</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #e0e0e0; padding-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .section h2 {{ color: #2c3e50; border-left: 4px solid #9b59b6; padding-left: 15px; }}
                .report-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .report-card {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-top: 4px solid #9b59b6; }}
                .report-title {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
                .report-description {{ color: #7f8c8d; margin-bottom: 15px; }}
                .report-link {{ display: inline-block; background-color: #9b59b6; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; transition: background-color 0.3s; }}
                .report-link:hover {{ background-color: #8e44ad; }}
                .summary-stats {{ background-color: #ecf0f1; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .stat-item {{ display: inline-block; margin: 10px 20px; text-align: center; }}
                .stat-number {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
                .stat-label {{ color: #7f8c8d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Комплексный отчет проекта YOLOv11</h1>
                    <p>Полный анализ данных, обучения и производительности модели</p>
                    <p>Сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="summary-stats">
                    <h3>📈 Краткая статистика проекта</h3>
                    <div class="stat-item">
                        <div class="stat-number">{len(report_sections.get('training_reports', []))}</div>
                        <div class="stat-label">Экспериментов обучения</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{'Да' if 'data_quality' in report_sections else 'Нет'}</div>
                        <div class="stat-label">Анализ данных</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{'Да' if 'model_comparison' in report_sections else 'Нет'}</div>
                        <div class="stat-label">Сравнение моделей</div>
                    </div>
                </div>

                <div class="section">
                    <h2>📋 Доступные отчеты</h2>
                    <div class="report-grid">
        """
        
        # Карточка отчета о данных
        if 'data_quality' in report_sections:
            html_content += """
                        <div class="report-card">
                            <div class="report-title">🔍 Качество данных</div>
                            <div class="report-description">
                                Анализ структуры датасета, качества изображений и аннотаций.
                                Включает рекомендации по улучшению данных.
                            </div>
                            <a href="../data_quality_report.html" class="report-link">Открыть отчет</a>
                        </div>
            """
        
        # Карточки отчетов об обучении
        for i, training_report in enumerate(report_sections.get('training_reports', [])):
            html_content += f"""
                        <div class="report-card">
                            <div class="report-title">🎯 Обучение модели #{i+1}</div>
                            <div class="report-description">
                                Детальный отчет о процессе обучения, метрики качества,
                                графики обучения и анализ результатов.
                            </div>
                            <a href="{Path(training_report).name}" class="report-link">Открыть отчет</a>
                        </div>
            """
        
        # Карточка сравнения моделей
        if 'model_comparison' in report_sections:
            html_content += """
                        <div class="report-card">
                            <div class="report-title">🏆 Сравнение моделей</div>
                            <div class="report-description">
                                Сравнительный анализ различных версий модели,
                                рекомендации по выбору лучшей модели.
                            </div>
                            <a href="model_comparison_report.html" class="report-link">Открыть отчет</a>
                        </div>
            """
        
        html_content += """
                    </div>
                </div>

                <div class="section">
                    <h2>💡 Общие рекомендации проекта</h2>
                    <ul>
                        <li>Регулярно проверяйте качество данных перед обучением новых моделей</li>
                        <li>Сравнивайте результаты разных экспериментов для выбора оптимальной конфигурации</li>
                        <li>Мониторьте производительность моделей на реальных данных</li>
                        <li>Документируйте все изменения в процессе разработки</li>
                        <li>Используйте версионирование для управления моделями</li>
                    </ul>
                </div>

                <div class="section">
                    <h2>📞 Контакты и поддержка</h2>
                    <p>
                        Для вопросов по отчетам и анализу обращайтесь к команде разработки.
                        Все отчеты автоматически генерируются системой мониторинга ML проекта.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content


def main():
    """Главная функция скрипта"""
    parser = argparse.ArgumentParser(
        description="Генерация отчетов для проекта YOLOv11",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Отчет о качестве данных
  python scripts/generate_reports.py --data-quality --dataset data/datasets/my_dataset

  # Отчет об обучении
  python scripts/generate_reports.py --training-report --experiment outputs/experiments/exp_001

  # Сравнение моделей
  python scripts/generate_reports.py --model-comparison --model-ids model_1 model_2 model_3

  # Отчет о производительности
  python scripts/generate_reports.py --performance --model models/trained/best.pt --test-images data/test/images

  # Комплексный отчет
  python scripts/generate_reports.py --comprehensive --dataset data/datasets/my_dataset --experiments outputs/experiments/exp_*

  # Все отчеты сразу
  python scripts/generate_reports.py --all --dataset data/datasets/my_dataset --experiments outputs/experiments/exp_*
        """
    )
    
    # Типы отчетов
    parser.add_argument(
        '--data-quality',
        action='store_true',
        help='Генерировать отчет о качестве данных'
    )
    
    parser.add_argument(
        '--training-report',
        action='store_true',
        help='Генерировать отчет об обучении'
    )
    
    parser.add_argument(
        '--model-comparison',
        action='store_true',
        help='Генерировать сравнительный отчет моделей'
    )
    
    parser.add_argument(
        '--performance',
        action='store_true',
        help='Генерировать отчет о производительности'
    )
    
    parser.add_argument(
        '--comprehensive',
        action='store_true',
        help='Генерировать комплексный отчет'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Генерировать все возможные отчеты'
    )
    
    # Пути и параметры
    parser.add_argument(
        '--dataset',
        type=str,
        help='Путь к датасету'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        help='Путь к директории эксперимента'
    )
    
    parser.add_argument(
        '--experiments',
        nargs='+',
        help='Пути к директориям экспериментов'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Путь к модели для анализа производительности'
    )
    
    parser.add_argument(
        '--model-ids',
        nargs='+',
        help='ID моделей для сравнения'
    )
    
    parser.add_argument(
        '--test-images',
        type=str,
        help='Путь к тестовым изображениям'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/reports',
        help='Директория для сохранения отчетов'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Уровень логирования'
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)
    
    logger.info("🚀 Запуск генератора отчетов")
    logger.info(f"Аргументы: {vars(args)}")
    
    try:
        # Инициализация генератора
        report_generator = ComprehensiveReportGenerator(args.output_dir)
        
        generated_reports = []
        
        # Генерация отчетов по типам
        if args.all or args.data_quality:
            if args.dataset:
                logger.info("Генерация отчета о качестве данных...")
                report_path = report_generator.generate_data_quality_report(Path(args.dataset))
                generated_reports.append(('Качество данных', report_path))
            else:
                logger.warning("Для отчета о данных необходимо указать --dataset")
        
        if args.all or args.training_report:
            if args.experiment:
                logger.info("Генерация отчета об обучении...")
                report_path = report_generator.generate_training_report(Path(args.experiment))
                generated_reports.append(('Обучение модели', report_path))
            elif args.experiments:
                for exp_path in args.experiments:
                    try:
                        logger.info(f"Генерация отчета об обучении для {exp_path}...")
                        report_path = report_generator.generate_training_report(Path(exp_path))
                        generated_reports.append(('Обучение модели', report_path))
                    except Exception as e:
                        logger.warning(f"Ошибка генерации отчета для {exp_path}: {e}")
            else:
                logger.warning("Для отчета об обучении необходимо указать --experiment или --experiments")
        
        if args.all or args.model_comparison:
            if args.model_ids:
                logger.info("Генерация сравнительного отчета моделей...")
                report_path = report_generator.generate_model_comparison_report(args.model_ids)
                generated_reports.append(('Сравнение моделей', report_path))
            else:
                logger.warning("Для сравнения моделей необходимо указать --model-ids")
        
        if args.all or args.performance:
            if args.model and args.test_images:
                logger.info("Генерация отчета о производительности...")
                report_path = report_generator.generate_performance_report(
                    Path(args.model), Path(args.test_images)
                )
                generated_reports.append(('Производительность', report_path))
            else:
                logger.warning("Для отчета о производительности необходимо указать --model и --test-images")
        
        if args.comprehensive:
            if args.dataset:
                experiments = []
                if args.experiments:
                    experiments = [Path(p) for p in args.experiments]
                elif args.experiment:
                    experiments = [Path(args.experiment)]
                
                logger.info("Генерация комплексного отчета...")
                report_path = report_generator.generate_comprehensive_report(
                    Path(args.dataset), experiments, args.model_ids
                )
                generated_reports.append(('Комплексный отчет', report_path))
            else:
                logger.warning("Для комплексного отчета необходимо указать --dataset")
        
        # Вывод результатов
        if generated_reports:
            logger.info("✅ ГЕНЕРАЦИЯ ОТЧЕТОВ ЗАВЕРШЕНА!")
            logger.info(f"Создано отчетов: {len(generated_reports)}")
            
            for report_type, report_path in generated_reports:
                logger.info(f"  📋 {report_type}: {report_path}")
            
            logger.info(f"📁 Все отчеты сохранены в: {args.output_dir}")
        else:
            logger.warning("Ни один отчет не был сгенерирован. Проверьте параметры запуска.")
            parser.print_help()
            sys.exit(1)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("Генерация отчетов прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()