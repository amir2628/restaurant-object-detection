"""
Скрипт для оценки обученной модели YOLOv11
Вычисляет метрики качества и создает детальные отчеты
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Добавление корневой директории в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger
from src.models.inference import YOLOInference
from src.utils.metrics import MetricsCalculator, COCOMetrics, evaluate_yolo_model
from src.utils.visualization import ReportGenerator
from ultralytics import YOLO
import json

def load_test_dataset_info(dataset_yaml: Path) -> Dict[str, Any]:
    """
    Загрузка информации о тестовом датасете
    
    Args:
        dataset_yaml: Путь к YAML конфигурации датасета
        
    Returns:
        Информация о датасете
    """
    import yaml
    
    logger = get_logger(__name__)
    
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Конфигурация датасета не найдена: {dataset_yaml}")
    
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    logger.info(f"Загружена конфигурация датасета: {dataset_yaml}")
    logger.info(f"Классов: {dataset_config.get('nc', 'неизвестно')}")
    
    return dataset_config

def run_official_evaluation(model_path: Path, 
                          dataset_yaml: Path,
                          output_dir: Path,
                          confidence_threshold: float = 0.001,
                          iou_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Запуск официальной оценки YOLO
    
    Args:
        model_path: Путь к модели
        dataset_yaml: Путь к конфигурации датасета
        output_dir: Директория для результатов
        confidence_threshold: Порог уверенности
        iou_threshold: Порог IoU
        
    Returns:
        Результаты оценки
    """
    logger = get_logger(__name__)
    
    logger.info("Запуск официальной оценки YOLO...")
    
    # Загрузка модели
    model = YOLO(str(model_path))
    
    # Создание директории для результатов
    eval_dir = output_dir / "official_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Запуск валидации
    results = model.val(
        data=str(dataset_yaml),
        conf=confidence_threshold,
        iou=iou_threshold,
        save_json=True,
        save_hybrid=False,
        plots=True,
        verbose=True,
        project=str(eval_dir),
        name="validation"
    )
    
    # Извлечение метрик
    metrics = {}
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
    
    # Структурирование результатов
    evaluation_results = {
        "model_path": str(model_path),
        "dataset_config": str(dataset_yaml),
        "evaluation_config": {
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold
        },
        "metrics": metrics,
        "output_directory": str(eval_dir)
    }
    
    # Сохранение результатов
    results_file = eval_dir / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"Официальная оценка завершена. Результаты в: {eval_dir}")
    
    return evaluation_results

def run_custom_evaluation(model_path: Path,
                        test_images_dir: Path,
                        test_annotations_dir: Path,
                        class_mapping: Dict[int, str],
                        output_dir: Path,
                        confidence_threshold: float = 0.25,
                        iou_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Запуск кастомной оценки с детальным анализом
    
    Args:
        model_path: Путь к модели
        test_images_dir: Директория с тестовыми изображениями
        test_annotations_dir: Директория с аннотациями
        class_mapping: Маппинг классов
        output_dir: Директория для результатов
        confidence_threshold: Порог уверенности
        iou_threshold: Порог IoU
        
    Returns:
        Результаты кастомной оценки
    """
    logger = get_logger(__name__)
    
    logger.info("Запуск кастомной оценки с детальным анализом...")
    
    # Создание директории для результатов
    custom_eval_dir = output_dir / "custom_evaluation"
    custom_eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Инициализация инференса
    inference = YOLOInference(model_path, confidence_threshold, iou_threshold)
    
    # Поиск тестовых изображений
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(test_images_dir.glob(f"*{ext}"))
        image_files.extend(test_images_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        raise ValueError(f"Тестовые изображения не найдены в: {test_images_dir}")
    
    logger.info(f"Найдено {len(image_files)} тестовых изображений")
    
    # Инициализация калькулятора метрик
    class_names = list(class_mapping.values())
    metrics_calc = MetricsCalculator(len(class_names), class_names)
    coco_metrics = COCOMetrics(len(class_names), class_names)
    
    # Загрузка ground truth
    from src.utils.metrics import load_ground_truth_from_yolo
    ground_truths = load_ground_truth_from_yolo(test_annotations_dir, class_mapping)
    
    logger.info(f"Загружено {len(ground_truths)} истинных аннотаций")
    
    # Выполнение инференса и сбор результатов
    logger.info("Выполнение инференса на тестовых изображениях...")
    
    from src.utils.metrics import DetectionResult
    detections = []
    
    inference_results = inference.predict_batch(image_files)
    
    for result in inference_results:
        image_id = Path(result.image_path).stem
        
        for detection in result.detections:
            det_result = DetectionResult(
                class_id=detection.class_id,
                confidence=detection.confidence,
                bbox=detection.bbox,
                image_id=image_id
            )
            detections.append(det_result)
    
    logger.info(f"Получено {len(detections)} детекций")
    
    # Добавление данных в калькуляторы метрик
    metrics_calc.add_detections(detections)
    metrics_calc.add_ground_truths(ground_truths)
    
    coco_metrics.add_detections(detections)
    coco_metrics.add_ground_truths(ground_truths)
    
    # Вычисление метрик
    logger.info("Вычисление метрик...")
    
    # Основные метрики
    overall_report = metrics_calc.generate_metrics_report(confidence_threshold, iou_threshold)
    
    # COCO метрики
    coco_results = coco_metrics.calculate_coco_map()
    
    # Построение графиков
    logger.info("Создание визуализаций...")
    
    # PR кривые
    pr_curves_dir = custom_eval_dir / "pr_curves"
    metrics_calc.plot_pr_curves(pr_curves_dir, iou_threshold)
    
    # Матрица ошибок
    from src.utils.visualization import MetricsVisualizer
    vis = MetricsVisualizer()
    
    confusion_matrix = metrics_calc.calculate_confusion_matrix(confidence_threshold, iou_threshold)
    class_names_with_bg = class_names + ["background"]
    
    vis.plot_confusion_matrix(
        confusion_matrix,
        class_names_with_bg,
        output_path=custom_eval_dir / "confusion_matrix.png"
    )
    
    # Распределение классов
    class_counts = overall_report["data_statistics"]["class_distribution"]
    vis.plot_class_distribution(
        class_counts,
        output_path=custom_eval_dir / "class_distribution.png"
    )
    
    # Объединение результатов
    custom_results = {
        "model_path": str(model_path),
        "test_images_directory": str(test_images_dir),
        "test_annotations_directory": str(test_annotations_dir),
        "evaluation_config": {
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "class_mapping": class_mapping
        },
        "overall_metrics": overall_report,
        "coco_metrics": coco_results,
        "performance_stats": inference.get_performance_stats(),
        "visualizations": {
            "pr_curves": str(pr_curves_dir),
            "confusion_matrix": str(custom_eval_dir / "confusion_matrix.png"),
            "class_distribution": str(custom_eval_dir / "class_distribution.png")
        }
    }
    
    # Сохранение результатов
    results_file = custom_eval_dir / "custom_evaluation_results.json"
    metrics_calc.save_metrics_report(overall_report, results_file)
    
    detailed_results_file = custom_eval_dir / "detailed_results.json"
    with open(detailed_results_file, 'w', encoding='utf-8') as f:
        json.dump(custom_results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"Кастомная оценка завершена. Результаты в: {custom_eval_dir}")
    
    return custom_results

def generate_comparison_report(evaluations: List[Dict[str, Any]], 
                             output_dir: Path) -> Path:
    """
    Генерация сравнительного отчета
    
    Args:
        evaluations: Список результатов оценки
        output_dir: Директория для отчета
        
    Returns:
        Путь к сравнительному отчету
    """
    logger = get_logger(__name__)
    
    logger.info("Генерация сравнительного отчета...")
    
    # Извлечение ключевых метрик
    comparison_data = []
    
    for eval_result in evaluations:
        eval_type = "official" if "official" in str(eval_result.get("output_directory", "")) else "custom"
        
        if eval_type == "official":
            metrics = eval_result.get("metrics", {})
            data = {
                "type": "Official YOLO",
                "map50": metrics.get("metrics/mAP50(B)", 0),
                "map50_95": metrics.get("metrics/mAP50-95(B)", 0),
                "precision": metrics.get("metrics/precision(B)", 0),
                "recall": metrics.get("metrics/recall(B)", 0),
                "model_path": eval_result.get("model_path", "")
            }
        else:
            overall_metrics = eval_result.get("overall_metrics", {}).get("overall_metrics", {})
            map_metrics = eval_result.get("overall_metrics", {}).get("map_metrics", {})
            
            data = {
                "type": "Custom Analysis",
                "map50": map_metrics.get("mAP@0.50", 0),
                "map50_95": map_metrics.get("mAP@0.5:0.95", 0),
                "precision": overall_metrics.get("precision", 0),
                "recall": overall_metrics.get("recall", 0),
                "f1_score": overall_metrics.get("f1_score", 0),
                "model_path": eval_result.get("model_path", "")
            }
        
        comparison_data.append(data)
    
    # Создание HTML отчета
    report_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Evaluation Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .section {{ margin-bottom: 30px; }}
            .metrics-table {{ border-collapse: collapse; width: 100%; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            .metrics-table th {{ background-color: #f2f2f2; font-weight: bold; }}
            .best-score {{ background-color: #d4edda; font-weight: bold; }}
            .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Model Evaluation Comparison Report</h1>
            <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Metrics Comparison</h2>
            <table class="metrics-table">
                <tr>
                    <th>Evaluation Type</th>
                    <th>mAP@0.5</th>
                    <th>mAP@0.5:0.95</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Model</th>
                </tr>
    """
    
    # Поиск лучших значений для выделения
    if comparison_data:
        best_map50 = max(data.get("map50", 0) for data in comparison_data)
        best_map50_95 = max(data.get("map50_95", 0) for data in comparison_data)
        best_precision = max(data.get("precision", 0) for data in comparison_data)
        best_recall = max(data.get("recall", 0) for data in comparison_data)
        best_f1 = max(data.get("f1_score", 0) for data in comparison_data)
        
        for data in comparison_data:
            map50_class = "best-score" if data.get("map50", 0) == best_map50 else ""
            map50_95_class = "best-score" if data.get("map50_95", 0) == best_map50_95 else ""
            precision_class = "best-score" if data.get("precision", 0) == best_precision else ""
            recall_class = "best-score" if data.get("recall", 0) == best_recall else ""
            f1_class = "best-score" if data.get("f1_score", 0) == best_f1 else ""
            
            model_name = Path(data["model_path"]).name if data["model_path"] else "Unknown"
            
            report_content += f"""
                <tr>
                    <td>{data["type"]}</td>
                    <td class="{map50_class}">{data.get("map50", 0):.4f}</td>
                    <td class="{map50_95_class}">{data.get("map50_95", 0):.4f}</td>
                    <td class="{precision_class}">{data.get("precision", 0):.4f}</td>
                    <td class="{recall_class}">{data.get("recall", 0):.4f}</td>
                    <td class="{f1_class}">{data.get("f1_score", 0):.4f}</td>
                    <td>{model_name}</td>
                </tr>
            """
    
    # Резюме
    summary_text = ""
    if comparison_data:
        official_data = [d for d in comparison_data if d["type"] == "Official YOLO"]
        custom_data = [d for d in comparison_data if d["type"] == "Custom Analysis"]
        
        if official_data and custom_data:
            official_map50 = official_data[0].get("map50", 0)
            custom_map50 = custom_data[0].get("map50", 0)
            
            if abs(official_map50 - custom_map50) < 0.01:
                summary_text = "✅ Отличная согласованность между официальной и кастомной оценкой"
            elif abs(official_map50 - custom_map50) < 0.05:
                summary_text = "⚠️ Небольшие расхождения между методами оценки"
            else:
                summary_text = "❌ Значительные расхождения требуют дополнительного анализа"
    
    report_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>📋 Summary</h2>
            <div class="summary">
                <p><strong>Evaluation Summary:</strong> {summary_text}</p>
                <p><strong>Total Evaluations:</strong> {len(comparison_data)}</p>
                <p><strong>Best Overall mAP@0.5:</strong> {best_map50:.4f}</p>
                <p><strong>Best Overall mAP@0.5:0.95:</strong> {best_map50_95:.4f}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🔍 Recommendations</h2>
            <ul>
    """
    
    # Рекомендации на основе результатов
    if comparison_data:
        avg_map50 = sum(d.get("map50", 0) for d in comparison_data) / len(comparison_data)
        
        if avg_map50 > 0.7:
            report_content += "<li>✅ Отличная производительность модели (mAP@0.5 > 70%)</li>"
        elif avg_map50 > 0.5:
            report_content += "<li>⚠️ Хорошая производительность, но есть место для улучшения</li>"
        else:
            report_content += "<li>❌ Требуется дополнительная настройка модели</li>"
        
        if best_precision > 0.8:
            report_content += "<li>✅ Высокая точность детекций</li>"
        else:
            report_content += "<li>⚠️ Рассмотрите увеличение порога уверенности для повышения точности</li>"
        
        if best_recall > 0.8:
            report_content += "<li>✅ Хорошее покрытие объектов</li>"
        else:
            report_content += "<li>⚠️ Рассмотрите снижение порога уверенности или дополнительную аугментацию</li>"
    
    report_content += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Сохранение отчета
    report_path = output_dir / "evaluation_comparison_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Сравнительный отчет создан: {report_path}")
    return report_path

def main():
    """Главная функция скрипта"""
    parser = argparse.ArgumentParser(
        description="Оценка модели YOLOv11 для детекции объектов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Базовая оценка
  python scripts/evaluate_model.py --model models/trained/best.pt --dataset data/dataset/dataset.yaml

  # Кастомная оценка с детальным анализом
  python scripts/evaluate_model.py --model models/trained/best.pt --test-images data/test/images --test-annotations data/test/labels

  # Полная оценка (официальная + кастомная)
  python scripts/evaluate_model.py --model models/trained/best.pt --dataset data/dataset/dataset.yaml --test-images data/test/images --test-annotations data/test/labels --full-evaluation

  # Оценка с настройкой порогов
  python scripts/evaluate_model.py --model models/trained/best.pt --dataset data/dataset/dataset.yaml --confidence 0.1 --iou 0.5
        """
    )
    
    # Обязательные аргументы
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Путь к обученной модели (.pt файл)'
    )
    
    # Конфигурация датасета
    parser.add_argument(
        '--dataset',
        type=str,
        help='Путь к YAML конфигурации датасета'
    )
    
    # Кастомная оценка
    parser.add_argument(
        '--test-images',
        type=str,
        help='Директория с тестовыми изображениями для кастомной оценки'
    )
    
    parser.add_argument(
        '--test-annotations',
        type=str,
        help='Директория с тестовыми аннотациями для кастомной оценки'
    )
    
    parser.add_argument(
        '--class-mapping',
        type=str,
        help='Путь к файлу маппинга классов (JSON)'
    )
    
    # Параметры оценки
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.001,
        help='Порог уверенности для оценки'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.6,
        help='Порог IoU для NMS'
    )
    
    # Опции вывода
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/evaluation',
        help='Директория для сохранения результатов'
    )
    
    parser.add_argument(
        '--full-evaluation',
        action='store_true',
        help='Выполнить полную оценку (официальная + кастомная)'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        default=True,
        help='Генерировать HTML отчет'
    )
    
    # Системные параметры
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
    
    logger.info("🎯 Запуск скрипта оценки модели YOLOv11")
    logger.info(f"Аргументы: {vars(args)}")
    
    try:
        # Проверка модели
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"Модель не найдена: {model_path}")
            sys.exit(1)
        
        # Создание выходной директории
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        evaluations = []
        
        # Официальная оценка YOLO
        if args.dataset:
            dataset_yaml = Path(args.dataset)
            if not dataset_yaml.exists():
                logger.error(f"Конфигурация датасета не найдена: {dataset_yaml}")
                sys.exit(1)
            
            logger.info("🔄 Запуск официальной оценки YOLO...")
            
            official_results = run_official_evaluation(
                model_path=model_path,
                dataset_yaml=dataset_yaml,
                output_dir=output_dir,
                confidence_threshold=args.confidence,
                iou_threshold=args.iou
            )
            
            evaluations.append(official_results)
            
            # Вывод основных метрик
            metrics = official_results.get("metrics", {})
            logger.info("📊 Основные метрики (официальная оценка):")
            logger.info(f"  mAP@0.5: {metrics.get('metrics/mAP50(B)', 0):.4f}")
            logger.info(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
            logger.info(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
            logger.info(f"  Recall: {metrics.get('metrics/recall(B)', 0):.4f}")
        
        # Кастомная оценка
        if args.test_images and args.test_annotations:
            logger.info("🔄 Запуск кастомной оценки...")
            
            test_images_dir = Path(args.test_images)
            test_annotations_dir = Path(args.test_annotations)
            
            if not test_images_dir.exists():
                logger.error(f"Директория тестовых изображений не найдена: {test_images_dir}")
                sys.exit(1)
            
            if not test_annotations_dir.exists():
                logger.error(f"Директория тестовых аннотаций не найдена: {test_annotations_dir}")
                sys.exit(1)
            
            # Загрузка маппинга классов
            class_mapping = {}
            if args.class_mapping:
                class_mapping_file = Path(args.class_mapping)
                if class_mapping_file.exists():
                    with open(class_mapping_file, 'r', encoding='utf-8') as f:
                        mapping_data = json.load(f)
                        class_mapping = {int(k): v for k, v in mapping_data.get('reverse_mapping', {}).items()}
                else:
                    logger.warning(f"Файл маппинга классов не найден: {class_mapping_file}")
            
            # Автоматическое создание маппинга, если не указан
            if not class_mapping:
                logger.info("Создание автоматического маппинга классов...")
                from config.config import config
                class_mapping = {i: name for i, name in enumerate(config.annotation.target_classes)}
            
            custom_results = run_custom_evaluation(
                model_path=model_path,
                test_images_dir=test_images_dir,
                test_annotations_dir=test_annotations_dir,
                class_mapping=class_mapping,
                output_dir=output_dir,
                confidence_threshold=args.confidence,
                iou_threshold=args.iou
            )
            
            evaluations.append(custom_results)
            
            # Вывод основных метрик
            overall_metrics = custom_results.get("overall_metrics", {}).get("overall_metrics", {})
            map_metrics = custom_results.get("overall_metrics", {}).get("map_metrics", {})
            
            logger.info("📊 Основные метрики (кастомная оценка):")
            logger.info(f"  mAP@0.5: {map_metrics.get('mAP@0.50', 0):.4f}")
            logger.info(f"  mAP@0.5:0.95: {map_metrics.get('mAP@0.5:0.95', 0):.4f}")
            logger.info(f"  Precision: {overall_metrics.get('precision', 0):.4f}")
            logger.info(f"  Recall: {overall_metrics.get('recall', 0):.4f}")
            logger.info(f"  F1-Score: {overall_metrics.get('f1_score', 0):.4f}")
        
        # Проверка, что хотя бы одна оценка была выполнена
        if not evaluations:
            logger.error("Не указаны параметры ни для официальной, ни для кастомной оценки")
            logger.error("Укажите --dataset для официальной оценки или --test-images и --test-annotations для кастомной")
            sys.exit(1)
        
        # Генерация сравнительного отчета
        if args.generate_report and len(evaluations) > 0:
            logger.info("📝 Генерация сравнительного отчета...")
            
            report_path = generate_comparison_report(evaluations, output_dir)
            
            logger.info(f"✅ Сравнительный отчет создан: {report_path}")
        
        # Итоговое резюме
        logger.info("🎉 ОЦЕНКА МОДЕЛИ ЗАВЕРШЕНА!")
        logger.info(f"Выполнено оценок: {len(evaluations)}")
        logger.info(f"Результаты сохранены в: {output_dir}")
        
        # Инструкции для следующих шагов
        logger.info("📋 Следующие шаги:")
        logger.info("1. Просмотрите HTML отчет для детального анализа")
        logger.info("2. Изучите PR-кривые и матрицу ошибок")
        logger.info("3. При необходимости настройте пороги confidence/IoU")
        logger.info("4. Рассмотрите дополнительную аугментацию для слабых классов")
        
        if args.generate_report:
            logger.info(f"5. Откройте отчет: {output_dir / 'evaluation_comparison_report.html'}")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("Оценка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    import pandas as pd
    main()