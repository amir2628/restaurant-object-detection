"""
Скрипт для запуска инференса обученной модели YOLOv11
Поддерживает обработку изображений, видео и реальное время
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Добавление корневой директории в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger
from src.models.inference import YOLOInference, benchmark_model
from src.utils.visualization import ReportGenerator
from config.config import config

def process_images(inference: YOLOInference, 
                  input_paths: List[Path], 
                  output_dir: Path,
                  save_visualizations: bool = True) -> List[dict]:
    """
    Обработка списка изображений
    
    Args:
        inference: Объект инференса
        input_paths: Пути к изображениям
        output_dir: Выходная директория
        save_visualizations: Сохранять визуализации
        
    Returns:
        Результаты обработки
    """
    logger = get_logger(__name__)
    
    logger.info(f"Обработка {len(input_paths)} изображений...")
    
    # Создание директорий
    if save_visualizations:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Пакетная обработка
    results = inference.predict_batch(input_paths)
    
    # Сохранение результатов
    all_results = []
    for i, result in enumerate(results):
        result_dict = result.to_dict()
        all_results.append(result_dict)
        
        # Сохранение индивидуального результата
        result_file = results_dir / f"result_{i:04d}.json"
        inference.save_results(result, result_file, format="json")
        
        # Визуализация
        if save_visualizations and result.detections:
            try:
                import cv2
                image = cv2.imread(result.image_path)
                if image is not None:
                    visualized = inference.visualizer.draw_detections(image, result.detections)
                    
                    vis_path = vis_dir / f"vis_{Path(result.image_path).stem}.jpg"
                    cv2.imwrite(str(vis_path), visualized)
            except Exception as e:
                logger.warning(f"Ошибка при визуализации {result.image_path}: {e}")
    
    # Сохранение сводных результатов
    summary_file = output_dir / "inference_summary.json"
    inference.save_results(all_results, summary_file, format="json")
    
    logger.info(f"Результаты сохранены в: {output_dir}")
    return all_results

def process_video(inference: YOLOInference,
                 video_path: Path,
                 output_dir: Path,
                 save_output_video: bool = True,
                 frame_skip: int = 1) -> dict:
    """
    Обработка видео файла
    
    Args:
        inference: Объект инференса
        video_path: Путь к видео
        output_dir: Выходная директория
        save_output_video: Сохранять обработанное видео
        frame_skip: Пропускать кадры
        
    Returns:
        Результаты обработки видео
    """
    logger = get_logger(__name__)
    
    logger.info(f"Обработка видео: {video_path}")
    
    # Подготовка путей
    output_video_path = None
    if save_output_video:
        output_video_path = output_dir / f"processed_{video_path.name}"
    
    # Обработка видео
    results = inference.predict_video(
        video_path=video_path,
        output_path=output_video_path,
        save_frames=False,
        frame_skip=frame_skip
    )
    
    # Сохранение результатов
    results_file = output_dir / f"video_results_{video_path.stem}.json"
    inference.save_results(results, results_file, format="json")
    
    logger.info(f"Видео обработано: {len(results['frame_results'])} кадров")
    return results

def run_realtime_inference(inference: YOLOInference,
                          camera_id: int = 0,
                          save_video: bool = False,
                          output_dir: Optional[Path] = None) -> None:
    """
    Запуск инференса в реальном времени
    
    Args:
        inference: Объект инференса
        camera_id: ID камеры
        save_video: Сохранять видео
        output_dir: Директория для сохранения
    """
    logger = get_logger(__name__)
    
    logger.info(f"Запуск инференса в реальном времени с камеры {camera_id}")
    
    output_path = None
    if save_video and output_dir:
        output_path = str(output_dir / "realtime_output.mp4")
    
    try:
        inference.predict_realtime(
            camera_id=camera_id,
            display_results=True,
            save_video=save_video,
            output_path=output_path
        )
    except KeyboardInterrupt:
        logger.info("Инференс в реальном времени остановлен пользователем")
    except Exception as e:
        logger.error(f"Ошибка при инференсе в реальном времени: {e}")

def generate_performance_report(inference: YOLOInference, 
                              output_dir: Path,
                              test_images: Optional[List[Path]] = None) -> None:
    """
    Генерация отчета о производительности
    
    Args:
        inference: Объект инференса
        output_dir: Выходная директория
        test_images: Тестовые изображения для бенчмарка
    """
    logger = get_logger(__name__)
    
    logger.info("Генерация отчета о производительности...")
    
    # Базовая статистика
    stats = inference.get_performance_stats()
    
    # Бенчмарк на тестовых изображениях
    benchmark_results = None
    if test_images and len(test_images) > 0:
        benchmark_results = benchmark_model(
            model_path=inference.model_path,
            test_images=test_images[:50],  # Ограничиваем для быстроты
            device=str(inference.device_manager.get_device())
        )
    
    # HTML отчет
    report_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv11 Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .section {{ margin-bottom: 30px; }}
            .stats-table {{ border-collapse: collapse; width: 100%; }}
            .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .stats-table th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>YOLOv11 Performance Report</h1>
            <p>Model: {inference.model_path.name}</p>
            <p>Device: {inference.device_manager.get_device()}</p>
            <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Performance Statistics</h2>
            <table class="stats-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Inferences</td><td>{stats['total_inferences']}</td></tr>
                <tr><td>Total Detections</td><td>{stats['total_detections']}</td></tr>
                <tr><td>Images Processed</td><td>{stats['images_processed']}</td></tr>
                <tr><td>Videos Processed</td><td>{stats['videos_processed']}</td></tr>
                <tr><td>Average Inference Time</td><td>{stats['average_inference_time']:.4f} seconds</td></tr>
                <tr><td>Estimated FPS</td><td>{1/stats['average_inference_time'] if stats['average_inference_time'] > 0 else 0:.1f}</td></tr>
            </table>
        </div>
    """
    
    if benchmark_results:
        report_content += f"""
        <div class="section">
            <h2>Benchmark Results</h2>
            <table class="stats-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Test Images</td><td>{benchmark_results['total_images']}</td></tr>
                <tr><td>Successful Predictions</td><td>{benchmark_results['successful_predictions']}</td></tr>
                <tr><td>Total Processing Time</td><td>{benchmark_results['total_time']:.2f} seconds</td></tr>
                <tr><td>Average Time per Image</td><td>{benchmark_results['average_time_per_image']:.4f} seconds</td></tr>
                <tr><td>Benchmark FPS</td><td>{benchmark_results['fps']:.2f}</td></tr>
                <tr><td>Total Detections</td><td>{benchmark_results['total_detections']}</td></tr>
                <tr><td>Avg Detections per Image</td><td>{benchmark_results['average_detections_per_image']:.2f}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Inference Time Statistics</h2>
            <table class="stats-table">
                <tr><th>Statistic</th><th>Value (seconds)</th></tr>
                <tr><td>Minimum</td><td>{benchmark_results['inference_time_stats']['min']:.4f}</td></tr>
                <tr><td>Maximum</td><td>{benchmark_results['inference_time_stats']['max']:.4f}</td></tr>
                <tr><td>Mean</td><td>{benchmark_results['inference_time_stats']['mean']:.4f}</td></tr>
                <tr><td>Standard Deviation</td><td>{benchmark_results['inference_time_stats']['std']:.4f}</td></tr>
            </table>
        </div>
        """
    
    report_content += """
    </body>
    </html>
    """
    
    # Сохранение отчета
    report_path = output_dir / "performance_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Отчет о производительности сохранен: {report_path}")

def main():
    """Главная функция скрипта"""
    parser = argparse.ArgumentParser(
        description="Инференс модели YOLOv11 для детекции объектов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Обработка изображений
  python scripts/run_inference.py --model models/trained/best.pt --images data/test/*.jpg --output results/

  # Обработка видео
  python scripts/run_inference.py --model models/trained/best.pt --video input.mp4 --output results/

  # Инференс в реальном времени
  python scripts/run_inference.py --model models/trained/best.pt --realtime --camera 0

  # Бенчмарк модели
  python scripts/run_inference.py --model models/trained/best.pt --benchmark --test-images data/test/

  # Обработка директории с изображениями
  python scripts/run_inference.py --model models/trained/best.pt --input-dir data/test/ --output results/
        """
    )
    
    # Обязательные аргументы
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Путь к обученной модели (.pt файл)'
    )
    
    # Входные данные
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--images',
        nargs='+',
        help='Пути к изображениям для обработки'
    )
    group.add_argument(
        '--input-dir',
        type=str,
        help='Директория с изображениями'
    )
    group.add_argument(
        '--video',
        type=str,
        help='Путь к видео файлу'
    )
    group.add_argument(
        '--realtime',
        action='store_true',
        help='Инференс в реальном времени с камеры'
    )
    group.add_argument(
        '--benchmark',
        action='store_true',
        help='Запуск бенчмарка модели'
    )
    
    # Выходные параметры
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/inference',
        help='Директория для сохранения результатов'
    )
    
    # Параметры инференса
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Порог уверенности для детекций'
    )
    
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.45,
        help='Порог IoU для NMS'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='Устройство для инференса'
    )
    
    # Параметры визуализации
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Не сохранять визуализации'
    )
    
    parser.add_argument(
        '--no-save-results',
        action='store_true',
        help='Не сохранять файлы результатов'
    )
    
    # Параметры для видео
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='Обрабатывать каждый N-й кадр видео'
    )
    
    parser.add_argument(
        '--no-output-video',
        action='store_true',
        help='Не сохранять обработанное видео'
    )
    
    # Параметры для реального времени
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='ID камеры для реального времени'
    )
    
    parser.add_argument(
        '--save-realtime-video',
        action='store_true',
        help='Сохранять видео с камеры'
    )
    
    # Параметры для бенчмарка
    parser.add_argument(
        '--test-images',
        type=str,
        help='Директория с тестовыми изображениями для бенчмарка'
    )
    
    # Отчеты
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Генерировать HTML отчет'
    )
    
    parser.add_argument(
        '--performance-report',
        action='store_true',
        help='Генерировать отчет о производительности'
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
    
    logger.info("🚀 Запуск скрипта инференса YOLOv11")
    logger.info(f"Аргументы: {vars(args)}")
    
    try:
        # Проверка модели
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"Модель не найдена: {model_path}")
            sys.exit(1)
        
        # Создание выходной директории
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализация инференса
        inference = YOLOInference(
            model_path=model_path,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou_threshold,
            device=args.device
        )
        
        # Выполнение задач
        if args.images:
            # Обработка списка изображений
            image_paths = [Path(img) for img in args.images]
            results = process_images(
                inference=inference,
                input_paths=image_paths,
                output_dir=output_dir,
                save_visualizations=not args.no_visualizations
            )
            
            if args.generate_report:
                report_gen = ReportGenerator()
                report_gen.generate_inference_report(results, output_dir)
        
        elif args.input_dir:
            # Обработка директории с изображениями
            input_dir = Path(args.input_dir)
            if not input_dir.exists():
                logger.error(f"Входная директория не найдена: {input_dir}")
                sys.exit(1)
            
            # Поиск изображений
            image_paths = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                image_paths.extend(input_dir.glob(f"*{ext}"))
                image_paths.extend(input_dir.glob(f"*{ext.upper()}"))
            
            if not image_paths:
                logger.error(f"Изображения не найдены в: {input_dir}")
                sys.exit(1)
            
            logger.info(f"Найдено {len(image_paths)} изображений")
            
            results = process_images(
                inference=inference,
                input_paths=image_paths,
                output_dir=output_dir,
                save_visualizations=not args.no_visualizations
            )
            
            if args.generate_report:
                report_gen = ReportGenerator()
                report_gen.generate_inference_report(results, output_dir)
        
        elif args.video:
            # Обработка видео
            video_path = Path(args.video)
            if not video_path.exists():
                logger.error(f"Видео файл не найден: {video_path}")
                sys.exit(1)
            
            results = process_video(
                inference=inference,
                video_path=video_path,
                output_dir=output_dir,
                save_output_video=not args.no_output_video,
                frame_skip=args.frame_skip
            )
        
        elif args.realtime:
            # Инференс в реальном времени
            run_realtime_inference(
                inference=inference,
                camera_id=args.camera,
                save_video=args.save_realtime_video,
                output_dir=output_dir if args.save_realtime_video else None
            )
        
        elif args.benchmark:
            # Бенчмарк модели
            test_images = []
            
            if args.test_images:
                test_dir = Path(args.test_images)
                if test_dir.exists():
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        test_images.extend(test_dir.glob(f"*{ext}"))
            
            if not test_images:
                logger.error("Для бенчмарка необходимы тестовые изображения (--test-images)")
                sys.exit(1)
            
            logger.info(f"Запуск бенчмарка на {len(test_images)} изображениях")
            
            benchmark_results = benchmark_model(
                model_path=model_path,
                test_images=test_images,
                device=args.device
            )
            
            # Сохранение результатов бенчмарка
            benchmark_file = output_dir / "benchmark_results.json"
            inference.save_results(benchmark_results, benchmark_file)
            
            logger.info(f"Результаты бенчмарка сохранены в: {benchmark_file}")
        
        # Генерация отчета о производительности
        if args.performance_report:
            test_images_for_report = []
            if args.test_images:
                test_dir = Path(args.test_images)
                if test_dir.exists():
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        test_images_for_report.extend(test_dir.glob(f"*{ext}"))
            
            generate_performance_report(
                inference=inference,
                output_dir=output_dir,
                test_images=test_images_for_report[:20]  # Ограничиваем для быстроты
            )
        
        # Финальная статистика
        stats = inference.get_performance_stats()
        logger.info("📊 Статистика выполнения:")
        logger.info(f"  - Общее количество инференсов: {stats['total_inferences']}")
        logger.info(f"  - Общее количество детекций: {stats['total_detections']}")
        logger.info(f"  - Среднее время инференса: {stats['average_inference_time']:.4f}с")
        
        if stats['average_inference_time'] > 0:
            fps = 1 / stats['average_inference_time']
            logger.info(f"  - Расчетная производительность: {fps:.1f} FPS")
        
        logger.info(f"✅ Результаты сохранены в: {output_dir}")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("Работа прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    import pandas as pd
    main()