import time
import cv2
import threading
import os

from processing.image_queue import ImageQueue
from processing.detector import Detector
from processing.cluster import Cluster
from processing.response import Response
from utils.logger import logger


def detection_thread(image_queue):
    detector = Detector()
    cluster = Cluster()
    response = Response()

    while True:
        if not image_queue.empty():
            image_tuple = image_queue.get_image()
            image, filename = image_tuple

            logger.info(
                f"Детекционный поток: Получено изображение '{filename}' для обработки.")

            detections, image_height = detector.detect(image)

            if detections:
                logger.info(
                    f"Детекционный поток: На изображении '{filename}' найдено {len(detections)} собак.")
                clusters = cluster.cluster_detections(detections, image_height)
                response.handle_pack(clusters)
                logger.info(
                    f"Детекционный поток: Обработка изображения '{filename}' завершена.")
            else:
                logger.info(
                    f"Детекционный поток: На изображении '{filename}' не обнаружено собак.")

            image_queue.queue.task_done()
        else:
            time.sleep(0.1)


def image_ingestion_thread(image_queue):
    while True:
        try:
            folder_path = "input_images"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            files = os.listdir(folder_path)
            images_to_add = []

            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, file)
                    image = cv2.imread(image_path)
                    if image is not None:
                        images_to_add.append((image, file))

            images_added_count = 0
            for image, file in images_to_add:
                if not image_queue.full():
                    image_queue.add_image((image, file))
                    image_path = os.path.join(folder_path, file)
                    os.remove(image_path)
                    logger.info(f"Добавлено изображение '{file}' в очередь.")
                    images_added_count += 1
                else:
                    logger.warning("Очередь заполнена, остановка добавления изображений.")
                    break

            if images_added_count == 0 and files:
                logger.debug("Очередь полна, ждем...")
            elif not files:
                logger.debug("Папка 'input_images' пуста.")

            time.sleep(2)
        except Exception as e:
            logger.error(f"Поток приема изображений: Произошла ошибка: {e}")
            time.sleep(10)


def main():
    image_queue = ImageQueue()

    ingestion_thread = threading.Thread(target=image_ingestion_thread, args=(image_queue,), daemon=True,
                                        name="ImageIngestionThread")
    ingestion_thread.start()

    processing_thread = threading.Thread(target=detection_thread, args=(image_queue,), daemon=True,
                                         name="DetectionThread")
    processing_thread.start()

    logger.info("Главный поток: Потоки приема и обработки изображений запущены.")
    print("Потоки приема и обработки изображений запущены...")

    try:
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        logger.info("Главный поток: Получен сигнал завершения программы.")
        logger.info("Главный поток: Завершение работы...")


if __name__ == "__main__":
    main()
