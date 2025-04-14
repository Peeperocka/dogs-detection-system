import asyncio
import cv2
import os
import config

from concurrent.futures import ThreadPoolExecutor
from processing.image_queue import ImageQueue
from processing.detector import Detector
from processing.cluster import Cluster
from processing.response import Response
from utils.logger import logger


class AsyncImageProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=config.THREAD_POOL_MAX_WORKERS)
        self.image_queue = ImageQueue()
        self.detector = Detector()
        self.cluster = Cluster()
        self.response = Response()

    async def async_read_image(self, image_path):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: (cv2.imread(image_path), os.path.basename(image_path))
        )

    async def async_file_operations(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            func,
            *args
        )

    async def image_ingestion_task(self):
        while True:
            try:
                folder_path = "input_images"
                await self.async_file_operations(os.makedirs, folder_path, exist_ok=True)

                files = await self.async_file_operations(os.listdir, folder_path)
                images_to_add = []

                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(folder_path, file)
                        image, filename = await self.async_read_image(image_path)
                        if image is not None:
                            images_to_add.append((image, filename))

                images_added_count = 0
                for image, file in images_to_add:
                    if not self.image_queue.full():
                        await self.async_file_operations(
                            self.image_queue.add_image,
                            (image, file)
                        )
                        await self.async_file_operations(os.remove, os.path.join(folder_path, file))
                        logger.info(f"Добавлено изображение '{file}' в очередь.")
                        images_added_count += 1
                    else:
                        logger.warning("Очередь заполнена, остановка добавления изображений.")
                        break

                if images_added_count == 0 and files:
                    logger.debug("Очередь полна, ждем...")
                elif not files:
                    logger.debug("Папка 'input_images' пуста.")

                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Ошибка приема изображений: {str(e)}")
                await asyncio.sleep(10)

    async def detection_task(self):
        while True:
            try:
                if not self.image_queue.empty():
                    image_tuple = await self.async_file_operations(self.image_queue.get_image)
                    image, filename = image_tuple

                    logger.info(f"Обработка изображения: {filename}")

                    loop = asyncio.get_running_loop()
                    detections, image_height = await loop.run_in_executor(
                        self.executor,
                        self.detector.detect,
                        image
                    )

                    if detections:
                        clusters = await loop.run_in_executor(
                            self.executor,
                            self.cluster.cluster_detections,
                            detections,
                            image_height
                        )
                        await loop.run_in_executor(
                            self.executor,
                            self.response.handle_pack,
                            clusters
                        )

                    await self.async_file_operations(self.image_queue.task_done)
                else:
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Ошибка обработки: {str(e)}")
                await asyncio.sleep(1)

    async def run(self):
        await asyncio.gather(
            self.image_ingestion_task(),
            self.detection_task()
        )


def main():
    processor = AsyncImageProcessor()

    try:
        asyncio.run(processor.run())
    except KeyboardInterrupt:
        logger.info("Завершение работы по запросу пользователя")
        processor.executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
