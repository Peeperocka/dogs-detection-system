# image_queue.py
import queue

import config


class ImageQueue:
    def __init__(self):
        self.queue = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)

    def add_image(self, image):
        self.queue.put(image, block=False)

    def get_image(self):
        return self.queue.get()

    def empty(self):
        return self.queue.empty()

    def full(self):
        return self.queue.full()

    def task_done(self):
        self.queue.task_done()
