import config
from ultralytics import YOLO


class Detector:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)
        self.model.eval()
        self.model.conf = config.CONFIDENCE_THRESHOLD
        self.model.iou = config.IOU_THRESHOLD

    def detect(self, image):
        results = self.model(image, verbose=False)
        detections = []
        boxes = results[0].boxes

        image_height, image_width = image.shape[:2]

        if boxes is not None and len(boxes):
            for box_data in boxes:
                xyxy = box_data.xyxy[0]
                conf = box_data.conf[0]
                cls = int(box_data.cls[0])

                if cls == 0:
                    detections.append({
                        "bbox": xyxy.tolist(),
                        "confidence": conf.item(),
                        "class": cls
                    })
        return detections, image_height
