import cv2
import config
import numpy as np
import onnxruntime as ort

from typing import List, Dict, Tuple
from utils.logger import logger


class Detector:
    def __init__(self):
        self.session = self._init_onnx_session()
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        input_shape = self.session.get_inputs()[0].shape
        self.input_size = (input_shape[2], input_shape[3])

        self.conf_threshold = config.CONFIDENCE_THRESHOLD
        self.iou_threshold = config.IOU_THRESHOLD

    @staticmethod
    def _init_onnx_session():
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 2

        return ort.InferenceSession(
            config.MODEL_PATH,
            providers=['CPUExecutionProvider'],
            sess_options=session_options
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img, _ = self._letterbox(img, new_shape=self.input_size)

        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    @staticmethod
    def _letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
        shape = im.shape[:2]

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
        return im, (r, r)

    def _postprocess(self, outputs: np.ndarray,
                     orig_shape: Tuple[int, int]) -> List[Dict]:
        detections = []
        outputs = np.squeeze(outputs).T
        rows = outputs.shape[0]

        orig_height, orig_width = orig_shape

        for i in range(rows):
            row = outputs[i]
            confidence = row[4]
            if confidence < self.conf_threshold:
                continue

            xc, yc, w, h = row[0], row[1], row[2], row[3]
            x1 = (xc - w / 2) / self.input_size[1] * orig_width
            y1 = (yc - h / 2) / self.input_size[0] * orig_height
            x2 = (xc + w / 2) / self.input_size[1] * orig_width
            y2 = (yc + h / 2) / self.input_size[0] * orig_height

            class_id = 0
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "class": class_id
            })

        return self._nms(detections)

    def _nms(self, detections: List[Dict]) -> List[Dict]:
        if len(detections) == 0:
            return []

        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]

    def detect(self, image: np.ndarray) -> Tuple[List[Dict], int]:
        try:
            orig_shape = image.shape[:2]
            blob = self._preprocess(image)

            outputs = self.session.run(
                self.output_names,
                {self.input_name: blob}
            )[0]

            detections = self._postprocess(outputs, orig_shape)
            return detections, orig_shape[0]

        except Exception as e:
            logger.error(f"Ошибка детекции: {str(e)}")
            return [], 0
