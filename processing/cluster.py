from sklearn.cluster import DBSCAN
import numpy as np
import config


class Cluster:
    def __init__(self):
        self.eps_fraction = config.EPS
        self.min_samples = config.MIN_SAMPLES

    def cluster_detections(self, detections, image_height):
        if not detections:
            return []

        centers = []
        for detection in detections:
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append([center_x, center_y])

        eps_pixels = image_height * self.eps_fraction  # Calculate EPS in pixels
        dbscan = DBSCAN(eps=eps_pixels, min_samples=self.min_samples)  # Use pixel EPS
        clusters = dbscan.fit_predict(np.array(centers))

        clustered_detections = [[] for _ in range(max(clusters) + 1)]
        for i, label in enumerate(clusters):
            if label != -1:
                clustered_detections[label].append(detections[i])

        return clustered_detections
