import cv2
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from sklearn.neighbors import NearestNeighbors
from mmdet.apis import inference_detector, init_detector
import torch

import settings


OK_COLOR = settings.visual['OK_COLOR']
HIGHLIGHT_COLOR = settings.visual['HIGHLIGHT_COLOR']
ALPHA = settings.visual['ALPHA']
BBOX_WIDTH = settings.visual['BBOX_WIDTH']
LINE_WIDTH = settings.visual['LINE_WIDTH']
POINT_SZ = settings.visual['POINT_SZ']

def box2centroid(box):
        l, u, r, b = box
        h, w = b-u, r-l
        return int(l+w/2), int(u+h/2)

def plot_boxes_and_distances(image, boxes, points, neighbors):
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw, 'RGBA')
    for i, (box, point, box_neighbors) in enumerate(zip(boxes, points, neighbors)):
        color = OK_COLOR if len(box_neighbors)==1 else HIGHLIGHT_COLOR

        # plot bb
        draw.rectangle(box, outline=color, width=BBOX_WIDTH, fill=color+(ALPHA,))

        # plot distance lines
        if len(neighbors)>1:
            x,y = point
            draw.ellipse([x-POINT_SZ/2, y-POINT_SZ/2, x+POINT_SZ/2, y+POINT_SZ/2], fill=color)

            for j in box_neighbors:
                if j > i:
                    draw.line((*points[i],*points[j]), width=LINE_WIDTH, fill=HIGHLIGHT_COLOR)
    return image_draw


class Perspective2Plane():
    def __init__(self, image_size, perspective_pts, distance_pts, distance_value):
        w, h = image_size
        perspective_pts = np.array(perspective_pts, dtype='float32')
        distance_pts = np.array(distance_pts, dtype='float32')
        pts_out = np.array([(0, 0), (100, 0), (100, 100), (0, 100)], dtype='float32')
        pts_img = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype='float32')
        
        M = cv2.getPerspectiveTransform(perspective_pts, pts_out)
        corners = cv2.perspectiveTransform(pts_img[None], M)
        bx, by, bwidth, bheight = cv2.boundingRect(corners)
        self.bwidth, self.bheight = bwidth, bheight
        A = np.float32([[1,0,-bx], [0,1,-by], [0,0,1]]) 
        self.F = A @ M
        (ax, ay), (bx, by) = cv2.perspectiveTransform(distance_pts[None], M)[0]
        distance_in_pixels = np.sqrt(np.array((ax-bx)**2 + (ay-by)**2))
        self.pixels_per_meter = distance_in_pixels / distance_value
        
    def transform_image(self, img):
        return cv2.warpPerspective(img, self.F, (self.bwidth, self.bheight))
    
    def transform_points(self, pts):
        pts = np.array(pts, dtype='float32')
        return cv2.perspectiveTransform(pts[None], self.F)[0]
    
    def get_neighbors(self, pts, radius):
        pts = self.transform_points(pts)
        radius = radius * self.pixels_per_meter
        nn = NearestNeighbors(n_neighbors=5, radius=radius, p=2)
        nn.fit(pts)
        return nn.radius_neighbors(pts, return_distance=False)


class DistancingDetector():
    def __init__(self, camera_calibration, distance_treshold, detection_treshold):
        self.perspective2plane = Perspective2Plane(**camera_calibration)
       
        self.detection_model = init_detector(
            config='Pedestron/cascade_hrnet.py',
            checkpoint=settings.detection_trained_model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu')
        self.distance_treshold = distance_treshold
        self.detection_treshold = detection_treshold

    def _detect_people(self, image, score_thr):
        results = inference_detector(self.detection_model, image)
        if isinstance(results, tuple): 
            bbox_result = results[0]
        else: 
            bbox_result = results
        bboxes = np.vstack(bbox_result)
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 5
        if score_thr:
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :4]
        return bboxes

    def detect(self, image):
        bboxes = self._detect_people(np.array(image), self.detection_treshold)
        centroids = [box2centroid(x) for x in bboxes]
        neighbors = self.perspective2plane.get_neighbors(centroids, radius=self.distance_treshold)
        image_draw = plot_boxes_and_distances(image, bboxes, centroids, neighbors)
        return image_draw