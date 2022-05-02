#!/usr/bin/env python
# coding: utf-8

from typing import Any, List, Tuple

import face_recognition
from facenet_pytorch import MTCNN
import numpy as np
from IPython.display import display
from PIL import Image, ImageDraw

# default value
downsample_scale = 4
epsilon = 0.001
crop_area = 1.6
down_dir_scale = 2
up_dir_scale = 2

# type definition
Point = Any
Bbox = Tuple[int, int, int, int]
FourPointBbox = Tuple[Point, Point, Point, Point]
Eye = Point
Eyes = Tuple[Eye, Eye]
Landmark = Any


def preprocess(raw_image: Image.Image, downsample: int = downsample_scale) -> Tuple[Image.Image, int]:  # 変形関数
    pimg = raw_image.convert("RGB").resize((raw_image.width // downsample, raw_image.height // downsample))
    return (pimg, downsample)


def map_processed(point_from: Point, downsample: int) -> Point:
    return point_from * downsample

def detect_face(image: Image.Image) -> List[Bbox]:
    try:
        mtcnn = MTCNN(keep_all=True, device='cuda:0', post_process=False)
        boxes, probs = mtcnn.detect(image)
    except Exception as e:
        print(e)
    if boxes is None: 
        return []
    return [(box[1],box[2], box[3], box[0]) for box, prob in zip(boxes, probs) if prob > 0.95]

def crop_face_area(image: Image.Image, bbox: Bbox, crop_area=crop_area) -> Tuple[Image.Image, Bbox]:  # 変形関数
    (top, right, bottom, left) = bbox

    fwidth = right - left
    fheight = bottom - top
    left -= int(fwidth // crop_area)
    right += int(fwidth // crop_area)
    top -= int(fheight // crop_area)
    bottom += int(fheight // crop_area)

    left = left if left > 0 else 0
    top = top if top > 0 else 0
    right = right if right < image.width else image.width
    bottom = bottom if bottom < image.height else image.height
    
    crop_bbox = (left, top, right, bottom)
    crop_bbox = tuple(map(int,crop_bbox))
    cropped = image.crop(crop_bbox)
    return (cropped, (top, right, bottom, left))


def map_cropped(point_from: Point, bbox: Bbox) -> Point:
    return point_from + np.array([bbox[3], bbox[0]])


def face_landmarks(image: Image.Image) -> List[Landmark]:
    lms = face_recognition.face_landmarks(np.array(image))
    return lms


def get_accurate_bbox(lm: Landmark) -> FourPointBbox:
    left_eye = (np.array(lm["left_eye"][0]) + np.array(lm["left_eye"][3])) / 2
    right_eye = (np.array(lm["right_eye"][0]) + np.array(lm["right_eye"][3])) / 2

    ltr_eye = right_eye - left_eye

    right_edge = right_eye + 1.5 * ltr_eye
    left_edge =  left_eye - 1.5 * ltr_eye 

    e_vec = np.array([1, -ltr_eye[0] / ltr_eye[1]] if abs(ltr_eye[1]) > epsilon else [0, 1])  # zerodivに気をつけて！
    e_vec = e_vec / np.linalg.norm(e_vec) * np.linalg.norm(ltr_eye)  # ltr_eyeと同じ長さにする

    left_up = left_edge + up_dir_scale * e_vec
    left_down = left_edge - e_vec * down_dir_scale
    right_up = right_edge + up_dir_scale * e_vec
    right_down = right_edge - e_vec * down_dir_scale
    return (left_up, left_down, right_down, right_up)


def draw_bbox(image: Image.Image, points: FourPointBbox) -> Image.Image:
    d = ImageDraw.Draw(image)
    d.polygon(
        points[0].tolist() + points[1].tolist() + points[2].tolist() + points[3].tolist(), outline=(0, 255, 0, 255),
    )
    return image


def get_abs_accurate_bbox(points: FourPointBbox, bbox: Bbox, downsample: int) -> FourPointBbox:
    return tuple(map_processed(map_cropped(pt, bbox), downsample) for pt in points)


def crop_raw_image(image: Image.Image, points: FourPointBbox) -> Image.Image:
    facesize = (
        int(np.linalg.norm(points[0] - points[3])),
        int(np.linalg.norm(points[0] - points[1])),
    )
    facepoly = points[0].tolist() + points[1].tolist() + points[2].tolist() + points[3].tolist()

    cropped = image.convert("RGBA").transform(facesize, Image.QUAD, tuple(facepoly), resample=Image.NEAREST, fillcolor=(0, 0, 0, 0),)
    if points[0][1] >= points[3][1]:
        cropped = cropped.transpose(Image.FLIP_TOP_BOTTOM)
    return cropped


def draw_fullsize_eye_heatmap(lm: Landmark, raw_image: Image.Image, bbox: Bbox, downsample: int) -> Tuple[Image.Image, Point, Point]:
    left_eye = [map_processed(map_cropped(np.array(eyepoint), bbox), downsample) for eyepoint in lm["left_eye"]]
    right_eye = [map_processed(map_cropped(np.array(eyepoint), bbox), downsample) for eyepoint in lm["right_eye"]]

    left_eye_center = (left_eye[0] + left_eye[3]) / 2
    right_eye_center = (right_eye[0] + right_eye[3]) / 2
    return (None, left_eye_center, right_eye_center)
    
    heatmap = Image.new("L", raw_image.size, color=0)

    d = ImageDraw.Draw(heatmap, "L")
    d.polygon([tuple(p) for p in left_eye], fill=100)
    d.polygon([tuple(p) for p in right_eye], fill=100)
    d.line([tuple(p) for p in (left_eye + [left_eye[0]])], fill=50, joint="curve", width=2)
    d.line(
        [tuple(p) for p in (right_eye + [right_eye[0]])], fill=50, joint="curve", width=2,
    )

    eyesize = np.linalg.norm(left_eye[0] - left_eye[3])
    sizevec = np.array([1, 1]) * eyesize / 16
    d.ellipse(
        (left_eye_center - sizevec).tolist() + (left_eye_center + sizevec).tolist(), fill=255, outline=150, width=1,
    )
    d.ellipse(
        (right_eye_center - sizevec).tolist() + (right_eye_center + sizevec).tolist(), fill=255, outline=150, width=1,
    )
    lip = [tuple(map_processed(map_cropped(np.array(eyepoint), bbox), downsample)) for eyepoint in lm["bottom_lip"]]
    d.polygon(lip, fill=255)
    display(heatmap)
    return (heatmap, left_eye_center, right_eye_center)


def copy_in_four_dir(image: Image.Image,) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    return tuple(image.rotate(a, expand=True) for a in (0, 90, 180, 270))


def common_process(filename: str, angle: int) -> Tuple[Image.Image, int, List[Tuple[Bbox, List[Landmark], Image.Image]]]:
    raw_image = Image.open(filename).rotate(angle, expand=True)
    (pimg, downsample) = preprocess(raw_image)
    face_bbox_list = detect_face(pimg)
    faces = []
    for face_bbox in face_bbox_list:
        (cropped_image, cropped_bbox) = crop_face_area(pimg, face_bbox)
        orig_face_bbox = [map_processed(pt, downsample) for pt in face_bbox]
        (cropped_orig_img, _) = crop_face_area(raw_image, orig_face_bbox)
        landmarks = face_landmarks(cropped_image)
        faces.append((cropped_bbox, landmarks, cropped_orig_img))
    return (raw_image, downsample, faces)


def process(filename: str, angle: int = 0) -> Tuple[Image.Image, List[Tuple[Image.Image, FourPointBbox, Image.Image, Point, Point]]]:
    (raw_image, downsample, faces) = common_process(filename, angle)
    results = []
    face_images = []
    for face in faces:
        (cropped_bbox, landmarks, cropped_orig_img) = face
        for landmark in landmarks:
            points = get_accurate_bbox(landmark)
            raw_points = get_abs_accurate_bbox(points, cropped_bbox, downsample)
            accurate_face = crop_raw_image(raw_image, raw_points)

            (_, left_eye_center, right_eye_center) = draw_fullsize_eye_heatmap(landmark, raw_image, cropped_bbox, downsample)
            results.append((accurate_face, raw_points, left_eye_center, right_eye_center))
        face_images.append(cropped_orig_img) # とってつけたような独自仕様
    return (raw_image, results, face_images)


def crop_accurate_face(filename: str) -> List[Tuple[Image.Image, FourPointBbox]]:
    (raw_image, downsample, faces) = common_process(filename)
    results = []
    for face in faces:
        (cropped_bbox, landmarks) = face
        for landmark in landmarks:
            points = get_accurate_bbox(landmark)
            raw_points = get_abs_accurate_bbox(points, cropped_bbox, downsample)
            accurate_face = crop_raw_image(raw_image, raw_points)
            results.append((accurate_face, raw_points))
    return (results, raw_image)


def generate_heatmap(filename: str) -> List[Tuple[Image.Image, Point, Point]]:
    (raw_image, downsample, faces) = common_process(filename)
    results = []
    for face in faces:
        (cropped_bbox, landmarks) = face
        for landmark in landmarks:
            (heatmap, left_eye_center, right_eye_center) = draw_fullsize_eye_heatmap(landmark, raw_image, cropped_bbox, downsample)
            results.append((heatmap, left_eye_center, right_eye_center))
    return results


def _process(filename: str):
    raw_image = Image.open(filename)
    (pimg, downsample) = preprocess(raw_image)
    face_bbox = detect_face(pimg)
    (cropped_image, cropped_bbox) = crop_face_area(pimg, face_bbox[0])
    landmarks = face_landmarks(cropped_image)

    points = get_accurate_bbox(landmarks[0])
    raw_points = get_abs_accurate_bbox(points, cropped_bbox, downsample)
    accurate_face = crop_raw_image(raw_image, raw_points)

    heatmap = draw_fullsize_eye_heatmap(landmarks[0], raw_image, cropped_bbox, downsample)
    heatmaps = copy_in_four_dir(heatmap)
    return (raw_image, accurate_face, raw_points, heatmaps)


def show_image(image: Image.Image, show_scale: int = 4):
    print(image.size)
    display(image.resize((image.width // show_scale, image.height // show_scale)))
