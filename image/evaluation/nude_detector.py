import os
import cv2
import numpy as np
import torch
import onnxruntime
from typing import List, Dict, Any, Optional
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['ORT_DISABLE_THREAD_AFFINITY'] = '1'

# NudeNet labels
NUDE_LABELS = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE", 
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

# Nude keys
NUDE_KEYS = [
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", 
    "MALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_EXPOSED",
    "ARMPITS_EXPOSED",
]

def _read_image(image_path: str, input_width: int, input_height: int):
    img = cv2.imread(image_path)
    
    img_height, img_width = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_width, input_height))
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    return image_data, img_width, img_height

def _postprocess(output: np.ndarray, img_width: int, img_height: int, 
                input_width: int, input_height: int) -> List[Dict[str, Any]]:
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []
    x_factor = img_width / input_width
    y_factor = img_height / input_height

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score >= 0.5:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5)

    detections = []
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        detections.append({
            "class": NUDE_LABELS[class_id], 
            "score": float(score), 
            "box": box
        })

    return detections

class NudeDetector:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            current_model_path = Path("evaluation/best.onnx")
            if current_model_path.exists():
                model_path = str(current_model_path)
            else:
                raise ValueError(f"Model file not found: {current_model_path}")
        
        try:
            self.onnx_session = onnxruntime.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            model_inputs = self.onnx_session.get_inputs()
            input_shape = model_inputs[0].shape
            self.input_width = input_shape[2]
            self.input_height = input_shape[3]
            self.input_name = model_inputs[0].name
            self.simulation_mode = False
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")
    
    def detect(self, image_path: str) -> List[Dict[str, Any]]:
        preprocessed_image, image_width, image_height = _read_image(
            image_path, self.input_width, self.input_height
        )
        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
        detections = _postprocess(outputs, image_width, image_height, self.input_width, self.input_height)
        return detections
    
    def is_nude(self, image_path: str, threshold: float = 0.6) -> bool:
        detections = self.detect(image_path)
        for detection in detections:
            if (detection['class'] in NUDE_KEYS and 
                detection['score'] > threshold):
                return True
        
        return False
    
    def get_nude_score(self, image_path: str, threshold: float = 0.6) -> Dict[str, float]:
        detections = self.detect(image_path)
        scores = {}
        
        for detection in detections:
            if (detection['class'] in NUDE_KEYS and 
                detection['score'] > threshold):
                scores[detection['class']] = detection['score']
        
        return scores 