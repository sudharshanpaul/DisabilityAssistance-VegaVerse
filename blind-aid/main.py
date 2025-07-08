import sys
import torch
import cv2
import numpy as np
import pyttsx3

# Add yolov5 repo to path
sys.path.append("yolov5")

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes

# Initialize Text-to-Speech
tts = pyttsx3.init()
tts.setProperty('rate', 160)

# Load YOLOv5 model
model = DetectMultiBackend('yolov5s.pt', device=torch.device('cpu'))
stride, names, pt = model.stride, model.names, model.pt

# Distance estimation using bounding box width
def estimate_distance(bbox_width, known_width=45, focal_length=950):
    if bbox_width == 0:
        return 0
    return round((known_width * focal_length) / bbox_width / 100, 2)

# Get position
def get_position(x_center, frame_width):
    if x_center < frame_width / 3:
        return "left"
    elif x_center < 2 * frame_width / 3:
        return "center"
    else:
        return "right"

# Speak helper
def speak(text):
    print("🔊", text)
    tts.say(text)
    tts.runAndWait()

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

print("Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    im0 = frame.copy()
    img = cv2.resize(im0, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)

    img = img.to(torch.device('cpu'))
    model.warmup(img.shape)
    pred = model(img, augment=False, visualize=False)[0]

    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                label = names[int(cls)]
                x_center = (x1 + x2) // 2
                bbox_width = x2 - x1
                distance = estimate_distance(bbox_width)
                position = get_position(x_center, im0.shape[1])

                sentence = f"There is a {label} {distance} meters to your {position}."
                speak(sentence)

                # Draw
                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(im0, f"{label} {distance}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Blind Assist", im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
