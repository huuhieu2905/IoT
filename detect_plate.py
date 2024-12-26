from flask import Flask, render_template, request, redirect, url_for
from yolov5 import detect  # import thư viện detect từ YOLOv5
import cv2
import pandas
from PIL import Image
import cv2
import torch
import math 
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import argparse
import function.helper as helper
import requests
import io


def push_img(img, upload_url="https://io-backend-final.onrender.com/upload" ):
    img = Image.fromarray(img)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")  # Lưu ảnh ở định dạng PNG
    buffer.seek(0)  # Đặt con trỏ về đầu buffer

    files = {"image":buffer}  # Tạo payload cho file
    response = requests.post(upload_url, files=files)

    # Kiểm tra phản hồi từ server
    if response.status_code == 200:
        print("Upload complete")
        print("Response from server", response.text)
    else:
        print("Upload incomplete", response.status_code)

# Hàm load model YOLO
def load_detect_lp_model():
    model = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
    return model

model_detect = load_detect_lp_model()

vid = cv2.VideoCapture(2)
idx = 0
while(True):
    ret, frame = vid.read()
    results = model_detect(frame, size=640)
    list_plates = results.pandas().xyxy[0].values.tolist()
    for plate in list_plates:
        flag = 0
        if plate[4] > 0.9:
        # print(plate)
            x = int(plate[0])
            y = int(plate[1])
            w = int(plate[2] - plate[0])
            h = int(plate[3] - plate[1])
            crop_img = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (255,0,225), thickness = 2)
            push_img(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    idx += 1
    





