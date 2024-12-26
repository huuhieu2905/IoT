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
        print("Upload thành công!")
        print("Phản hồi từ server:", response.text)
    else:
        print("Upload thất bại. Mã lỗi:", response.status_code)

# Hàm load model YOLO
def load_detect_lp_model():
    model = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
    return model

def load_ocr_license_plate_model():
    model = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
    return model

model_detect = load_detect_lp_model()
model_ocr = load_ocr_license_plate_model()

# Hàm dự đoán và phát hiện đối tượng
def detect_objects(img):
    

    return img


vid = cv2.VideoCapture(2)
idx = 0
while(True):
    ret, frame = vid.read()
    results = model_detect(frame, size=640)
    list_plates = results.pandas().xyxy[0].values.tolist()
    for plate in list_plates:
        flag = 0
        # print(plate)
        x = int(plate[0])
        y = int(plate[1])
        w = int(plate[2] - plate[0]) # xmax - xmin
        h = int(plate[3] - plate[1]) # ymax - ymin  
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (255,0,225), thickness = 2)
        lp = ""
        img_plate = crop_img.copy()
        list_read_plates = set()
        for cc in range(0,2):
            for ct in range(0,2):
                lp = helper.read_plate(model_ocr, utils_rotate.deskew(crop_img, cc, ct))
                print(lp)
                if lp != "unknown":
                    list_read_plates.add(lp)
                    cv2.putText(frame, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    flag = 1
                    push_img(img=frame)
                    break
                if flag == 1:
                    break
              

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    idx += 1
    





