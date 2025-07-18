# backend/main.py
# 这是一个功能强大的后端API，使用了FastAPI框架。
# 它负责接收前端上传的图片/视频，调用YOLOv8模型进行分析，并返回结构化的JSON结果。

import asyncio
import cv2
import io
import numpy as np
import os
import tempfile
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
from typing import List, Dict
from ultralytics import YOLO
import uvicorn

# --- 应用初始化 ---
app = FastAPI(
    title="智能安全检测系统 API",
    description="提供基于YOLOv8的安全行为检测服务，支持图片、视频和实时流。",
    version="2.0.0"
)

# --- 中间件配置 (CORS) ---
# 允许所有来源的跨域请求，方便本地开发。
# 在生产环境中，应将其替换为您的前端域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 全局变量和配置 ---
try:
    # 加载您的YOLOv8模型
    MODEL = YOLO('models/best.pt')
except Exception as e:
    print(f"错误：无法加载模型 'models/best.pt'。请确保模型文件存在。 {e}")
    MODEL = None

# 类别名称到中文的映射
CLASS_NAMES_CN = {
    'person': '人员',
    'helmet': '安全帽',
    'safety_vest': '安全服',
    'no_helmet': '未戴安全帽',
    'no_vest': '未穿安全服',
    'head': '头部',
    'blur_head': '模糊头部',
    'blur_clothes': '模糊衣物',
    'looking_around': '东张西望'
}

# --- 核心处理函数 ---
def analyze_frame(image_np: np.ndarray) -> Dict:
    """对单个图像帧进行分析并返回结果"""
    if MODEL is None:
        raise RuntimeError("模型未成功加载，无法进行检测。")
        
    results = MODEL(image_np)
    
    detections = []
    is_danger = False
    look_around = False
    
    # 收集所有检测到的类别
    detected_classes = []
    
    if results and results[0].boxes:
        for box in results[0].boxes:
            try:
                cls_id = int(box.cls[0])
                class_name = MODEL.names[cls_id]
                conf = float(box.conf[0])
                
                # 过滤低置信度的结果
                if conf < 0.5:
                    continue
                
                class_name_cn = CLASS_NAMES_CN.get(class_name, class_name)
                detected_classes.append(class_name)
                
                detections.append({
                    'class': class_name_cn,
                    'confidence': conf,
                    'bbox': [round(coord) for coord in box.xyxy[0].tolist()]
                })
            except (IndexError, KeyError) as e:
                print(f"处理检测框时出错: {e}")
                continue


    # 危险判断逻辑
    if 'no_helmet' in detected_classes or 'no_vest' in detected_classes:
        is_danger = True
    
    # 东张西望判断
    if 'looking_around' in detected_classes:
        look_around = True
        
    return {
        "detections": detections,
        "is_danger": is_danger,
        "look_around": look_around
    }

# --- API 端点 (Endpoints) ---

@app.get("/", summary="API根节点")
async def root():
    """提供API的基本信息和健康状态。"""
    return {
        "message": "欢迎使用智能安全检测系统API",
        "model_loaded": MODEL is not None,
        "version": app.version
    }

@app.post("/detect", summary="检测图片或视频文件")
async def detect_file(file: UploadFile = File(...)):
    """
    接收一个文件（图片或视频），进行安全检测。
    - **图片**: 直接返回检测结果。
    - **视频**: (此简化版本将处理第一帧) 返回第一帧的检测结果。
    """
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024: # 50MB
        raise HTTPException(status_code=413, detail="文件大小超过50MB限制")

    file_ext = Path(file.filename).suffix.lower()
    
    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        
        result_data = analyze_frame(image_np)
        
        return {
            "type": "image" if file_ext in ['.jpg', '.jpeg', '.png'] else "video",
            **result_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """通过WebSocket接收视频帧进行实时检测。"""
    await websocket.accept()
    try:
        while True:
            bytes_data = await websocket.receive_bytes()
            image_np = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result_data = analyze_frame(img_rgb)
                await websocket.send_json({
                    **result_data,
                    "timestamp": datetime.now().isoformat()
                })
    except Exception as e:
        print(f"WebSocket 连接中断: {e}")
    finally:
        await websocket.close()

# --- 运行服务器 ---
if __name__ == "__main__":
    # 使用 uvicorn 启动服务器, host="0.0.0.0" 使其可以被局域网访问
    uvicorn.run(app, host="0.0.0.0", port=8000)
