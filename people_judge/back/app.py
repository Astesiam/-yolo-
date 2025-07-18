from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境用 *, 生产请改为指定前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('best.pt')
# 英文类别到中文类别映射
name_map = {
    "person": "人员",
    "helmet": "安全帽",
    "self_clothes": "自身服装",
    "safety_clothes": "安全服",
    "head": "没帽子",
    "blur_head": "模糊帽子",
    "blur_clothes": "模糊服装",
}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    results = model(img)

    detections = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        for box, score, cls in zip(boxes, scores, classes):
            label_en = model.names[int(cls)] if hasattr(model, 'names') else str(int(cls))
            label_cn = name_map.get(label_en, label_en)  # 有映射用中文，否则用英文
            detections.append({
                "box": box.tolist(),
                "confidence": float(score),
                "class": int(cls),
                "name_en": label_en,
                "name": label_cn  # 返回给前端的是中文名
            })

    CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值，可以调整

    has_person = any(d['name_en'] == 'person' and d['confidence'] >= CONFIDENCE_THRESHOLD for d in detections)
    has_helmet = any(d['name_en'] == 'helmet' and d['confidence'] >= CONFIDENCE_THRESHOLD for d in detections)
    has_clothes = any(d['name_en'] == 'safety_clothes' and d['confidence'] >= CONFIDENCE_THRESHOLD for d in detections)

    is_danger = has_person and (not has_helmet or not has_clothes)

    return JSONResponse(content={
        "detections": detections,
        "is_danger": is_danger
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)