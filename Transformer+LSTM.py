import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --------- 混合LSTM+Transformer模型定义 ---------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class LSTMTransformerPredictor(nn.Module):
    def __init__(self, input_dim=2, lstm_hidden=64, d_model=128, nhead=4, num_layers=2, pred_len=12):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.input_proj = nn.Linear(lstm_hidden * 2, d_model)  # 双向LSTM输出乘2
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, pred_len * 2)
        self.pred_len = pred_len

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  
        x = self.input_proj(lstm_out)  
        x = self.pos_encoder(x)         # 加位置编码
        x = x.permute(1, 0, 2)          
        transformer_out = self.transformer(x)
        pooled = transformer_out.mean(dim=0)  
        out = self.output_proj(pooled)        
        return out.view(-1, self.pred_len, 2)  

# --------- 初始化模型及其他工具 ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMTransformerPredictor().to(device)
model.eval()  # 预测时设为eval

# 加载自训练YOLOv8模型
yolo_model = YOLO("train14/weights/best.pt")

# 初始化DeepSORT多目标跟踪
tracker = DeepSort(max_age=30)

# 危险区域示例
danger_zone = (500, 200, 700, 400)  # (x1, y1, x2, y2)

# 用于存储各目标历史轨迹
trajectories = {}

# --------- 视频处理主循环 ---------
cap = cv2.VideoCapture("crowds_zara02.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8目标检测
    results = yolo_model(frame)[0]
    detections = []
    for det in results.boxes:
        cls = int(det.cls)
        if yolo_model.names[cls] == "person":
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = det.conf.item()
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    # DeepSORT多目标跟踪
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # 处理每个track
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        bbox = track.to_tlbr() 

        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        # 更新轨迹缓存，只保留最近8帧
        if tid not in trajectories:
            trajectories[tid] = []
        trajectories[tid].append([cx, cy])
        if len(trajectories[tid]) > 8:
            trajectories[tid] = trajectories[tid][-8:]

        # 轨迹预测
        if len(trajectories[tid]) == 8:
            traj_np = np.array(trajectories[tid], dtype=np.float32)
            traj_tensor = torch.tensor(traj_np, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 8, 2]

            with torch.no_grad():
                pred_future = model(traj_tensor).cpu().numpy()[0]  # [12, 2]

            # 判断未来轨迹是否进入危险区域
            danger = any(
                (danger_zone[0] <= x <= danger_zone[2]) and (danger_zone[1] <= y <= danger_zone[3])
                for x, y in pred_future
            )

            # 画轨迹箭头（可选）
            for i in range(1, len(pred_future)):
                pt1 = tuple(pred_future[i - 1].astype(int))
                pt2 = tuple(pred_future[i].astype(int))
                cv2.arrowedLine(frame, pt1, pt2, (0, 0, 255) if danger else (0, 255, 0), 2, tipLength=0.3)

            # 根据危险状态画框和标注
            color = (0, 0, 255) if danger else (0, 255, 0)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            if danger:
                cv2.putText(frame, "DANGER", (int(cx), int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            # 轨迹不够时正常画框
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

    # 画危险区域（示例）
    cv2.rectangle(frame, (danger_zone[0], danger_zone[1]), (danger_zone[2], danger_zone[3]), (0, 0, 255), 2)
    cv2.putText(frame, "DANGER ZONE", (danger_zone[0], danger_zone[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Trajectory Prediction", frame)
    if cv2.waitKey(1) == 27:  # ESC退出
        break

cap.release()
cv2.destroyAllWindows()