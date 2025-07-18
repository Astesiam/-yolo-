import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math

# ---------- Transformer 模型定义 ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=128, nhead=4, num_layers=2, pred_len=12):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, pred_len * input_dim)
        self.pred_len = pred_len
        self.input_dim = input_dim

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        out = self.output_proj(x)
        return out.view(-1, self.pred_len, self.input_dim)

# ---------- 初始化模型与工具 ----------
model = YOLO("train14/weights/best.pt")
tracker = DeepSort(max_age=30)
transformer_model = TrajectoryTransformer()
transformer_model.eval()  # 推理模式

cap = cv2.VideoCapture("crowds_zara02.mp4")  # 替换为你的路径
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
trajectories = {}  # {track_id: [(x, y), ...]}

# ---------- 设置危险区域为视频中央区域 ----------
margin_x, margin_y = 0.25, 0.25
x1 = int(frame_width * 0.1)
y1 = int(frame_height * 0.5)
x2 = int(frame_width * 0.6)
y2 = int(frame_height * 0.8)
danger_zone = ((x1, y1), (x2, y2))

# ---------- 视频处理主循环 ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    detections = []
    for r in results.boxes.data:
        x1, y1, x2, y2, conf, cls = r
        if int(cls) == 0:  # person 类
            detections.append(([x1.item(), y1.item(), x2.item() - x1.item(), y2.item() - y1.item()], conf.item(), "person"))

    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        x1, y1, x2, y2 = track.to_ltrb()
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        if tid not in trajectories:
            trajectories[tid] = []
        trajectories[tid].append((cx, cy))
        if len(trajectories[tid]) > 8:
            trajectories[tid] = trajectories[tid][-8:]

        color = (0, 255, 0)  # 默认绿色
        danger_flag = False

        # ---------- 轨迹预测 ----------
        if len(trajectories[tid]) == 8:
            traj_input = torch.tensor(trajectories[tid], dtype=torch.float32).unsqueeze(0)
            pred = transformer_model(traj_input).detach().numpy()[0]
            for i in range(len(pred)):
                px, py = int(pred[i][0]), int(pred[i][1])
                if i > 0:
                    p_prev = (int(pred[i - 1][0]), int(pred[i - 1][1]))
                    p_curr = (px, py)
                    cv2.arrowedLine(frame, p_prev, p_curr, (255, 0, 0), 2, tipLength=0.3)
                if danger_zone[0][0] < px < danger_zone[1][0] and danger_zone[0][1] < py < danger_zone[1][1]:
                    danger_flag = True

            if danger_flag:
                color = (0, 0, 255)
                cv2.putText(frame, f"⚠ Track {tid} heading to danger!", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ---------- 绘制目标框 ----------
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"ID:{tid}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---------- 绘制危险区域 ----------
    cv2.rectangle(frame, danger_zone[0], danger_zone[1], (0, 0, 255), 2)
    cv2.imshow("Intent Recognition", frame)
    if cv2.waitKey(1) == 27:  # Esc 键退出，不再循环
        break

cap.release()
cv2.destroyAllWindows()