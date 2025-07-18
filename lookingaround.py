import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
import time

# 初始化 MediaPipe 姿态模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7)

# 参数配置
WINDOW_SIZE = 25          # 增大窗口（更看重长期趋势）
VARIANCE_THRESHOLD = 80.0 # 调高方差阈值（允许更大波动）
ALERT_DURATION = 0.5      # 缩短报警显示时间
SAVE_ALERT_FRAMES = True  # 是否保存报警截图
save_dir = 'alert_frames'
os.makedirs(save_dir, exist_ok=True)

# 初始化数据结构
yaw_history = deque(maxlen=WINDOW_SIZE)  # 滑动窗口记录Yaw角度
alert_active = False                     # 当前是否处于报警状态
alert_start_time = 0                     # 报警开始时间

# 读取视频
video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

def calculate_yaw(nose, left_shoulder, right_shoulder, frame_width, frame_height):
    """计算头部偏航角（改进版，增加图像尺寸归一化）"""
    # 将MediaPipe的归一化坐标转换为像素坐标
    nose = np.array([nose.x * frame_width, nose.y * frame_height])
    left_sh = np.array([left_shoulder.x * frame_width, left_shoulder.y * frame_height])
    right_sh = np.array([right_shoulder.x * frame_width, right_shoulder.y * frame_height])
    
    shoulder_mid = (left_sh + right_sh) / 2
    dx = nose[0] - shoulder_mid[0]
    dy = nose[1] - shoulder_mid[1]
    return np.degrees(np.arctan2(dy, dx))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    frame_height, frame_width = frame.shape[:2]
    
    # 转换为RGB格式并进行姿态检测
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    label = "Normal"
    color = (0, 255, 0)  # 默认绿色
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # 计算当前帧的Yaw角度（使用鼻尖和双肩关键点）
        yaw = calculate_yaw(lm[0], lm[11], lm[12], frame_width, frame_height)
        yaw_history.append(yaw)
        
        # 显示实时角度（左上角）
        cv2.putText(frame, f"Yaw: {yaw:.1f}°", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 当窗口填满时开始检测
        if len(yaw_history) == WINDOW_SIZE:
            yaw_variance = np.var(yaw_history)
            
            # 显示方差值（调试用）
            cv2.putText(frame, f"Variance: {yaw_variance:.1f}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
            
            # 方差超过阈值触发报警
            if yaw_variance > VARIANCE_THRESHOLD:
                if not alert_active:
                    alert_active = True
                    alert_start_time = time.time()
                    if SAVE_ALERT_FRAMES:
                        cv2.imwrite(f"{save_dir}/alert_{frame_idx}.jpg", frame)
            else:
                alert_active = False
        
        # 处理报警状态显示
        if alert_active:
            label = "LOOKING AROUND"
            color = (0, 0, 255)  # 红色
            
            # 自动结束报警
            if time.time() - alert_start_time > ALERT_DURATION:
                alert_active = False
    else:
        cv2.putText(frame, "No person detected", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
    
    # 在画面中央显示状态标签
    cv2.putText(frame, label, 
               (frame_width//2 - 100, frame_height//2),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # 可视化关键点（调试用）
    mp.solutions.drawing_utils.draw_landmarks(
        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow("Head Pose Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()