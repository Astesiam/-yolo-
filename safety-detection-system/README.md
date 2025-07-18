# 智能安全检测系统 (Safety Detection System)

这是一个基于深度学习（YOLOv8）和现代Web技术（Vue3, FastAPI）的全栈项目，旨在提供一个实时的作业人员安全行为监测平台。

## ✨ 主要功能

- **多种输入方式**: 支持图片、视频文件上传和实时摄像头捕捉。
- **实时检测**: 后端采用 YOLOv8 模型，能够高效识别多种安全相关的目标，如：
  - `人员`
  - `安全帽` / `未戴安全帽`
  - `安全服` / `未穿安全服`
  - `东张西望` 等异常行为
- **现代化前端**:
  - 使用 Vue 3 (Composition API) 构建，组件化、易维护。
  - 响应式设计，适配不同尺寸的屏幕。
  - 清晰的数据可视化仪表盘，直观展示检测结果。
- **高性能后端**:
  - 基于 FastAPI 构建，性能卓越，支持异步处理。
  - 支持通过 WebSocket 进行低延迟的实时视频流分析。
- **清晰的项目结构**: 前后端分离，职责明确，便于团队协作和独立部署。

## 🚀 技术栈

- **前端**: Vue 3, Vite, Axios, Remixicon
- **后端**: Python 3, FastAPI, Uvicorn, YOLOv8 (Ultralytics), OpenCV
- **核心算法**: YOLOv8 目标检测

## 🔧 环境搭建与运行

### 1. 克隆项目

```bash
git clone <your-repository-url>
cd safety-detection-system
```

### 2. 后端设置

**首先，将您训练好的 `best.pt` 模型文件放入 `backend/models/` 目录下。**

然后，进入后端目录，创建虚拟环境并安装依赖：

```bash
cd backend

# 创建Python虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate


# 安装依赖
pip install -r requirements.txt
```

### 3. 前端设置

打开一个新的终端，进入前端目录并安装依赖：

```bash
cd frontend
npm install
```

### 4. 运行项目

你需要**同时运行**前端和后端服务。

- **启动后端服务** (在 `backend` 目录下):
  ```bash
  uvicorn main:app --reload
  ```
  服务器将在 `http://localhost:8000` 启动。

- **启动前端开发服务** (在 `frontend` 目录下):
  ```bash
  npm run dev
  ```
  前端应用将在 `http://localhost:5173` (或另一个可用端口) 启动。

现在，在浏览器中打开前端应用的地址，即可开始使用！

## 📈 未来展望

- **用户认证系统**: 增加登录和注册功能，实现多用户管理。
- **历史记录与数据分析**: 将检测结果持久化到数据库，并提供历史查询和统计分析图表。
- **优化视频流处理**: 实现更高效的视频流分析，例如在后端直接处理 RTSP 流。
- **模型管理**: 在前端提供上传和切换不同版本检测模型的功能。
- **部署**: 提供 Docker 容器化部署方案，简化生产环境的部署流程。
