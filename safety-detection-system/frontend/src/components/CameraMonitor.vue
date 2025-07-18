<!-- src/components/CameraMonitor.vue -->
<!--
  这个组件负责处理实时摄像头视频流。
  - 它会请求浏览器摄像头权限并显示视频流。
  - 用户可以“开始/暂停检测”、“拍照检测”和“关闭摄像头”。
  - 当进行检测时，它会从视频流中捕获一帧图像。
  - 然后将该帧图像发送到后端的 /detect 接口。
  - 成功或失败时，通过 `emit` 向父组件（App.vue）发送事件。
-->
<template>
  <div class="camera-component">
    <div class="status-indicator">
      <span class="status-dot" :class="isCameraActive ? 'active' : 'inactive'"></span>
      <span>{{ statusText }}</span>
    </div>

    <div class="camera-container">
      <video ref="videoElementRef" v-show="isCameraActive" autoplay playsinline></video>
      <canvas ref="canvasElementRef" style="display: none;"></canvas>
      <div v-if="!isCameraActive" class="camera-placeholder">
        <i class="ri-camera-off-line"></i>
        <p>摄像头未开启</p>
      </div>
    </div>

    <div class="camera-controls">
      <button v-if="!isCameraActive" class="btn btn-primary" @click="startCamera">
        <i class="ri-camera-line"></i> 开启摄像头
      </button>
      <template v-else>
        <button class="btn btn-primary" @click="toggleDetection" :disabled="isCapturing">
          <span v-if="isCapturing" class="loading-spinner"></span>
          <i v-else :class="isDetecting ? 'ri-pause-line' : 'ri-play-line'"></i>
          {{ isDetecting ? '暂停检测' : '开始检测' }}
        </button>
        <button class="btn btn-secondary" @click="captureAndDetect" :disabled="isCapturing || isDetecting">
          <i class="ri-camera-3-line"></i> 拍照检测
        </button>
        <button class="btn btn-danger" @click="stopCamera">
          <i class="ri-stop-line"></i> 关闭摄像头
        </button>
      </template>
    </div>
  </div>
</template>

<script>
import { ref, computed, onBeforeUnmount } from 'vue';
import axios from 'axios';

export default {
  name: 'CameraMonitor',
  props: {
    apiUrl: { type: String, required: true },
  },
  emits: ['detection-complete', 'error'],
  setup(props, { emit }) {
    const videoElementRef = ref(null);
    const canvasElementRef = ref(null);
    const isCameraActive = ref(false);
    const isDetecting = ref(false);
    const isCapturing = ref(false);
    const detectionInterval = 2000; // 2秒
    let stream = null;
    let detectionTimer = null;

    const statusText = computed(() => {
      if (!isCameraActive.value) return '摄像头未连接';
      if (isDetecting.value) return `实时检测中... (${detectionInterval / 1000}s/次)`;
      return '摄像头已连接，待机中';
    });

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
        });
        if (videoElementRef.value) {
          videoElementRef.value.srcObject = stream;
          isCameraActive.value = true;
        }
      } catch (err) {
        emit('error', '无法访问摄像头，请检查权限。');
      }
    };

    const stopCamera = () => {
      if (detectionTimer) clearInterval(detectionTimer);
      if (stream) stream.getTracks().forEach((track) => track.stop());
      isCameraActive.value = false;
      isDetecting.value = false;
      stream = null;
    };

    const captureAndDetect = async () => {
      if (!videoElementRef.value || isCapturing.value) return;
      isCapturing.value = true;

      const video = videoElementRef.value;
      const canvas = canvasElementRef.value;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);

      try {
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg'));
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');
        const response = await axios.post(`${props.apiUrl}/detect`, formData);
        emit('detection-complete', response.data);
      } catch (err) {
        const message = err.response?.data?.detail || '检测失败';
        emit('error', message);
      } finally {
        isCapturing.value = false;
      }
    };

    const toggleDetection = () => {
      isDetecting.value = !isDetecting.value;
      if (isDetecting.value) {
        detectionTimer = setInterval(captureAndDetect, detectionInterval);
      } else {
        if (detectionTimer) clearInterval(detectionTimer);
      }
    };

    onBeforeUnmount(stopCamera);

    return {
      videoElementRef,
      canvasElementRef,
      isCameraActive,
      isDetecting,
      isCapturing,
      statusText,
      startCamera,
      stopCamera,
      toggleDetection,
      captureAndDetect,
    };
  },
};
</script>

<style scoped>
/* 样式与 App.vue 中的通用样式保持一致，并添加组件特定样式 */
.btn {
  padding: 12px 24px; border: none; border-radius: 10px; font-size: 1rem;
  font-weight: 500; cursor: pointer; transition: all 0.3s ease;
  display: inline-flex; align-items: center; gap: 8px;
}
.btn-primary { background: var(--primary-color); color: white; }
.btn-primary:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
.btn-primary:disabled, .btn-secondary:disabled { opacity: 0.6; cursor: not-allowed; }
.btn-secondary { background: #f0f0f0; color: var(--text-light); }
.btn-secondary:hover:not(:disabled) { background: #e0e0e0; }
.btn-danger { background: var(--danger-color); color: white; }
.btn-danger:hover { background: #e13a48; }

.status-indicator {
  display: flex; align-items: center; gap: 10px; padding: 10px 15px;
  background: #f8f9fa; border-radius: 10px; margin-bottom: 20px;
}
.status-dot {
  width: 10px; height: 10px; border-radius: 50%;
  animation: pulse 2s ease-in-out infinite;
}
.status-dot.active { background: var(--success-color); }
.status-dot.inactive { background: var(--danger-color); animation: none; }

.camera-container {
  position: relative; width: 100%; background: #000;
  border-radius: 15px; overflow: hidden; min-height: 300px;
  display: flex; align-items: center; justify-content: center;
}
video { width: 100%; height: auto; display: block; }
.camera-placeholder { text-align: center; color: #666; }
.camera-placeholder i { font-size: 3rem; margin-bottom: 10px; }

.camera-controls {
  display: flex; gap: 10px; margin-top: 20px;
  justify-content: center; flex-wrap: wrap;
}
.loading-spinner {
  width: 16px; height: 16px; border: 2px solid #fff; border-radius: 50%;
  border-top-color: transparent; animation: spin 0.8s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
</style>
