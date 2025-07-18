<!-- App.vue -->
<!-- 
  这是应用的主组件，扮演着“指挥中心”的角色。
  它负责：
  1. 整体页面布局（顶部统计、左右功能区）。
  2. 管理核心状态，如 `currentResults`。
  3. 在 `ImageVideoUpload` 和 `CameraMonitor` 组件之间切换。
  4. 监听子组件发出的事件（如 `@detection-complete`），并更新状态。
  5. 将更新后的状态通过 props 传递给 `DetectionResult` 组件。
-->
<template>
  <div id="app-container">
    <header class="app-header">
      <div class="container header-content">
        <div class="logo">
          <i class="ri-shield-check-line"></i>
          <h1>智能安全检测系统</h1>
        </div>
        <div class="user-info">
          <span class="status-indicator online"></span>
          <span>{{ currentTime }}</span>
        </div>
      </div>
    </header>

    <main class="app-main">
      <div class="container">
        <!-- 主功能区 -->
        <div class="main-grid">
          <!-- 左侧输入区 -->
          <div class="input-section card">
            <div class="input-tabs">
              <button
                class="tab-btn"
                :class="{ active: activeInput === 'upload' }"
                @click="activeInput = 'upload'"
              >
                <i class="ri-upload-cloud-2-line"></i>
                文件上传
              </button>
              <button
                class="tab-btn"
                :class="{ active: activeInput === 'camera' }"
                @click="activeInput = 'camera'"
              >
                <i class="ri-camera-line"></i>
                摄像头监测
              </button>
            </div>

            <!-- 组件容器 -->
            <transition name="fade" mode="out-in">
              <ImageVideoUpload
                v-if="activeInput === 'upload'"
                :api-url="apiUrl"
                @detection-complete="handleDetectionResult"
                @error="handleError"
              />
              <CameraMonitor
                v-else
                :api-url="apiUrl"
                :use-web-socket="false"
                @detection-complete="handleDetectionResult"
                @error="handleError"
              />
            </transition>
          </div>

          <!-- 右侧结果区 -->
          <div class="result-section card">
            <DetectionResult
              :results="currentResults"
              @clear="clearResults"
              @export="exportResults"
            />
          </div>
        </div>
      </div>
    </main>

    <!-- 全局通知 -->
    <transition-group name="notification" tag="div" class="notifications">
      <div
        v-for="notification in notifications"
        :key="notification.id"
        class="notification"
        :class="notification.type"
      >
        <i :class="getNotificationIcon(notification.type)"></i>
        <div class="notification-content">
          <p class="notification-title">{{ notification.title }}</p>
          <p class="notification-message">{{ notification.message }}</p>
        </div>
        <button class="notification-close" @click="removeNotification(notification.id)">
          <i class="ri-close-line"></i>
        </button>
      </div>
    </transition-group>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted } from 'vue';
import ImageVideoUpload from './components/ImageVideoUpload.vue';
import CameraMonitor from './components/CameraMonitor.vue';
import DetectionResult from './components/DetectionResult.vue';

export default {
  name: 'App',
  components: {
    ImageVideoUpload,
    CameraMonitor,
    DetectionResult,
  },
  setup() {
    // 核心状态
    const activeInput = ref('upload');
    const currentResults = ref(null);
    const apiUrl = ref('http://localhost:8000'); // 后端API地址
    const currentTime = ref('');
    const notifications = ref([]);
    let notificationId = 0;

    // --- 方法 ---

    // 处理子组件传递的检测结果
    const handleDetectionResult = (result) => {
      console.log('Received detection result:', result);
      currentResults.value = result;
      addNotification({
        type: 'success',
        title: '检测完成',
        message: `检测到 ${result.detections?.length || 0} 个对象。`,
      });
    };

    // 处理子组件传递的错误信息
    const handleError = (errorMessage) => {
      console.error('Received error:', errorMessage);
      addNotification({
        type: 'error',
        title: '发生错误',
        message: errorMessage,
      });
    };

    // 清除结果
    const clearResults = () => {
      currentResults.value = null;
    };

    // 导出结果
    const exportResults = (data) => {
      const jsonData = JSON.stringify(data, null, 2);
      const blob = new Blob([jsonData], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `detection-report-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      addNotification({
        type: 'info',
        title: '导出成功',
        message: '检测报告已保存到本地。',
      });
    };

    // --- 通知系统 ---

    const addNotification = (notification) => {
      const id = notificationId++;
      notifications.value.push({ id, ...notification });
      setTimeout(() => removeNotification(id), 5000);
    };

    const removeNotification = (id) => {
      const index = notifications.value.findIndex((n) => n.id === id);
      if (index > -1) {
        notifications.value.splice(index, 1);
      }
    };

    const getNotificationIcon = (type) => {
      const icons = {
        success: 'ri-checkbox-circle-line',
        error: 'ri-close-circle-line',
        info: 'ri-information-line',
      };
      return icons[type] || icons.info;
    };

    // --- 时间更新 ---
    const updateTime = () => {
      currentTime.value = new Date().toLocaleTimeString('zh-CN');
    };

    let timeInterval = null;
    onMounted(() => {
      updateTime();
      timeInterval = setInterval(updateTime, 1000);
    });

    onUnmounted(() => {
      if (timeInterval) clearInterval(timeInterval);
    });

    return {
      activeInput,
      currentResults,
      apiUrl,
      currentTime,
      notifications,
      handleDetectionResult,
      handleError,
      clearResults,
      exportResults,
      removeNotification,
      getNotificationIcon,
    };
  },
};
</script>

<style>
/* 全局和布局样式 */
:root {
  --primary-color: #667eea;
  --secondary-color: #764ba2;
  --danger-color: #ff4757;
  --warning-color: #ffa502;
  --success-color: #2ed573;
  --light-bg: #f5f6fa;
  --card-bg: #ffffff;
  --text-color: #333;
  --text-light: #666;
  --border-color: #e0e0e0;
  --card-shadow: 0 10px 30px rgba(0, 0, 0, 0.07);
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  background-color: var(--light-bg);
  color: var(--text-color);
  margin: 0;
}

#app-container {
  min-height: 100vh;
}

.container {
  max-width: 1600px;
  margin: 0 auto;
  padding: 0 20px;
}

.card {
  background: var(--card-bg);
  border-radius: 15px;
  padding: 25px;
  box-shadow: var(--card-shadow);
}

/* 头部 */
.app-header {
  background: var(--card-bg);
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  position: sticky;
  top: 0;
  z-index: 100;
  height: 70px;
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 100%;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
}
.logo i {
  font-size: 2rem;
  color: var(--primary-color);
}
.logo h1 {
  font-size: 1.5rem;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 15px;
  color: var(--text-light);
}
.status-indicator.online {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--success-color);
  animation: pulse 2s infinite;
}

/* 主内容区 */
.app-main {
  padding: 30px 0;
}

.main-grid {
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: 30px;
}

/* 输入区 */
.input-tabs {
  display: flex;
  gap: 10px;
  margin-bottom: 25px;
}
.tab-btn {
  flex: 1;
  padding: 12px 20px;
  border: 1px solid var(--border-color);
  background: transparent;
  color: var(--text-light);
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 1rem;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
.tab-btn:hover {
  background: #f8f9fa;
  color: var(--primary-color);
}
.tab-btn.active {
  background: var(--primary-color);
  border-color: var(--primary-color);
  color: white;
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
}

/* 通知 */
.notifications {
  position: fixed;
  top: 90px;
  right: 20px;
  z-index: 200;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.notification {
  background: white;
  border-radius: 10px;
  padding: 15px 20px;
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
  display: flex;
  align-items: flex-start;
  gap: 15px;
  min-width: 300px;
  max-width: 400px;
  position: relative;
  border-left: 4px solid;
}
.notification.success { border-color: var(--success-color); }
.notification.success i { color: var(--success-color); }
.notification.error { border-color: var(--danger-color); }
.notification.error i { color: var(--danger-color); }
.notification.info { border-color: var(--primary-color); }
.notification.info i { color: var(--primary-color); }

.notification i { font-size: 1.5rem; }
.notification-title { font-weight: 600; margin-bottom: 3px; }
.notification-message { font-size: 0.9rem; color: var(--text-light); }
.notification-close {
  position: absolute;
  top: 10px;
  right: 10px;
  background: none;
  border: none;
  color: #999;
  cursor: pointer;
}

/* 动画 */
.fade-enter-active, .fade-leave-active { transition: opacity 0.3s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
.notification-enter-active, .notification-leave-active { transition: all 0.3s ease; }
.notification-enter-from, .notification-leave-to { transform: translateX(100%); opacity: 0; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

/* 响应式 */
@media (max-width: 1024px) {
  .main-grid {
    grid-template-columns: 1fr;
  }
}
</style>
