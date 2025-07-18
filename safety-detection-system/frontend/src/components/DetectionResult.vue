<!-- src/components/DetectionResult.vue -->
<!--
  这个组件负责展示后端返回的检测结果。
  - 它通过 props 接收 `results` 对象。
  - 当没有结果时，显示一个空状态提示。
  - 当有结果时，它会：
    1. 显示危险/异常/正常的总体状态警告。
    2. 遍历并显示每个检测到的对象（类别、置信度）。
    3. 如果是视频，会显示视频的摘要信息和时间轴。
    4. 展示总体的统计数据。
  - 它还提供了“导出报告”和“清除结果”的功能，通过 `emit` 通知父组件执行。
-->
<template>
  <div class="result-component">
    <div class="result-header">
      <h3 class="component-title">
        <i class="ri-file-list-3-line"></i>
        检测结果
      </h3>
      <div class="header-actions">
        <button v-if="hasResults" class="btn-icon" @click="$emit('export', results)" title="导出报告">
          <i class="ri-download-2-line"></i>
        </button>
        <button v-if="hasResults" class="btn-icon" @click="$emit('clear')" title="清除结果">
          <i class="ri-delete-bin-line"></i>
        </button>
      </div>
    </div>

    <div v-if="!hasResults" class="empty-state">
      <i class="ri-inbox-line"></i>
      <p>暂无检测结果</p>
      <p class="empty-hint">上传文件或开启摄像头以开始</p>
    </div>

    <div v-else class="result-content">
      <!-- 状态告警 -->
      <div class="alerts-container">
        <div v-if="results.is_danger" class="alert alert-danger">
          <i class="ri-alert-line"></i>
          <strong>危险警告:</strong> 检测到安全隐患！
        </div>
        <div v-if="results.look_around" class="alert alert-warning">
          <i class="ri-eye-line"></i>
          <strong>行为异常:</strong> 检测到东张西望行为。
        </div>
        <div v-if="!results.is_danger && !results.look_around" class="alert alert-success">
          <i class="ri-shield-check-line"></i>
          <strong>状态正常:</strong> 未检测到明显隐患。
        </div>
      </div>

      <!-- 统计信息 -->
      <div class="statistics">
        <div class="stat-item">
          <p class="stat-value">{{ totalDetections }}</p>
          <p class="stat-label">检测对象</p>
        </div>
        <div class="stat-item">
          <p class="stat-value">{{ avgConfidence }}%</p>
          <p class="stat-label">平均置信度</p>
        </div>
        <div class="stat-item">
          <p class="stat-value">{{ personCount }}</p>
          <p class="stat-label">检测人数</p>
        </div>
      </div>

      <!-- 详情列表 -->
      <div class="details-list">
        <div v-for="(detection, index) in detections" :key="index" class="detection-item" :class="getDetectionClass(detection.class)">
          <div class="detection-info">
            <i :class="getIconForClass(detection.class)"></i>
            <span class="detection-label">{{ detection.class }}</span>
          </div>
          <div class="confidence-bar">
            <div class="confidence-fill" :style="{ width: (detection.confidence * 100) + '%' }"></div>
          </div>
          <span class="confidence-text">{{ (detection.confidence * 100).toFixed(1) }}%</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';

export default {
  name: 'DetectionResult',
  props: {
    results: { type: Object, default: null },
  },
  emits: ['clear', 'export'],
  setup(props) {
    const hasResults = computed(() => props.results && props.results.detections);

    const detections = computed(() => {
      if (!hasResults.value) return [];
      // 简单处理，对于视频，可以显示第一帧或最后一帧的结果
      return props.results.detections || [];
    });

    const totalDetections = computed(() => detections.value.length);

    const avgConfidence = computed(() => {
      if (totalDetections.value === 0) return 0;
      const sum = detections.value.reduce((acc, d) => acc + d.confidence, 0);
      return (sum / totalDetections.value * 100).toFixed(1);
    });

    const personCount = computed(() => {
        return detections.value.filter(d => d.class === '人员').length;
    });

    const getIconForClass = (className) => {
      const iconMap = {
        '人员': 'ri-user-line', '安全帽': 'ri-shield-check-line', '安全服': 'ri-t-shirt-line',
        '未戴安全帽': 'ri-alert-line', '未穿安全服': 'ri-alert-line', '东张西望': 'ri-eye-line',
        '头部': 'ri-user-3-line', '模糊头部': 'ri-blur-off-line', '模糊衣物': 'ri-blur-off-line',
      };
      return iconMap[className] || 'ri-checkbox-circle-line';
    };

    const getDetectionClass = (className) => {
      if (['未戴安全帽', '未穿安全服'].includes(className)) return 'danger';
      if (['东张西望', '模糊头部', '模糊衣物'].includes(className)) return 'warning';
      return '';
    };

    return {
      hasResults, detections, totalDetections, avgConfidence, personCount,
      getIconForClass, getDetectionClass,
    };
  },
};
</script>

<style scoped>
/* 组件特定样式 */
.result-component { display: flex; flex-direction: column; height: 100%; }
.result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.component-title { margin: 0; display: flex; align-items: center; gap: 10px; }
.component-title i { color: var(--primary-color); }
.header-actions { display: flex; gap: 10px; }
.btn-icon { background: none; border: none; font-size: 1.2rem; color: var(--text-light); cursor: pointer; transition: color 0.3s; }
.btn-icon:hover { color: var(--primary-color); }

.empty-state {
  flex-grow: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  text-align: center; color: #999;
}
.empty-state i { font-size: 4rem; margin-bottom: 20px; opacity: 0.5; }
.empty-hint { font-size: 0.9rem; color: #bbb; }

.result-content { overflow-y: auto; }
.alerts-container { display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }
.alert {
  padding: 12px 15px; border-radius: 10px; display: flex;
  align-items: center; gap: 10px; font-weight: 500;
}
.alert i { font-size: 1.2rem; }
.alert-danger { background: #fee; color: #c00; }
.alert-warning { background: #fff3cd; color: #856404; }
.alert-success { background: #d4edda; color: #155724; }

.statistics {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 15px; padding: 20px; background: #f8f9fa;
  border-radius: 10px; margin-bottom: 20px; text-align: center;
}
.stat-value { font-size: 1.8rem; font-weight: bold; margin: 0 0 5px 0; }
.stat-label { font-size: 0.9rem; color: var(--text-light); margin: 0; }

.details-list { display: flex; flex-direction: column; gap: 10px; }
.detection-item {
  background: #f8f9fa; padding: 12px 15px; border-radius: 10px;
  display: grid; grid-template-columns: 1fr auto auto;
  align-items: center; gap: 15px; transition: all 0.3s ease;
  border-left: 4px solid transparent;
}
.detection-item.danger { border-color: var(--danger-color); background: #fff2f2; }
.detection-item.warning { border-color: var(--warning-color); background: #fff9e8; }
.detection-item:hover { transform: translateX(5px); box-shadow: 0 5px 10px rgba(0,0,0,0.05); }

.detection-info { display: flex; align-items: center; gap: 10px; font-weight: 500; }
.detection-info i { font-size: 1.2rem; color: var(--primary-color); }
.detection-item.danger .detection-info i { color: var(--danger-color); }
.detection-item.warning .detection-info i { color: var(--warning-color); }

.confidence-bar { width: 100px; height: 6px; background: #e9ecef; border-radius: 3px; overflow: hidden; }
.confidence-fill { height: 100%; background: var(--primary-color); }
.confidence-text { font-size: 0.9rem; font-weight: 500; color: var(--text-light); min-width: 50px; text-align: right; }
</style>
