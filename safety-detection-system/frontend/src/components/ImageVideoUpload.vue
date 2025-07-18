<!-- src/components/ImageVideoUpload.vue -->
<!--
  这个组件负责处理图片和视频文件的上传。
  - 它包含一个拖拽区域，也支持点击选择文件。
  - 文件选中后，会显示预览。
  - 点击“开始检测”后，它会使用 axios 将文件发送到后端 /detect 接口。
  - 上传过程中会显示进度条。
  - 成功或失败时，通过 `emit` 向父组件（App.vue）发送事件。
-->
<template>
  <div class="upload-component">
    <div
      class="upload-area"
      :class="{ 'dragover': isDragging, 'has-file': selectedFile }"
      @click="triggerFileInput"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop.prevent="handleDrop"
    >
      <template v-if="!selectedFile">
        <i class="ri-upload-2-line upload-icon"></i>
        <p class="upload-text">拖拽文件到此处或点击上传</p>
        <p class="upload-hint">支持图片和视频 (最大50MB)</p>
      </template>
      <template v-else>
        <div class="preview-wrapper">
          <img v-if="isImage" :src="previewUrl" class="preview-content" alt="Image Preview" />
          <video v-else-if="isVideo" :src="previewUrl" controls class="preview-content"></video>
          <div class="file-info">
            <i :class="isImage ? 'ri-image-line' : 'ri-video-line'"></i>
            <span>{{ selectedFile.name }}</span>
            <span class="file-size">({{ formatFileSize(selectedFile.size) }})</span>
          </div>
        </div>
      </template>
    </div>

    <input
      ref="fileInputRef"
      type="file"
      class="hidden-input"
      @change="handleFileSelect"
      accept="image/*,video/*"
    />

    <div v-if="uploadProgress > 0 && isProcessing" class="progress-bar">
      <div class="progress-fill" :style="{ width: uploadProgress + '%' }"></div>
      <span class="progress-text">{{ uploadProgress }}%</span>
    </div>

    <div v-if="selectedFile" class="action-buttons">
      <button class="btn btn-primary" @click="uploadFile" :disabled="isProcessing">
        <span v-if="isProcessing" class="loading-spinner"></span>
        <span v-else><i class="ri-scan-line"></i> 开始检测</span>
      </button>
      <button class="btn btn-secondary" @click="clearFile" :disabled="isProcessing">
        <i class="ri-delete-bin-line"></i> 清除
      </button>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import axios from 'axios';

export default {
  name: 'ImageVideoUpload',
  props: {
    apiUrl: {
      type: String,
      required: true,
    },
  },
  emits: ['detection-complete', 'error'],
  setup(props, { emit }) {
    const selectedFile = ref(null);
    const previewUrl = ref(null);
    const isDragging = ref(false);
    const isProcessing = ref(false);
    const uploadProgress = ref(0);
    const fileInputRef = ref(null);

    const isImage = computed(() => selectedFile.value?.type.startsWith('image/'));
    const isVideo = computed(() => selectedFile.value?.type.startsWith('video/'));

    const formatFileSize = (bytes) => {
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };
    
    const triggerFileInput = () => {
        if (!selectedFile.value) {
            fileInputRef.value.click();
        }
    };

    const processFile = (file) => {
      if (!file) return;
      if (file.size > 50 * 1024 * 1024) {
        emit('error', '文件大小不能超过50MB');
        return;
      }
      selectedFile.value = file;
      if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
      previewUrl.value = URL.createObjectURL(file);
    };

    const handleFileSelect = (event) => processFile(event.target.files[0]);
    const handleDrop = (event) => {
      isDragging.value = false;
      processFile(event.dataTransfer.files[0]);
    };

    const clearFile = () => {
      selectedFile.value = null;
      if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
      previewUrl.value = null;
      uploadProgress.value = 0;
      if (fileInputRef.value) fileInputRef.value.value = '';
    };

    const uploadFile = async () => {
      if (!selectedFile.value) return;

      isProcessing.value = true;
      uploadProgress.value = 0;
      const formData = new FormData();
      formData.append('file', selectedFile.value);

      try {
        const response = await axios.post(`${props.apiUrl}/detect`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (e) => {
            if (e.total) {
              uploadProgress.value = Math.round((e.loaded * 100) / e.total);
            }
          },
        });
        emit('detection-complete', response.data);
      } catch (err) {
        const message = err.response?.data?.detail || '检测失败，请检查网络或后端服务';
        emit('error', message);
      } finally {
        isProcessing.value = false;
        // Keep the file for viewing results, clear it manually
        // clearFile(); 
      }
    };

    return {
      selectedFile,
      previewUrl,
      isDragging,
      isProcessing,
      uploadProgress,
      fileInputRef,
      isImage,
      isVideo,
      formatFileSize,
      handleFileSelect,
      handleDrop,
      uploadFile,
      clearFile,
      triggerFileInput,
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
.btn-primary:disabled { background: #a9b3f4; cursor: not-allowed; }
.btn-secondary { background: #f0f0f0; color: var(--text-light); }
.btn-secondary:hover { background: #e0e0e0; }

.upload-area {
  border: 2px dashed var(--border-color);
  border-radius: 15px;
  padding: 20px;
  text-align: center;
  background: #fafafa;
  transition: all 0.3s ease;
  cursor: pointer;
  min-height: 250px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.upload-area:hover, .upload-area.dragover {
  border-color: var(--primary-color);
  background: #f4f5fe;
}
.upload-area.has-file { cursor: default; }

.upload-icon { font-size: 3rem; color: var(--primary-color); margin-bottom: 15px; }
.upload-text { font-size: 1.1rem; color: var(--text-light); margin-bottom: 10px; }
.upload-hint { color: #999; font-size: 0.9rem; }
.hidden-input { display: none; }

.preview-wrapper { width: 100%; text-align: center; }
.preview-content { max-width: 100%; max-height: 250px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
.file-info { margin-top: 15px; display: flex; align-items: center; justify-content: center; gap: 8px; color: var(--text-light); }
.file-size { color: #999; font-size: 0.9rem; }

.action-buttons { display: flex; gap: 10px; margin-top: 20px; justify-content: center; }
.loading-spinner {
  width: 16px; height: 16px; border: 2px solid #fff; border-radius: 50%;
  border-top-color: transparent; animation: spin 0.8s linear infinite;
}
.progress-bar {
  margin-top: 15px; height: 8px; background: #e9ecef; border-radius: 4px;
  overflow: hidden; position: relative;
}
.progress-fill { height: 100%; background: var(--primary-color); transition: width 0.3s ease; }
.progress-text {
  position: absolute; width: 100%; text-align: center; top: -20px;
  font-size: 0.8rem; color: var(--text-light); font-weight: 500;
}

@keyframes spin { to { transform: rotate(360deg); } }
</style>
