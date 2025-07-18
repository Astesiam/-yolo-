<template>
  <div class="app-container">
    <h1>图片检测系统</h1>
    <ImageUpload @detected="onDetected" />
    <DetectionResult :detections="detections" />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import ImageUpload from './components/ImageUpload.vue'
import DetectionResult from './components/DetectionResult.vue'

const detections = ref([])

const onDetected = (result) => {
  detections.value = result
}
</script>

<style scoped>
.app-container {
  max-width: 600px;
  margin: 2em auto;
  font-family: Arial, sans-serif;
}
</style>
