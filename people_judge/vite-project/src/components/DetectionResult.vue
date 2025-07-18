<template>
  <div v-if="detections.length" class="result-container">
    <h3>检测结果：</h3>
    <ul>
      <li v-for="(item, index) in detections" :key="index">
        类别：{{ item.name }}，置信度：{{ (item.confidence * 100).toFixed(1) }}%
      </li>
    </ul>
    <p>是否危险：<strong>{{ danger ? '是' : '否' }}</strong></p>
  </div>
</template>

<script setup>
import { computed, defineProps } from 'vue'

const props = defineProps({
  detections: {
    type: Array,
    default: () => []
  }
})

const danger = computed(() => {
  return props.detections.some(item => item.name === 'danger' || item.name === 'person')
})
</script>

<style scoped>
.result-container {
  margin-top: 1.5em;
  font-family: Arial, sans-serif;
}
</style>