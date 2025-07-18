<template>
  <div>
    <input type="file" @change="onFileChange" accept="image/*" />
    <div v-if="previewUrl" style="margin: 10px 0;">
      <p>图片预览：</p>
      <img :src="previewUrl" alt="图片预览" style="max-width: 300px;" />
    </div>
    <button @click="submit" :disabled="!file">上传并识别</button>

    <div v-if="detections.length > 0" style="margin-top: 20px;">
      <h3>检测结果：</h3>
      <ul>
        <li v-for="(item, index) in detections" :key="index">
          类别：{{ item.name }}，置信度：{{ (item.confidence * 100).toFixed(1) }}%
        </li>
      </ul>
      <p style="font-weight: bold; color: red;" v-if="isDanger">是否危险：是 ⚠️</p>
      <p style="font-weight: bold; color: green;" v-else>是否危险：否 ✅</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const file = ref(null)
const previewUrl = ref(null)
const detections = ref([])
const isDanger = ref(false)

const onFileChange = (e) => {
  const selected = e.target.files[0]
  if (selected) {
    file.value = selected
    previewUrl.value = URL.createObjectURL(selected)
    detections.value = []
    isDanger.value = false
  } else {
    file.value = null
    previewUrl.value = null
  }
}

const submit = async () => {
  if (!file.value) return
  const formData = new FormData()
  formData.append('file', file.value)
  try {
    const res = await axios.post('http://localhost:8000/detect', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    detections.value = res.data.detections || []
    isDanger.value = res.data.is_danger || false
  } catch (err) {
    console.error('上传失败:', err)
    detections.value = []
    isDanger.value = false
  }
}
</script>