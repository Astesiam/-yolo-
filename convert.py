import json
import os

# 读取 JSON 文件
with open("annotations/val.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 构建 image_id 到 file_name 和尺寸的映射
id2file = {img["id"]: (img["file_name"], img["width"], img["height"]) for img in data["images"]}

# 创建保存路径
os.makedirs("labels/val", exist_ok=True)

# 按图像整理 annotation
image_annotations = {}
for ann in data["annotations"]:
    img_id = ann["image_id"]
    if img_id not in image_annotations:
        image_annotations[img_id] = []
    image_annotations[img_id].append(ann)

# 写入每张图像对应的 .txt 文件
for img_id, anns in image_annotations.items():
    file_name, width, height = id2file[img_id]
    txt_name = os.path.splitext(file_name)[0] + ".txt"
    txt_path = os.path.join("labels/val", txt_name)

    with open(txt_path, "w") as f:
        for ann in anns:
            cat_id = ann["category_id"]
            bbox = ann["bbox"]  # [x_min, y_min, w, h]
            x_center = (bbox[0] + bbox[2] / 2) / width
            y_center = (bbox[1] + bbox[3] / 2) / height
            w = bbox[2] / width
            h = bbox[3] / height
            f.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
