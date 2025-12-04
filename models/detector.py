# models/detector.py

import onnxruntime as ort
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image

# 导入配置
from config import MODEL_PATH, INPUT_SIZE, CONF_THRESHOLD, TV_CLASS

class TVDetector:
    def __init__(self, model_path=MODEL_PATH):
        providers = ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(model_path, providers=providers)
        print(f"✅ 电视检测模型加载成功: {model_path}")
        
    def detect_tv(self, frame):
        """检测电视屏幕"""
        orig_h, orig_w = frame.shape[:2]
        
        im = Image.fromarray(frame) # frame 是 BGR 格式，PIL 默认处理 RGB
        im_resized = im.resize(INPUT_SIZE)
        im_data = ToTensor()(im_resized)[None]
        
        # 保持与 ONNX 导出时的 input_feed 结构一致
        size_tensor = torch.tensor([[INPUT_SIZE[1], INPUT_SIZE[0]]]) # 注意: 有些 ONNX 模型需要 (H, W)
        
        try:
            output = self.sess.run(
                output_names=None,
                input_feed={
                    'images': im_data.numpy(),
                    'orig_target_sizes': size_tensor.numpy()
                }
            )
        except Exception as e:
            # print(f"❌ 推理错误: {str(e)}")
            return np.array([]), np.array([]), np.array([])
            
        labels, boxes, scores = output

        scale_x = orig_w / INPUT_SIZE[0]
        scale_y = orig_h / INPUT_SIZE[1]

        valid_indices = scores[0] > CONF_THRESHOLD
        labels = labels[0][valid_indices]
        boxes = boxes[0][valid_indices]
        scores = scores[0][valid_indices]
        
        # 仅保留 TV 类别
        tv_indices = [i for i, label in enumerate(labels) if int(label) == TV_CLASS]
        labels = labels[tv_indices]
        boxes = boxes[tv_indices]
        scores = scores[tv_indices]
        
        scaled_boxes = []
        for b in boxes:
            # 这里的 boxes 已经是 x0, y0, x1, y1 格式
            x0, y0, x1, y1 = [float(coord) for coord in b]
            x0 *= scale_x
            x1 *= scale_x
            y0 *= scale_y
            y1 *= scale_y
            scaled_boxes.append([x0, y0, x1, y1])
        
        if len(scaled_boxes) > 0:
            scaled_boxes = np.array(scaled_boxes)
        else:
            scaled_boxes = np.array([])
            
        return scaled_boxes, labels, scores
    
    def calculate_tv_center(self, boxes):
        """计算电视中心点和尺寸"""
        if len(boxes) == 0:
            return None
        
        # 找到最大的 TV 框
        largest_box = max(boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
        
        center_x = (largest_box[0] + largest_box[2]) / 2
        center_y = (largest_box[1] + largest_box[3]) / 2
        width = largest_box[2] - largest_box[0]
        height = largest_box[3] - largest_box[1]
        
        return center_x, center_y, width, height