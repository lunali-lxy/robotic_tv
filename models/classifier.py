# models/classifier.py

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
from queue import Queue
import threading
import time

# 导入配置
from config import CLASSIFICATION_WEIGHTS, CLASSIFICATION_DEVICE, CLASS_MAP

# 简化的屏幕状态分类预处理
classification_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ScreenClassifier:
    def __init__(self, weights_path=CLASSIFICATION_WEIGHTS, device=CLASSIFICATION_DEVICE):
        """初始化屏幕状态分类器"""
        self.device = device
        self.model = self._load_model(weights_path)
    
    def _load_model(self, weights_path):
        """加载分类模型权重"""
        model = models.efficientnet_b0(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 8)
        model.to(self.device)
        model.eval()

        print(f"加载分类模型权重: {weights_path}")
        try:
            map_location = torch.device('cpu') if not torch.cuda.is_available() else None
            checkpoint = torch.load(weights_path, map_location=map_location)

            state_dict = checkpoint.get('state_dict') or checkpoint.get('model') or checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)
            print("✅ 屏幕状态分类模型加载成功")
            return model
        except Exception as e:
            print(f"❌ 分类模型加载失败: {str(e)}")
            return None
    
    def preprocess_roi_fast(self, tv_roi):
        """快速预处理电视ROI区域"""
        if tv_roi is None or tv_roi.size == 0 or tv_roi.shape[0] < 20 or tv_roi.shape[1] < 20:
            return None
            
        try:
            # 从 BGR (OpenCV) 转换为 RGB (PIL)
            pil_img = Image.fromarray(cv2.cvtColor(tv_roi, cv2.COLOR_BGR2RGB))
            return pil_img
        except Exception as e:
            # print(f"预处理 ROI 失败: {e}")
            return None
    
    def analyze_screen_fast(self, tv_roi):
        """快速分析TV屏幕状态"""
        if self.model is None or tv_roi is None:
            return "waiting", 0.0

        try:
            pil_img = self.preprocess_roi_fast(tv_roi)
            if pil_img is None:
                return "waiting", 0.0
                
            img_tensor = classification_preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                probs = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                pred_prob = probs[0, pred_class].item()

            if pred_class in CLASS_MAP:
                return CLASS_MAP[pred_class], pred_prob
            else:
                return f"class_{pred_class}", pred_prob
        except Exception as e:
            # print(f"分类错误: {e}")
            return "error", 0.0


class ClassificationWorker:
    """分类工作线程"""
    def __init__(self, classifier):
        self.classifier = classifier
        # 使用 maxsize=1 确保任务队列只存储最新帧
        self.task_queue = Queue(maxsize=1) 
        self.result_queue = Queue(maxsize=1)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.running = True
        self.worker_thread.start()
    
    def _worker(self):
        while self.running:
            try:
                # 尝试非阻塞地获取任务
                tv_roi = self.task_queue.get(block=False) 
                if tv_roi is not None:
                    status, confidence = self.classifier.analyze_screen_fast(tv_roi)
                    
                    # 确保结果队列中只有最新的结果
                    if not self.result_queue.empty():
                        self.result_queue.get_nowait()
                    self.result_queue.put((status, confidence))
            except: # 队列为空时会抛出异常
                time.sleep(0.001)
    
    def submit_task(self, tv_roi):
        """提交分类任务"""
        try:
            # 清理旧任务，只放入新任务
            if not self.task_queue.empty():
                self.task_queue.get_nowait()
            self.task_queue.put(tv_roi, block=False)
        except:
            pass # 队列已满或异常
    
    def get_result(self):
        """获取最新的分类结果"""
        try:
            return self.result_queue.get_nowait()
        except:
            return None
    
    def stop(self):
        """停止工作线程"""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)