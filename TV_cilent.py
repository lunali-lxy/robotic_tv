#!/usr/bin/python3
# coding=utf8
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决OpenMP冲突

import cv2
import numpy as np
import json
import socket
import time
import onnxruntime as ort
import math
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.transforms import ToTensor
from collections import deque
import threading
from queue import Queue

# 配置参数
MODEL_PATH = "model2.onnx"  # RT-DETR模型路径
SERVER_IP = "192.168.51.180"  # 树莓派IP地址
SERVER_PORT = 6000  # 树莓派端口
CAMERA_INDEX = 1  # 摄像头索引
CONF_THRESHOLD = 0.6  # 置信度阈值，与参考代码一致
IMAGE_WIDTH = 640  # 图像宽度
IMAGE_HEIGHT = 480  # 图像高度
FLIP_CAMERA = True  # 是否翻转摄像头画面
INPUT_SIZE = (640, 640)  # 模型输入尺寸 (W, H)

# 屏幕状态分类模型配置
CLASSIFICATION_WEIGHTS = "Last_Epoch016.pth"  # 分类模型权重路径
CLASSIFICATION_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 8个类别定义 - 按照您提供的映射
CLASS_MAP = {0: "red", 1: "green", 2: "black", 3: "white",
             4: "pink", 5: "blue", 6: "fault", 7: "snow"}

# 机械臂控制参数 - 修正方向
SEND_INTERVAL = 0.5  # 发送间隔
MOVE_THRESHOLD = 20  # 像素偏移阈值

# 运动平滑参数
SMOOTHING_FACTOR = 0.7
MAX_HISTORY = 5

# 搜索模式参数
SEARCH_MOVE_RANGE = 10
SEARCH_MOVE_SPEED = 0.2
RESET_WAIT_TIME = 3.0

# 只定义电视类别
TV_CLASS = 0  # 电视整体

# 简化的屏幕状态分类预处理
classification_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ScreenClassifier:
    def __init__(self, weights_path, device="cpu"):
        """初始化屏幕状态分类器"""
        self.device = device
        
        # 使用与训练时相同的模型结构
        self.model = models.efficientnet_b0(pretrained=False)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, 8)
        self.model.to(self.device)
        self.model.eval()

        # 加载分类模型权重
        print(f"加载分类模型权重: {weights_path}")
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(weights_path)
            else:
                checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            try:
                self.model.load_state_dict(state_dict)
                print("✅ 模型权重加载成功")
            except RuntimeError as e:
                print(f"❌ 权重加载错误: {e}")
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
                print(f"✅ 部分权重加载成功: {len(pretrained_dict)}/{len(state_dict)} 个参数")

            print(f"✅ 屏幕状态分类模型加载成功")
            
        except Exception as e:
            print(f"❌ 分类模型加载失败: {str(e)}")
            self.model = None
    
    def preprocess_roi_fast(self, tv_roi):
        """快速预处理电视ROI区域"""
        if tv_roi.size == 0:
            return None
            
        try:
            pil_img = Image.fromarray(cv2.cvtColor(tv_roi, cv2.COLOR_BGR2RGB))
            if pil_img.width < 20 or pil_img.height < 20:
                return None
            return pil_img
            
        except Exception as e:
            return None
    
    def analyze_screen_fast(self, tv_roi):
        """快速分析TV屏幕状态"""
        if self.model is None or tv_roi is None or tv_roi.size == 0:
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
            return "error", 0.0


class FrameDifferenceDetector:
    """基于帧间差异的内容变化检测器"""
    def __init__(self, threshold=10, min_contour_area=500):
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.prev_frame = None
    
    def detect_change(self, current_frame):
        """检测当前帧与前一帧的内容变化"""
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return True
        
        # 转换为灰度图
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # 计算帧间差异
        frame_diff = cv2.absdiff(prev_gray, current_gray)
        
        # 应用阈值
        _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 检查是否有足够大的变化区域
        content_changed = False
        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                content_changed = True
                break
        
        # 更新前一帧
        self.prev_frame = current_frame.copy()
        
        return content_changed


class ClassificationWorker:
    """分类工作线程"""
    def __init__(self, classifier):
        self.classifier = classifier
        self.task_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.running = True
        self.worker_thread.start()
    
    def _worker(self):
        while self.running:
            try:
                tv_roi = self.task_queue.get(block=False)
                if tv_roi is not None:
                    status, confidence = self.classifier.analyze_screen_fast(tv_roi)
                    if not self.result_queue.empty():
                        self.result_queue.get()
                    self.result_queue.put((status, confidence))
            except:
                time.sleep(0.001)
    
    def submit_task(self, tv_roi):
        try:
            if not self.task_queue.empty():
                self.task_queue.get(block=False)
            self.task_queue.put(tv_roi, block=False)
        except:
            pass
    
    def get_result(self):
        try:
            if not self.result_queue.empty():
                return self.result_queue.get(block=False)
        except:
            pass
        return None
    
    def stop(self):
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)


class TVDetector:
    def __init__(self, model_path):
        providers = ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(model_path, providers=providers)
        print(f"✅ 电视检测模型加载成功: {model_path}")
        
    def detect_tv(self, frame):
        """检测电视屏幕"""
        orig_h, orig_w = frame.shape[:2]
        
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        im_resized = im.resize(INPUT_SIZE)
        im_data = ToTensor()(im_resized)[None]
        size_tensor = torch.tensor([[INPUT_SIZE[0], INPUT_SIZE[1]]])

        try:
            output = self.sess.run(
                output_names=None,
                input_feed={
                    'images': im_data.numpy(),
                    'orig_target_sizes': size_tensor.numpy()
                }
            )
        except Exception as e:
            print(f"❌ 推理错误: {str(e)}")
            return np.array([]), np.array([]), np.array([])
            
        labels, boxes, scores = output

        scale_x = orig_w / INPUT_SIZE[0]
        scale_y = orig_h / INPUT_SIZE[1]

        valid_indices = scores[0] > CONF_THRESHOLD
        labels = labels[0][valid_indices]
        boxes = boxes[0][valid_indices]
        scores = scores[0][valid_indices]
        
        tv_indices = [i for i, label in enumerate(labels) if int(label) == TV_CLASS]
        labels = labels[tv_indices]
        boxes = boxes[tv_indices]
        scores = scores[tv_indices]
        
        scaled_boxes = []
        for b in boxes:
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
    
    def calculate_tv_center(self, boxes, labels):
        """计算电视中心点和尺寸"""
        if len(boxes) == 0:
            return None
        
        largest_box = max(boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
        
        center_x = (largest_box[0] + largest_box[2]) / 2
        center_y = (largest_box[1] + largest_box[3]) / 2
        width = largest_box[2] - largest_box[0]
        height = largest_box[3] - largest_box[1]
        
        return center_x, center_y, width, height


class CorrectArmController:
    """正确方向的机械臂控制器"""
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.sock = None
        self.connected = False
        self.last_send_time = 0
        self.last_command = None
        
        self.connect_to_server()
    
    def connect_to_server(self):
        """连接到树莓派服务器"""
        try:
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
                
            print(f"🔌 尝试连接到 {self.server_ip}:{self.server_port}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3.0)
            self.sock.connect((self.server_ip, self.server_port))
            self.sock.settimeout(5.0)
            self.connected = True
            print(f"✅ 连接成功")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {str(e)}")
            self.connected = False
            return False
    
    def send_command(self, x, y, z):
        """发送命令到机械臂"""
        current_time = time.time()
        
        # 检查发送间隔
        if current_time - self.last_send_time < SEND_INTERVAL:
            return True
            
        if not self.connected:
            if not self.connect_to_server():
                return False
        
        command = {'x': round(x, 1), 'y': round(y, 1), 'z': round(z, 1)}
        
        try:
            # 发送命令
            command_str = json.dumps(command)
            self.sock.sendall(command_str.encode('utf-8'))
            print(f"🎯 发送: {command_str}")
            
            # 尝试接收响应
            self.sock.settimeout(1.0)
            try:
                response = self.sock.recv(128)
                if response:
                    response_str = response.decode().strip()
                    print(f"📥 响应: {response_str}")
            except socket.timeout:
                pass
            except Exception as e:
                print(f"⚠️ 读取响应错误: {str(e)}")
            
            # 恢复超时设置
            self.sock.settimeout(5.0)
            
            self.last_send_time = current_time
            self.last_command = (x, y, z)
            return True
            
        except socket.timeout:
            print("⚠️ 发送超时")
            self.connected = False
            return False
        except Exception as e:
            print(f"❌ 发送失败: {str(e)}")
            self.connected = False
            return False
    
    def stop(self):
        """停止控制器"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass


class TVTracker:
    def __init__(self, model_path, server_ip, server_port, classifier_weights):
        self.detector = TVDetector(model_path)
        self.classifier = ScreenClassifier(classifier_weights, CLASSIFICATION_DEVICE)
        self.classification_worker = ClassificationWorker(self.classifier)
        self.frame_difference_detector = FrameDifferenceDetector(threshold=15, min_contour_area=300)
        
        # 使用正确方向的机械臂控制器
        self.arm_controller = CorrectArmController(server_ip, server_port)
        
        # 位置历史
        self.x_history = deque(maxlen=MAX_HISTORY)
        self.y_history = deque(maxlen=MAX_HISTORY)
        self.z_history = deque(maxlen=MAX_HISTORY)
        
        # 目标位置
        self.target_x = 0
        self.target_y = 15
        self.target_z = 15
        
        # 运行标志
        self.running = True
        self.tv_detected = False
        self.last_tv_time = 0
        
        # 搜索模式
        self.search_mode = False
        self.search_start_time = 0
        self.search_phase = 0
        
        # 屏幕状态
        self.screen_status = "waiting"
        self.screen_confidence = 0.0
        self.content_changed = False
        self.force_classification = True
        self.classification_count = 0
        
        # 帧率统计
        self.frame_count = 0
        self.start_time = time.time()
        self.avg_fps = 0
        
        # 尝试加载字体
        try:
            self.font = ImageFont.truetype("arial.ttf", 16)
        except:
            self.font = None
    
    def calculate_target_position(self, center_x, center_y, tv_w, tv_h):
        """计算机械臂目标位置 """
        image_center_x = IMAGE_WIDTH / 2
        image_center_y = IMAGE_HEIGHT / 2
        
        offset_x = center_x - image_center_x
        offset_y = center_y - image_center_y
        
        # 如果偏移量小于阈值，则机械臂不动
        if abs(offset_x) < MOVE_THRESHOLD and abs(offset_y) < MOVE_THRESHOLD:
            return None, None, None
        
        # 方向逻辑：
        # 1. 电视在画面左边（offset_x < 0）→ 机械臂向左移动（减少x）
        # 2. 电视在画面右边（offset_x > 0）→ 机械臂向右移动（增加x）
        # 3. 电视在画面上方（offset_y < 0）→ 机械臂向上移动（增加z）
        # 4. 电视在画面下方（offset_y > 0）→ 机械臂向下移动（减少z）
        
        # 将偏移量转换为机械臂移动量
        # X轴: 电视中心偏左，机械臂需要向左移动（减小x）
        #       电视中心偏右，机械臂需要向右移动（增加x）
        x_adjust = offset_x / image_center_x * 4  # 缩放因子，保持正值
        
        # Z轴: 电视中心偏上，机械臂需要向上移动（增加z）
        #       电视中心偏下，机械臂需要向下移动（减少z）
        z_adjust = offset_y / image_center_y * 3  # 缩放因子，保持正值
        
        # Y轴: 根据电视大小调整距离
        #       电视太大 → 后退（增加y）
        #       电视太小 → 前进（减少y）
        # 注意：y轴是距离，增加y表示后退，减少y表示前进
        tv_size_ratio = tv_w / IMAGE_WIDTH
        if tv_size_ratio > 0.4:  # 电视太大，需要后退
            y_adjust = 2
        elif tv_size_ratio < 0.2:  # 电视太小，需要前进
            y_adjust = -2
        else:
            y_adjust = 0
        
        # 计算新的目标位置
        new_x = self.target_x + x_adjust
        new_y = self.target_y + y_adjust
        new_z = self.target_z - z_adjust  # 注意：z轴方向需要取反，因为图像坐标y向下为正
        
        # 限制机械臂运动范围
        new_x = max(-20, min(20, new_x))
        new_y = max(10, min(25, new_y))
        new_z = max(10, min(20, new_z))
        
        return new_x, new_y, new_z
    
    def smooth_position(self, x, y, z):
        """平滑位置变化"""
        self.x_history.append(x)
        self.y_history.append(y)
        self.z_history.append(z)

        if len(self.x_history) == 0:
            return x, y, z

        # 应用加权平均
        smooth_x = sum(self.x_history) / len(self.x_history)
        smooth_y = sum(self.y_history) / len(self.y_history)
        smooth_z = sum(self.z_history) / len(self.z_history)

        return smooth_x, smooth_y, smooth_z
    
    def calculate_search_position(self):
        """计算搜索模式下的位置"""
        current_time = time.time()
        elapsed = current_time - self.search_start_time
        
        phase = elapsed * SEARCH_MOVE_SPEED
        self.search_phase = phase
        
        # 搜索时左右移动
        search_x = SEARCH_MOVE_RANGE * math.sin(phase)
        search_y = 15
        search_z = 15
        
        return search_x, search_y, search_z
    
    def send_to_arm(self, x, y, z):
        """发送坐标到机械臂"""
        return self.arm_controller.send_command(x, y, z)
    
    def extract_tv_roi(self, frame, boxes):
        """提取电视ROI区域"""
        if len(boxes) == 0:
            return None
            
        largest_box = max(boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
        x1, y1, x2, y2 = [int(coord) for coord in largest_box]
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        
        tv_roi = frame[y1:y2, x1:x2]
        
        if tv_roi.size == 0:
            return None
        
        return tv_roi
    
    def update_screen_status(self):
        """更新屏幕状态"""
        result = self.classification_worker.get_result()
        if result:
            self.screen_status, self.screen_confidence = result
            self.classification_count += 1
    
    def visualize_detections(self, frame, boxes, labels, scores):
        """在帧上可视化检测结果"""
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(im)
        
        for idx, b in enumerate(boxes):
            x0, y0, x1, y1 = [float(coord) for coord in b]
            
            draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
            
            score = scores[idx]
            text = f"TV: {score:.2f}"
            if self.font:
                draw.text((x0, y0), text=text, fill='red', font=self.font)
            else:
                draw.text((x0, y0), text=text, fill='red')
            
            status_text = f"Status: {self.screen_status}"
            if self.font:
                draw.text((x0, y0 + 20), text=status_text, fill='yellow', font=self.font)
            else:
                draw.text((x0, y0 + 20), text=status_text, fill='yellow')
        
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    
    def update_fps(self):
        """更新帧率统计"""
        self.frame_count += 1
        if self.frame_count >= 30:
            elapsed = time.time() - self.start_time
            self.avg_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
    
    def process_frame(self, frame):
        """处理视频帧并检测电视"""
        self.update_fps()
        
        display_frame = frame.copy()
        
        boxes, labels, scores = self.detector.detect_tv(frame)

        if len(boxes) > 0:
            tv_roi = self.extract_tv_roi(frame, boxes)
            
            if tv_roi is not None:
                try:
                    self.content_changed = self.frame_difference_detector.detect_change(tv_roi)
                    if self.content_changed or self.force_classification:
                        self.classification_worker.submit_task(tv_roi)
                        self.force_classification = False
                except Exception as e:
                    self.frame_difference_detector.prev_frame = None
                    self.content_changed = True
            
            self.update_screen_status()
            
            display_frame = self.visualize_detections(display_frame, boxes, labels, scores)
            
            tv_info = self.detector.calculate_tv_center(boxes, labels)
            
            if tv_info:
                center_x, center_y, tv_w, tv_h = tv_info
                
                cv2.circle(display_frame, (int(center_x), int(center_y)), 10, (0, 0, 255), -1)
                cv2.circle(display_frame, (int(IMAGE_WIDTH/2), int(IMAGE_HEIGHT/2)), 5, (255, 0, 0), -1)
                cv2.line(display_frame, 
                        (int(IMAGE_WIDTH/2), int(IMAGE_HEIGHT/2)), 
                        (int(center_x), int(center_y)), 
                        (255, 255, 0), 2)
                
                new_x, new_y, new_z = self.calculate_target_position(center_x, center_y, tv_w, tv_h)

                if new_x is not None and new_y is not None and new_z is not None:
                    self.target_x, self.target_y, self.target_z = self.smooth_position(
                        new_x, new_y, new_z)

                    self.send_to_arm(self.target_x, self.target_y, self.target_z)

                self.tv_detected = True
                self.last_tv_time = time.time()
                
                if self.search_mode:
                    self.search_mode = False
                    print("🎯 退出搜索模式")

                # 计算当前偏移量
                offset_x = center_x - IMAGE_WIDTH/2
                offset_y = center_y - IMAGE_HEIGHT/2
                current_offset = math.sqrt(offset_x**2 + offset_y**2)
                
                # 判断电视位置
                if offset_x < 0:
                    x_position = "LEFT"
                else:
                    x_position = "RIGHT"
                    
                if offset_y < 0:
                    y_position = "UP"
                else:
                    y_position = "DOWN"
                
                # 显示信息
                cv2.putText(display_frame, f"X: {self.target_x:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Y: {self.target_y:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Z: {self.target_z:.1f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"TVs: {len(boxes)}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "CORRECT TRACKING", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(display_frame, f"Screen: {self.screen_status}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Confidence: {self.screen_confidence:.2f}", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                conn_status = "CONNECTED" if self.arm_controller.connected else "DISCONNECTED"
                cv2.putText(display_frame, f"Arm: {conn_status}", (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.putText(display_frame, f"TV Position: {x_position}, {y_position}", (10, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Offset X: {offset_x:.1f}", (10, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Offset Y: {offset_y:.1f}", (10, 330),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # 显示移动方向
                if new_x is not None:
                    if new_x < self.target_x:
                        move_x = "← LEFT"
                    else:
                        move_x = "→ RIGHT"
                else:
                    move_x = "HOLD"
                    
                if new_z is not None:
                    if new_z < self.target_z:
                        move_z = "↓ DOWN"
                    else:
                        move_z = "↑ UP"
                else:
                    move_z = "HOLD"
                
                cv2.putText(display_frame, f"Move X: {move_x}", (10, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Move Z: {move_z}", (10, 390),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            self.frame_difference_detector.prev_frame = None
            self.force_classification = True
            
            if self.tv_detected and (time.time() - self.last_tv_time > RESET_WAIT_TIME):
                self.search_mode = True
                self.tv_detected = False
                self.search_start_time = time.time()
                print("🔍 进入搜索模式")
            
            if self.search_mode:
                search_x, search_y, search_z = self.calculate_search_position()
                self.send_to_arm(search_x, search_y, search_z)
                self.target_x, self.target_y, self.target_z = search_x, search_y, search_z
                
                cv2.putText(display_frame, "SEARCHING", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, f"X: {self.target_x:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                if self.tv_detected and (time.time() - self.last_tv_time > 2.0):
                    self.target_x = 0
                    self.target_y = 15
                    self.target_z = 15
                    self.send_to_arm(self.target_x, self.target_y, self.target_z)
                    self.tv_detected = False

                cv2.putText(display_frame, "No TV detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(display_frame, f"FPS: {self.avg_fps:.1f}", (10, IMAGE_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return display_frame
    
    def run(self):
        """运行电视追踪"""
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头: {CAMERA_INDEX}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("🚀 开始电视追踪（修正方向版）...")
        print("🛑 按 'q' 键退出")
        print("🔧 按 'r' 键重置机械臂位置")
        print("🔌 按 'd' 键重新连接机械臂")
        print("📊 按 't' 键测试移动方向")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头帧")
                break
            
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)
            
            processed_frame = self.process_frame(frame)
            
            cv2.imshow('TV Tracking - Correct Direction', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.target_x = 0
                self.target_y = 15
                self.target_z = 15
                self.send_to_arm(self.target_x, self.target_y, self.target_z)
                print("🔄 重置机械臂位置")
            elif key == ord('d'):
                self.arm_controller.connect_to_server()
                print("🔄 尝试重新连接机械臂")
            elif key == ord('t'):
                # 测试移动方向
                print("🧪 开始方向测试...")
                print("1. 向左移动 (x = -10)")
                self.send_to_arm(-10, 15, 15)
                time.sleep(1)
                print("2. 向右移动 (x = 10)")
                self.send_to_arm(10, 15, 15)
                time.sleep(1)
                print("3. 向上移动 (z = 18)")
                self.send_to_arm(0, 15, 18)
                time.sleep(1)
                print("4. 向下移动 (z = 12)")
                self.send_to_arm(0, 15, 12)
                time.sleep(1)
                print("5. 返回中心")
                self.send_to_arm(0, 15, 15)
                print("✅ 方向测试完成")
        
        cap.release()
        cv2.destroyAllWindows()
        self.classification_worker.stop()
        self.arm_controller.stop()
        print("🛑 电视追踪已停止")


if __name__ == '__main__':
    print("=" * 50)
    print("电视屏幕追踪客户端 - 修正方向版本")
    print("=" * 50)
    print("- 电视太大 → 后退 (y增加)")
    print("- 电视太小 → 前进 (y减少)")
    print("=" * 50)
    
    tracker = TVTracker(MODEL_PATH, SERVER_IP, SERVER_PORT, CLASSIFICATION_WEIGHTS)
    tracker.run()