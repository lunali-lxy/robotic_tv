# utils/frame_utils.py

import cv2
import numpy as np
import time
import math
from PIL import Image, ImageDraw, ImageFont

# 导入配置
from config import IMAGE_WIDTH, IMAGE_HEIGHT, CLASS_MAP

class FrameDifferenceDetector:
    """基于帧间差异的内容变化检测器"""
    def __init__(self, threshold=15, min_contour_area=300):
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.prev_frame = None
    
    def detect_change(self, current_frame):
        """检测当前帧与前一帧的内容变化"""
        if self.prev_frame is None:
            # 首次运行，初始化并认为内容改变
            self.prev_frame = current_frame.copy()
            return True
        
        # 统一大小，防止 ROI 尺寸变化导致错误
        if current_frame.shape != self.prev_frame.shape:
             # 如果尺寸不匹配，直接更新前一帧并返回变化
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
        
        # 如果有变化，更新前一帧
        if content_changed:
            self.prev_frame = current_frame.copy()
            
        return content_changed

def visualize_detections(frame, boxes, scores, screen_status, screen_confidence, arm_controller, font=None):
    """在帧上可视化检测和状态结果"""
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)
    
    # 绘制 TV 框和检测信息
    for idx, b in enumerate(boxes):
        x0, y0, x1, y1 = [float(coord) for coord in b]
        
        draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
        
        score = scores[idx]
        text = f"TV: {score:.2f}"
        
        # 状态信息
        status_text = f"Status: {screen_status} ({screen_confidence:.2f})"
        
        if font:
            draw.text((x0, y0 - 15), text=text, fill='red', font=font)
            draw.text((x0, y1 + 5), text=status_text, fill='yellow', font=font)
        else:
            draw.text((x0, y0 - 15), text=text, fill='red')
            draw.text((x0, y1 + 5), text=status_text, fill='yellow')
            
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def draw_info_on_frame(frame, tracker):
    """在帧上绘制机械臂和 FPS 信息"""
    # 绘制中心点和 TV 中心
    if tracker.tv_detected and len(tracker.detector.calculate_tv_center(tracker.last_boxes)) > 0:
        center_x, center_y, _, _ = tracker.detector.calculate_tv_center(tracker.last_boxes)
        cv2.circle(frame, (int(center_x), int(center_y)), 10, (0, 0, 255), -1)
        cv2.circle(frame, (int(IMAGE_WIDTH/2), int(IMAGE_HEIGHT/2)), 5, (255, 0, 0), -1)
        cv2.line(frame, 
                 (int(IMAGE_WIDTH/2), int(IMAGE_HEIGHT/2)), 
                 (int(center_x), int(center_y)), 
                 (255, 255, 0), 2)
    
    # 绘制状态信息
    status_color = (0, 255, 0) if tracker.tv_detected else (0, 0, 255)
    
    info = [
        (f"FPS: {tracker.avg_fps:.1f}", (10, IMAGE_HEIGHT - 20), (255, 255, 255)),
        (f"X: {tracker.target_x:.1f} | Y: {tracker.target_y:.1f} | Z: {tracker.target_z:.1f}", (10, 30), status_color),
        (f"Tracking: {'ON' if tracker.tv_detected else 'OFF'}", (10, 60), status_color),
        (f"Mode: {'SEARCHING' if tracker.search_mode else 'TRACKING' if tracker.tv_detected else 'IDLE'}", (10, 90), (0, 255, 255)),
        (f"Arm: {'CONNECTED' if tracker.arm_controller.connected else 'DISCONNECTED'}", (10, 120), (255, 255, 0)),
    ]
    
    for text, pos, color in info:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    return frame

def update_fps(tracker):
    """Update FPS calculation"""
    tracker.frame_count += 1
    if tracker.frame_count >= 30:
        elapsed = time.time() - tracker.start_time
        tracker.avg_fps = tracker.frame_count / elapsed
        tracker.frame_count = 0
        tracker.start_time = time.time()