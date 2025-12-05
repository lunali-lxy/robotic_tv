# main.py
# 主入口：TV 屏幕跟踪与机械臂控制主流程
# 该文件负责：
# - 初始化检测器/分类器/机械臂控制器/帧差检测等组件
# - 在主循环中读取摄像头帧，执行检测/分类/控制逻辑
# - 提供可视化与调试按键（重置、重连、方向测试）

from PIL import ImageFont
import cv2
import numpy as np
import time
import math
from collections import deque
from config import *
from models.detector import TVDetector
from models.classifier import ScreenClassifier, ClassificationWorker
from control.arm_controller import CorrectArmController
from utils.frame_utils import FrameDifferenceDetector, visualize_detections, draw_info_on_frame, update_fps
from utils.logger_utils import setup_logger, DEFAULT_LOG_FILE
import threading
from queue import Queue, Empty

# 配置日志记录器，日志文件位置由 `utils/logger_utils` 提供
log = setup_logger(log_file=DEFAULT_LOG_FILE)

class TVTracker:
    def __init__(self):
        # 初始化检测器与分类器
        self.detector = TVDetector(MODEL_PATH)  # 目标检测器（TV 检测）
        self.classifier = ScreenClassifier(CLASSIFICATION_WEIGHTS, CLASSIFICATION_DEVICE)  # 屏幕内容分类器
        self.classification_worker = ClassificationWorker(self.classifier)  # 异步分类任务工作线程

        # 帧差检测用于检测屏幕内容是否发生显著变化以触发重新分类
        self.frame_difference_detector = FrameDifferenceDetector()

        # 机械臂控制器（通过网络/服务器控制机械臂）
        self.arm_controller = CorrectArmController(SERVER_IP, SERVER_PORT)

        # 平滑历史记录（用于平滑目标位置，防抖）
        self.x_history = deque(maxlen=MAX_HISTORY)
        self.y_history = deque(maxlen=MAX_HISTORY)
        self.z_history = deque(maxlen=MAX_HISTORY)

        # 目标默认位置（机械臂坐标系），根据实际机械臂设定
        self.target_x = 0
        self.target_y = 15
        self.target_z = 15

        # 运行状态控制变量
        self.running = True
        self.tv_detected = False
        self.last_tv_time = 0
        self.last_boxes = np.array([])

        # 搜索模式（丢失目标时机械臂搜索）相关状态
        self.search_mode = False
        self.search_start_time = 0
        self.search_phase = 0

        # 屏幕分类结果与置信度
        self.screen_status = "waiting"
        self.screen_confidence = 0.0

        # 内容变化检测与强制分类标志（首次检测或内容变化时触发分类）
        self.content_changed = False
        self.force_classification = True

        # 用于 FPS 计算和统计
        self.frame_count = 0
        self.start_time = time.time()
        self.avg_fps = 0

        # 使用线程队列实现主线程与处理线程的轻量通信（避免主循环被处理阻塞）
        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.process_thread = threading.Thread(target=self._process_thread_worker, daemon=True)
        self.process_thread.start()

        # 对齐流程相关（当机械臂到达目标中心点时标记完成）
        self.align_complete = False
        self.final_tv_center = None

        # 尝试加载字体用于可视化，若失败则使用默认绘制方式
        try:
            self.font = ImageFont.truetype("arial.ttf", 16)
        except:
            self.font = None

    def verify_alignment(self, tv_center, view_center):
        if tv_center is None:
            return False

        # 计算 TV 中心与视图中心的欧式距离，判断是否在阈值内
        dx = tv_center[0] - view_center[0]
        dy = tv_center[1] - view_center[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        return distance <= MOVE_THRESHOLD

    def calculate_target_position(self, center_x, center_y, tv_w, tv_h):
        """ The target position calculation for the robotic arm """
        image_center_x = IMAGE_WIDTH / 2
        image_center_y = IMAGE_HEIGHT / 2
        
        offset_x = center_x - image_center_x
        offset_y = center_y - image_center_y
        
        # 如果先前已标记为对齐完成，则跳过移动
        if self.align_complete:
            return None, None, None

        # 当偏移量在阈值内时，判断为到达对齐位置
        if math.sqrt(offset_x ** 2 + offset_y ** 2) <= MOVE_THRESHOLD:
            self.align_complete = True
            self.final_tv_center = (center_x, center_y)
            # 再次校验距离以确保稳定
            if not self.verify_alignment(self.final_tv_center, (image_center_x, image_center_y)):
                self.align_complete = False
            else:
                return None, None, None

        # X 轴调整：根据像素偏移量映射到机械臂的 X 轴增量（比例缩放）
        x_adjust = offset_x / image_center_x * 4

        # Z 轴调整：垂直方向偏移映射到 Z 轴（注意方向约定）
        z_adjust = offset_y / image_center_y * 3 

        # Y 轴调整：通过检测到的 TV 宽度占比判断远近，决定前进/后退
        tv_size_ratio = tv_w / IMAGE_WIDTH
        if tv_size_ratio > 0.4:
            y_adjust = 2    # 屏幕占比过大，向后移动
        elif tv_size_ratio < 0.2:
            y_adjust = -2   # 屏幕占比过小，向前移动
        else:
            y_adjust = 0

        # 组合得到新的目标位置（注意 Z 方向可能需要反向处理）
        new_x = self.target_x + x_adjust
        new_y = self.target_y + y_adjust
        new_z = self.target_z - z_adjust # Z 轴方向在此实现中取反

        # 限位，防止命令超出机械臂安全范围
        new_x = max(-20, min(20, new_x))
        new_y = max(10, min(25, new_y))
        new_z = max(10, min(20, new_z))

        return new_x, new_y, new_z
    
    def smooth_position(self, x, y, z):
        """Smooth the target position using historical data"""
        # 如果是首次平滑，初始化历史数据并直接返回
        if not self.x_history:
            self.x_history.append(x)
            self.y_history.append(y)
            self.z_history.append(z)
            return x, y, z

        # 使用指数加权移动平均（EWMA）实现位置平滑，降低抖动
        smooth_x = self.target_x * (1 - SMOOTHING_FACTOR) + x * SMOOTHING_FACTOR
        smooth_y = self.target_y * (1 - SMOOTHING_FACTOR) + y * SMOOTHING_FACTOR
        smooth_z = self.target_z * (1 - SMOOTHING_FACTOR) + z * SMOOTHING_FACTOR

        # 将平滑结果加入历史，用于后续可能的回退或参考
        self.x_history.append(smooth_x)
        self.y_history.append(smooth_y)
        self.z_history.append(smooth_z)

        return smooth_x, smooth_y, smooth_z
    
    def calculate_search_position(self):
        """Calculate position in search mode"""
        current_time = time.time()
        elapsed = current_time - self.search_start_time
        
        # 使用正弦波在 X 轴上摆动进行左右搜索
        phase = elapsed * SEARCH_MOVE_SPEED
        search_x = SEARCH_MOVE_RANGE * math.sin(phase)
        # 搜索时保持 Y/Z 的默认近似值
        search_y = 15
        search_z = 15

        return search_x, search_y, search_z
    
    def extract_tv_roi(self, frame, boxes):
        """Extract TV ROI region from the frame"""
        if len(boxes) == 0:
            return None
        # 选择最大的检测框作为 TV 区域（忽略小噪声框）
        largest_box = max(boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
        x1, y1, x2, y2 = [int(coord) for coord in largest_box]

        # 边界裁剪，防止越界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)

        # 过滤过小的 ROI
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        # 返回 ROI 的拷贝，避免对原帧造成副作用
        tv_roi = frame[y1:y2, x1:x2].copy()

        return tv_roi
    
    def update_screen_status(self):
        """Update screen status from classification worker"""
        result = self.classification_worker.get_result()
        if result:
            self.screen_status, self.screen_confidence = result
    
    def process_frame(self, frame):
        """Process video frame and execute tracking/control logic (Pipeline)"""
        
        # 1. 更新并记录 FPS 统计信息
        update_fps(self)
        
        display_frame = frame.copy()
        
        # 2. 目标检测（检测画面中的 TV）
        boxes, labels, scores = self.detector.detect_tv(frame)
        # 保存检测框用于可视化或后续处理
        self.last_boxes = boxes
        
        if len(boxes) > 0:
            # --- 跟踪模式：找到了 TV ---
            self.tv_detected = True
            self.last_tv_time = time.time()
            # 如果此前处于搜索模式，切换回跟踪模式
            if self.search_mode:
                self.search_mode = False
                log.info("Exit search mode")

            # 提取 TV 区域用于内容分类
            tv_roi = self.extract_tv_roi(frame, boxes)

            # 3. 异步执行屏幕内容分类：仅在内容变更或强制分类时提交任务
            if tv_roi is not None:
                self.content_changed = self.frame_difference_detector.detect_change(tv_roi)
                if self.content_changed or self.force_classification:
                    self.classification_worker.submit_task(tv_roi)
                    self.force_classification = False

            # 更新分类结果（若已有结果则读取）
            self.update_screen_status()

            # 计算 TV 的中心位置与尺寸用于机械臂控制
            tv_info = self.detector.calculate_tv_center(boxes)

            # 4. 根据检测到的中心与尺寸计算新的机械臂目标位置
            if tv_info:
                center_x, center_y, tv_w, tv_h = tv_info

                new_x, new_y, new_z = self.calculate_target_position(center_x, center_y, tv_w, tv_h)

                if new_x is not None and new_y is not None and new_z is not None:
                    # 5. 对新目标位置做平滑处理后下发给机械臂
                    self.target_x, self.target_y, self.target_z = self.smooth_position(new_x, new_y, new_z)
                    self.arm_controller.send_command(self.target_x, self.target_y, self.target_z)
                    
        else:
            # --- 未检测到目标：进入搜索或待机模式 ---
            # 清除帧差检测历史，准备下一次检测
            self.frame_difference_detector.prev_frame = None
            # 下次检测时强制进行分类以重新获取屏幕状态
            self.force_classification = True

            # 如果刚刚丢失目标并超过设定等待时间，则进入搜索模式
            if self.tv_detected and (time.time() - self.last_tv_time > RESET_WAIT_TIME):
                self.search_mode = True
                self.tv_detected = False
                self.search_start_time = time.time()
                log.info("Entering search mode")

            if self.search_mode:
                # 6. 搜索模式：执行周期性的左右摆动移动
                search_x, search_y, search_z = self.calculate_search_position()
                self.arm_controller.send_command(search_x, search_y, search_z)
                self.target_x, self.target_y, self.target_z = search_x, search_y, search_z
            else:
                # 7. 无目标且不在搜索中：回到默认位置并保持待机状态
                if self.target_x != 0 or self.target_y != 15 or self.target_z != 15:
                    self.target_x = 0
                    self.target_y = 15
                    self.target_z = 15
                    self.arm_controller.send_command(self.target_x, self.target_y, self.target_z)
                self.tv_detected = False
                
        # 8. 可视化：在画面上绘制检测框、分类状态、机械臂信息等
        if self.tv_detected:
            display_frame = visualize_detections(display_frame, boxes, scores,
                                                 self.screen_status, self.screen_confidence,
                                                 self.arm_controller, self.font)

        # 在画面上绘制 FPS、目标位置等调试信息
        display_frame = draw_info_on_frame(display_frame, self)

        return display_frame

    def _process_thread_worker(self):
        while True:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                processed_frame = self.process_frame(frame)
                if not self.result_queue.empty():
                    self.result_queue.get()
                self.result_queue.put(processed_frame)
                if self.align_complete:
                    self.align_complete = False
                    self.x_history.clear()
                    self.y_history.clear()
                    self.z_history.clear()
                    time.sleep(3600)
            except Empty:
                continue
            except Exception as e:
                # 线程内部异常打印，避免进程中断
                print(f"Thread error: {e}")
                continue

    def run(self):
        """Run the main loop for TV tracking"""
        log.info("Step1: Starting camera...")
        start_time = time.time()
        cap = cv2.VideoCapture(CAMERA_INDEX)
        while time.time() - start_time < 60:
            if cap.isOpened():
                is_opened = True
                break
            time.sleep(0.01)
        if is_opened:
            # Set camera parameters
            log.info("Camera opened successfully, setting parameters...")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            log.info("Unable to open camera")
            cap.release()
            return
        duration = time.time() - start_time
        log.info(f"Camera initialized in {duration:.2f} seconds")
        
        # 启动时初始化机械臂到默认位置，给机械臂一定的响应时间
        log.info("Step2: Initializing robotic arm to default position...")
        start_time = time.time()
        self.arm_controller.send_command(self.target_x, self.target_y, self.target_z)
        time.sleep(1)
        duration = time.time() - start_time
        log.info(f"Robotic arm initialized in {duration:.2f} seconds")
    
        MAX_READ_FAILURES = 30
        consecutive_failures = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                log.info("Unable to read camera frame")
                time.sleep(0.1)
                if consecutive_failures >= MAX_READ_FAILURES:
                    log.error("Maximum camera read failures reached, exiting...")
                    break
                continue
            
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            #USB camera may be flipped
            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)
            
            # Main pipeline call
            #processed_frame = self.process_frame(frame)
            #1205
            # 使用线程队列将帧传给处理线程并获取已处理帧，以降低主循环阻塞
            if not self.frame_queue.empty():
                self.frame_queue.get()
            self.frame_queue.put(frame)
            processed_frame = self.result_queue.get(block=True)

            cv2.imshow('TV Tracking', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.target_x, self.target_y, self.target_z = 0, 15, 15
                self.arm_controller.send_command(self.target_x, self.target_y, self.target_z)
                log.info("Reset robotic arm position")
            elif key == ord('d'):
                self.arm_controller.connect_to_server()
                log.info("Attempting to reconnect robotic arm")
            elif key == ord('t'):
                # Direction test
                self.test_arm_movement()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.classification_worker.stop()
        self.arm_controller.stop()
        log.info("TV tracking stopped")

    def test_arm_movement(self):
        """Test the four directional movements of the robotic arm"""
        log.info("Starting direction test...")
        
        # Reset to center
        self.arm_controller.send_command(0, 15, 15)
        time.sleep(1)
        
        log.info("1. Move left (x = -10)")
        self.arm_controller.send_command(-10, 15, 15)
        time.sleep(1)
        
        log.info("2. Move right (x = 10)")
        self.arm_controller.send_command(10, 15, 15)
        time.sleep(1)
        
        log.info("3. Move up (z = 18)")
        self.arm_controller.send_command(0, 15, 18)
        time.sleep(1)
        
        log.info("4. Move down (z = 12)")
        self.arm_controller.send_command(0, 15, 12)
        time.sleep(1)
        
        log.info("5. Return to center")
        self.arm_controller.send_command(0, 15, 15)
        log.info("Direction test completed")


if __name__ == '__main__':
    log.info("=" * 50)
    log.info("TV Screen Tracking Client - Main Pipeline")
    log.info("=" * 50)
    try:
        tracker = TVTracker()
        tracker.run()
    except Exception as e:
        log.info(f"Fatal error: {e}")