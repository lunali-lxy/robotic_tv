# main.py
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import time
import math
from collections import deque
from config import *
from models.detector import TVDetector
import logging
from models.classifier import ScreenClassifier, ClassificationWorker
from control.arm_controller import CorrectArmController
from utils.frame_utils import FrameDifferenceDetector, visualize_detections, draw_info_on_frame, update_fps
from utils.logger_utils import setup_logger, DEFAULT_LOG_FILE

log = setup_logger(log_file=DEFAULT_LOG_FILE)

class TVTracker:
    def __init__(self):
        self.detector = TVDetector(MODEL_PATH)
        self.classifier = ScreenClassifier(CLASSIFICATION_WEIGHTS, CLASSIFICATION_DEVICE)
        self.classification_worker = ClassificationWorker(self.classifier)
        self.frame_difference_detector = FrameDifferenceDetector()
        self.arm_controller = CorrectArmController(SERVER_IP, SERVER_PORT)
        self.x_history = deque(maxlen=MAX_HISTORY)
        self.y_history = deque(maxlen=MAX_HISTORY)
        self.z_history = deque(maxlen=MAX_HISTORY)
        self.target_x = 0
        self.target_y = 15
        self.target_z = 15
        self.running = True
        self.tv_detected = False
        self.last_tv_time = 0
        self.last_boxes = np.array([])
        self.search_mode = False
        self.search_start_time = 0
        self.search_phase = 0
        self.screen_status = "waiting"
        self.screen_confidence = 0.0
        self.content_changed = False
        self.force_classification = True
        self.frame_count = 0
        self.start_time = time.time()
        self.avg_fps = 0
        try:
            self.font = ImageFont.truetype("arial.ttf", 16)
        except:
            self.font = None
    
    def calculate_target_position(self, center_x, center_y, tv_w, tv_h):
        """ The target position calculation for the robotic arm """
        image_center_x = IMAGE_WIDTH / 2
        image_center_y = IMAGE_HEIGHT / 2
        
        offset_x = center_x - image_center_x
        offset_y = center_y - image_center_y
        
        # If the offset is smaller than the threshold, the robotic arm does not move
        if abs(offset_x) < MOVE_THRESHOLD and abs(offset_y) < MOVE_THRESHOLD:
            return None, None, None
        
        # X axis adjustment: target is to the left (offset_x < 0) -> robotic arm decreases X (move left)
        x_adjust = offset_x / image_center_x * 4
        
        # Z axis adjustment: target is up (offset_y < 0) -> robotic arm increases Z (move up)
        z_adjust = offset_y / image_center_y * 3 
        
        # Y axis adjustment: target size
        tv_size_ratio = tv_w / IMAGE_WIDTH
        if tv_size_ratio > 0.4:
            y_adjust = 2    # Too large, move backward
        elif tv_size_ratio < 0.2:
            y_adjust = -2   # Too small, move forward
        else:
            y_adjust = 0
        
        # Calculate new target position
        new_x = self.target_x + x_adjust
        new_y = self.target_y + y_adjust
        new_z = self.target_z - z_adjust # Z axis direction is inverted
        
        # Limit robotic arm movement range
        new_x = max(-20, min(20, new_x))
        new_y = max(10, min(25, new_y))
        new_z = max(10, min(20, new_z))
        
        return new_x, new_y, new_z
    
    def smooth_position(self, x, y, z):
        """Smooth the target position using historical data"""
        
        if not self.x_history:
            self.x_history.append(x)
            self.y_history.append(y)
            self.z_history.append(z)
            return x, y, z
        
        # Exponentially Weighted Moving Average (EWMA)
        smooth_x = self.target_x * (1 - SMOOTHING_FACTOR) + x * SMOOTHING_FACTOR
        smooth_y = self.target_y * (1 - SMOOTHING_FACTOR) + y * SMOOTHING_FACTOR
        smooth_z = self.target_z * (1 - SMOOTHING_FACTOR) + z * SMOOTHING_FACTOR
        
        # Update history to have a reasonable base position when the target is lost
        self.x_history.append(smooth_x)
        self.y_history.append(smooth_y)
        self.z_history.append(smooth_z)

        return smooth_x, smooth_y, smooth_z
    
    def calculate_search_position(self):
        """Calculate position in search mode"""
        current_time = time.time()
        elapsed = current_time - self.search_start_time
        
        phase = elapsed * SEARCH_MOVE_SPEED
        # Left-right swing
        search_x = SEARCH_MOVE_RANGE * math.sin(phase)
        search_y = 15
        search_z = 15
        
        return search_x, search_y, search_z
    
    def extract_tv_roi(self, frame, boxes):
        """Extract TV ROI region from the frame"""
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
        
        tv_roi = frame[y1:y2, x1:x2].copy() # Use .copy() to avoid modifying the ROI
        
        return tv_roi
    
    def update_screen_status(self):
        """Update screen status from classification worker"""
        result = self.classification_worker.get_result()
        if result:
            self.screen_status, self.screen_confidence = result
    
    def process_frame(self, frame):
        """Process video frame and execute tracking/control logic (Pipeline)"""
        
        # 1. FPS statistics
        update_fps(self)
        
        display_frame = frame.copy()
        
        # 2. Target detection
        boxes, labels, scores = self.detector.detect_tv(frame)
        self.last_boxes = boxes # Save box information for visualization
        
        if len(boxes) > 0:
            # --- Tracking mode ---
            
            self.tv_detected = True
            self.last_tv_time = time.time()
            if self.search_mode:
                self.search_mode = False
                log.info("Exit search mode")
            
            tv_roi = self.extract_tv_roi(frame, boxes)
            
            # 3. Screen status classification (asynchronous)
            if tv_roi is not None:
                self.content_changed = self.frame_difference_detector.detect_change(tv_roi)
                if self.content_changed or self.force_classification:
                    self.classification_worker.submit_task(tv_roi)
                    self.force_classification = False
            
            self.update_screen_status()

            tv_info = self.detector.calculate_tv_center(boxes)
            
            # 4. Robotic arm control calculation
            if tv_info:
                center_x, center_y, tv_w, tv_h = tv_info
                
                new_x, new_y, new_z = self.calculate_target_position(center_x, center_y, tv_w, tv_h)

                if new_x is not None and new_y is not None and new_z is not None:
                    # 5. Smooth and send command
                    self.target_x, self.target_y, self.target_z = self.smooth_position(new_x, new_y, new_z)
                    self.arm_controller.send_command(self.target_x, self.target_y, self.target_z)
                    
        else:
            # --- Search or standby mode ---
            self.frame_difference_detector.prev_frame = None # Clear difference detection history
            self.force_classification = True
            
            if self.tv_detected and (time.time() - self.last_tv_time > RESET_WAIT_TIME):
                # Target lost for longer than the set time, enter search mode
                self.search_mode = True
                self.tv_detected = False
                self.search_start_time = time.time()
                log.info("Entering search mode")
            
            if self.search_mode:
                # 6. Search movement
                search_x, search_y, search_z = self.calculate_search_position()
                self.arm_controller.send_command(search_x, search_y, search_z)
                self.target_x, self.target_y, self.target_z = search_x, search_y, search_z
            else:
                # 7. Return to default position and standby
                if self.target_x != 0 or self.target_y != 15 or self.target_z != 15:
                    self.target_x = 0
                    self.target_y = 15
                    self.target_z = 15
                    self.arm_controller.send_command(self.target_x, self.target_y, self.target_z)
                self.tv_detected = False
                
        # 8. Visualization
        if self.tv_detected:
            display_frame = visualize_detections(display_frame, boxes, scores, 
                                                 self.screen_status, self.screen_confidence, 
                                                 self.arm_controller, self.font)
        
        display_frame = draw_info_on_frame(display_frame, self)
        
        return display_frame
    
    def run(self):
        """Run the main loop for TV tracking"""
        start_time = time.time()
        cap = cv2.VideoCapture(CAMERA_INDEX)
        while time.time() - start_time < 30:
            if cap.isOpened():
                is_opened = True
                break
            time.sleep(0.01)
        if is_opened:
            # Set camera parameters
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            log.info("Unable to open camera")
            cap.release()
            return
        
        log.info("Starting TV tracking client...")
        log.info("Press 'q' to quit | Press 'r' to reset robotic arm | Press 'd' to reconnect | Press 't' to test directions")
        
        # Initialize robotic arm to default position
        log.info("Initializing robotic arm to default position...")
        self.arm_controller.send_command(self.target_x, self.target_y, self.target_z)
        time.sleep(1)

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
            processed_frame = self.process_frame(frame)
            
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