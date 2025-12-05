# config.py

import os
import torch

# --- 环境变量/通用配置 ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决OpenMP冲突

# --- 模型文件路径 ---
MODEL_PATH = "model2.onnx"                 # RT-DETR模型路径
CLASSIFICATION_WEIGHTS = "Last_Epoch016.pth" # 分类模型权重路径

# --- 网络/硬件配置 ---
SERVER_IP = "192.168.51.180"               # 树莓派IP地址
SERVER_PORT = 6000                         # 树莓派端口
CAMERA_INDEX = 1                           # 摄像头索引
CLASSIFICATION_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 图像/检测配置 ---
IMAGE_WIDTH = 640                          # 图像宽度
IMAGE_HEIGHT = 480                         # 图像高度
FLIP_CAMERA = True                         # 是否翻转摄像头画面
INPUT_SIZE = (640, 640)                    # 模型输入尺寸 (W, H)
CONF_THRESHOLD = 0.6                       # 置信度阈值
TV_CLASS = 0                               # 电视类别索引

# --- 屏幕状态分类类别映射 ---
CLASS_MAP = {
    0: "red", 1: "green", 2: "black", 3: "white",
    4: "pink", 5: "blue", 6: "fault", 7: "snow"
}

# --- 机械臂控制参数 ---
SEND_INTERVAL = 0.5                        # 发送间隔 (s)
MOVE_THRESHOLD = 20                        # 像素偏移阈值 (px)

# --- 运动平滑参数 ---
SMOOTHING_FACTOR = 0.7
MAX_HISTORY = 5

# --- 搜索模式参数 ---
SEARCH_MOVE_RANGE = 10                     # 搜索时 X 轴移动范围
SEARCH_MOVE_SPEED = 0.2                    # 搜索时的移动速度因子
RESET_WAIT_TIME = 3.0                      # 丢失目标多久后进入搜索模式

BLUR_THRESHOLD = 120 