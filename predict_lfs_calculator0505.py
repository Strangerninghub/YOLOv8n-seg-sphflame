import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy import interpolate
import os
from pathlib import Path
import glob
import time
import torch
from ultralytics import YOLO
import argparse
import keyboard

"""
    该版本代码扩充了关于卡尔曼滤波的所有内容
    完善了图窗显示的全部内容，可使用左右键控制回放
"""


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 新增卡尔曼滤波类
class KalmanFilter:
    """
    卡尔曼滤波器用于平滑火焰半径数据
    """

    def __init__(self, process_noise=1e-5, measurement_noise=1e-1):
        # 状态向量: [半径, 半径变化率]
        self.state = np.zeros(2)
        self.covariance = np.eye(2)  # 状态协方差矩阵

        # 状态转移矩阵 (假设匀速运动)
        self.transition_matrix = np.array([[1, 1],
                                           [0, 1]])

        # 观测矩阵 (我们只能观测到半径)
        self.observation_matrix = np.array([[1, 0]])

        # 过程噪声协方差
        self.process_noise_cov = np.eye(2) * process_noise

        # 观测噪声协方差
        self.measurement_noise_cov = np.array([[measurement_noise]])

        # 初始化标志
        self.initialized = False

    def init(self, first_measurement):
        """初始化滤波器"""
        self.state = np.array([first_measurement, 0])  # [初始半径, 初始速度=0]
        self.covariance = np.eye(2) * 1.0  # 较大的初始不确定性
        self.initialized = True

    def predict(self):
        """预测步骤"""
        if not self.initialized:
            return

        # 状态预测
        self.state = self.transition_matrix @ self.state
        # 协方差预测
        self.covariance = (self.transition_matrix @ self.covariance @
                           self.transition_matrix.T + self.process_noise_cov)

    def update(self, measurement):
        """更新步骤"""
        if not self.initialized:
            self.init(measurement)
            return self.state[0]

        # 计算卡尔曼增益
        innovation_cov = (self.observation_matrix @ self.covariance @
                          self.observation_matrix.T + self.measurement_noise_cov)
        kalman_gain = (self.covariance @ self.observation_matrix.T @
                       np.linalg.inv(innovation_cov))

        # 状态更新
        innovation = measurement - self.observation_matrix @ self.state
        self.state = self.state + kalman_gain @ innovation

        # 协方差更新
        self.covariance = (np.eye(2) - kalman_gain @ self.observation_matrix) @ self.covariance

        return self.state[0]

    def filter_sequence(self, measurements):
        """对整个序列进行滤波"""
        if not measurements:
            return measurements

        filtered = []
        for meas in measurements:
            self.predict()
            filtered.append(self.update(meas))

        return filtered


class IntegratedFlameAnalyzer:
    def __init__(self, model_path, input_path, output_dir="./integrated_results",
                 min_radius_mm=8, max_radius_mm=25, show_processing=False,
                 fps=20000, input_type="auto", conf_threshold=0.5, contour_selection_ratio=0.6,
                 deal_gap=1, kalman_iterations=1, process_noise=1e-5, use_kalman_after_loess=True, measurement_noise=1e-1,
                 # 新增的数据清洗参数
                 window_frac=0.0011, threshold_factor=3, interpolation_method='spline',
                 loess_frac_rt=0.25, loess_frac_sbt=0.25,
                 # 新增：视窗检测方式选择参数
                 window_detection_method="yolo", manual_pixel_to_mm=None, traditional_kernel_size=18):
        """
        初始化火焰分析器
        """
        self.model_path = model_path
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 加载模型
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")

        # 存储计算结果的列表 - 这些将只包含有效范围内的数据
        self.frame_numbers = []
        self.flame_radii = []  # 像素单位
        self.flame_areas = []
        self.propagation_speeds = []
        self.stretch_rates = []

        # 存储每帧的处理结果用于可视化
        self.processed_frames = []
        self.original_frames = []

        # 物理参数
        self.fps = fps
        self.window_diameter_mm = 80  # 已知视窗直径为80mm
        self.pixel_to_mm = None  # 将在第一帧中计算
        self.default_pixel_to_mm = 91.0 / 768.0  # 默认比例尺 (768像素=91mm)

        # 火焰半径范围 (单位: mm)
        self.min_radius_mm = min_radius_mm
        self.max_radius_mm = max_radius_mm
        self.show_processing = show_processing

        # 窗口检测结果
        self.window_detected = False
        self.window_center = None
        self.window_radius_pixels = None
        self.use_default_ratio = False  # 标记是否使用默认比例尺

        # 确定输入类型
        self.input_type = self._determine_input_type(input_type)

        # 存储原始帧尺寸
        self.original_frame_size = None

        # 存储最终结果
        self.laminar_flame_speed_method1 = None  # 线性拟合
        self.laminar_flame_speed_method2 = None  # Chen非线性拟合
        self.laminar_flame_speed_method3 = None  # Kelly非线性拟合
        self.markstein_length_method1 = None
        self.markstein_length_method2 = None
        self.markstein_length_method3 = None

        # 轮廓选择比例
        self.contour_selection_ratio = contour_selection_ratio

        # 间隔读取参数
        self.deal_gap = deal_gap

        # 新增卡尔曼滤波参数
        self.kalman_iterations = kalman_iterations
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # 初始化卡尔曼滤波器
        self.kalman_filter = KalmanFilter(process_noise, measurement_noise)
        self.use_kalman_after_loess = use_kalman_after_loess

        # 数据清洗参数
        self.window_frac = window_frac
        self.threshold_factor = threshold_factor
        self.interpolation_method = interpolation_method
        self.loess_frac_rt = loess_frac_rt
        self.loess_frac_sbt = loess_frac_sbt

        # 存储所有原始数据（包括被剔除的）
        self.all_frame_numbers = []
        self.all_flame_radii = []
        self.all_flame_areas = []
        self.all_propagation_speeds = []
        self.all_stretch_rates = []

        # 视窗检测方式
        self.window_detection_method = window_detection_method  # "yolo", "traditional", "manual"
        self.manual_pixel_to_mm = manual_pixel_to_mm
        self.traditional_kernel_size = traditional_kernel_size

        # 确保检测方式变量被正确设置
        if self.window_detection_method == "manual":
            if self.manual_pixel_to_mm is not None:
                self.pixel_to_mm = self.manual_pixel_to_mm
                print(f"手动模式：使用用户指定比例尺 {self.pixel_to_mm:.6f} mm/像素")
            else:
                print("警告: 手动模式未提供 manual_pixel_to_mm，将使用默认比例尺")
                self.pixel_to_mm = self.default_pixel_to_mm
                self.use_default_ratio = True
            self.window_detected = True

        if self.window_detection_method == "manual":
            if self.manual_pixel_to_mm is not None:
                self.pixel_to_mm = self.manual_pixel_to_mm
                print(f"手动模式：使用用户指定比例尺 {self.pixel_to_mm:.6f} mm/像素")
            else:
                print("警告: 手动模式未提供比例尺，将使用默认比例尺")
                self.pixel_to_mm = self.default_pixel_to_mm
                self.use_default_ratio = True
            self.window_detected = True


        # 提取输入路径的最后一级文件夹名（用于输出文件命名）
        if os.path.isdir(self.input_path):
            self.last_folder_name = os.path.basename(os.path.normpath(self.input_path))
        else:
            # 如果是文件，则取父目录名
            self.last_folder_name = os.path.basename(os.path.dirname(os.path.normpath(self.input_path)))

    def export_fit_curves_to_excel(self):
        """生成三条拟合线的完整数据点并导出为 CSV，保存在 .\curve4origin 文件夹下"""
        import numpy as np
        from pathlib import Path

        curve_dir = Path("curve4origin")
        curve_dir.mkdir(exist_ok=True)

        input_path_raw = self.input_path.rstrip("/\\")
        folder_name = Path(input_path_raw).name
        if not folder_name:
            folder_name = "unknown"

        # ---------- 准备实验数据 ----------
        radii_cm = np.array(self.flame_radii_physical)   # cm
        speeds_cm_s = np.array(self.propagation_speeds_smoothed)   # cm/s

        E_exp = 2.0 * speeds_cm_s / radii_cm
        valid = (E_exp > 0) & (speeds_cm_s > 0) & (radii_cm > 0)
        E_exp = E_exp[valid]
        R_exp = radii_cm[valid]
        Sb_exp = speeds_cm_s[valid]

        # ---------- 拟合参数 ----------
        Sb0_1, Lb_1 = self.laminar_flame_speed_method1, self.markstein_length_method1
        Sb0_2, Lb_2 = self.laminar_flame_speed_method2, self.markstein_length_method2
        Sb0_3, Lb_3 = self.laminar_flame_speed_method3, self.markstein_length_method3

        N_pts = 1000

        # ---- 线性模型 ----
        if Lb_1 > 0:
            E_max1 = Sb0_1 / Lb_1
        else:
            E_max1 = np.max(E_exp) * 1.2
        E_lin = np.linspace(0, min(E_max1, np.max(E_exp) * 1.2), N_pts)
        Sb_lin = Sb0_1 - Lb_1 * E_lin
        mask = (E_lin >= 0) & (Sb_lin >= 0)
        E_lin, Sb_lin = E_lin[mask], Sb_lin[mask]

        # ---- Frankel ----
        if abs(Lb_2) > 1e-12:
            if Lb_2 > 0:
                Sb_f = np.linspace(Sb0_2, max(1e-4 * abs(Sb0_2), 1e-4), N_pts)
            else:
                Sb_max = max(np.max(Sb_exp) * 1.2, Sb0_2 * 1.5)
                Sb_f = np.linspace(Sb0_2, Sb_max, N_pts)
            E_f = (Sb0_2 * Sb_f - Sb_f ** 2) / (Sb0_2 * Lb_2)
            mask = (E_f >= 0) & (Sb_f > 0)
            E_f, Sb_f = E_f[mask], Sb_f[mask]
        else:
            E_f, Sb_f = np.array([]), np.array([])

        # ---- Kelly ----
        if abs(Lb_3) > 1e-12:
            if Lb_3 > 0:
                u = np.logspace(0, -4, N_pts)
            else:
                u_max = max(np.max(Sb_exp) / Sb0_3 * 1.2, 1.5)
                u = np.linspace(1, u_max, N_pts)
            E_k = -(Sb0_3 / Lb_3) * u ** 2 * np.log(u)
            Sb_k = u * Sb0_3
            mask = (E_k >= 0) & (Sb_k > 0)
            E_k, Sb_k = E_k[mask], Sb_k[mask]
        else:
            E_k, Sb_k = np.array([]), np.array([])

        # ---- 保存 CSV ----
        csv_exp = curve_dir / f"LFS_data_{folder_name}_exp.csv"
        csv_fit = curve_dir / f"LFS_data_{folder_name}_fit.csv"
        csv_params = curve_dir / f"LFS_data_{folder_name}_params.csv"

        # 实验数据
        np.savetxt(csv_exp,
                   np.column_stack([E_exp, R_exp, Sb_exp]),
                   header='E_exp(1/s),R_exp(cm),Sb_exp(cm/s)', delimiter=',', comments='')
        # 拟合曲线（补齐到相同长度）
        max_len = max(len(E_lin), len(E_f), len(E_k))
        def pad(arr, length):
            return np.append(arr, [np.nan] * (length - len(arr)))
        np.savetxt(csv_fit,
                   np.column_stack([pad(E_lin, max_len), pad(Sb_lin, max_len),
                                    pad(E_f, max_len), pad(Sb_f, max_len),
                                    pad(E_k, max_len), pad(Sb_k, max_len)]),
                   header='E_Linear,Sb_Linear,E_Frankel,Sb_Frankel,E_Kelly,Sb_Kelly',
                   delimiter=',', comments='')
        # 参数
        param_arr = np.array([[Sb0_1, Lb_1], [Sb0_2, Lb_2], [Sb0_3, Lb_3]])
        np.savetxt(csv_params, param_arr,
                   header='Sb0 (cm/s),Lb (cm)', delimiter=',', comments='')

        print(f"拟合数据已导出为 CSV 文件，保存于: {curve_dir}")


    def _determine_input_type(self, input_type):
        """确定输入类型"""
        if input_type != "auto":
            return input_type

        path = Path(self.input_path)
        if path.is_file():
            if path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                return "video"
            else:
                raise ValueError(f"不支持的视频格式: {path.suffix}")
        elif path.is_dir():
            image_files = list(path.glob("*.tif")) + list(path.glob("*.tiff")) + \
                          list(path.glob("*.jpg")) + list(path.glob("*.png"))
            if image_files:
                return "images"
            else:
                raise ValueError("目录中未找到支持的图片文件")
        else:
            raise ValueError("输入路径不存在")

    def detect_window(self, frame):
        """
        根据选择的检测方式检测视窗
        """
        if self.window_detection_method == "yolo":
            return self.detect_window_yolo(frame)
        elif self.window_detection_method == "traditional":
            return self.detect_window_traditional(frame)
        elif self.window_detection_method == "manual":
            return self.detect_window_manual()
        else:
            print(f"未知的视窗检测方式: {self.window_detection_method}，将改用YOLO检测")
            return self.detect_window_yolo(frame)

    def detect_window_yolo(self, frame):
        # 使用 YOLO 检测
        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False, device=self.device)
        result = results[0]
        window_detected = False
        window_center = None
        window_radius_pixels = None

        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                if class_id == 1:  # window 类
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    # 扩大 ROI 区域一点，确保窗口完整
                    margin = 10
                    roi_x1 = max(0, x1 - margin)
                    roi_y1 = max(0, y1 - margin)
                    roi_x2 = min(frame.shape[1], x2 + margin)
                    roi_y2 = min(frame.shape[0], y2 + margin)
                    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    if roi.size == 0:
                        continue
                    # 在 ROI 内使用传统方法寻找圆形
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    # 使用 Hough Circle 检测（参数可根据实际情况调整）
                    circles = cv2.HoughCircles(gray_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                               param1=50, param2=30, minRadius=10, maxRadius=0)
                    if circles is not None:
                        circles = np.round(circles[0, :]).astype("int")
                        # 选择半径最大的圆（通常窗口是最大的）
                        circles = sorted(circles, key=lambda c: c[2], reverse=True)
                        cx, cy, r = circles[0]
                        # 转换回原图坐标
                        window_center = (cx + roi_x1, cy + roi_y1)
                        window_radius_pixels = r
                    else:
                        # 回退到原始矩形框方法
                        window_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        window_radius_pixels = int((x2 - x1) / 2)
                    # 计算比例尺
                    self.pixel_to_mm = self.window_diameter_mm / (2 * window_radius_pixels)
                    window_detected = True
                    print(
                        f"YOLO方法视窗检测: 窗口中心{window_center}, 半径:{window_radius_pixels}像素, 比例尺:{self.pixel_to_mm:.6f}mm/像素")
                    break
        return window_detected, window_center, window_radius_pixels

    def detect_window_traditional(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 高斯模糊降噪
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            # 使用 Canny 边缘检测
            edges = cv2.Canny(blurred, 50, 150)
            # 霍夫圆检测
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                       param1=50, param2=30, minRadius=50, maxRadius=300)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # 选择半径最大的圆（窗口）
                largest_circle = max(circles, key=lambda c: c[2])
                x, y, r = largest_circle
                window_center = (int(x), int(y))
                window_radius_pixels = int(r)
            else:
                # 回退到原轮廓方法
                _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
                kernel = np.ones((self.traditional_kernel_size, self.traditional_kernel_size), np.uint8)
                opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return False, None, None
                largest_contour = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                window_center = (int(x), int(y))
                window_radius_pixels = int(radius)
            self.pixel_to_mm = self.window_diameter_mm / (2 * window_radius_pixels)
            print(
                f"传统形态学视窗检测: 中心{window_center}, 半径{window_radius_pixels}像素, 比例尺{self.pixel_to_mm:.6f}mm/像素")
            return True, window_center, window_radius_pixels
        except Exception as e:
            print(f"传统视窗检测失败: {e}")
            return False, None, None

    def detect_window_manual(self):
        """
        手动输入比例尺
        """

        # 设置检测方式
        self.window_detection_method = "manual"

        if self.manual_pixel_to_mm is not None:
            self.pixel_to_mm = self.manual_pixel_to_mm
            print(f"手动设置像素到毫米转换系数: {self.pixel_to_mm:.6f} mm/像素")
            return True, None, None  # 手动方式没有具体的中心点和半径
        else:
            print("错误: 选择了手动方式但未提供manual_pixel_to_mm参数")
            return False, None, None


    def extract_flame_contour(self, frame):
        """
        从分割帧中提取火焰轮廓 - 使用可调节比例的顶部和底部点进行圆拟合
        """
        # 使用模型进行预测
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device
        )

        result = results[0]

        # 获取分割结果图像
        seg_img = result.plot()

        # 查找火焰类 (class_id=0)
        flame_detected = False
        flame_radius = 0
        flame_area = 0
        flame_center = None
        all_contour_points = None  # 存储所有轮廓点
        selected_points = None  # 存储用于拟合的点

        if result.masks is not None:
            # 找到置信度最高的火焰实例
            max_confidence = 0
            best_mask_idx = -1

            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                if class_id == 0:  # flame类
                    confidence = box.conf[0].cpu().numpy()
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_mask_idx = i

            if 0 <= best_mask_idx < len(result.masks.data):
                # 获取最佳火焰的mask
                flame_mask = result.masks.data[best_mask_idx].cpu().numpy()

                # 确保数据类型正确并转换为uint8
                if flame_mask.dtype == np.uint16:
                    flame_mask = (flame_mask / 256).astype(np.uint8)  # 16位转8位
                elif flame_mask.dtype == np.float32 or flame_mask.dtype == np.float64:
                    flame_mask = (flame_mask * 255).astype(np.uint8)
                else:
                    flame_mask = (flame_mask * 255).astype(np.uint8)

                # 调整mask大小与原图匹配
                flame_mask_resized = cv2.resize(flame_mask, (frame.shape[1], frame.shape[0]))

                # 对mask进行二值化处理
                _, binary_mask = cv2.threshold(flame_mask_resized, 127, 255, cv2.THRESH_BINARY)

                # 改进的形态学处理：先填充空洞，再进行闭运算
                # 1. 填充空洞
                filled_mask = self.fill_holes(binary_mask)

                # 2. 形态学闭运算：先膨胀后腐蚀，填充小孔洞和连接断开的边界
                kernel_size = 29  # 可以根据实际情况调整核大小
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

                # 执行闭运算
                closed_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)

                # 3. 可选：再进行一次开运算去除小噪声（先腐蚀后膨胀）
                closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

                # 查找轮廓 - 使用与OpenCV 3.4兼容的方法
                contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # 选择最大的轮廓作为火焰
                    main_contour = max(contours, key=cv2.contourArea)

                    # 将轮廓点转换为二维数组
                    contour_points = main_contour.reshape(-1, 2)
                    all_contour_points = contour_points  # 保存所有轮廓点

                    if len(contour_points) >= 5:
                        # 按y坐标排序
                        sorted_points = contour_points[np.argsort(contour_points[:, 1])]

                        # 计算选择点的数量 (顶部和底部各选择 contour_selection_ratio/2 比例的点)
                        n_points = len(sorted_points)
                        n_select = int((self.contour_selection_ratio / 2) * n_points)

                        # 确保至少选择1个点
                        n_select = max(1, n_select)

                        # 选择顶部和底部的点
                        top_points = sorted_points[:n_select]  # 顶部
                        bottom_points = sorted_points[-n_select:]  # 底部

                        # 合并选中的点
                        selected_points = np.vstack([top_points, bottom_points])

                        # 使用选中的点进行圆拟合
                        (x, y), radius_fit = cv2.minEnclosingCircle(
                            selected_points.reshape(-1, 1, 2).astype(np.float32)
                        )

                        flame_center = (int(x), int(y))
                        flame_radius = radius_fit

                        # 使用拟合的半径计算球形表面积
                        flame_area = 4 * np.pi * (flame_radius ** 2)

                        flame_detected = True
                    else:
                        # 如果点数太少，回退到使用整个轮廓
                        (x, y), radius_fit = cv2.minEnclosingCircle(main_contour)
                        flame_center = (int(x), int(y))
                        flame_radius = radius_fit
                        flame_area = 4 * np.pi * (flame_radius ** 2)
                        flame_detected = True
                        selected_points = contour_points  # 使用所有点作为选中的点

        return flame_radius, flame_area, flame_center, flame_radius, seg_img, flame_detected, all_contour_points, selected_points

    def fill_holes(self, binary_mask):
        """
        填充二值图像中的空洞
        """
        # 复制mask
        mask_copy = binary_mask.copy()

        # 创建一个稍大的内核用于填充
        h, w = mask_copy.shape[:2]

        # 创建一个临时图像用于洪水填充
        temp_mask = np.zeros((h + 2, w + 2), np.uint8)

        # 从边界点开始填充背景
        cv2.floodFill(mask_copy, temp_mask, (0, 0), 255)

        # 反转图像，这样空洞就变成了前景
        holes = cv2.bitwise_not(mask_copy)

        # 合并原始mask和填充的空洞
        filled_mask = binary_mask | holes

        return filled_mask


    # 新增卡尔曼滤波应用函数
    def apply_kalman_filter(self, radii, iterations):
        """
        应用卡尔曼滤波平滑半径数据
        """
        if len(radii) < 3 or iterations < 1:
            return radii

        print(f"应用卡尔曼滤波，迭代次数: {iterations}")

        filtered_radii = radii.copy()

        for i in range(iterations):
            # 每次迭代都重新初始化滤波器
            self.kalman_filter = KalmanFilter(self.process_noise, self.measurement_noise)
            filtered_radii = self.kalman_filter.filter_sequence(filtered_radii)

            print(f"卡尔曼滤波第{i + 1}次迭代完成")
            print(f"===== 数据清洗完成 =====\n")

        return filtered_radii


    def calculate_propagation_speed(self, radii_cm, time_interval):
        """计算火焰传播速度 - 单位: cm/s"""
        speeds = []
        for i in range(1, len(radii_cm)):
            # drb/dt (mm/s)，然后转换为 cm/s
            dr_dt = (radii_cm[i] - radii_cm[i - 1]) / time_interval  # cm/s
            speeds.append(dr_dt)  # cm/s

        if len(speeds) > 0:
            speeds.insert(0, speeds[0])
        else:
            speeds = [0]

        return speeds

    def calculate_stretch_rate(self, radii_mm, areas, time_interval):
        """计算拉伸率 - 单位: 1/s"""
        stretch_rates = []

        for i in range(1, len(areas)):
            if areas[i] > 0 and radii_mm[i] > 0:
                dA_dt = (areas[i] - areas[i - 1]) / time_interval
                alpha = (1.0 / areas[i]) * dA_dt
                stretch_rates.append(alpha)
            else:
                stretch_rates.append(0)

        if len(stretch_rates) > 0:
            stretch_rates.insert(0, stretch_rates[0])
        else:
            stretch_rates = [0]

        return stretch_rates

    def calculate_all_propagation_speeds(self):
        """计算所有数据的传播速度"""
        if len(self.all_flame_radii) < 2:
            return [0]

        # 转换为物理单位
        all_radii_mm = [r * self.pixel_to_mm for r in self.all_flame_radii]

        # 关键修改：所有数据是连续帧，时间间隔为 1/fps
        # 因为all_frame_numbers存储的是原始帧号
        time_interval = self.deal_gap / self.fps

        # 计算传播速度
        speeds = []
        for i in range(1, len(all_radii_mm)):
            dr_dt = (all_radii_mm[i] - all_radii_mm[i - 1]) / time_interval  # mm/s
            speeds.append(dr_dt / 10.0)  # 转换为 cm/s

        if len(speeds) > 0:
            speeds.insert(0, speeds[0])
        else:
            speeds = [0]

        return speeds


    def method1_linear_fit(self, stretch_rates_1s, propagation_speeds_cms):
        """
        方法1: 线性拟合法 Sb = Sb0 - Lb * α
        使用平滑后的数据进行拟合
        """
        # 筛选有效数据点 - 使用平滑后的数据
        valid_data = [(alpha, sb) for alpha, sb in zip(stretch_rates_1s, propagation_speeds_cms)
                      if alpha > 0 and sb > 0 and not np.isnan(alpha) and not np.isnan(sb)]

        if len(valid_data) < 2:
            print("方法1: 有效数据点不足")
            return 0, 0

        alpha_valid, sb_valid = zip(*valid_data)

        # 线性拟合: Sb = Sb0 - Lb * α
        try:
            coeffs = np.polyfit(alpha_valid, sb_valid, 1)
            Sb0 = coeffs[1]  # 截距即为Sb0
            Lb = -coeffs[0]  # 斜率为 -Lb

            print(f"方法1 (线性拟合) - Sb0: {Sb0:.2f} cm/s, Lb: {Lb:.4f} cm")
            return Sb0, Lb
        except Exception as e:
            print(f"方法1拟合失败: {e}")
            return 0, 0

    def method2_nonlinear_fit(self, radii_cm, propagation_speeds_cms):
        """
        方法2: 陈正非线性拟合法 Sb = Sb0 - Sb0 * Lb * 2 / Rf
        Sb: 测量的传播速度
        Sb0: 待求的无拉伸火焰速度
        """
        # 筛选有效数据点
        valid_indices = [i for i in range(len(radii_cm))
                         if radii_cm[i] > 0 and propagation_speeds_cms[i] > 0
                         and not np.isnan(radii_cm[i]) and not np.isnan(propagation_speeds_cms[i])]

        if len(valid_indices) < 3:
            print("方法2: 有效数据点不足")
            return 0, 0

        radii_valid = np.array([radii_cm[i] for i in valid_indices])
        speeds_valid = np.array([propagation_speeds_cms[i] for i in valid_indices])

        def model2(rb, Sb0, Lb):
            # Sb = Sb0 - Sb0 * Lb * 2 / Rf
            return Sb0 - Sb0 * Lb * 2 / rb

        try:
            # 初始猜测
            initial_Sb0 = np.max(speeds_valid) * 1.2  # Sb0应该大于所有测量值
            initial_Lb = 0.01

            # 边界条件
            bounds = ([0.1, -100000], [np.max(speeds_valid) * 3, 100000])

            popt, pcov = curve_fit(model2, radii_valid, speeds_valid,
                                   p0=[initial_Sb0, initial_Lb],
                                   bounds=bounds, maxfev=5000)

            Sb0 = popt[0]
            Lb = popt[1]

            print(f"方法2 (Chen非线性) - Sb0: {Sb0:.2f} cm/s, Lb: {Lb:.4f} cm")
            return Sb0, Lb
        except Exception as e:
            print(f"方法2拟合失败: {e}")
            return 0, 0

    def method3_chen_nonlinear_extrapolation(self, radii_cm, propagation_speeds_cms):
        """
        方法3: 非线性外推法 ln(Sb) = ln(Sb0) - Sb0 * Lb * 2/(Rf * Sb)
        隐式方程: Sb = Sb0 * exp(-Sb0 * Lb * 2/(Rf * Sb))
        Sb: 测量的传播速度
        Sb0: 待求的无拉伸火焰速度
        """
        valid_indices = [i for i in range(len(radii_cm))
                         if radii_cm[i] > 0 and propagation_speeds_cms[i] > 0
                         and not np.isnan(radii_cm[i]) and not np.isnan(propagation_speeds_cms[i])]

        if len(valid_indices) < 3:
            print("方法3: 有效数据点不足")
            return 0, 0

        radii_valid = np.array([radii_cm[i] for i in valid_indices])
        speeds_valid = np.array([propagation_speeds_cms[i] for i in valid_indices])

        def chen_model(rb, Sb0, Lb):
            Sb_values = []
            for r in rb:
                # 使用迭代法求解隐式方程
                # 初始猜测使用测量值Sb，而不是Sb0
                Sb_guess = np.mean(speeds_valid)  # 使用平均测量值作为初始猜测

                for iteration in range(100):
                    # 隐式方程: Sb = Sb0 * exp(-2 * Sb0 * Lb / (r * Sb))
                    if r * Sb_guess == 0:
                        break

                    exponent = -2 * Sb0 * Lb / (r * Sb_guess)

                    # 防止数值溢出
                    if exponent > 100:
                        exponent = 100
                    elif exponent < -100:
                        exponent = -100

                    Sb_new = Sb0 * np.exp(exponent)

                    # 检查收敛
                    if abs(Sb_guess - Sb_new) < 1e-8:
                        Sb_guess = Sb_new
                        break

                    Sb_guess = Sb_new

                Sb_values.append(Sb_guess)
            return np.array(Sb_values)

        try:
            # 初始猜测
            initial_Sb0 = np.max(speeds_valid) * 1.2
            initial_Lb = 0.01

            # 边界条件
            bounds = ([np.max(speeds_valid) * 0.8, -100000],
                      [np.max(speeds_valid) * 5, 100000])

            popt, pcov = curve_fit(chen_model, radii_valid, speeds_valid,
                                   p0=[initial_Sb0, initial_Lb],
                                   bounds=bounds, maxfev=10000)

            Sb0 = popt[0]
            Lb = popt[1]

            print(f"方法3 (Kelly非线性) - Sb0: {Sb0:.2f} cm/s, Lb: {Lb:.4f} cm")
            return Sb0, Lb

        except Exception as e:
            print(f"方法3拟合失败: {e}")
            # 备选方案：使用更简单的初始猜测
            try:
                initial_Sb0 = np.mean(speeds_valid) * 1.5
                initial_Lb = 0.001
                bounds = ([0.1, 0], [np.max(speeds_valid) * 10, 1])

                popt, pcov = curve_fit(chen_model, radii_valid, speeds_valid,
                                       p0=[initial_Sb0, initial_Lb],
                                       bounds=bounds, maxfev=10000)
                Sb0 = popt[0]
                Lb = popt[1]
                print(f"方法3 (备选拟合) - Sb0: {Sb0:.6f} cm/s, Lb: {Lb:.6f} cm")
                return Sb0, Lb
            except Exception as e2:
                print(f"方法3第二次尝试也失败: {e2}")
                return 0, 0


    # 改进清洗函数
    def clean_data_improved(self, radii, areas, frame_numbers,
                            window_frac=0.0011, threshold_factor=3,
                            interpolation_method='spline'):
        """
        改进的数据清洗函数
        - 使用移动中位数检测离群值
        - 使用插值填充离群值
        """
        if len(radii) < 5:
            return radii, areas, frame_numbers

        radii_arr = np.array(radii)
        areas_arr = np.array(areas)
        frame_arr = np.array(frame_numbers)

        # 1. 移动中位数离群值检测
        window_size = max(3, int(len(radii_arr) * window_frac))
        if window_size % 2 == 0:
            window_size += 1  # 确保窗口大小为奇数

        print(f"===== 数据清洗参数 =====")
        print(f"移动中位数窗口大小: {window_size}")

        # 计算移动中位数
        median_filtered = []
        for i in range(len(radii_arr)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(radii_arr), i + window_size // 2 + 1)
            window_data = radii_arr[start_idx:end_idx]
            median_filtered.append(np.median(window_data))

        median_filtered = np.array(median_filtered)

        # 计算残差和动态阈值
        residuals = np.abs(radii_arr - median_filtered)
        mad = np.median(residuals)  # 中位数绝对偏差
        threshold = threshold_factor * 1.4826 * mad  # 转换为标准差估计

        outliers = residuals > threshold
        print(f"检测到 {np.sum(outliers)} 个离群值 (阈值: {threshold:.4f})\n")

        # 2. 插值填充离群值
        if np.any(outliers):
            valid_indices = np.where(~outliers)[0]

            if len(valid_indices) > 3:
                if interpolation_method == 'spline':
                    # 三次样条插值
                    try:
                        radius_interp = interpolate.UnivariateSpline(
                            frame_arr[valid_indices],
                            radii_arr[valid_indices],
                            s=0.5 * len(valid_indices)
                        )
                        area_interp = interpolate.UnivariateSpline(
                            frame_arr[valid_indices],
                            areas_arr[valid_indices],
                            s=0.5 * len(valid_indices)
                        )
                    except:
                        # 样条失败时回退到线性插值
                        radius_interp = interpolate.interp1d(
                            frame_arr[valid_indices],
                            radii_arr[valid_indices],
                            kind='linear',
                            fill_value='extrapolate'
                        )
                        area_interp = interpolate.interp1d(
                            frame_arr[valid_indices],
                            areas_arr[valid_indices],
                            kind='linear',
                            fill_value='extrapolate'
                        )
                else:
                    # 线性插值
                    radius_interp = interpolate.interp1d(
                        frame_arr[valid_indices],
                        radii_arr[valid_indices],
                        kind='linear',
                        fill_value='extrapolate'
                    )
                    area_interp = interpolate.interp1d(
                        frame_arr[valid_indices],
                        areas_arr[valid_indices],
                        kind='linear',
                        fill_value='extrapolate'
                    )

                # 填充离群值
                radii_arr[outliers] = radius_interp(frame_arr[outliers])
                areas_arr[outliers] = area_interp(frame_arr[outliers])

        return radii_arr.tolist(), areas_arr.tolist(), frame_arr.tolist()

    def loess_smooth(self, x, y, frac=0.25, it=3):
        """
        局部二次回归(Loess)平滑
        """
        if len(x) < 5:
            return y

        try:
            # 使用statsmodels的lowess函数
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(y, x, frac=frac, it=it, return_sorted=False)
            return smoothed
        except ImportError:
            print("警告: 未找到statsmodels，使用移动平均代替")
            # 简单的移动平均作为备选
            window_size = max(3, int(len(x) * frac))
            if window_size % 2 == 0:
                window_size += 1

            from scipy.signal import savgol_filter
            return savgol_filter(y, window_size, 2)


    def create_display_frame(self, original_frame, seg_frame, radius, center,
                             frame_count, total_frames, is_valid=False,
                             all_contour_points=None, selected_points=None):
        """
        创建显示帧，包含原始图像、分割结果和拟合结果
        """
        # 创建显示画布
        display_frame = original_frame.copy()

        # 绘制分割结果（半透明叠加）
        seg_resized = cv2.resize(seg_frame, (display_frame.shape[1], display_frame.shape[0]))
        display_frame = cv2.addWeighted(display_frame, 0.7, seg_resized, 0.3, 0)

        # 绘制所有轮廓点（红色）
        if all_contour_points is not None:
            for point in all_contour_points:
                x, y = int(point[0]), int(point[1])
                cv2.circle(display_frame, (x, y), 1, (0, 0, 255), -1)  # 白色小圆点 BGR

        # 绘制用于拟合的点（绿色）
        if selected_points is not None:
            for point in selected_points:
                x, y = int(point[0]), int(point[1])
                cv2.circle(display_frame, (x, y), 1, (0, 255, 0), -1)  # 绿色稍大的点

        # 绘制拟合的圆（白色）
        if center is not None and radius > 0:
            cv2.circle(display_frame, center, int(radius), (255, 255, 255), 1)  # 蓝色圆

            # 显示半径信息
            if self.pixel_to_mm is not None:
                radius_mm = radius * self.pixel_to_mm
                radius_text = f"Radius: {radius_mm:.2f} mm"
                cv2.putText(display_frame, radius_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示帧号
        frame_text = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(display_frame, frame_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示进度
        progress = (frame_count / total_frames) * 100
        progress_text = f"Progress: {progress:.1f}%"
        cv2.putText(display_frame, progress_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示半径范围
        range_text = f"Range: {self.min_radius_mm}-{self.max_radius_mm} mm"
        cv2.putText(display_frame, range_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示轮廓选择比例
        contour_ratio_text = f"Contour Ratio: {self.contour_selection_ratio}"
        cv2.putText(display_frame, contour_ratio_text, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示间隔参数
        gap_text = f"Deal Gap: {self.deal_gap}"
        cv2.putText(display_frame, gap_text, (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示是否在有效范围内
        if is_valid:
            status_text = "Status: VALID"
            color = (0, 255, 0)
        else:
            status_text = "Status: INVALID"
            color = (0, 0, 255)

        cv2.putText(display_frame, status_text, (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 绘制窗口检测结果
        if self.window_detected and self.window_center is not None and self.window_radius_pixels is not None:
            if self.window_detection_method == "yolo":
                # 绘制矩形检测框（红色）
                x1 = self.window_center[0] - self.window_radius_pixels
                y1 = self.window_center[1] - self.window_radius_pixels
                x2 = self.window_center[0] + self.window_radius_pixels
                y2 = self.window_center[1] + self.window_radius_pixels
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 绘制拟合的圆（绿色）
                cv2.circle(display_frame, self.window_center, self.window_radius_pixels, (0, 255, 0), 2)
                cv2.putText(display_frame, "Window (AI)",
                            (self.window_center[0] - 30, self.window_center[1] - self.window_radius_pixels - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif self.window_detection_method == "traditional":
                # 绘制黄色圆
                cv2.circle(display_frame, self.window_center, self.window_radius_pixels, (0, 255, 255), 2)
                cv2.putText(display_frame, "Window",
                            (self.window_center[0] - 30, self.window_center[1] - self.window_radius_pixels - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            # manual 模式不绘制

        # 显示视窗检测方式
        detection_method_text = f"Radius detection: "
        if self.window_detection_method == "yolo":
            detection_method_text += "YOLO"
        elif self.window_detection_method == "traditional":
            detection_method_text += "Traditional"
        elif self.window_detection_method == "manual":
            detection_method_text += "Defined"
        else:
            detection_method_text += "Unknown"

        #cv2.putText(display_frame, detection_method_text, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 如果使用默认比例尺，显示警告（保留这个，但只在真正使用默认时显示）
        if self.use_default_ratio and self.window_detection_method != "manual":
            warning_text = "WARNING: Using default ratio"
            cv2.putText(display_frame, warning_text, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 添加图例说明
        legend_text0 = "Legend: "
        legend_text1 = "Red: Unselected points"
        legend_text2 = "Green: Selected points"
        legend_text3 = "White: Fitted circle"

        cv2.putText(display_frame, legend_text0, (10, display_frame.shape[0] - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(display_frame, legend_text1, (10, display_frame.shape[0] - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(display_frame, legend_text2, (10, display_frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(display_frame, legend_text3, (10, display_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return display_frame

    def display_processing_frame(self, display_frame):
        """
        显示处理帧，不控制延迟，让算力决定帧率
        """
        # 创建窗口并设置大小
        cv2.namedWindow('Flame Analysis', cv2.WINDOW_NORMAL)
        if self.original_frame_size:
            cv2.resizeWindow('Flame Analysis', self.original_frame_size[1], self.original_frame_size[0])
        else:
            cv2.resizeWindow('Flame Analysis', 1000, 1000)

        cv2.imshow('Flame Analysis', display_frame)

        # 使用最小延迟，让算力决定帧率
        key = cv2.waitKey(1) & 0xFF

        # 按ESC键提前退出处理
        if key == 27:
            return False

        return True

    def play_processed_frames(self):
        """
        循环播放处理后的帧，支持键盘控制
        - 空格键：暂停/继续播放
        - 左箭头：后退一帧
        - 右箭头：前进一帧
        - ESC键：退出播放
        """
        if not self.processed_frames:
            print("没有可播放的处理帧")
            return

        print("开始播放处理结果")
        print("控制说明:")
        print("  - 空格键: 暂停/继续播放")
        print("  - 左箭头: 后退一帧")
        print("  - 右箭头: 前进一帧")
        print("  - ESC键: 退出播放")
        show_overlay = True  # 新增：默认显示标注
        print("  - H键: 清除/展示信息")

        # 设置播放帧率
        if self.input_type == "video":
            delay = int(1000 / self.fps)  # 使用视频自带帧率
        else:
            delay = 33  # 图片序列使用30fps (1000ms/30 ≈ 33ms)

        # 创建窗口并设置大小
        cv2.namedWindow('Flame Analysis Results', cv2.WINDOW_NORMAL)
        if self.original_frame_size:
            cv2.resizeWindow('Flame Analysis Results', self.original_frame_size[1], self.original_frame_size[0])
        else:
            cv2.resizeWindow('Flame Analysis Results', 1000, 1000)

        current_frame = 0
        total_frames = len(self.processed_frames)
        paused = False
        show_overlay = True  # 是否显示标注信息，默认显示
        loop_playback = True  # 是否循环播放

        while loop_playback:
            i = current_frame  # 使用单独的变量控制当前帧
            last_frame_time = time.time()

            while i < total_frames:
                # 显示当前帧（根据 show_overlay 选择原图或标注图）
                if show_overlay:
                    frame = self.processed_frames[i].copy()
                else:
                    frame = self.original_frames[i].copy()

                if paused:
                    status_text = "PAUSED (Press SPACE to continue)"
                    cv2.putText(frame, status_text, (240, frame.shape[0] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow('Flame Analysis Results', frame)

                # 使用非阻塞等待，以便同时检测 keyboard 按键
                key = cv2.waitKey(delay) & 0xFF

                # 检测 keyboard 库的按键
                if keyboard.is_pressed('c'):
                    print(f"C键触发......")
                    # 这里添加C键的功能
                    # time.sleep(0.2)

                if keyboard.is_pressed('left') or keyboard.is_pressed('up'):
                    i = max(0, i - 1)
                    current_frame = i
                    paused = True
                    # time.sleep(0.02)
                    continue

                if keyboard.is_pressed('right') or keyboard.is_pressed('down'):
                    i = min(total_frames - 1, i + 1)
                    current_frame = i
                    paused = True
                    # time.sleep(0.02)
                    continue

                # 处理 OpenCV 检测到的按键
                if key == 27:  # ESC键 - 退出
                    cv2.destroyAllWindows()
                    return
                elif key == 32:  # 空格键 - 暂停/继续
                    paused = not paused
                    # time.sleep(0.02)

                # 如果不是暂停状态，自动前进到下一帧
                if not paused:
                    current_time = time.time()
                    if current_time - last_frame_time >= 1.0 / self.fps:
                        i += 1
                        current_frame = i
                        last_frame_time = current_time
                else:
                    # 暂停状态下，保持当前帧
                    current_frame = i

                # H键
                if keyboard.is_pressed('h'):
                    show_overlay = not show_overlay
                    print(f"H键触发: 标注显示 {'开启' if show_overlay else '关闭'}")
                    time.sleep(0.2)  # 防止按键重复触发



            # 播放完一轮后询问是否继续循环
            if loop_playback:
                print("播放完成，按任意键继续循环播放，按ESC退出...")

                key = cv2.waitKey(0) & 0xFF
                if key == 27:  # ESC键
                    loop_playback = False
                else:
                    current_frame = 0  # 重置到第一帧
                    paused = False  # 重置播放状态

        cv2.destroyAllWindows()

    def process_video_file(self):
        """处理视频文件"""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.input_path}")

        # 获取视频属性
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps > 0:
            self.fps = video_fps
            print(f"使用视频自带帧率: {self.fps} fps")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        time_interval = 1.0 / self.fps

        print(f"视频信息: {total_frames} 帧, 帧率: {self.fps} fps")
        print(f"间隔处理参数: {self.deal_gap} (每{self.deal_gap}帧处理1帧)")

        frame_count = 0
        processed_count = 0

        # 存储有效数据的临时列表 - 只包含半径范围内的数据
        valid_frame_numbers = []
        valid_flame_radii = []
        valid_flame_areas = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 根据间隔参数跳过帧
            if (frame_count - 1) % self.deal_gap != 0:
                continue

            processed_count += 1

            # 记录原始帧尺寸
            if self.original_frame_size is None:
                self.original_frame_size = frame.shape[:2]

            # 在第一帧检测窗口 - 使用选择的检测方式
            if processed_count == 1 and not self.window_detected:
                if self.window_detection_method == "manual":
                    # 比例尺已在 __init__ 中设置，此处只需标记已检测
                    self.window_detected = True
                    if self.pixel_to_mm is not None:
                        print(f"手动模式: 使用比例尺 {self.pixel_to_mm:.6f} mm/像素")
                    else:
                        print("错误：手动模式下比例尺未正确初始化，请检查输入参数")
                        raise ValueError("pixel_to_mm is None in manual mode")
                else:
                    self.window_detected, self.window_center, self.window_radius_pixels = self.detect_window(frame)

                if not self.window_detected  and self.window_detection_method != "manual":
                    print("警告: 第一帧未检测到视窗，使用默认比例尺")
                    self.pixel_to_mm = self.default_pixel_to_mm
                    self.use_default_ratio = True
                    print(f"使用默认像素到毫米转换系数: {self.pixel_to_mm:.6f} mm/像素")

            if processed_count % 10 == 0:
                print(f"处理进度: {processed_count}/{total_frames // self.deal_gap}")

            # 提取火焰轮廓和计算半径
            radius, area, center, _, seg_frame, flame_detected, all_contour_points, selected_points = self.extract_flame_contour(
                frame)


            # 添加保存所有数据的代码：
            if flame_detected:
                # 无论是否在有效范围内，都保存数据
                self.all_frame_numbers.append(frame_count)
                self.all_flame_radii.append(radius)
                self.all_flame_areas.append(area)


            # 转换为物理单位 (mm)
            if self.pixel_to_mm is not None:
                radius_mm = radius * self.pixel_to_mm
            else:
                radius_mm = 0

            # 检查是否在有效半径范围内 - 严格筛选
            is_valid = (self.min_radius_mm <= radius_mm <= self.max_radius_mm and
                        flame_detected and radius_mm > 0)

            if is_valid:
                valid_frame_numbers.append(frame_count)
                valid_flame_radii.append(radius)  # 存储像素单位的半径
                valid_flame_areas.append(area)

            # 创建显示帧
            display_frame = self.create_display_frame(
                frame, seg_frame, radius, center,
                frame_count, total_frames, is_valid,
                all_contour_points, selected_points
            )

            # 存储处理后的帧用于后续播放
            self.original_frames.append(frame.copy())
            self.processed_frames.append(display_frame)

            # 显示处理过程
            if self.show_processing:
                continue_processing = self.display_processing_frame(display_frame)
                if not continue_processing:
                    print("用户中断处理过程")
                    break

        cap.release()

        print(f"处理帧数: {processed_count}")
        print(f"有效帧数: {len(valid_flame_radii)}\n")


        # 数据清洗 - 使用改进的方法
        if len(valid_flame_radii) > 0:
            cleaned_radii, cleaned_areas, cleaned_frame_numbers = self.clean_data_improved(
                valid_flame_radii, valid_flame_areas, valid_frame_numbers,
                window_frac=self.window_frac,
                threshold_factor=self.threshold_factor,
                interpolation_method=self.interpolation_method
            )

            # 应用卡尔曼滤波（如果启用）
            if self.kalman_iterations > 0:
                cleaned_radii = self.apply_kalman_filter(cleaned_radii, self.kalman_iterations)

            # 转换为物理单位并再次确认在范围内
            final_frame_numbers = []
            final_flame_radii = []
            final_flame_areas = []

            for i in range(len(cleaned_radii)):
                radius_mm = cleaned_radii[i] * self.pixel_to_mm
                if self.min_radius_mm <= radius_mm <= self.max_radius_mm:
                    final_frame_numbers.append(cleaned_frame_numbers[i])
                    final_flame_radii.append(cleaned_radii[i])
                    final_flame_areas.append(cleaned_areas[i])

            print(f"数据清洗后最终有效帧数: {len(self.flame_radii)}")
        else:
            self.frame_numbers = []
            self.flame_radii = []
            self.flame_areas = []

        return total_frames, len(self.flame_radii)

    def process_image_folder(self):
        """处理图片文件夹"""
        path = Path(self.input_path)

        # 获取所有图片文件
        image_files = list(path.glob("*.tif")) + list(path.glob("*.tiff")) + \
                      list(path.glob("*.jpg")) + list(path.glob("*.png"))

        if not image_files:
            raise ValueError("目录中未找到支持的图片文件")

        # 按文件名排序
        image_files.sort()
        total_frames = len(image_files)
        time_interval = 1.0 / self.fps

        print(f"图片序列信息: {total_frames} 帧")
        print(f"间隔处理策略: 每{self.deal_gap}帧处理1帧")

        frame_count = 0
        processed_count = 0

        # 存储有效数据的临时列表 - 只包含半径范围内的数据
        valid_frame_numbers = []
        valid_flame_radii = []
        valid_flame_areas = []

        for i, image_file in enumerate(image_files):
            frame_count += 1

            # 根据间隔参数跳过帧
            if (i) % self.deal_gap != 0:
                continue

            processed_count += 1

            # 读取图片 - 使用与OpenCV 3.4兼容的方法
            frame = cv2.imread(str(image_file))
            if frame is None:
                print(f"无法读取图片: {image_file}")
                continue

            # 确保图像数据类型为uint8
            if frame.dtype == np.uint16:
                frame = (frame / 256).astype(np.uint8)  # 将16位转换为8位

            # 将单通道灰度图转换为三通道图像（YOLO模型需要3通道输入）
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # 记录原始帧尺寸
            if self.original_frame_size is None:
                self.original_frame_size = frame.shape[:2]

            # 在第一帧检测窗口 - 使用选择的检测方式
            if processed_count == 1 and not self.window_detected:
                if self.window_detection_method == "manual":
                    # 比例尺已在 __init__ 中设置，此处只需标记已检测
                    self.window_detected = True
                    if self.pixel_to_mm is not None:
                        print(f"手动模式: 使用比例尺 {self.pixel_to_mm:.6f} mm/像素")
                    else:
                        print("错误：手动模式下比例尺未正确初始化，请检查输入参数")
                        raise ValueError("pixel_to_mm is None in manual mode")
                else:
                    self.window_detected, self.window_center, self.window_radius_pixels = self.detect_window(frame)

                if not self.window_detected and self.window_detection_method != "manual":
                    print("警告: 第一帧未检测到视窗，使用默认比例尺")
                    self.pixel_to_mm = self.default_pixel_to_mm
                    self.use_default_ratio = True
                    print(f"使用默认像素到毫米转换系数: {self.pixel_to_mm:.6f} mm/像素")

            if processed_count % 10 == 0:
                print(f"处理进度: {processed_count}/{total_frames // self.deal_gap}")

            # 提取火焰轮廓和计算半径
            radius, area, center, _, seg_frame, flame_detected, all_contour_points, selected_points = self.extract_flame_contour(
                frame)

            # 转换为物理单位 (mm)
            if self.pixel_to_mm is not None:
                radius_mm = radius * self.pixel_to_mm
            else:
                radius_mm = 0

            # 在这里添加保存所有数据的代码：
            if flame_detected:
                # 无论是否在有效范围内，都保存数据
                self.all_frame_numbers.append(frame_count)
                self.all_flame_radii.append(radius)
                self.all_flame_areas.append(area)



            # 检查是否在有效半径范围内 - 严格筛选
            is_valid = (self.min_radius_mm <= radius_mm <= self.max_radius_mm and
                        flame_detected and radius_mm > 0)

            if is_valid:
                valid_frame_numbers.append(frame_count)
                valid_flame_radii.append(radius)  # 存储像素单位的半径
                valid_flame_areas.append(area)

            # 创建显示帧
            display_frame = self.create_display_frame(
                frame, seg_frame, radius, center,
                frame_count, total_frames, is_valid,
                all_contour_points, selected_points
            )

            # 存储处理后的帧用于后续播放
            self.original_frames.append(frame.copy())
            self.processed_frames.append(display_frame)

            # 显示处理过程
            if self.show_processing:
                continue_processing = self.display_processing_frame(display_frame)
                if not continue_processing:
                    print("用户中断处理过程")
                    break

        print(f"处理帧数: {processed_count}")
        print(f"有效帧数: {len(valid_flame_radii)}\n")

        # 数据清洗 - 使用改进的方法
        if len(valid_flame_radii) > 0:
            cleaned_radii, cleaned_areas, cleaned_frame_numbers = self.clean_data_improved(
                valid_flame_radii, valid_flame_areas, valid_frame_numbers,
                window_frac=self.window_frac,
                threshold_factor=self.threshold_factor,
                interpolation_method=self.interpolation_method
            )

            # 应用卡尔曼滤波（如果启用）
            if self.kalman_iterations > 0:
                cleaned_radii = self.apply_kalman_filter(cleaned_radii, self.kalman_iterations)

            # 转换为物理单位并再次确认在范围内
            final_frame_numbers = []
            final_flame_radii = []
            final_flame_areas = []

            for i in range(len(cleaned_radii)):
                radius_mm = cleaned_radii[i] * self.pixel_to_mm
                if self.min_radius_mm <= radius_mm <= self.max_radius_mm:
                    final_frame_numbers.append(cleaned_frame_numbers[i])
                    final_flame_radii.append(cleaned_radii[i])
                    final_flame_areas.append(cleaned_areas[i])

            self.frame_numbers = final_frame_numbers
            self.flame_radii = final_flame_radii
            self.flame_areas = final_flame_areas

            print(f"数据清洗后最终有效帧数: {len(self.flame_radii)}")
        else:
            self.frame_numbers = []
            self.flame_radii = []
            self.flame_areas = []

        return total_frames, len(self.flame_radii)

    def process_input(self):
        """
        处理输入并计算层流火焰速度
        """
        print(f"初始工况: {self.input_path}")
        print(f"输入类型: {self.input_type}")
        print(f"视窗检测方式: {self.window_detection_method}")
        print(f"火焰半径有效范围: {self.min_radius_mm:.0f} - {self.max_radius_mm:.0f} mm")
        print(f"帧率: {self.fps} FPS")
        print(f"间隔处理: {self.deal_gap}")

        if self.pixel_to_mm is not None:
            print(f"比例尺: {self.pixel_to_mm:.6f} mm/像素\n")
        else:
            print("比例尺未计算。\n")

        if self.use_default_ratio:
            print("使用默认比例尺进行计算。\n")

        # 如果需要显示处理过程，创建窗口
        if self.show_processing:
            cv2.namedWindow('Flame Analysis', cv2.WINDOW_NORMAL)

        # 根据输入类型处理
        if self.input_type == "video":
            total_frames, valid_frames = self.process_video_file()
        else:  # images
            total_frames, valid_frames = self.process_image_folder()

        if self.show_processing:
            cv2.destroyAllWindows()

        # 播放处理后的帧（循环播放）
        if self.processed_frames:
            self.play_processed_frames()

        if len(self.flame_radii) < 3:
            raise ValueError("有效帧数不足，无法计算火焰速度")

        print(f"有效帧数: {len(self.flame_radii)}/{total_frames}")

        print("计算火焰传播速度和拉伸率...")

        # 转换为物理单位 (cm) —— 与 MATLAB 保持一致
        if self.pixel_to_mm is not None:
            self.flame_radii_physical = [r * self.pixel_to_mm / 10.0 for r in self.flame_radii]  # mm -> cm
        else:
            self.flame_radii_physical = [0.0 for r in self.flame_radii]

        # 关键修改：使用实际的时间间隔，考虑deal_gap参数
        actual_time_interval = self.deal_gap / self.fps

        print(f"\n时间间隔信息:")
        print(f"  - 帧率: {self.fps} fps")
        print(f"  - 原始时间间隔: {1.0 / self.fps:.6f} s")
        print(f"  - 实际时间间隔 (考虑deal_gap={self.deal_gap}): {actual_time_interval:.6f} s")

        # 计算传播速度 - 使用实际时间间隔
        self.propagation_speeds_physical = self.calculate_propagation_speed(
            self.flame_radii_physical, actual_time_interval)  # 单位: cm/s

        # 计算拉伸率 - 同样使用实际时间间隔
        self.stretch_rates = self.calculate_stretch_rate(
            self.flame_radii_physical, self.flame_areas, actual_time_interval)  # 单位: 1/s

        # 应用Loess平滑到r-t数据
        if len(self.flame_radii_physical) > 5:
            times_for_smooth = [i / self.fps * 1000 for i in self.frame_numbers]
            self.flame_radii_smoothed = self.loess_smooth(
                times_for_smooth, self.flame_radii_physical, self.loess_frac_rt)   # cm
        else:
            self.flame_radii_smoothed = self.flame_radii_physical  # cm

        # 应用Loess平滑到Sb-t数据
        if len(self.propagation_speeds_physical) > 5:
            self.propagation_speeds_smoothed = self.loess_smooth(
                times_for_smooth, self.propagation_speeds_physical, self.loess_frac_sbt)
        else:
            self.propagation_speeds_smoothed = self.propagation_speeds_physical


        print("使用三种方法计算层流火焰速度和马克斯坦长度...")

        # 计算有效数据的拉伸率（使用平滑后的数据）
        stretch_rates_1s_smoothed = []
        for sb, rb in zip(self.propagation_speeds_smoothed, self.flame_radii_smoothed):  # cm/s cm
            if rb > 0 and sb > 0:
                stretch_rates_1s_smoothed.append(2 * sb / (rb) )  # rb:cm
            else:
                stretch_rates_1s_smoothed.append(0)

        # 筛选有效数据点用于拟合
        valid_indices = [i for i in range(len(stretch_rates_1s_smoothed))
                         if stretch_rates_1s_smoothed[i] > 0 and self.propagation_speeds_smoothed[i] > 0
                         and self.min_radius_mm / 10.0 <= self.flame_radii_smoothed[i] <= self.max_radius_mm / 10.0]   # 临界半径mm → cm

        if valid_indices:
            alpha_valid_smoothed = [stretch_rates_1s_smoothed[i] for i in valid_indices]
            sb_valid_smoothed = [self.propagation_speeds_smoothed[i] for i in valid_indices]
            radii_valid_smoothed = [self.flame_radii_smoothed[i] for i in valid_indices]

            # 方法1: 线性拟合 - 使用平滑后的数据
            self.laminar_flame_speed_method1, self.markstein_length_method1 = self.method1_linear_fit(
                alpha_valid_smoothed, sb_valid_smoothed)

            # # 修正后：直接使用 mm 进行拟合
            # radii_valid_mm = radii_valid_smoothed  # 单位 mm

            # 方法2
            Sb0_2, Lb_2 = self.method2_nonlinear_fit(radii_valid_smoothed, sb_valid_smoothed)
            self.laminar_flame_speed_method2 = Sb0_2
            self.markstein_length_method2 = Lb_2    # cm

            # 方法3
            Sb0_3, Lb_3 = self.method3_chen_nonlinear_extrapolation(radii_valid_smoothed, sb_valid_smoothed)
            self.laminar_flame_speed_method3 = Sb0_3
            self.markstein_length_method3 = Lb_3    # cm

        else:
            print("警告: 没有有效数据点进行拟合")
            self.laminar_flame_speed_method1, self.markstein_length_method1 = 0, 0
            self.laminar_flame_speed_method2, self.markstein_length_method2 = 0, 0
            self.laminar_flame_speed_method3, self.markstein_length_method3 = 0, 0

        # 计算所有数据的传播速度和拉伸率（用于灰色显示）
        self.all_propagation_speeds = self.calculate_all_propagation_speeds()

        # 计算所有数据的拉伸率
        self.all_stretch_rates = []
        for sb, rb in zip(self.all_propagation_speeds, [r * self.pixel_to_mm / 10.0 for r in self.all_flame_radii]):
            if rb > 0 and sb > 0:
                self.all_stretch_rates.append(2 * sb / rb)
            else:
                self.all_stretch_rates.append(0)

    def visualize_results(self):
        """
        可视化计算结果 - 显示所有数据点，但只对有效数据进行平滑和拟合
        """
        if not self.frame_numbers:
            raise ValueError("没有可用的计算结果，请先运行 process_input() 进行层流火焰速度计算")

        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        # 获取时间数据
        times_ms_valid = [i / self.fps * 1000 for i in self.frame_numbers]    # ms

        # 计算所有帧的时间（用于显示被剔除的数据点）
        all_times_ms = [i / self.fps * 1000 for i in self.all_frame_numbers]   # ms

        # 计算所有数据的物理半径
        all_radii_mm = [r * self.pixel_to_mm for r in self.all_flame_radii] # mm
        flame_radii_physical_mm = [r * self.pixel_to_mm for r in self.flame_radii]

        # 1. rb-t 图像
        # 显示所有数据点（灰色）和有效数据点（蓝色）
        ax1.scatter(all_times_ms, all_radii_mm, alpha=0.3, s=15, color='gray', label='Invalid data')
        ax1.scatter(times_ms_valid, flame_radii_physical_mm, alpha=0.7, s=20, color='blue', label='Valid data')
        ax1.set_xlabel('时间 (ms)')
        ax1.set_ylabel('火焰半径 (mm)')
        ax1.set_title('火焰半径-时间关系')
        ax1.set_ylim(0, max(all_radii_mm) * 1.1)  # 从0开始显示
        ax1.legend()
        ax1.grid(True)

        # 2. Sb-rb 图像
        # 计算所有数据的传播速度
        all_speeds = self.calculate_propagation_speed(all_radii_mm, 1.0 / self.fps)

        #ax2.scatter(all_radii_mm, all_speeds, alpha=0.3, s=15, color='gray', label='Invalid data')
        ax2.scatter(flame_radii_physical_mm, self.propagation_speeds_physical, alpha=0.7, s=20, color='blue',
                    label='Valid data')
        ax2.set_xlabel('火焰半径 (mm)')
        ax2.set_ylabel('层流火焰速度 (cm/s)')
        ax2.set_title('层流火焰速度-火焰半径关系')
        #ax2.set_xlim(min(self.flame_radii_mm)*0.8, max(self.flame_radii_mm * 1.3)  # 从0开始显示
        ax2.set_ylim(0, max(self.propagation_speeds_physical) * 1.3)  # 从0开始显示
        ax2.legend()
        ax2.grid(True)

        # 3. Sb-t 图像
        # ax3.scatter(all_times_ms, all_speeds, alpha=0.3, s=15, color='gray', label='Invalid data')
        ax3.scatter(times_ms_valid, self.propagation_speeds_physical, alpha=0.7, s=20, color='blue', label='Valid data')
        ax3.set_xlabel('时间 (ms)')
        ax3.set_ylabel('层流火焰速度 (cm/s)')
        ax3.set_title('层流火焰速度-时间关系')
        ax3.set_ylim(0, max(self.propagation_speeds_physical) * 1.3)  # 从0开始显示
        ax3.legend()
        ax3.grid(True)

        # 4. Sb-α 图像 (三种拟合方法)
        # 计算所有数据的拉伸率
        all_stretch_rates = []
        for sb, rb in zip(all_speeds, all_radii_mm):
            if rb > 0 and sb > 0:
                all_stretch_rates.append(2 * sb / rb)
            else:
                all_stretch_rates.append(0)

        # 计算有效数据的拉伸率（使用平滑后的数据）
        stretch_rates_1s = []
        for sb, rb in zip(self.propagation_speeds_smoothed, self.flame_radii_smoothed):
            if rb > 0 and sb > 0:
                stretch_rates_1s.append(2 * sb / (rb))  # rb mm → cm
            else:
                stretch_rates_1s.append(0)

        # 此时 self.flame_radii_smoothed 已经是 mm，直接比较
        valid_indices = [i for i in range(len(stretch_rates_1s))
                         if stretch_rates_1s[i] > 0 and self.propagation_speeds_smoothed[i] > 0
                         and self.min_radius_mm / 10.0 <= self.flame_radii_smoothed[i] <= self.max_radius_mm / 10.0]

        if valid_indices:
            alpha_valid = [stretch_rates_1s[i] for i in valid_indices]
            sb_valid = [self.propagation_speeds_smoothed[i] for i in valid_indices]
            radii_valid = [self.flame_radii_smoothed[i] for i in valid_indices]

            # 显示所有数据点（灰色）和有效数据点（蓝色）
            # ax4.scatter(all_stretch_rates, all_speeds, alpha=0.3, s=15, color='gray', label='Invalid data')
            ax4.scatter(alpha_valid, sb_valid, alpha=0.7, s=20, color='blue', label='Valid data')

            # 扩展α范围用于完整显示拟合线
            alpha_min, alpha_max = min(alpha_valid), max(alpha_valid)
            alpha_range_extended = np.linspace(0, alpha_max * 2.0, 300)  # 扩展到2倍最大α值

            # ---------- 使用参数化公式生成完整拟合线（与MATLAB一致）----------
            # 方法1: 线性拟合
            if self.laminar_flame_speed_method1 > 0:
                Sb0_1 = self.laminar_flame_speed_method1
                Lb_1 = self.markstein_length_method1
                if Lb_1 > 0:
                    E_max1 = Sb0_1 / Lb_1
                else:
                    E_max1 = max(alpha_valid) * 1.2
                E_lin = np.linspace(0, min(E_max1, max(alpha_valid)*1.2), 500)
                Sb_lin = Sb0_1 - Lb_1 * E_lin
                mask = (E_lin >= 0) & (Sb_lin >= 0)
                ax4.plot(E_lin[mask], Sb_lin[mask], 'r-', linewidth=1,
                         label=f'LM: Sb0={Sb0_1:.2f} cm/s, Lb={Lb_1:.4f} cm')

            # 方法2: Frankel & 陈正模型 (NMI)
            if self.laminar_flame_speed_method2 > 0 and abs(self.markstein_length_method2) > 1e-12:
                Sb0_2 = self.laminar_flame_speed_method2
                Lb_2 = self.markstein_length_method2
                if Lb_2 > 0:
                    Sb_f = np.linspace(Sb0_2, max(1e-4*abs(Sb0_2), 1e-4), 500)
                else:
                    Sb_max = max(max(sb_valid)*1.2, Sb0_2*1.5)
                    Sb_f = np.linspace(Sb0_2, Sb_max, 500)
                E_f = (Sb0_2 * Sb_f - Sb_f**2) / (Sb0_2 * Lb_2)
                mask = (E_f >= 0) & (Sb_f > 0)
                ax4.plot(E_f[mask], Sb_f[mask], 'y-', linewidth=1,
                         label=f'NMI: Sb0={Sb0_2:.2f} cm/s, Lb={Lb_2:.4f} cm')

            # 方法3: Kelly 模型 (NMII)
            if self.laminar_flame_speed_method3 > 0 and abs(self.markstein_length_method3) > 1e-12:
                Sb0_3 = self.laminar_flame_speed_method3
                Lb_3 = self.markstein_length_method3
                if Lb_3 > 0:
                    u = np.logspace(0, -4, 500)   # 1 到 1e-4
                else:
                    u_max = max(max(sb_valid)/Sb0_3*1.2, 1.5)
                    u = np.linspace(1, u_max, 500)
                E_k = -(Sb0_3 / Lb_3) * u**2 * np.log(u)
                Sb_k = u * Sb0_3
                mask = (E_k >= 0) & (Sb_k > 0)
                ax4.plot(E_k[mask], Sb_k[mask], 'g-', linewidth=1,
                         label=f'NMII: Sb0={Sb0_3:.2f} cm/s, Lb={Lb_3:.4f} cm')

            # 设置坐标轴范围（从 0 开始，留出空白）
            ax4.set_xlim(0, alpha_max * 2)
            ax4.set_ylim(0, max(sb_valid) * 1.3)
            ax4.set_xlabel('火焰拉伸率 (1/s)')
            ax4.set_ylabel('层流火焰速度 (cm/s)')
            ax4.set_title('层流火焰速度-火焰拉伸率关系')
            ax4.legend()
            ax4.grid(True)


        plt.tight_layout()
        plt.savefig(self.output_dir / 'flame_analysis_results.png', dpi=300, bbox_inches='')
        plt.show()



    def save_results(self):
        """
        保存计算结果到文件 - 只保存有效范围内的数据
        """
        results_file = self.output_dir / 'flame_analysis_results.csv'

        with open(results_file, 'w') as f:
            f.write("帧号,时间(ms),火焰半径(mm),火焰面积(像素),传播速度(cm/s),拉伸率(1/s)\n")

            for i in range(len(self.frame_numbers)):
                time_ms = self.frame_numbers[i] / self.fps * 1000  # 转换为毫秒
                radius = self.flame_radii_physical[i] if i < len(self.flame_radii_physical) else 0
                area = self.flame_areas[i] if i < len(self.flame_areas) else 0
                speed = self.propagation_speeds_physical[i] if i < len(self.propagation_speeds_physical) else 0
                stretch = self.stretch_rates[i] if i < len(self.stretch_rates) else 0

                f.write(f"{self.frame_numbers[i]},{time_ms:.6f},{radius:.6f},{area:.2f},{speed:.6f},{stretch:.6f}\n")

        # 保存层流火焰速度结果
        summary_file = self.output_dir / 'flame_analysis_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("层流火焰速度计算结果汇总\n")
            f.write("=" * 50 + "\n")
            f.write(f"分析输入: {self.input_path}\n")
            f.write(f"输入类型: {self.input_type}\n")
            f.write(f"帧率: {self.fps} fps\n")
            f.write(f"间隔处理参数: {self.deal_gap}\n")
            if self.pixel_to_mm is not None:
                f.write(f"像素到毫米转换系数: {self.pixel_to_mm:.6f} mm/像素\n")
            else:
                f.write(f"像素到毫米转换系数: 未计算\n")
            f.write(f"火焰半径有效范围: {self.min_radius_mm:.1f} - {self.max_radius_mm:.1f} mm\n")
            f.write(f"总分析帧数: {len(self.frame_numbers)}\n")
            f.write("\n三种拟合方法结果:\n")
            f.write(
                f"方法1 (线性拟合): Sb0 = {self.laminar_flame_speed_method1:.6f} cm/s, Lb = {self.markstein_length_method1:.6f} cm\n")
            f.write(
                f"方法2 (Chen非线性): Sb0 = {self.laminar_flame_speed_method2:.6f} cm/s, Lb = {self.markstein_length_method2:.6f} cm\n")
            f.write(
                f"方法3 (Kelly非线性): Sb0 = {self.laminar_flame_speed_method3:.6f} cm/s, Lb = {self.markstein_length_method3:.6f} cm\n")
            if self.use_default_ratio:
                f.write("注意: 使用了默认比例尺进行计算\n")

        print(f"结果已保存到: {self.output_dir}")

        # 导出拟合曲线数据
        try:
            self.export_fit_curves_to_excel()
        except Exception as e:
            print(f"导出 Excel 失败: {e}")
            import traceback
            traceback.print_exc()



if __name__ == "__main__":
    # ========== 可修改的参数区域 ==========
    # 1. 模型权重文件路径
    MODEL_PATH = "weights/YOLOv8n-seg-sphflame-v1.0.pt"

    # 2. 输入路径（视频文件或TIFF图片文件夹）
    # INPUT_PATH = "./datasets/cine2tiff/293K_1atm_a0.9_f1.0/"
    # INPUT_PATH = "./datasets/cine2tiff/363K_1atm_a0.2_f1.0/"
    # INPUT_PATH = "./datasets/cine2tiff/363K_1atm_a0.9_f0.6/"
    # INPUT_PATH = "./datasets/cine2tiff/363K_1atm_a1.0_f1.0/"
    INPUT_PATH = "./datasets/1.3_363K_1atm_a0.9_f1.4-2.mp4"
    # INPUT_PATH = "./datasets/cine2tiff/test"

    # 3. 输出目录
    OUTPUT_DIR = "./integrated_results"

    # 4. 火焰半径范围 (单位: mm)
    MIN_RADIUS_MM = 8
    MAX_RADIUS_MM = 25

    # 5. 帧率 (对于图片文件夹输入必需)
    FPS = 20000

    # 6. 置信度阈值
    CONFIDENCE_THRESHOLD = 0.05

    # 7. 是否显示处理过程
    SHOW_PROCESSING = False

    # 8. 输入类型 ("auto"自动检测, "video"视频文件, "images"图片文件夹)
    INPUT_TYPE = "auto"

    # 9. 轮廓选择比例
    CONTOUR_RATIO = 0.6

    # 10. 间隔处理参数 (每deal_gap帧处理1帧)
    DEAL_GAP = 1

    # 11. 新增卡尔曼滤波参数
    KALMAN_ITERATIONS = 1  # 滤波迭代次数 (0表示不使用)
    PROCESS_NOISE = 1e-5  # 过程噪声
    MEASUREMENT_NOISE = 1e-1  # 观测噪声

    # 12. 数据清洗参数
    WINDOW_FRAC = 0.0011  # 移动中位数窗口比例
    THRESHOLD_FACTOR = 3.0  # 离群值检测阈值因子
    INTERPOLATION_METHOD = 'spline'  # 插值方法: 'spline' 或 'linear'
    LOESS_FRAC_RT = 0.45  # r-t数据Loess平滑因子
    LOESS_FRAC_SBT = 0.25  # Sb-t数据Loess平滑因子

    # 13. 视窗检测方式选择
    WINDOW_DETECTION_METHOD = "manual"  # 可选: "yolo", "traditional", "manual"
    MANUAL_PIXEL_TO_MM = 0.098490  # 当选择manual时使用，例如: 91.0/768.0
    TRADITIONAL_KERNEL_SIZE = 18  # 传统方法的开运算核大小


    # ========== END ==========

    # 创建参数解析器
    parser = argparse.ArgumentParser(description='集成火焰分析')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='模型权重路径')
    parser.add_argument('--source', type=str, default=INPUT_PATH,
                        help='输入源: 可以是图片路径、图片目录、视频路径或摄像头ID')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='输出目录')
    parser.add_argument('--min-radius', type=float, default=MIN_RADIUS_MM,
                        help='最小有效火焰半径 (mm)')
    parser.add_argument('--max-radius', type=float, default=MAX_RADIUS_MM,
                        help='最大有效火焰半径 (mm)')
    parser.add_argument('--fps', type=float, default=FPS,
                        help='帧率 (对图片文件夹输入必需)')
    parser.add_argument('--conf', type=float, default=CONFIDENCE_THRESHOLD,
                        help='置信度阈值 (0-1)')
    parser.add_argument('--show-processing', action='store_true', default=SHOW_PROCESSING,
                        help='实时显示处理过程')
    parser.add_argument('--input-type', type=str, default=INPUT_TYPE,
                        choices=['auto', 'video', 'images'],
                        help='输入类型: auto, video, 或 images')
    parser.add_argument('--contour-ratio', type=float, default=CONTOUR_RATIO,
                        help='轮廓点选择比例 (0-1), 例如0.6表示选择顶部30%%和底部30%%的点')
    parser.add_argument('--deal-gap', type=int, default=DEAL_GAP,
                        help='间隔处理参数 (每deal_gap帧处理1帧)，用于处理高帧率数据')
    # Kalman滤波
    parser.add_argument('--use-kalman-after-loess', action='store_true', default=True,
                        help='是否对Loess平滑后的半径应用卡尔曼滤波')
    parser.add_argument('--kalman-iterations', type=int, default=KALMAN_ITERATIONS,
                        help='卡尔曼滤波迭代次数 (0表示不使用)')
    parser.add_argument('--process-noise', type=float, default=1e-5,
                        help='卡尔曼滤波过程噪声')
    parser.add_argument('--measurement-noise', type=float, default=1e-1,
                        help='卡尔曼滤波观测噪声')
    # 数据清洗参数
    parser.add_argument('--window-frac', type=float, default=0.0011,
                        help='移动中位数窗口比例')
    parser.add_argument('--threshold-factor', type=float, default=3.0,
                        help='离群值检测阈值因子')
    parser.add_argument('--interpolation-method', type=str, default='spline',
                        choices=['spline', 'linear'],
                        help='插值方法: spline 或 linear')
    parser.add_argument('--loess-frac-rt', type=float, default=0.25,
                        help='r-t数据Loess平滑因子')
    parser.add_argument('--loess-frac-sbt', type=float, default=0.25,
                        help='Sb-t数据Loess平滑因子')
    # 视窗检测参数
    parser.add_argument('--window-method', type=str, default=WINDOW_DETECTION_METHOD,
                        choices=['yolo', 'traditional', 'manual'],
                        help='视窗检测方式: yolo(目标检测), traditional(传统图像处理), manual(手动输入)')
    parser.add_argument('--manual-pixel-to-mm', type=float, default=MANUAL_PIXEL_TO_MM,
                        help='手动输入像素到毫米转换系数 (当window-method=manual时使用)')
    parser.add_argument('--traditional-kernel', type=int, default=18,
                        help='传统图像处理方法开运算核大小')


    args = parser.parse_args()

    try:
        # 创建分析器实例
        analyzer = IntegratedFlameAnalyzer(
            model_path=args.model,
            input_path=args.source,
            output_dir=args.output,
            min_radius_mm=args.min_radius,
            max_radius_mm=args.max_radius,
            show_processing=args.show_processing,
            fps=args.fps,
            input_type=args.input_type,
            conf_threshold=args.conf,
            contour_selection_ratio=args.contour_ratio,
            deal_gap=args.deal_gap,
            kalman_iterations=args.kalman_iterations,  # 改为kalman_iterations
            window_detection_method=args.window_method,
            manual_pixel_to_mm=args.manual_pixel_to_mm,
            traditional_kernel_size=args.traditional_kernel
        )

        # 处理输入并计算
        analyzer.process_input()

        # 可视化结果
        analyzer.visualize_results()

        # 保存结果
        analyzer.save_results()

        print(f"\n===== Laminar burning velocity =====")
        print(
            f"LM (Wu and Law et al.): Sb0 = {analyzer.laminar_flame_speed_method1:.2f} cm/s, Lb = {analyzer.markstein_length_method1:.4f} cm")
        print(
            f"NMI (Frankel and Chen): Sb0 = {analyzer.laminar_flame_speed_method2:.2f} cm/s, Lb = {analyzer.markstein_length_method2:.4f} cm")
        print(
            f"NMII (Kelly and Law):   Sb0 = {analyzer.laminar_flame_speed_method3:.2f} cm/s, Lb = {analyzer.markstein_length_method3:.4f} cm\n")

        average_Sb0 = 1/3 * (analyzer.laminar_flame_speed_method1 + analyzer.laminar_flame_speed_method2 + analyzer.laminar_flame_speed_method3)
        average_Lb  = 1/3 * (analyzer.markstein_length_method1 + analyzer.markstein_length_method2 + analyzer.markstein_length_method3)
        print(
            f"Average result of LBV:  Sb0 = {average_Sb0:.2f} cm/s, Lb = {average_Lb:.4f} cm")

    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback

        traceback.print_exc()