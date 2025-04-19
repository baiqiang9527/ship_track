import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
import pandas as pd
from datetime import datetime
import json


# ================== 增强配置参数 ==================
class EnhancedConfig:
    """增强跟踪系统配置参数"""
    # 基础参数
    KALMAN_PROCESS_NOISE = 0.05  # 基础过程噪声
    MAX_AGE = 300  # 最大丢失帧数
    MIN_VISIBILITY = 0.3  # 最小可见比例
    DETECTION_CONFIDENCE = 0.71  # 检测置信度阈值

    # 过程噪声自适应参数
    ACCEL_NOISE_FACTOR = 0.3  # 加速度噪声系数
    MAX_ACCEL = 8.0  # 最大有效加速度(m/s²)

    # 多模态匹配参数
    # 修改后的多模态匹配参数
    MOTION_WEIGHT = 0.25  # 原0.6
    APPEARANCE_WEIGHT = 0.25  # 原0.3
    ANGLE_WEIGHT = 0.25  # 原0.1
    TRAJ_WEIGHT = 0.25  # 新增轨迹匹配权重
    HIST_BINS = 64  # 直方图bin数
    FEATURE_UPDATE_RATE = 0.8  # 特征更新率

    # 预测参数
    PREDICT_STEPS = 20  # 适当减少步数以平衡精度和性能
    PREDICT_INTERVAL = 0.3  # 更精细的时间间隔
    MATCH_DISTANCE_THRESHOLD = 200  # 匹配距离阈值(像素)


# ================== 增强跟踪器类 ==================
class EnhancedTrack:
    def __init__(self, track_id, centroid, bbox, dt):
        self.id = track_id
        self.centroid = centroid  # 显式保存质心坐标
        self.bbox = bbox
        self.age = 1
        self.total_visible = 1
        self.consecutive_invisible = 0
        self.position_history = deque([centroid], maxlen=20)
        self.velocity_history = deque(maxlen=30)
        self.dt = dt
        self.appearance_feat = None
        # 添加这两行初始化
        self.optimal_state_history = []  # 存储历史最优状态
        self.optimal_state_frames = []  # 存储对应的帧号q
        self.frame_counter = 0  # 新增帧计数器
        self.fps = 1.0 / dt  # 确保这行代码存在
        self.bbox_history = deque([bbox], maxlen=int(10 * self.fps))  # 调整历史记录长度为10秒

        # 初始化卡尔曼滤波器
        self.kf = cv2.KalmanFilter(4, 2)
        self._init_kalman(dt)
        # 在现有初始化代码后添加
        self.fps = 1.0 / dt  # 添加fps属性
        self.forced_prediction = False  # 标记是否因面积变化强制进入预测状态
        self.normal_detection_lost = False  # 标记是否因置信度损失而丢失检测
        self.forced_prediction = False
        self.last_normal_velocity = None
        self.last_normal_bbox = None
        self.normal_detection_lost = False
        self.enable_area_detection = True  # 新添加的状态标志

        # 添加新的状态变量保存强制预测位置
        self.forced_prediction_position = None
        self.forced_prediction = False
        self.last_normal_velocity = None
        self.last_normal_bbox = None
        # 添加新状态变量
        self.consecutive_high_conf_frames = 0  # 连续高置信度帧计数
        self.in_normal_prediction = False  # 是否处于普通预测状态

        # 添加状态冷却相关变量
        self.state_transition_cooling = 0  # 状态冷却计数器
        self.state_cooling_period = 45  # 状态冷却期（帧数）
        self.consecutive_high_conf_frames = 0  # 连续高置信度帧计数
        self.consecutive_low_conf_frames = 0  # 连续低置信度帧计数
        self.in_normal_prediction = False  # 是否处于普通预测状态



    def _init_kalman(self, dt):
        """初始化卡尔曼滤波器参数（修复类型问题）"""
        # 确保所有矩阵使用float32类型
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * EnhancedConfig.KALMAN_PROCESS_NOISE

        # 使用保存的centroid变量
        self.kf.statePost = np.array([
            [self.centroid[0]],
            [self.centroid[1]],
            [0.0],  # 初始速度x
            [0.0]  # 初始速度y
        ], dtype=np.float32)

    def update_process_noise(self):
        """动态调整过程噪声（修复类型问题）"""
        if len(self.velocity_history) >= 30:  # 改为30帧窗口
            current_v = np.array(self.velocity_history[-1], dtype=np.float32)
            prev_v = np.array(self.velocity_history[-30], dtype=np.float32)
            dt_window = self.dt * 30  # 时间窗口跨度
            accel = np.linalg.norm(current_v - prev_v) / dt_window
        else:
            accel = 0.0

        noise_scale = 1 + EnhancedConfig.ACCEL_NOISE_FACTOR * accel
        new_noise = EnhancedConfig.KALMAN_PROCESS_NOISE * noise_scale
        print(f"Accel: {accel:.2f}, NoiseScale: {noise_scale:.2f}, NewNoise: {new_noise:.4f}")
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * new_noise  # 确保类型正确

    def predict(self):
        """单步预测（修复类型问题）"""
        self.update_process_noise()
        prediction = self.kf.predict()
        return prediction.astype(np.float32)  # 确保输出类型正确

    def correct(self, measurement, reliability=0.5):
        """测量校正（增加可靠性权重参数）"""
        if measurement.dtype != np.float32:
            measurement = measurement.astype(np.float32)

        # 根据可靠性调整测量噪声
        if reliability < 0.5:
            # 临时增加测量噪声
            original_noise = self.kf.measurementNoiseCov.copy()
            # 降低可靠性会增加测量噪声，减少对不可靠测量的依赖
            self.kf.measurementNoiseCov *= (2.0 - reliability)
            self.kf.correct(measurement)
            # 恢复原始噪声设置
            self.kf.measurementNoiseCov = original_noise
        else:
            self.kf.correct(measurement)

    def detect_partial_occlusion(self, bbox, frame_shape):
        """检测目标是否部分遮挡"""
        x, y, w, h = bbox

        # 检查是否靠近图像边缘
        near_edge = (x < 20 or y < 20 or
                     x + w > frame_shape[1] - 20 or
                     y + h > frame_shape[0] - 20)

        # 检查边界框大小是否突然变化
        if len(self.bbox_history) > 5:
            prev_w, prev_h = self.bbox_history[-1][2:]
            size_change_ratio = abs((w * h) / (prev_w * prev_h) - 1.0)
            sudden_size_change = size_change_ratio > 0.3  # 面积变化超过30%
        else:
            sudden_size_change = False

        return near_edge or sudden_size_change

    def predict_future(self, steps=5, dt=None):
        """多步轨迹预测"""
        if dt is None:
            dt = self.dt

        # 保存原始参数
        original_state = self.kf.statePost.copy()
        original_trans = self.kf.transitionMatrix.copy()

        predictions = []
        current_state = original_state.copy()

        for _ in range(steps):
            # 更新转移矩阵时间参数（修复方法名）
            self._init_kalman(dt)  # 修改为正确的方法名

            # 预测下一状态
            current_state = self.kf.transitionMatrix.dot(current_state)
            x = int(current_state[0, 0])
            y = int(current_state[1, 0])
            predictions.append((x, y))

        # 恢复原始参数
        self.kf.transitionMatrix = original_trans
        self.kf.statePost = original_state

        return predictions

    def get_velocity(self):
        """改进的速度计算方法 - 带限幅功能"""
        # 使用卡尔曼滤波器的速度估计
        kalman_velocity = self.kf.statePost.flatten()[2:4]

        # 限幅处理 - 防止速度突变
        if len(self.velocity_history) > 0:
            prev_vx, prev_vy = self.velocity_history[-1]
            curr_vx, curr_vy = float(kalman_velocity[0]), float(kalman_velocity[1])

            # 计算加速度
            ax = (curr_vx - prev_vx) / self.dt
            ay = (curr_vy - prev_vy) / self.dt

            # 限制最大加速度
            max_accel = 15.0  # 根据场景调整
            accel_magnitude = np.sqrt(ax * ax + ay * ay)

            if accel_magnitude > max_accel:
                # 如果加速度过大，限制速度变化
                scale = max_accel / accel_magnitude
                curr_vx = prev_vx + (curr_vx - prev_vx) * scale
                curr_vy = prev_vy + (curr_vy - prev_vy) * scale
                return (curr_vx, curr_vy)

        return (float(kalman_velocity[0]), float(kalman_velocity[1]))

    def update_appearance(self, new_feature):
        """更新外观特征"""
        if self.appearance_feat is None:
            self.appearance_feat = new_feature
        else:
            self.appearance_feat = EnhancedConfig.FEATURE_UPDATE_RATE * self.appearance_feat + \
                                   (1 - EnhancedConfig.FEATURE_UPDATE_RATE) * new_feature

    def update_optimal_state_history(self, frame_num):
        """记录当前最优状态及对应帧号"""
        current_state = self.kf.statePost.copy()
        self.optimal_state_history.append(current_state)
        self.optimal_state_frames.append(frame_num)

        # 仅保留最近1000帧的记录（可根据内存需求调整）
        if len(self.optimal_state_history) > 1000:
            self.optimal_state_history.pop(0)
            self.optimal_state_frames.pop(0)


    def get_historical_velocity(self, current_frame, start_offset=90, end_offset=150):
        """获取历史帧范围内的平均速度

        Args:
            current_frame: 当前帧号
            start_offset: 开始偏移量（多少帧之前）
            end_offset: 结束偏移量（多少帧之前）

        Returns:
            历史平均速度(vx, vy)
        """
        if len(self.optimal_state_history) < 10:
            # 如果历史记录不足，返回当前速度
            current_state = self.kf.statePost.copy()
            return (float(current_state[2]), float(current_state[3]))

        # 计算目标帧范围
        target_start = current_frame - end_offset
        target_end = current_frame - start_offset

        # 查找符合范围的状态
        vx_values = []
        vy_values = []

        for i, frame_num in enumerate(self.optimal_state_frames):
            if target_start <= frame_num <= target_end:
                state = self.optimal_state_history[i]
                vx_values.append(float(state[2]))
                vy_values.append(float(state[3]))

        # 如果找到了符合条件的记录
        if vx_values and vy_values:
            # 使用中位数避免异常值影响
            vx = np.median(vx_values)
            vy = np.median(vy_values)
            return (vx, vy)

        # 如果没有找到历史记录，使用最近的几个状态
        recent_count = min(5, len(self.optimal_state_history))
        if recent_count > 0:
            recent_states = self.optimal_state_history[-recent_count:]
            vx = np.median([float(s[2]) for s in recent_states])
            vy = np.median([float(s[3]) for s in recent_states])
            return (vx, vy)

        # 最后的后备方案：当前速度
        current_state = self.kf.statePost.copy()
        return (float(current_state[2]), float(current_state[3]))

    def apply_historical_velocity_for_prediction(self, current_frame):
        """在开始预测时应用历史稳定速度"""
        # 获取历史速度
        hist_vx, hist_vy = self.get_historical_velocity(current_frame)

        # 修改卡尔曼状态中的速度分量
        state = self.kf.statePost.copy()
        state[2] = hist_vx
        state[3] = hist_vy
        self.kf.statePost = state

        # 可选：减小速度过程噪声以保持稳定
        noise = self.kf.processNoiseCov.copy()
        noise[2:4, 2:4] *= 0.3
        self.kf.processNoiseCov = noise

    def detect_rapid_area_change(self):
        frames_4sec = int(4.0 * self.fps)

        if len(self.bbox_history) < frames_4sec:
            return False

        current_bbox = self.bbox_history[-1]
        current_area = current_bbox[2] * current_bbox[3]

        past_bbox = list(self.bbox_history)[-frames_4sec]
        past_area = past_bbox[2] * past_bbox[3]

        if past_area < 1e-6:
            return False

        area_change_ratio = (past_area - current_area) / past_area

        # 添加调试信息
        print(f"Track ID: {self.id}")
        print(f"Current Area: {current_area:.2f}")
        print(f"Past Area: {past_area:.2f}")
        print(f"Change Ratio: {area_change_ratio:.3f}")

        return area_change_ratio >= 0.3

    def reset_area_history(self):
        """重置边界框面积历史"""
        current_bbox = self.bbox
        self.bbox_history.clear()
        self.bbox_history.append(current_bbox)


# ================== 增强跟踪系统 ==================
class EnhancedTrackerSystem:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.tracks = []
        self.next_id = 0
        # 在EnhancedTrackerSystem的__init__方法中修改columns定义
        columns = ['timestamp', 'frame_num', 'track_id', 'x', 'y',
                   'width', 'height', 'speed', 'noise_level',
                   'optimal_x', 'optimal_y', 'optimal_vx', 'optimal_vy',
                   'is_predicted']
        # 添加预测坐标列（根据PREDICT_STEPS自动生成）
        for step in range(1, EnhancedConfig.PREDICT_STEPS + 1):
            columns.extend([f'pred_x{step}', f'pred_y{step}'])
        self.detection_data = pd.DataFrame(columns=columns)
        self.current_frame = 0

        # 视频参数
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.dt = 1.0 / self.fps if self.fps > 0 else 1 / 30
        print(f"视频参数: {self.fps:.2f}FPS, 时间间隔: {self.dt:.3f}s")

        # 添加输出目录设置
        self.OUTPUT_DIR = 'results/'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # 添加视频保存功能
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{self.OUTPUT_DIR}tracking_{timestamp}.mp4"

        # 获取视频分辨率
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 考虑info_panel的高度
        output_height = height + 80  # info_panel_height

        # 创建视频写入器（尝试多种编解码器）
        try:
            # 首先尝试H.264编解码器
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            self.video_writer = cv2.VideoWriter(
                video_filename,
                fourcc,
                self.fps,
                (width, output_height)
            )

            # 检查是否成功创建
            if not self.video_writer.isOpened():
                # 如果H.264失败，尝试mp4v
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    video_filename,
                    fourcc,
                    self.fps,
                    (width, output_height)
                )

            # 如果还是失败，尝试XVID
            if not self.video_writer.isOpened():
                video_filename = f"{self.OUTPUT_DIR}tracking_{timestamp}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(
                    video_filename,
                    fourcc,
                    self.fps,
                    (width, output_height)
                )

            print(f"跟踪视频将保存至: {os.path.abspath(video_filename)}")
        except Exception as e:
            print(f"视频保存初始化错误: {e}")
            self.video_writer = None

    def detect_objects(self, frame):
        """增强的目标检测（返回特征）"""
        results = self.model(frame)
        centroids = []
        bboxes = []
        features = []
        self.detection_confidences = []  # 添加置信度列表
        print(f"\n-- 第{self.current_frame}帧检测结果 --")
        detect_count = 0

        for detection in results[0].boxes:
            xmin, ymin, xmax, ymax = detection.xyxy[0].cpu().numpy()
            confidence = detection.conf.item()  # 获取置信度

            if confidence > EnhancedConfig.DETECTION_CONFIDENCE:
                w, h = int(xmax - xmin), int(ymax - ymin)
                if w > 0 and h > 0:
                    # 提取ROI区域
                    roi = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

                    # 特征提取
                    hist = self._extract_hsv_hist(roi)

                    bbox = (int(xmin), int(ymin), w, h)
                    centroid = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))

                    bboxes.append(bbox)
                    centroids.append(centroid)
                    features.append(hist)
                    self.detection_confidences.append(confidence)  # 保存置信度
                    # 打印检测信息
                    print(
                        f"检测到目标 #{detect_count}: 置信度={confidence:.3f}, 位置=({int((xmin + xmax) / 2)}, {int((ymin + ymax) / 2)}), 尺寸={w}x{h}")
        print(f"总共检测到{len(self.detection_confidences)}个目标")
        return centroids, bboxes, features

    def _extract_hsv_hist(self, roi):
        """提取HSV颜色直方图特征（增加容错处理）"""
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            return np.zeros(EnhancedConfig.HIST_BINS * 3, dtype=np.float32)

        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [EnhancedConfig.HIST_BINS], [0, 180]).flatten()
            hist_s = cv2.calcHist([hsv], [1], None, [EnhancedConfig.HIST_BINS], [0, 256]).flatten()
            hist_v = cv2.calcHist([hsv], [2], None, [EnhancedConfig.HIST_BINS], [0, 256]).flatten()
            return np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        except:
            return np.zeros(EnhancedConfig.HIST_BINS * 3, dtype=np.float32)

    def _predict_detection_traj(self, centroid, steps, dt):
        """生成检测目标的假设轨迹（匀速模型）"""
        if len(self.tracks) > 0:
            # 使用现有跟踪器的平均速度作为参考
            avg_vx = np.mean([t.get_velocity()[0] for t in self.tracks])
            avg_vy = np.mean([t.get_velocity()[1] for t in self.tracks])
        else:
            avg_vx, avg_vy = 10, 10  # 默认初始速度

        return [
            (
                int(centroid[0] + avg_vx * i * dt),
                int(centroid[1] + avg_vy * i * dt)
            )
            for i in range(1, steps + 1)
        ]

    def _dtw_distance(self, traj1, traj2):
        """动态时间规整距离计算"""
        n, m = len(traj1), len(traj2)
        if n == 0 or m == 0:
            return 0.0

        # 转换为numpy数组提高计算效率
        traj1 = np.array(traj1)
        traj2 = np.array(traj2)

        # 初始化DTW矩阵
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(traj1[i - 1] - traj2[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

        return dtw[n, m] / (n + m)  # 归一化处理

    def data_association(self, centroids, features):
        """改进的多模态数据关联（整合轨迹预测）"""
        cost_matrix = np.zeros((len(self.tracks), len(centroids)), dtype=np.float32)

        for i, track in enumerate(self.tracks):
            # 获取跟踪器的多步预测轨迹
            track_traj = track.predict_future(
                EnhancedConfig.PREDICT_STEPS,
                EnhancedConfig.PREDICT_INTERVAL
            )

            for j, (centroid, feature) in enumerate(zip(centroids, features)):
                # 基础运动代价（单步预测）
                single_step_pred = track.predict().flatten()[:2]
                motion_cost = np.linalg.norm(single_step_pred - centroid)

                # 外观相似度
                if track.appearance_feat is not None and feature is not None:
                    appearance_sim = 1 - cosine(track.appearance_feat, feature)
                else:
                    appearance_sim = 0.0

                # 方向一致性代价
                track_v = track.get_velocity()
                current_v = (centroid[0] - track.position_history[-1][0],
                             centroid[1] - track.position_history[-1][1])
                angle_diff = self._vector_angle_diff(track_v, current_v)

                # 轨迹匹配代价（新增部分）
                detection_traj = self._predict_detection_traj(
                    centroid,
                    EnhancedConfig.PREDICT_STEPS,
                    EnhancedConfig.PREDICT_INTERVAL
                )
                traj_cost = self._dtw_distance(track_traj, detection_traj)

                # 综合加权代价
                cost_matrix[i, j] = (
                        EnhancedConfig.MOTION_WEIGHT * motion_cost +
                        EnhancedConfig.APPEARANCE_WEIGHT * (1 - appearance_sim) +
                        EnhancedConfig.ANGLE_WEIGHT * angle_diff +
                        EnhancedConfig.TRAJ_WEIGHT * traj_cost
                )

        # 匈牙利算法匹配（后续保持不变）
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments = []
        unassigned_tracks = []
        unassigned_dets = []

        # 过滤有效匹配
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < EnhancedConfig.MATCH_DISTANCE_THRESHOLD:
                assignments.append((r, c))
            else:
                unassigned_tracks.append(r)
                unassigned_dets.append(c)

        # 收集未匹配项
        assigned_tracks = set(r for r, _ in assignments)
        assigned_dets = set(c for _, c in assignments)
        unassigned_tracks += [r for r in range(len(self.tracks)) if r not in assigned_tracks]
        unassigned_dets += [c for c in range(len(centroids)) if c not in assigned_dets]

        return assignments, unassigned_tracks, unassigned_dets

    def _vector_angle_diff(self, v1, v2):
        """计算向量角度差"""
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(cos_theta, -1, 1)) / np.pi  # 归一化到[0,1]

    # 完整修改后的update_tracks函数

    def update_tracks(self, centroids, bboxes, features, assignments, unassigned_tracks, unassigned_dets, frame):
        """更新跟踪器状态"""
        # 遍历所有跟踪器检查面积变化
        for track in self.tracks:
            # 如果还没有进入强制预测状态，则检查面积变化
            if track.enable_area_detection and not track.forced_prediction:
                if track.detect_rapid_area_change():
                    track.forced_prediction = True
                    track.last_normal_velocity = track.get_velocity()
                    track.last_normal_bbox = track.bbox
                    # 添加：记录进入强制预测状态时的位置
                    state = track.kf.statePost.flatten()
                    track.forced_prediction_position = (float(state[0]), float(state[1]))
                    print(f"ID {track.id}: 检测到边界框压缩，进入强制预测状态")

                    state = track.kf.statePost.copy()
                    state[2] = track.last_normal_velocity[0]
                    state[3] = track.last_normal_velocity[1]
                    track.kf.statePost = state

            # 如果处于强制预测状态，持续更新最新位置
            if track.forced_prediction:
                # 使用卡尔曼状态中的位置
                current_state = track.kf.statePost.flatten()
                track.forced_prediction_position = (float(current_state[0]), float(current_state[1]))

        # 处理未匹配的跟踪器
        for track_idx in unassigned_tracks:
            track = self.tracks[track_idx]

            # 减少状态冷却计数器
            if track.state_transition_cooling > 0:
                track.state_transition_cooling -= 1

            if track.consecutive_invisible == 0:
                track.normal_detection_lost = True
                # 重置高置信度计数
                track.consecutive_high_conf_frames = 0

            # 从强制预测切换到普通预测状态时的处理
            if track.forced_prediction and track.normal_detection_lost and track.state_transition_cooling == 0:
                # 保存强制预测最后位置和速度
                last_forced_position = track.forced_prediction_position
                last_forced_velocity = track.get_velocity()  # 获取当前速度估计

                # 更改状态标志
                track.forced_prediction = False
                track.enable_area_detection = False  # 禁用面积检测
                track.in_normal_prediction = True  # 设置为普通预测状态
                track.consecutive_high_conf_frames = 0  # 重置高置信度帧计数
                track.consecutive_low_conf_frames = 0  # 重置低置信度帧计数
                track.state_transition_cooling = track.state_cooling_period  # 设置冷却期
                print(f"ID {track.id}: 从强制预测切换到普通预测状态，进入{track.state_cooling_period}帧冷却期")

                # 确保平滑过渡到正常预测状态
                if last_forced_position is not None:
                    # 更新卡尔曼状态，保持位置和速度的连续性
                    state = track.kf.statePost.copy()
                    state[0] = last_forced_position[0]  # 更新位置x
                    state[1] = last_forced_position[1]  # 更新位置y
                    state[2] = last_forced_velocity[0]  # 保持速度vx
                    state[3] = last_forced_velocity[1]  # 保持速度vy
                    track.kf.statePost = state

                    # 临时降低速度过程噪声，保持轨迹稳定
                    process_noise = track.kf.processNoiseCov.copy()
                    process_noise[2:4, 2:4] *= 0.5  # 降低速度噪声
                    track.kf.processNoiseCov = process_noise

                    # 直接使用同步后的位置
                    predicted_x = float(last_forced_position[0])
                    predicted_y = float(last_forced_position[1])
                    predicted_centroid = (int(predicted_x), int(predicted_y))

                    # 更新位置历史，确保连续性
                    track.position_history.append(predicted_centroid)
                    track.velocity_history.append(last_forced_velocity)

                    # 更新边界框
                    w, h = track.last_normal_bbox[2:]
                    new_x = int(predicted_x - w // 2)
                    new_y = int(predicted_y - h // 2)
                    track.bbox = (new_x, new_y, w, h)
                    track.bbox_history.append((new_x, new_y, w, h))

                    track.consecutive_invisible += 1
                    track.age += 1

                    # 已经处理完状态转换，继续下一个跟踪器
                    continue

            # 正常预测逻辑
            prediction = track.predict()
            predicted_x = float(prediction[0])
            predicted_y = float(prediction[1])
            predicted_centroid = (int(predicted_x), int(predicted_y))

            # 更新位置历史
            track.position_history.append(predicted_centroid)
            track.velocity_history.append(track.get_velocity())

            # 如果在强制预测状态，使用最后的正常边界框大小
            if track.forced_prediction:
                w, h = track.last_normal_bbox[2:]
            else:
                w, h = track.bbox[2:]

            new_x = int(predicted_x - w // 2)
            new_y = int(predicted_y - h // 2)
            track.bbox = (new_x, new_y, w, h)
            track.bbox_history.append((new_x, new_y, w, h))

            track.consecutive_invisible += 1
            track.age += 1

        # 更新已匹配目标
        # 处理已匹配的目标
        for track_idx, det_idx in assignments:
            track = self.tracks[track_idx]
            centroid = centroids[det_idx]
            bbox = bboxes[det_idx]

            # 获取当前检测置信度
            detection_confidence = self.detection_confidences[det_idx] if hasattr(self,
                                                                                  'detection_confidences') else 1.0

            # 打印详细匹配信息
            print(f"ID {track.id}: 匹配到检测结果 #{det_idx + 1}, 置信度={detection_confidence:.3f}, "
                  f"位置=({centroid[0]}, {centroid[1]}), "
                  f"高置信度帧={track.consecutive_high_conf_frames}, "
                  f"低置信度帧={track.consecutive_low_conf_frames}, "
                  f"{'普通预测状态' if track.in_normal_prediction else '正常检测状态'}")

            # 重置连续不可见帧计数
            track.consecutive_invisible = 0

            # 更新置信度计数器
            if detection_confidence >= EnhancedConfig.DETECTION_CONFIDENCE:
                track.consecutive_high_conf_frames += 1
                track.consecutive_low_conf_frames = 0
                print(
                    f"ID {track.id}: 检测置信度={detection_confidence:.3f} >= 阈值{EnhancedConfig.DETECTION_CONFIDENCE}，"
                    f"高置信度帧计数增加到{track.consecutive_high_conf_frames}")
            else:
                track.consecutive_low_conf_frames += 1
                track.consecutive_high_conf_frames = 0
                print(
                    f"ID {track.id}: 检测置信度={detection_confidence:.3f} < 阈值{EnhancedConfig.DETECTION_CONFIDENCE}，"
                    f"低置信度帧计数增加到{track.consecutive_low_conf_frames}")

            # 减少状态冷却计数器
            if track.state_transition_cooling > 0:
                track.state_transition_cooling -= 1

            # 如果还没有进入强制预测状态，则检查面积变化
            if track.enable_area_detection and not track.forced_prediction:
                if track.detect_rapid_area_change():
                    track.forced_prediction = True
                    track.last_normal_velocity = track.get_velocity()
                    track.last_normal_bbox = track.bbox
                    # 记录进入强制预测状态时的位置
                    state = track.kf.statePost.flatten()
                    track.forced_prediction_position = (float(state[0]), float(state[1]))
                    print(f"ID {track.id}: 检测到边界框压缩，进入强制预测状态")

                    state = track.kf.statePost.copy()
                    state[2] = track.last_normal_velocity[0]
                    state[3] = track.last_normal_velocity[1]
                    track.kf.statePost = state

            # 处理不同状态下的更新逻辑
            if track.forced_prediction:
                # 强制预测状态的处理
                prediction = track.predict()
                predicted_x = float(prediction[0])
                predicted_y = float(prediction[1])
                predicted_centroid = (int(predicted_x), int(predicted_y))

                # 更新强制预测位置
                track.forced_prediction_position = (predicted_x, predicted_y)

                # 使用最后的正常边界框大小
                w, h = track.last_normal_bbox[2:]
                new_x = int(predicted_x - w // 2)
                new_y = int(predicted_y - h // 2)

                track.position_history.append(predicted_centroid)
                track.bbox = (new_x, new_y, w, h)
                track.bbox_history.append((new_x, new_y, w, h))

                # 在强制预测状态下重置高置信度帧计数
                track.consecutive_high_conf_frames = 0
                track.consecutive_low_conf_frames = 0
                track.in_normal_prediction = False
            elif track.in_normal_prediction:
                # 普通预测状态的处理
                # 检查是否达到状态转换条件
                if track.consecutive_high_conf_frames >= 30 and track.state_transition_cooling == 0:
                    print(f"ID {track.id}: 状态转换条件满足! 连续{track.consecutive_high_conf_frames}帧高置信度，"
                          f"从普通预测状态切换回正常检测状态")
                    track.in_normal_prediction = False
                    track.state_transition_cooling = track.state_cooling_period
                    track.consecutive_high_conf_frames = 5  # 不完全重置

                    # 添加这行代码重新启用面积检测
                    track.enable_area_detection = True  # 重新启用面积检测
                    print(f"ID {track.id}: 重新启用面积检测功能")

                    # 现在可以使用检测位置更新
                    measurement = np.array([[centroid[0]], [centroid[1]]], dtype=np.float32)
                    track.correct(measurement)
                    track.position_history.append(centroid)
                    track.bbox = bbox
                    track.bbox_history.append(bbox)
                else:
                    # 未达到转换条件，继续使用预测位置
                    prediction = track.predict()
                    predicted_x = float(prediction[0])
                    predicted_y = float(prediction[1])
                    predicted_centroid = (int(predicted_x), int(predicted_y))

                    # 更新位置历史
                    track.position_history.append(predicted_centroid)

                    # 更新边界框
                    w, h = track.bbox[2:]
                    new_x = int(predicted_x - w // 2)
                    new_y = int(predicted_y - h // 2)
                    track.bbox = (new_x, new_y, w, h)
                    track.bbox_history.append((new_x, new_y, w, h))

                    print(f"ID {track.id}: 普通预测状态中，继续使用预测位置 ({predicted_x:.1f}, {predicted_y:.1f})，"
                          f"而非检测位置 ({centroid[0]}, {centroid[1]})")
            else:
                # 正常检测状态的处理
                # 使用检测位置更新
                measurement = np.array([[centroid[0]], [centroid[1]]], dtype=np.float32)
                track.correct(measurement)
                track.position_history.append(centroid)
                track.bbox = bbox
                track.bbox_history.append(bbox)

                # 检查是否需要切换到普通预测状态
                if track.consecutive_low_conf_frames >= 3 and track.state_transition_cooling == 0:
                    print(f"ID {track.id}: 状态转换条件满足! 连续{track.consecutive_low_conf_frames}帧低置信度，"
                          f"从正常检测状态切换到普通预测状态")
                    track.in_normal_prediction = True
                    track.state_transition_cooling = track.state_cooling_period // 2
                    track.consecutive_low_conf_frames = 1  # 不完全重置

            # 无论何种状态，都更新以下通用属性
            track.total_visible += 1
            track.age += 1
            track.velocity_history.append(track.get_velocity())

            # 仅在非强制预测状态下更新外观特征
            if not track.forced_prediction and features is not None and len(features) > det_idx:
                track.update_appearance(features[det_idx])

            # 更新最优状态历史
            track.update_optimal_state_history(self.current_frame)

            # 如果在普通预测状态下且启用了面积检测，检查面积变化
            if not track.forced_prediction and track.enable_area_detection:
                # 面积变化检测
                if track.detect_rapid_area_change() and track.state_transition_cooling == 0:
                    print(f"ID {track.id}: 检测到面积急剧变化，强制进入预测状态")
                    # 保存当前状态
                    track.forced_prediction = True
                    track.last_normal_velocity = track.get_velocity()
                    track.last_normal_bbox = track.bbox
                    track.forced_prediction_position = (float(track.position_history[-1][0]),
                                                        float(track.position_history[-1][1]))
                    # 应用历史速度
                    track.apply_historical_velocity_for_prediction(self.current_frame)
                    track.state_transition_cooling = track.state_cooling_period  # 设置冷却期

        # 创建新跟踪器
        for det_idx in unassigned_dets:
            centroid = centroids[det_idx]
            bbox = bboxes[det_idx]
            new_track = EnhancedTrack(self.next_id, centroid, bbox, self.dt)
            # 初始化新添加的强制预测位置变量
            new_track.forced_prediction_position = None
            new_track.update_optimal_state_history(self.current_frame)

            if features is not None and len(features) > det_idx:
                new_track.update_appearance(features[det_idx])

            self.tracks.append(new_track)
            self.next_id += 1

        # 清理丢失目标
        self.tracks = [t for t in self.tracks
                       if t.consecutive_invisible < EnhancedConfig.MAX_AGE
                       and (t.total_visible / max(t.age, 1)) > EnhancedConfig.MIN_VISIBILITY]


    def record_data(self, frame):
        """记录跟踪数据（增加最优估计值保存）"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        for track in self.tracks:
            if not track.position_history:
                continue

            # 获取预测轨迹
            predictions = track.predict_future(
                EnhancedConfig.PREDICT_STEPS,
                EnhancedConfig.PREDICT_INTERVAL
            )

            # 计算动态噪声水平
            noise_level = np.mean(np.diag(track.kf.processNoiseCov))

            # 获取卡尔曼最优估计状态（增加部分）
            optimal_state = track.kf.statePost.flatten()
            optimal_x = float(optimal_state[0])
            optimal_y = float(optimal_state[1])
            optimal_vx = float(optimal_state[2])
            optimal_vy = float(optimal_state[3])

            # 构建数据行
            new_row = {
                'timestamp': timestamp,
                'frame_num': self.current_frame,
                'track_id': track.id,
                'x': track.position_history[-1][0],
                'y': track.position_history[-1][1],
                'width': track.bbox[2],
                'height': track.bbox[3],
                'speed': np.sqrt(sum(v ** 2 for v in track.get_velocity())) / self.dt,
                'noise_level': noise_level,
                # 添加最优估计值
                'optimal_x': optimal_x,
                'optimal_y': optimal_y,
                'optimal_vx': optimal_vx,
                'optimal_vy': optimal_vy,
                # 增加标记字段，标识此记录是实际观测还是预测
                'is_predicted': track.consecutive_invisible > 0
            }

            # 添加预测坐标
            for i, (px, py) in enumerate(predictions, 1):
                new_row[f'pred_x{i}'] = px
                new_row[f'pred_y{i}'] = py

            self.detection_data = pd.concat([self.detection_data, pd.DataFrame([new_row])],
                                            ignore_index=True)

    def _get_color_by_id(self, track_id):
        """根据跟踪ID生成唯一的颜色"""
        np.random.seed(track_id * 9999)
        # 避免太暗或太亮的颜色
        hue = np.random.randint(0, 180)
        saturation = np.random.randint(100, 255)
        value = np.random.randint(150, 255)

        # 将HSV转换为BGR
        hsv = np.array([[[hue, saturation, value]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return tuple(map(int, bgr[0][0]))

    def _draw_grid(self, frame, grid_size=50):
        """在画面上绘制网格线"""
        h, w = frame.shape[:2]

        # 绘制水平线
        for y in range(0, h, grid_size):
            cv2.line(frame, (0, y), (w, y), (50, 50, 50), 1)

        # 绘制垂直线
        for x in range(0, w, grid_size):
            cv2.line(frame, (x, 0), (x, h), (50, 50, 50), 1)

    def visualize(self, frame):
        """增强可视化效果（添加预测状态标记并优化标签布局）"""
        display_frame = frame.copy()

        # 添加全局信息面板
        info_panel_height = 80
        info_panel = np.zeros((info_panel_height, frame.shape[1], 3), dtype=np.uint8)

        # 绘制当前跟踪的全局信息
        cv2.putText(info_panel, f"Total_item: {len(self.tracks)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_panel, f"Fps: {self.current_frame} | FPS: {self.fps:.1f}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_panel, f"Time: {datetime.now().strftime('%H:%M:%S')}",
                    (frame.shape[1] - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 创建标签占用区域的记录，用于防止重叠
        occupied_regions = []

        # 绘制轨迹和目标
        for track in self.tracks:
            if track.total_visible < 3:
                continue

            track_color = self._get_color_by_id(track.id)
            x, y, w, h = track.bbox

            # 计算跟踪状态数据
            speed = np.linalg.norm(track.get_velocity()) / self.dt
            noise = np.mean(np.diag(track.kf.processNoiseCov))
            visibility = track.total_visible / max(track.age, 1) * 100

            # 根据跟踪状态决定显示效果
            if track.forced_prediction:
                # 强制预测状态 - 使用黄色虚线框
                for i in range(0, h, 4):
                    cv2.line(display_frame, (x, y + i), (x + 4, y + i), (0, 255, 255), 1)
                    cv2.line(display_frame, (x + w - 4, y + i), (x + w, y + i), (0, 255, 255), 1)

                for i in range(0, w, 4):
                    cv2.line(display_frame, (x + i, y), (x + i, y + 4), (0, 255, 255), 1)
                    cv2.line(display_frame, (x + i, y + h - 4), (x + i, y + h), (0, 255, 255), 1)

                cv2.putText(display_frame, "FORCED PREDICT", (x, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                optimal_state = track.kf.statePost.flatten()
                optimal_x = int(optimal_state[0])
                optimal_y = int(optimal_state[1])
                cv2.circle(display_frame, (optimal_x, optimal_y), 5, (0, 255, 255), -1)

            elif track.consecutive_invisible > 0:
                # 常规预测状态 - 使用红色虚线框
                for i in range(0, h, 4):
                    cv2.line(display_frame, (x, y + i), (x + 4, y + i), track_color, 1)
                    cv2.line(display_frame, (x + w - 4, y + i), (x + w, y + i), track_color, 1)

                for i in range(0, w, 4):
                    cv2.line(display_frame, (x + i, y), (x + i, y + 4), track_color, 1)
                    cv2.line(display_frame, (x + i, y + h - 4), (x + i, y + h), track_color, 1)

                cv2.putText(display_frame, "PREDICT", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                optimal_state = track.kf.statePost.flatten()
                optimal_x = int(optimal_state[0])
                optimal_y = int(optimal_state[1])
                cv2.circle(display_frame, (optimal_x, optimal_y), 5, (0, 0, 255), -1)

            else:
                # 正常跟踪状态 - 实线框
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), track_color, 2)
                cx, cy = x + w // 2, y + h // 2
                cv2.circle(display_frame, (cx, cy), 5, track_color, -1)

            # 多行信息显示
            info_text = [
                f"ID: {track.id}",
                f"speed: {speed:.1f}px/s",
                f"visb: {visibility:.0f}%"
            ]

            # 添加状态信息
            if track.forced_prediction:
                info_text.append("Status: Forced Pred")
            elif track.in_normal_prediction:
                info_text.append(f"Status: Normal Pred ({track.consecutive_high_conf_frames}/30)")
                if track.state_transition_cooling > 0:
                    info_text.append(f"Cooling: {track.state_transition_cooling}")
            elif track.consecutive_invisible > 0:
                info_text.append(f"Missing: {track.consecutive_invisible}")
            else:
                info_text.append("Status: Normal")
                if track.state_transition_cooling > 0:
                    info_text.append(f"Cooling: {track.state_transition_cooling}")

            # 创建标签参数
            text_width = 150
            line_height = 20
            text_bg_height = len(info_text) * line_height + 10

            # 计算合适的标签位置
            label_positions = [
                (x, y - text_bg_height - 5),
                (x, y + h + 5),
                (x - text_width - 5, y),
                (x + w + 5, y)
            ]

            # 检查每个可能位置与已占用区域的碰撞情况
            best_pos = None
            min_overlap = float('inf')

            for pos_x, pos_y in label_positions:
                if pos_x < 0:
                    pos_x = 0
                if pos_y < 0:
                    pos_y = 0
                if pos_x + text_width > display_frame.shape[1]:
                    pos_x = display_frame.shape[1] - text_width
                if pos_y + text_bg_height > display_frame.shape[0]:
                    pos_y = display_frame.shape[0] - text_bg_height

                curr_region = (pos_x, pos_y, pos_x + text_width, pos_y + text_bg_height)
                overlap = 0
                for region in occupied_regions:
                    x_overlap = max(0, min(curr_region[2], region[2]) - max(curr_region[0], region[0]))
                    y_overlap = max(0, min(curr_region[3], region[3]) - max(curr_region[1], region[1]))
                    overlap += x_overlap * y_overlap

                if overlap < min_overlap:
                    min_overlap = overlap
                    best_pos = (pos_x, pos_y)

            text_pos = best_pos
            occupied_regions.append((text_pos[0], text_pos[1],
                                     text_pos[0] + text_width,
                                     text_pos[1] + text_bg_height))

            # 绘制半透明背景
            overlay = display_frame.copy()
            cv2.rectangle(overlay,
                          (text_pos[0], text_pos[1]),
                          (text_pos[0] + text_width, text_pos[1] + text_bg_height),
                          (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

            # 绘制文本内容
            for i, txt in enumerate(info_text):
                y_offset = text_pos[1] + 20 + i * line_height
                cv2.putText(display_frame, txt, (text_pos[0] + 5, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 2)

            # 历史轨迹 - 渐变颜色效果
            alpha_base = 0.4
            line_thickness = 2
            if len(track.position_history) > 1:
                for i in range(1, len(track.position_history)):
                    alpha = alpha_base + (1.0 - alpha_base) * i / len(track.position_history)
                    color = tuple([int(c * alpha) for c in track_color])
                    cv2.line(display_frame,
                             track.position_history[i - 1],
                             track.position_history[i],
                             color,
                             line_thickness)

            # 预测轨迹 - 虚线效果
            future = track.predict_future(EnhancedConfig.PREDICT_STEPS, EnhancedConfig.PREDICT_INTERVAL)
            if len(future) > 1:
                for i in range(1, len(future)):
                    if i % 2 == 0:
                        pred_color = (0, 165, 255)
                        cv2.line(display_frame, future[i - 1], future[i],
                                 pred_color, 1, lineType=cv2.LINE_AA)
                cv2.circle(display_frame, future[-1], 3, (0, 0, 255), -1)

        # 绘制网格线
        self._draw_grid(display_frame)

        # 合并信息面板和主显示
        final_display = np.vstack([info_panel, display_frame])

        # 添加时间戳水印
        cv2.putText(final_display, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    (10, final_display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)

        # 保存当前帧到视频文件
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            try:
                if self.video_writer.isOpened():
                    self.video_writer.write(final_display)
            except Exception as e:
                print(f"保存视频帧错误: {e}")

        cv2.imshow('增强型船舶跟踪系统', final_display)

    def run(self):
        """主运行循环"""
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                # 增强检测（返回特征）
                centroids, bboxes, features = self.detect_objects(frame)
                # 多模态数据关联
                assignments, unassigned_tracks, unassigned_dets = \
                    self.data_association(centroids, features)
                # 增强更新 - 修改这行，添加frame参数
                self.update_tracks(centroids, bboxes, features,
                                   assignments, unassigned_tracks, unassigned_dets, frame)
                self.record_data(frame)
                self.visualize(frame)
                self.current_frame += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                print(f"\n=============== 处理第{self.current_frame}帧 ===============")
        finally:
            self.cap.release()
            if hasattr(self, 'video_writer') and self.video_writer is not None:
                try:
                    if self.video_writer.isOpened():
                        self.video_writer.release()
                        print(f"视频文件已成功保存")
                except Exception as e:
                    print(f"关闭视频写入器错误: {e}")
            cv2.destroyAllWindows()
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.OUTPUT_DIR}final_model{timestamp}.xlsx"
            self.detection_data.to_excel(filename, index=False, engine='openpyxl')
            print(f"数据已保存至: {os.path.abspath(filename)}")


if __name__ == "__main__":
    tracker = EnhancedTrackerSystem(
        model_path=r'E:\program file\python file\Yolo\YOLO_V11\runs\detect\train6\weights\best.pt',
        # model_path=r'E:\program file\python file\Yolo\YOLO_V11\yolo11l.pt',
        video_path=r'E:\program file\data\ship\zhedang_ship/test05.mp4'
    )
    tracker.run()