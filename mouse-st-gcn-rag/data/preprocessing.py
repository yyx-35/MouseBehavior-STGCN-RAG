import numpy as np
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from shapely.geometry import Point, Polygon, LineString
import math


class DataPreprocessor:
    """数据预处理器（9 类别原始版本）"""

    # 行为类别映射
    BEHAVIOR_MAPPING = {
        '行走': 0,
        '静止': 1,
        '喝水': 2,
        '进食': 3,
        '嗅探': 4,
        '挖洞': 5,
        '不支撑直立': 6,
        '支撑直立': 7,
        '梳理': 8,
    }

    # 反向映射
    ID_TO_BEHAVIOR = {v: k for k, v in BEHAVIOR_MAPPING.items()}

    # 有效的类别ID
    VALID_CLASS_IDS = list(BEHAVIOR_MAPPING.values())  # [0,1,2,3,4,5,6,7,8]

    # 原始ID到新ID的映射
    ORIGINAL_TO_NEW_MAPPING = {
        1: 0,  # 行走
        2: 1,  # 静止
        3: 2,  # 喝水
        4: 3,  # 进食
        5: 4,  # 嗅探
        6: 6,  # 不支撑直立
        7: 7,  # 支撑直立
        8: 8,  # 梳理
        9: 5,  # 挖洞
    }

    # 注意：原始ID 6 映射到新ID 6（不支撑直立），7->7，8->8，9->5（挖洞）
    # 这样新ID 0~8 正好对应9个类别，顺序为：行走,静止,喝水,进食,嗅探,挖洞,不支撑直立,支撑直立,梳理

    @staticmethod
    def parse_keypoints_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """解析关键点文件（与之前相同，无需修改）"""
        frames_kps = []
        frames_centers = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                try:
                    values = list(map(float, parts[1:]))
                except ValueError:
                    continue

                if len(values) >= 16:
                    x1, y1, x2, y2 = values[0:4]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    frames_centers.append([center_x, center_y])

                    kp_coords = values[4:16]  # 12个值 (6个关键点 * 2)
                    frame_kps = np.array(kp_coords).reshape(6, 2)
                    frames_kps.append(frame_kps)

        if frames_kps:
            return np.stack(frames_kps, axis=0), np.array(frames_centers)
        return np.array([]), np.array([])

    @staticmethod
    def parse_annotation_file(filepath: str, num_frames: int = 76) -> Tuple[List[int], List[int]]:
        """
        解析标注文件，返回合并后的类别标签列表和边界帧索引列表。
        现在返回 9 类标签。
        """
        behavior_labels = np.full(num_frames, -1, dtype=np.int64)
        boundaries = []

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = r'(\d+)-(\d+):(\d+)\.(.+)'
        matches = re.findall(pattern, content)

        for start, end, behavior_id, behavior_name in matches:
            start = int(start) - 1
            end = int(end) - 1
            original_id = int(behavior_id)

            if original_id in DataPreprocessor.ORIGINAL_TO_NEW_MAPPING:
                new_id = DataPreprocessor.ORIGINAL_TO_NEW_MAPPING[original_id]
                behavior_labels[start:end + 1] = new_id
                if start > 0:
                    boundaries.append(start)
            elif original_id == 10:  # 废片
                behavior_labels[start:end + 1] = -1
            else:
                print(f"警告: 未知的行为ID: {original_id}, 行为名称: {behavior_name}")

        return behavior_labels.tolist(), boundaries

    @staticmethod
    def get_num_classes():
        return len(DataPreprocessor.BEHAVIOR_MAPPING) 

    @staticmethod
    def get_class_name(class_id: int) -> str:
        return DataPreprocessor.ID_TO_BEHAVIOR.get(class_id, f"未知类别_{class_id}")

    @staticmethod
    def parse_time_file(filepath: str) -> float:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) >= 2:
            time_str = lines[1]
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s
        else:
            raise ValueError(f"时间文件格式错误: {filepath}")

    @staticmethod
    def parse_environment_json(json_path: str) -> Dict[str, Polygon]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        polygons = {}
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            if points[0] != points[-1]:
                points.append(points[0])
            polygons[label] = Polygon(points)
        return polygons

    @staticmethod
    def sample_points_on_polygon(polygon: Polygon, num_points: int) -> List[Point]:
        if polygon.is_empty:
            return [Point(0, 0)] * num_points
        boundary = polygon.boundary
        if boundary.geom_type == 'MultiLineString':
            coords = []
            for line in boundary.geoms:
                coords.extend(line.coords)
            boundary = LineString(coords)
        length = boundary.length
        if length == 0:
            return [Point(0, 0)] * num_points
        distances = np.linspace(0, length, num_points, endpoint=False)
        points = [boundary.interpolate(d) for d in distances]
        return points

    @staticmethod
    def compute_environment_features(
            keypoints_frame: np.ndarray,
            bbox_center: np.ndarray,
            env_polygons: Dict[str, Polygon]
    ) -> np.ndarray:
        mouse_points = []
        for i in range(keypoints_frame.shape[0]):
            mouse_points.append(Point(keypoints_frame[i, 0], keypoints_frame[i, 1]))
        mouse_points.append(Point(bbox_center[0], bbox_center[1]))

        obj_points = {}
        if '食槽' in env_polygons and not env_polygons['食槽'].is_empty:
            poly = env_polygons['食槽']
            exterior = poly.exterior
            coords = list(exterior.coords)[:-1]
            if len(coords) >= 4:
                corners = [Point(coords[0]), Point(coords[1]), Point(coords[2]), Point(coords[3])]
            else:
                corners = DataPreprocessor.sample_points_on_polygon_fixed(poly, 4)
            center = Point(poly.centroid.x, poly.centroid.y)
            uniform_points = DataPreprocessor.sample_points_on_polygon_fixed(poly, 5)
            obj_points['食槽'] = corners + [center] + uniform_points
        else:
            obj_points['食槽'] = [Point(0, 0)] * 10

        if '饮水器' in env_polygons and not env_polygons['饮水器'].is_empty:
            poly = env_polygons['饮水器']
            exterior = poly.exterior
            coords = list(exterior.coords)[:-1]
            if len(coords) >= 4:
                corners = [Point(coords[0]), Point(coords[1]), Point(coords[2]), Point(coords[3])]
            else:
                corners = DataPreprocessor.sample_points_on_polygon_fixed(poly, 4)
            center = Point(poly.centroid.x, poly.centroid.y)
            uniform_points = DataPreprocessor.sample_points_on_polygon_fixed(poly, 2)
            obj_points['饮水器'] = corners + [center] + uniform_points
        else:
            obj_points['饮水器'] = [Point(0, 0)] * 7

        if '基座' in env_polygons and not env_polygons['基座'].is_empty:
            poly = env_polygons['基座']
            exterior = poly.exterior
            coords = list(exterior.coords)[:-1]
            if len(coords) >= 4:
                corners = [Point(coords[0]), Point(coords[1]), Point(coords[2]), Point(coords[3])]
            else:
                corners = DataPreprocessor.sample_points_on_polygon_fixed(poly, 4)
            center = Point(poly.centroid.x, poly.centroid.y)
            uniform_points = DataPreprocessor.sample_points_on_polygon_fixed(poly, 3)
            obj_points['基座'] = corners + [center] + uniform_points
        else:
            obj_points['基座'] = [Point(0, 0)] * 8

        features = []
        for obj_name in ['食槽', '饮水器', '基座']:
            for mouse_pt in mouse_points:
                for obj_pt in obj_points[obj_name]:
                    dist = mouse_pt.distance(obj_pt)
                    features.append(dist)
        return np.array(features, dtype=np.float32)

    @staticmethod
    def sample_points_on_polygon_fixed(polygon: Polygon, num_points: int) -> List[Point]:
        if polygon.is_empty:
            return [Point(0, 0)] * num_points
        boundary = polygon.boundary
        if boundary.geom_type == 'MultiLineString':
            coords = []
            for line in boundary.geoms:
                coords.extend(line.coords)
            boundary = LineString(coords)
        length = boundary.length
        if length == 0:
            return [Point(0, 0)] * num_points
        distances = np.linspace(0, length, num_points, endpoint=False)
        points = [boundary.interpolate(d) for d in distances]
        return points

    @staticmethod
    def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
        if len(keypoints) == 0:
            return keypoints
        neck_points = keypoints[:, 3, :]
        normalized = keypoints.copy()
        for t in range(len(keypoints)):
            center = neck_points[t]
            distances = np.linalg.norm(keypoints[t] - center, axis=1)
            max_dist = np.max(distances)
            if max_dist > 0:
                normalized[t] = (keypoints[t] - center) / max_dist
        return normalized

    @staticmethod
    def create_adjacency_matrix(num_joints: int = 6) -> np.ndarray:
        edges = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (0, 1), (0, 2)]
        adj_matrix = np.zeros((num_joints, num_joints))
        for i, j in edges:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
        np.fill_diagonal(adj_matrix, 1)
        return adj_matrix

    # 数据增强辅助函数
    @staticmethod
    def flip_keypoints_horizontal(keypoints, bbox_centers, image_width):
        flipped_kp = keypoints.copy()
        flipped_kp[..., 0] = image_width - flipped_kp[..., 0]
        flipped_centers = bbox_centers.copy()
        flipped_centers[..., 0] = image_width - flipped_centers[..., 0]
        return flipped_kp, flipped_centers

    @staticmethod
    def rotate_keypoints(keypoints, bbox_centers, angle_deg, center):
        theta = np.deg2rad(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        cx, cy = center

        def rotate_points(points):
            translated = points - np.array([cx, cy])
            x = translated[..., 0]
            y = translated[..., 1]
            x_rot = x * cos_t - y * sin_t
            y_rot = x * sin_t + y * cos_t
            return np.stack([x_rot, y_rot], axis=-1) + np.array([cx, cy])

        rotated_kp = rotate_points(keypoints)
        rotated_centers = rotate_points(bbox_centers)
        return rotated_kp, rotated_centers

    @staticmethod
    def apply_keypoint_occlusion(keypoints, num_joints_to_occlude=2, mode='zero', noise_std=0.05):
        occluded = keypoints.copy()
        T, V, _ = occluded.shape
        for t in range(T):
            indices = np.random.choice(V, size=min(num_joints_to_occlude, V), replace=False)
            if mode == 'zero':
                occluded[t, indices, :] = 0.0
            elif mode == 'noise':
                noise = np.random.randn(len(indices), 2) * noise_std
                occluded[t, indices, :] += noise
        return occluded