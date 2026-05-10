import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import random
import json
import os

from .preprocessing import DataPreprocessor


class BehaviorDataset(Dataset):
    """行为分割数据集"""

    # 允许时序倒序的行为类别
    REVERSIBLE_CLASSES = {1, 4, 5, 6}   # 静止(1), 嗅探(4), 直立(5), 梳理(6)

    def __init__(self, config, split='train', transform=None):
        self.config = config
        self.split = split
        self.transform = transform
        self.preprocessor = DataPreprocessor()

        # 是否启用增强
        self.use_augmentation = getattr(config, 'use_augmentation', False) and split == 'train'

        # 预加载环境多边形
        self.env_polygons = self.preprocessor.parse_environment_json(config.env_json_path)

        # 加载样本列表
        self.samples = self._load_samples_from_split()

        # 邻接矩阵
        self.adj_matrix = torch.tensor(
            self.preprocessor.create_adjacency_matrix(),
            dtype=torch.float32
        )

        print(f"{split}数据集加载完成，共 {len(self.samples)} 个样本"
              f"{'（启用数据增强）' if self.use_augmentation else ''}")

    def _load_samples_from_split(self):
        """加载指定文件列表"""
        split_file = getattr(self.config, 'data_split_json', None)
        if split_file is None:
            split_file = getattr(self.config, 'split_file', None)
        if split_file is None or not os.path.exists(split_file):
            raise FileNotFoundError(f"未找到数据划分文件: {split_file}。请先准备好 data_split.json。")

        with open(split_file, 'r', encoding='utf-8') as f:
            split_dict = json.load(f)

        if self.split not in split_dict:
            raise KeyError(f"划分文件中不存在键 '{self.split}'，可用键: {list(split_dict.keys())}")

        file_names = split_dict[self.split]

        keypoints_dir = Path(self.config.keypoints_dir)
        annotations_dir = Path(self.config.annotations_dir)
        time_dir = Path(self.config.time_dir)

        samples = []
        for fname in file_names:
            if not fname.endswith('.txt'):
                fname = fname + '.txt'
            kp_file = keypoints_dir / fname
            anno_file = annotations_dir / fname
            time_file = time_dir / fname

            if not (kp_file.exists() and anno_file.exists() and time_file.exists()):
                print(f"警告: 文件不完整，跳过 {fname}")
                continue

            sample = self._load_single_sample(str(kp_file), str(anno_file), str(time_file), fname)
            if sample is not None:
                samples.append(sample)

        return samples

    def _load_single_sample(self, kp_file, anno_file, time_file, filename):
        """加载单个样本，返回原始坐标及标签"""
        try:
            keypoints, bbox_centers = self.preprocessor.parse_keypoints_file(str(kp_file))
            if len(keypoints) == 0:
                print(f"警告: {filename} 关键点数据为空，跳过")
                return None

            # 解析标注
            behavior_labels, boundaries = self.preprocessor.parse_annotation_file(
                str(anno_file), len(keypoints)
            )

            # 解析时间戳
            time_seconds = self.preprocessor.parse_time_file(str(time_file))
            time_seconds_array = np.full(len(keypoints), time_seconds, dtype=np.float32)


            return {
                'keypoints_raw': keypoints.copy(),               # 原始坐标 (T, V, 2)
                'bbox_centers_raw': bbox_centers.copy(),         # (T, 2)
                'time_seconds_raw': time_seconds_array.copy(),   # (T,)
                'behavior_labels_raw': behavior_labels.copy(),   # list of int, length T
                'boundaries_raw': boundaries.copy(),             # list of int
                'filename': filename
            }

        except Exception as e:
            print(f"加载样本 {filename} 时出错: {str(e)}")
            return None

    def _create_targets(self, boundaries: List[int], behavior_labels: List[int]) -> Dict:
        """创建训练目标"""
        num_heads = self.config.max_boundaries
        num_classes = self.config.num_classes

        boundary_masks = np.zeros(num_heads, dtype=np.float32)
        boundary_positions = np.zeros(num_heads, dtype=np.float32)
        behavior_classes = np.zeros(num_heads, dtype=np.int64)
        confidences = np.zeros(num_heads, dtype=np.float32)

        T = len(behavior_labels)
        for i, boundary in enumerate(boundaries[:num_heads]):
            if i < num_heads:
                boundary_masks[i] = 1.0
                boundary_positions[i] = boundary / (T - 1) if T > 1 else 0

                if boundary < T:
                    class_label = behavior_labels[boundary]
                    if class_label >= 0 and class_label < num_classes:
                        behavior_classes[i] = class_label
                        confidences[i] = 1.0
                    else:
                        behavior_classes[i] = 0
                        confidences[i] = 0.5
                else:
                    behavior_classes[i] = 0
                    confidences[i] = 0.5

        return {
            'boundary_masks': boundary_masks,
            'boundary_positions': boundary_positions,
            'behavior_classes': behavior_classes,
            'confidences': confidences
        }

    def _pad_or_crop_with_start(self, arr: np.ndarray, start: int) -> np.ndarray:
        """根据指定起始位置进行裁剪或填充"""
        T_orig = arr.shape[0]
        target_len = self.config.num_frames
        if T_orig > target_len:
            arr = arr[start:start + target_len]
        elif T_orig < target_len:
            pad_len = target_len - T_orig
            if T_orig > 0:
                last = arr[-1:]
                pad = np.repeat(last, pad_len, axis=0)
                arr = np.concatenate([arr, pad], axis=0)
            else:
                shape = list(arr.shape)
                shape[0] = pad_len
                arr = np.concatenate([arr, np.zeros(shape, dtype=arr.dtype)], axis=0)
        return arr

    def _pad_or_crop_list(self, lst: list, start: int) -> list:
        """对列表进行裁剪或填充，填充值为 -1"""
        T_orig = len(lst)
        target_len = self.config.num_frames
        if T_orig > target_len:
            lst = lst[start:start + target_len]
        elif T_orig < target_len:
            pad_len = target_len - T_orig
            lst = lst + [-1] * pad_len
        return lst

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 获取原始副本
        keypoints = sample['keypoints_raw'].copy()          # (T, V, 2)
        bbox_centers = sample['bbox_centers_raw'].copy()    # (T, 2)
        time_seconds_arr = sample['time_seconds_raw'].copy()# (T,)
        behavior_labels = sample['behavior_labels_raw'].copy()
        boundaries = sample['boundaries_raw'].copy()
        T_orig = keypoints.shape[0]

        # ---------- 数据增强----------
        apply_time_reverse = False
        if self.use_augmentation:
            # 1. 镜像翻转
            if np.random.rand() < getattr(self.config, 'mirror_prob', 0.0):
                keypoints, bbox_centers = self.preprocessor.flip_keypoints_horizontal(
                    keypoints, bbox_centers, self.config.image_width
                )

            # 2. 时序倒序
            if np.random.rand() < getattr(self.config, 'time_reverse_prob', 0.3):
                # 检查该样本中的有效行为类别是否属于允许倒序的集合
                unique_labels = set(behavior_labels) - {-1}
                if unique_labels.issubset(self.REVERSIBLE_CLASSES):
                    apply_time_reverse = True


            if apply_time_reverse:
                keypoints = keypoints[::-1].copy()
                bbox_centers = bbox_centers[::-1].copy()
                time_seconds_arr = time_seconds_arr[::-1].copy()  
                behavior_labels = behavior_labels[::-1]
                boundaries = [T_orig - 1 - b for b in boundaries[::-1]]

            # 3. 关键点坐标小范围扰动
            if np.random.rand() < getattr(self.config, 'jitter_prob', 0.2):
                num_joints_to_jitter = np.random.randint(0, 4)  # 0~3个点
                if num_joints_to_jitter > 0:
                    keypoints = self._apply_keypoint_jitter(
                        keypoints,
                        num_joints=num_joints_to_jitter,
                        noise_std=getattr(self.config, 'jitter_noise_std', 20.0)
                    )

        # ---------- 重新计算环境特征 ----------
        env_features_list = []
        for t in range(T_orig):
            frame_kps = keypoints[t]
            frame_center = bbox_centers[t]
            env_feat = self.preprocessor.compute_environment_features(
                frame_kps, frame_center, self.env_polygons
            )
            env_features_list.append(env_feat)
        env_features = np.stack(env_features_list, axis=0)  # (T, env_feat_dim)

        # ---------- 归一化关键点 ----------
        keypoints_norm = self.preprocessor.normalize_keypoints(keypoints)

        # ---------- 减弱随机裁剪范围 ----------
        if self.split == 'train':
            max_offset = int(0.2 * max(0, T_orig - self.config.num_frames))
            start = np.random.randint(0, max_offset + 1) if T_orig > self.config.num_frames else 0
        else:
            start = 0

        # 裁剪/填充到固定长度
        keypoints_norm = self._pad_or_crop_with_start(keypoints_norm, start)
        bbox_centers = self._pad_or_crop_with_start(bbox_centers, start)
        env_features = self._pad_or_crop_with_start(env_features, start)
        time_seconds_arr = self._pad_or_crop_with_start(time_seconds_arr, start)
        behavior_labels = self._pad_or_crop_list(behavior_labels, start)

        # 调整边界索引
        T_fixed = self.config.num_frames
        adjusted_boundaries = []
        for b in boundaries:
            new_b = b - start
            if 0 <= new_b < T_fixed:
                adjusted_boundaries.append(new_b)

        # 创建目标
        targets = self._create_targets(adjusted_boundaries, behavior_labels)

        # ---------- 构造返回张量 ----------
        keypoints_tensor = torch.tensor(keypoints_norm, dtype=torch.float32)
        # 调整形状为 (1, C, T, V)
        keypoints_tensor = keypoints_tensor.permute(2, 0, 1).unsqueeze(0)

        time_seconds_tensor = torch.tensor(time_seconds_arr, dtype=torch.float32)
        env_features_tensor = torch.tensor(env_features, dtype=torch.float32)
        behavior_labels_tensor = torch.tensor(behavior_labels, dtype=torch.long)

        return {
            'keypoints': keypoints_tensor,
            'time_seconds': time_seconds_tensor,
            'env_features': env_features_tensor,
            'behavior_labels': behavior_labels_tensor,
            'boundary_masks': torch.tensor(targets['boundary_masks'], dtype=torch.float32),
            'boundary_positions': torch.tensor(targets['boundary_positions'], dtype=torch.float32),
            'behavior_classes': torch.tensor(targets['behavior_classes'], dtype=torch.long),
            'confidences': torch.tensor(targets['confidences'], dtype=torch.float32),
            'boundaries': adjusted_boundaries,
            'filename': sample['filename']
        }

    def _apply_keypoint_jitter(self, keypoints: np.ndarray, num_joints: int, noise_std: float):
        """
        随机选取 num_joints 个关键点，对其坐标添加高斯噪声
        keypoints: (T, V, 2)
        """
        T, V, _ = keypoints.shape
        keypoints = keypoints.copy()
        for t in range(T):
            # 每帧独立选取要扰动的关节点
            chosen = np.random.choice(V, size=min(num_joints, V), replace=False)
            noise = np.random.normal(0, noise_std, size=(len(chosen), 2))
            keypoints[t, chosen] += noise
        return keypoints