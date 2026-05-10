import torch
import os

class STGCNConfig:
    def __init__(self):
        # 基础配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.val_interval = 1
        self.data_dir = r"\path\to\json"
        self.keypoints_dir = r"\path\to\pose"
        self.annotations_dir = r"\path\to\labels"
        self.time_dir = r"\path\to\times"
        self.env_json_path = r"\path\to\frame.json"
        self.output_dir = 'results'         

        # 数据划分文件
        self.data_split_json = os.path.join(self.data_dir, "data_split.json")

        # 模型架构
        self.hidden_dim = 256
        self.num_classes = 9
        self.num_features = 2
        self.num_frames = 75
        self.num_joints = 6
        self.dropout = 0.2
        self.behavior_names = [
            'walk', 'stop', 'drinking', 'feeding', 'sniffing',
            'unsupported rearing', 'supported rearing', 'grooming', 'digging'
        ]
        self.max_boundaries = 10
        self.use_spatial_attention = True
        self.spatial_attn_reduction = 16

        # 时间编码配置
        self.time_embedding_dim = 128

        # 环境特征配置
        self.env_feature_dim = 175
        self.env_embedding_dim = 128

        # 交叉注意力头数
        self.num_attention_heads = 16

        # 损失函数参数
        self.focal_gamma = 2.0
        self.temporal_weight = 0.5
        self.boundary_weight = 0.1
        self.class_weights = [1.2, 1.0, 1.0, 1.0, 1.0, 1.2, 1.0, 1.0, 1.0]

        # 训练参数
        self.grad_clip = 1.5
        self.patience = 10

        # 数据增强配置
        self.use_augmentation = True
        self.image_width = 1920
        self.image_height = 1080
        self.mirror_prob = 0.0
        self.time_reverse_prob = 0
        self.jitter_prob = 0
        self.jitter_noise_std = 20.0

        # 评估指标 Top-K
        self.topk = 3