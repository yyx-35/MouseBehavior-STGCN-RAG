import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ---------- 标准空间图卷积----------
class SpatialGraphConv(nn.Module):
    """标准空间图卷积层，使用归一化邻接矩阵"""
    def __init__(self, in_channels, out_channels, adj_matrix):
        super().__init__()
        self.register_buffer('adj_matrix', adj_matrix)
        self.adj_size = adj_matrix.size(0)
        self.conv = nn.Conv2d(in_channels, out_channels * self.adj_size, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        B, C, T, V = x.shape
        x = self.conv(x)  # (B, out_channels*V, T, V)
        x = x.view(B, self.adj_size, -1, T, V)
        x = torch.einsum('baktv,av->bktv', x, self.adj_matrix)  # 图卷积
        x = x.reshape(B, -1, T, V)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# ---------- 单尺度时间卷积块----------
class TemporalConv(nn.Module):
    """时间卷积块"""
    def __init__(self, channels, kernel_size=5, stride=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                      padding=(padding, 0), stride=(stride, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.conv(x)


# ---------- 标准ST-GCN块（官方风格）----------
class STGCNBlock(nn.Module):
    """单个ST-GCN块：空间图卷积 + 时间卷积 + 残差连接"""
    def __init__(self, in_channels, out_channels, adj_matrix,
                 temporal_stride=1, dropout=0.1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.temporal_stride = temporal_stride

        # 空间图卷积
        self.spatial_conv = SpatialGraphConv(in_channels, out_channels, adj_matrix)

        # 时间卷积
        self.temporal_conv = TemporalConv(out_channels, stride=temporal_stride, dropout=dropout)

        # 残差连接
        if self.use_residual:
            if in_channels != out_channels or temporal_stride != 1:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(temporal_stride, 1)),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.residual = nn.Identity()
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)
        out = self.spatial_conv(x)
        out = self.temporal_conv(out)
        out = out + residual
        return self.relu(out)


# ---------- 预测头时间注意力----------
class TemporalAttention(nn.Module):
    """轻量级时间注意力"""
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                               padding=kernel_size//2, groups=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv(x)
        return x * self.sigmoid(attn)


# ---------- 轻量级空间注意力模块 ----------
class LightweightSpatialAttention(nn.Module):
    """轻量级空间注意力：对关节维度进行加权"""
    def __init__(self, channels, num_joints, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, None))  # (B, C, 1, V)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, T, V = x.shape
        pooled = self.global_pool(x)
        attn = self.fc(pooled)
        return x * attn


# ---------- TCN 模块----------
class TemporalBlock(nn.Module):
    """TCN 残差块"""
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                                padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                                padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return out + residual


class TemporalConvNet(nn.Module):
    """堆叠多个 TCN 残差块，膨胀率逐层增加"""
    def __init__(self, channels, num_layers=4, kernel_size=5, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.layers.append(
                TemporalBlock(channels, kernel_size, dilation, dropout)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------- 层次化特征提取器----------
class HierarchicalFeatureExtractor(nn.Module):
    """多层ST-GCN特征提取器，输出融合多尺度的时空特征图"""
    def __init__(self, config, adj_matrix):
        super().__init__()
        self.config = config
        self.input_layer = nn.Sequential(
            nn.Conv2d(config.num_features, config.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(config.hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layers = nn.ModuleList([
            STGCNBlock(config.hidden_dim, config.hidden_dim, adj_matrix,
                       temporal_stride=1, dropout=config.dropout),
            STGCNBlock(config.hidden_dim, config.hidden_dim * 2, adj_matrix,
                       temporal_stride=2, dropout=config.dropout),
            STGCNBlock(config.hidden_dim * 2, config.hidden_dim * 2, adj_matrix,
                       temporal_stride=1, dropout=config.dropout),
            STGCNBlock(config.hidden_dim * 2, config.hidden_dim * 4, adj_matrix,
                       temporal_stride=2, dropout=config.dropout)
        ])

        self.fusion_convs = nn.ModuleList()
        for i, layer in enumerate(self.layers):
            if i == 0:
                in_ch = config.hidden_dim
            elif i == 1:
                in_ch = config.hidden_dim * 2
            elif i == 2:
                in_ch = config.hidden_dim * 2
            else:
                in_ch = config.hidden_dim * 4
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, config.hidden_dim, kernel_size=1),
                    nn.BatchNorm2d(config.hidden_dim),
                    nn.ReLU(inplace=True)
                )
            )

        self.fusion_out = nn.Sequential(
            nn.Conv2d(config.hidden_dim * len(self.layers), config.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.input_layer(x)
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        target_size = features[0].shape[2:]
        upsampled_features = []
        for i, feat in enumerate(features):
            feat_proj = self.fusion_convs[i](feat)
            if feat.shape[2:] != target_size:
                feat_proj = F.interpolate(feat_proj, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(feat_proj)

        fused = torch.cat(upsampled_features, dim=1)
        output = self.fusion_out(fused)
        return output


# ---------- 预测头----------
class EnhancedFramePredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temporal_attention = TemporalAttention(config.hidden_dim)
        self.tcn = TemporalConvNet(
            channels=config.hidden_dim,
            num_layers=getattr(config, 'tcn_layers', 4),
            kernel_size=getattr(config, 'tcn_kernel_size', 5),
            dropout=config.dropout
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(config.hidden_dim, config.hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.hidden_dim // 2, config.hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(config.hidden_dim // 4, config.num_classes, kernel_size=3, padding=1)
        )
        self.boundary_detector = nn.Sequential(
            nn.Conv1d(config.hidden_dim, config.hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(config.hidden_dim // 2, 1, kernel_size=3, padding=1)
        )

    def forward(self, temporal_features, target_length=None):
        B, C, T = temporal_features.shape
        output_length = target_length if target_length is not None else T * 4

        attended = self.temporal_attention(temporal_features)
        tcn_out = self.tcn(attended)
        combined = attended + tcn_out
        upsampled = F.interpolate(combined, size=output_length, mode='linear', align_corners=False)

        frame_logits = self.classifier(upsampled)
        boundary = self.boundary_detector(upsampled)

        frame_logits_t = frame_logits.transpose(1, 2)
        frame_predictions = torch.argmax(frame_logits_t, dim=-1)
        frame_probs = F.softmax(frame_logits_t, dim=-1)

        return {
            'frame_logits': frame_logits_t,
            'frame_predictions': frame_predictions,
            'frame_probs': frame_probs,
            'boundary_features': boundary.squeeze(1),
            'frame_features': upsampled,
            'output_length': output_length
        }


# ---------- 多模态模块（位置编码 + 双重查询注意力）----------
class TimeEncoder(nn.Module):
    def __init__(self, d_model, max_len=86400):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, time_seconds):
        if time_seconds.dim() == 2:
            time_seconds = time_seconds.unsqueeze(-1)
        indices = time_seconds.long().clamp(0, self.pe.size(0)-1)
        return self.pe[indices.squeeze(-1)]


class EnvironmentEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, env_features):
        return self.fc(env_features)


class CrossAttentionFusion(nn.Module):
    """交叉注意力"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = dim ** -0.5

    def forward(self, query, key, value, key_padding_mask=None):
        B, T, _ = query.shape
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        head_dim = query.size(-1) // self.num_heads
        Q = Q.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


# ---------- 骨骼特征计算 ----------
def compute_bone_features(joint_feat, adj_matrix):
    B, C, T, V = joint_feat.shape
    adj = adj_matrix[None, None, None, :, :]
    joint_expand_j = joint_feat.unsqueeze(-1)
    joint_expand_i = joint_feat.unsqueeze(-2)
    diff = joint_expand_j - joint_expand_i
    masked_diff = diff * adj
    neighbor_count = adj_matrix.sum(dim=1, keepdim=True)
    neighbor_count = neighbor_count.view(1, 1, 1, V)
    bone_feat = masked_diff.sum(dim=-2) / (neighbor_count + 1e-8)
    return bone_feat


# ---------- 双流特征提取器----------
class DualStreamSTGCNFeatureExtractor(nn.Module):
    def __init__(self, config, adj_matrix):
        super().__init__()
        self.register_buffer('adj_matrix', adj_matrix)
        self.joint_stream = HierarchicalFeatureExtractor(config, adj_matrix)
        self.bone_stream = HierarchicalFeatureExtractor(config, adj_matrix)

        use_spatial_attn = getattr(config, 'use_spatial_attention', True)
        if use_spatial_attn:
            self.spatial_attn = LightweightSpatialAttention(
                channels=config.hidden_dim,
                num_joints=config.num_joints,
                reduction=getattr(config, 'spatial_attn_reduction', 16)
            )
        else:
            self.spatial_attn = None

    def forward(self, x):
        bone_feat = compute_bone_features(x, self.adj_matrix)
        joint_out = self.joint_stream(x)
        bone_out = self.bone_stream(bone_feat)
        fused = joint_out + bone_out

        if self.spatial_attn is not None:
            fused = self.spatial_attn(fused)
        return fused


# ---------- 统一模型入口----------
class EnhancedUnifiedSTGCN(nn.Module):
    def __init__(self, config, adj_matrix):
        super().__init__()
        self.config = config
        self.feature_extractor = DualStreamSTGCNFeatureExtractor(config, adj_matrix)
        self.temporal_pool = nn.AdaptiveAvgPool2d((None, 1))

        self.time_encoder = TimeEncoder(d_model=config.time_embedding_dim)
        self.env_encoder = EnvironmentEncoder(
            input_dim=config.env_feature_dim,
            output_dim=config.env_embedding_dim,
            dropout=config.dropout
        )

        self.time_fusion_mlp = nn.Linear(config.hidden_dim + config.time_embedding_dim, config.hidden_dim)
        self.env_fusion_mlp = nn.Linear(config.hidden_dim + config.env_embedding_dim, config.hidden_dim)

        self.time_cross_attn = CrossAttentionFusion(dim=config.hidden_dim,
                                                     num_heads=config.num_attention_heads,
                                                     dropout=config.dropout)
        self.env_cross_attn = CrossAttentionFusion(dim=config.hidden_dim,
                                                    num_heads=config.num_attention_heads,
                                                    dropout=config.dropout)

        self.fusion_ffn = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        self.frame_head = EnhancedFramePredictionHead(config)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, time_seconds=None, env_features=None, target_length=None):
        B, _, C, T, V = x.shape
        x = x.squeeze(1)
        spatial_features = self.feature_extractor(x)
        temporal_features = self.temporal_pool(spatial_features).squeeze(-1)
        pose_feat = temporal_features.transpose(1, 2)

        T_max = temporal_features.shape[2]

        if time_seconds is not None:
            time_down = F.interpolate(time_seconds.unsqueeze(1), size=T_max,
                                      mode='linear', align_corners=False).squeeze(1)
            time_encoded = self.time_encoder(time_down)
        else:
            time_encoded = torch.zeros(B, T_max, self.config.time_embedding_dim, device=x.device)

        if env_features is not None:
            env_down = F.interpolate(env_features.transpose(1, 2), size=T_max,
                                      mode='linear', align_corners=False).transpose(1, 2)
            env_encoded = self.env_encoder(env_down)
        else:
            env_encoded = torch.zeros(B, T_max, self.config.env_embedding_dim, device=x.device)

        time_concat = torch.cat([pose_feat, time_encoded], dim=-1)
        time_query = self.time_fusion_mlp(time_concat)

        env_concat = torch.cat([pose_feat, env_encoded], dim=-1)
        env_query = self.env_fusion_mlp(env_concat)

        pose_time = self.time_cross_attn(time_query, pose_feat, pose_feat)
        pose_env = self.env_cross_attn(env_query, pose_feat, pose_feat)

        fused = torch.cat([pose_time, pose_env], dim=-1)
        fused = self.fusion_ffn(fused)
        fused = fused.transpose(1, 2)

        return self.frame_head(fused, target_length)


class EnhancedBehaviorSegmentationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_joints = config.num_joints
        adj_matrix = torch.zeros(num_joints, num_joints, dtype=torch.float32)
        for i in range(num_joints):
            for j in range(num_joints):
                if i != j:
                    adj_matrix[i, j] = 1.0
            adj_matrix[i, i] = 1.0
        adj_matrix = adj_matrix / (adj_matrix.sum(dim=1, keepdim=True) + 1e-8)
        adj_matrix = adj_matrix.to(config.device)
        self.model = EnhancedUnifiedSTGCN(config, adj_matrix)

    def forward(self, x, time_seconds=None, env_features=None, target_length=None):
        return self.model(x, time_seconds, env_features, target_length)


# ---------- 损失函数----------
class EnhancedFrameClassificationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.focal_gamma = getattr(config, 'focal_gamma', 0.0)
        self.boundary_weight = getattr(config, 'boundary_weight', 0.1)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.temporal_weight = getattr(config, 'temporal_weight', 0.05)

        # 类别权重
        class_weights = getattr(config, 'class_weights', None)
        if class_weights is not None:
            self.register_buffer('class_weight_tensor', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weight_tensor = None

    def focal_loss(self, logits, targets, gamma, weight=None):
        """带类别权重的 Focal Loss"""
        ce_loss = F.cross_entropy(logits, targets, weight=weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def compute_temporal_consistency_loss(self, pred_logits, target_labels):
        B, T, C = pred_logits.shape
        device = pred_logits.device
        loss = torch.tensor(0.0, device=device)
        valid_pairs = 0
        for b in range(B):
            batch_pred = pred_logits[b]
            batch_target = target_labels[b]
            valid = (batch_target != -1)
            indices = torch.where(valid)[0]
            if len(indices) >= 2:
                valid_preds = batch_pred[indices]
                num = len(indices)
                if num > 1:
                    p = F.log_softmax(valid_preds[:-1], dim=-1)
                    q = F.softmax(valid_preds[1:], dim=-1)
                    loss += F.kl_div(p, q, reduction='batchmean')
                    valid_pairs += (num - 1)
        return loss / valid_pairs if valid_pairs > 0 else torch.tensor(0.0, device=device)

    def forward(self, predictions, targets):
        pred_logits = predictions['frame_logits']
        target_labels = targets['behavior_labels'].long()
        num_classes = self.config.num_classes
        target_labels = torch.where(target_labels >= num_classes,
                                    torch.tensor(-1, device=target_labels.device),
                                    target_labels)
        boundary_pred = predictions.get('boundary_features')
        boundary_target = targets.get('boundary_masks')

        B_pred, T_pred, C = pred_logits.shape
        B_target, T_target = target_labels.shape
        if B_pred != B_target:
            raise ValueError(f"Batch size mismatch: pred {B_pred} vs target {B_target}")

        if T_pred != T_target:
            pred_logits_adj = F.interpolate(pred_logits.transpose(1, 2), size=T_target,
                                            mode='linear', align_corners=False).transpose(1, 2)
            if boundary_pred is not None:
                boundary_adj = F.interpolate(boundary_pred.unsqueeze(1), size=T_target,
                                             mode='linear', align_corners=False).squeeze(1)
            else:
                boundary_adj = None
            pred_classes_adj = F.interpolate(predictions['frame_predictions'].unsqueeze(1).float(),
                                             size=T_target, mode='nearest').squeeze(1).long()
        else:
            pred_logits_adj = pred_logits
            boundary_adj = boundary_pred
            pred_classes_adj = predictions['frame_predictions']

        B, T, C = pred_logits_adj.shape
        flat_logits = pred_logits_adj.reshape(-1, C)
        flat_target = target_labels.reshape(-1)
        valid_mask = (flat_target != -1)

        # 获取类别权重张量
        weight = self.class_weight_tensor.to(flat_logits.device) if self.class_weight_tensor is not None else None

        if valid_mask.sum() == 0:
            cls_loss = torch.tensor(0.0, device=pred_logits_adj.device)
        else:
            if self.focal_gamma > 0:
                cls_loss = self.focal_loss(flat_logits[valid_mask], flat_target[valid_mask],
                                           self.focal_gamma, weight)
            else:
                cls_loss = F.cross_entropy(flat_logits[valid_mask], flat_target[valid_mask],
                                           weight=weight, label_smoothing=0.1)

        boundary_loss = torch.tensor(0.0, device=pred_logits_adj.device)
        if boundary_adj is not None and boundary_target is not None:
            if boundary_target.shape[1] != T:
                boundary_target_adj = F.interpolate(boundary_target.unsqueeze(1), size=T,
                                                    mode='linear', align_corners=False).squeeze(1)
                boundary_target_adj = torch.clamp(boundary_target_adj, 0.0, 1.0)
            else:
                boundary_target_adj = boundary_target
            valid_boundary = (target_labels != -1).float()
            if valid_boundary.sum() > 0:
                boundary_loss = self.bce_loss(boundary_adj * valid_boundary,
                                              boundary_target_adj * valid_boundary)

        temporal_loss = self.compute_temporal_consistency_loss(pred_logits_adj, target_labels)

        total_loss = cls_loss + self.boundary_weight * boundary_loss + self.temporal_weight * temporal_loss

        valid_acc = (target_labels != -1)
        if valid_acc.any():
            accuracy = ((pred_classes_adj == target_labels) & valid_acc).float().sum() / valid_acc.float().sum()
        else:
            accuracy = torch.tensor(0.0, device=pred_logits_adj.device)

        loss_dict = {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'boundary_loss': boundary_loss,
            'temporal_consistency_loss': temporal_loss,
            'accuracy': accuracy,
            'pred_time_length': T_pred,
            'target_time_length': T_target
        }
        return total_loss, loss_dict


class SimpleEnhancedFrameClassificationLoss(EnhancedFrameClassificationLoss):
    pass