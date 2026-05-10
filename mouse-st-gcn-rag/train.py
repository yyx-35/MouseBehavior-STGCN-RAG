import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
import seaborn as sns
import random

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stgcn import EnhancedBehaviorSegmentationModel, EnhancedFrameClassificationLoss
from data.dataset import BehaviorDataset


def custom_collate_fn(batch):
    """自定义collate函数，支持多模态数据"""
    collated_batch = {}
    keys = batch[0].keys()

    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            tensors = [item[key] for item in batch]

            if key == 'keypoints':
                stacked = []
                for tensor in tensors:
                    if tensor.dim() == 4 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)
                    stacked.append(tensor)
                collated_batch[key] = torch.stack(stacked).unsqueeze(1)
            elif key in ['time_seconds', 'env_features', 'behavior_labels']:
                collated_batch[key] = torch.stack(tensors)
            else:
                collated_batch[key] = torch.stack(tensors)
        elif key == 'filename':
            collated_batch[key] = [item[key] for item in batch]
        else:
            collated_batch[key] = [item[key] for item in batch]

    return collated_batch


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.output_dir = getattr(config, 'output_dir', 'resultsv1')

        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self.model = EnhancedBehaviorSegmentationModel(config).to(self.device)
        self.print_model_summary()

        self.criterion = EnhancedFrameClassificationLoss(config)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-2,
            betas=(0.9, 0.999)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )

        self.grad_clip = config.grad_clip

        self.train_history = defaultdict(list)

        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_val_topk = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.patience = getattr(config, 'patience', 15)

        # 存储最佳模型的 state_dict
        self.best_model_state_dict = None

        self.setup_directories()

    def print_model_summary(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n{'='*50}\n模型信息:\n{'='*50}")
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"设备: {self.device}")
        print(f"输出根目录: {self.output_dir}\n{'='*50}\n")

    def setup_directories(self):
        dirs = [
            os.path.join(self.output_dir, "visualizations"),
            os.path.join(self.output_dir, "models"),
            os.path.join(self.output_dir, "logs"),
            os.path.join(self.output_dir, "test_results")
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        total_cls_loss = 0
        total_boundary_loss = 0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f'训练 Epoch {epoch}/{self.config.num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            keypoints = batch['keypoints'].to(self.device)
            behavior_labels = batch['behavior_labels'].to(self.device)
            time_seconds = batch.get('time_seconds', None)
            env_features = batch.get('env_features', None)
            if time_seconds is not None:
                time_seconds = time_seconds.to(self.device)
            if env_features is not None:
                env_features = env_features.to(self.device)

            targets = {'behavior_labels': behavior_labels}
            if 'boundary_masks' in batch:
                targets['boundary_masks'] = batch['boundary_masks'].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(keypoints, time_seconds, env_features)
            loss, loss_dict = self.criterion(predictions, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            total_accuracy += loss_dict['accuracy'].item()
            total_cls_loss += loss_dict['classification_loss'].item()
            if 'boundary_loss' in loss_dict:
                total_boundary_loss += loss_dict['boundary_loss'].item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{loss_dict['accuracy'].item():.4f}"
            })

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_boundary_loss = total_boundary_loss / num_batches
        return avg_loss, avg_accuracy, avg_cls_loss, avg_boundary_loss

    @torch.no_grad()
    def evaluate(self, data_loader, split_name='val'):
        """通用评估函数，返回指标字典和百分比混淆矩阵"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_cls_loss = 0
        total_boundary_loss = 0
        num_batches = len(data_loader)

        confusion_matrix = np.zeros((self.config.num_classes, self.config.num_classes), dtype=np.int32)
        all_frame_probs = []
        all_valid_targets = []

        for batch in tqdm(data_loader, desc=f'评估 {split_name}'):
            keypoints = batch['keypoints'].to(self.device)
            behavior_labels = batch['behavior_labels'].to(self.device)
            time_seconds = batch.get('time_seconds', None)
            env_features = batch.get('env_features', None)
            if time_seconds is not None:
                time_seconds = time_seconds.to(self.device)
            if env_features is not None:
                env_features = env_features.to(self.device)

            targets = {'behavior_labels': behavior_labels}
            if 'boundary_masks' in batch:
                targets['boundary_masks'] = batch['boundary_masks'].to(self.device)

            target_length = targets['behavior_labels'].shape[1]
            predictions = self.model(keypoints, time_seconds, env_features, target_length=target_length)
            loss, loss_dict = self.criterion(predictions, targets)

            total_loss += loss.item()
            total_accuracy += loss_dict['accuracy'].item()
            total_cls_loss += loss_dict['classification_loss'].item()
            if 'boundary_loss' in loss_dict:
                total_boundary_loss += loss_dict['boundary_loss'].item()

            # 收集概率用于 Top-K
            frame_probs = predictions['frame_probs']  # (B, T, C)
            B_pred, T_pred, C = frame_probs.shape
            B_target, T_target = behavior_labels.shape
            if T_pred != T_target:
                frame_probs_adj = F.interpolate(frame_probs.transpose(1,2), size=T_target,
                                                mode='linear', align_corners=False).transpose(1,2)
            else:
                frame_probs_adj = frame_probs

            for i in range(B_pred):
                target = behavior_labels[i].cpu().numpy()
                valid = target != -1
                if valid.any():
                    all_frame_probs.append(frame_probs_adj[i][valid].cpu())
                    all_valid_targets.append(behavior_labels[i][valid].cpu())

                pred = predictions['frame_predictions'][i].cpu().numpy()
                min_len = min(len(pred), len(target))
                pred = pred[:min_len]
                target = target[:min_len]
                valid = target != -1
                if valid.any():
                    for p, t in zip(pred[valid], target[valid]):
                        if 0 <= t < self.config.num_classes and 0 <= p < self.config.num_classes:
                            confusion_matrix[t, p] += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_boundary_loss = total_boundary_loss / num_batches if total_boundary_loss > 0 else 0

        # 计算 Top-1 和 Top-K
        if all_frame_probs:
            all_probs = torch.cat(all_frame_probs, dim=0)
            all_targets = torch.cat(all_valid_targets, dim=0)
            _, top1_pred = all_probs.topk(1, dim=1)
            top1_acc = (top1_pred.squeeze() == all_targets).float().mean().item()
            topk = min(self.config.topk, self.config.num_classes)
            _, topk_pred = all_probs.topk(topk, dim=1)
            topk_acc = topk_pred.eq(all_targets.unsqueeze(1)).any(dim=1).float().mean().item()
        else:
            top1_acc = topk_acc = 0.0

        # 计算各类指标
        metrics = self.compute_metrics_from_confusion(confusion_matrix)
        metrics.update({
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'avg_cls_loss': avg_cls_loss,
            'avg_boundary_loss': avg_boundary_loss,
            'top1_accuracy': top1_acc,
            f'top{topk}_accuracy': topk_acc,
            'confusion_matrix': confusion_matrix.tolist()
        })

        # 生成百分比混淆矩阵
        percent_cm = self.confusion_to_percent(confusion_matrix)

        return metrics, percent_cm

    def confusion_to_percent(self, confusion_matrix):
        """将计数混淆矩阵转换为行归一化百分比矩阵"""
        cm = np.array(confusion_matrix)
        row_sums = cm.sum(axis=1, keepdims=True)
        percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
        return percent

    def compute_metrics_from_confusion(self, confusion_matrix):
        num_classes = confusion_matrix.shape[0]
        metrics = {}
        class_metrics = {}
        for cls_idx in range(num_classes):
            tp = confusion_matrix[cls_idx, cls_idx]
            fp = confusion_matrix[:, cls_idx].sum() - tp
            fn = confusion_matrix[cls_idx, :].sum() - tp
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            support = confusion_matrix[cls_idx, :].sum()
            class_metrics[f'{self.config.behavior_names[cls_idx]}({cls_idx})'] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(support)
            }

        macro_precision = np.mean([m['precision'] for m in class_metrics.values()]) if class_metrics else 0.0
        macro_recall = np.mean([m['recall'] for m in class_metrics.values()]) if class_metrics else 0.0
        macro_f1 = np.mean([m['f1'] for m in class_metrics.values()]) if class_metrics else 0.0
        total_samples = confusion_matrix.sum()
        micro_precision = confusion_matrix.trace() / total_samples if total_samples > 0 else 0.0

        metrics.update({
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'micro_precision': float(micro_precision),
            'class_metrics': class_metrics
        })
        return metrics

    def save_percent_confusion_matrix(self, percent_cm, epoch, split_name):
        """保存百分比混淆矩阵图"""
        if epoch % 5 == 0:
            plt.figure(figsize=(12, 10))
            labels = self.config.behavior_names
            sns.heatmap(percent_cm, annot=True, fmt='.1f', cmap='Blues',
                        xticklabels=labels, yticklabels=labels,
                        cbar_kws={'label': 'Percentage (%)'})
            plt.title(f'{split_name.capitalize()} Confusion Matrix (Percentage) - Epoch {epoch}')
            plt.xlabel('Predict')
            plt.ylabel('Ground Truth')
            plt.tight_layout()
            filename = f"{split_name}_confusion_matrix_percent_epoch_{epoch:03d}.png"
            save_path = os.path.join(self.output_dir, "visualizations", filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"已保存 {split_name} 百分比混淆矩阵: {save_path}")

    def validate(self, val_loader, epoch):
        metrics, percent_cm = self.evaluate(val_loader, split_name='val')
        val_accuracy = metrics['avg_accuracy']
        val_loss = metrics['avg_loss']
        val_topk = metrics[f'top{self.config.topk}_accuracy']

        # 保存验证集百分比混淆矩阵
        self.save_percent_confusion_matrix(percent_cm, epoch, split_name='val')

        return val_loss, val_accuracy, metrics

    def train(self, train_loader, val_loader, test_loader):
        print(f"\n开始训练 - 多模态行为预测模型")
        print(f"训练集批次: {len(train_loader)}, 验证集批次: {len(val_loader)}, 测试集批次: {len(test_loader)}")
        print(f"行为类别数: {self.config.num_classes}")

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n{'='*40}\nEpoch {epoch}/{self.config.num_epochs}")

            # 训练一个epoch
            train_loss, train_acc, train_cls_loss, train_boundary_loss = self.train_epoch(train_loader, epoch)
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")

            # 记录训练历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['train_cls_loss'].append(train_cls_loss)
            self.train_history['train_boundary_loss'].append(train_boundary_loss)
            self.train_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # ========== 验证集评估 ==========
            val_loss, val_accuracy, val_metrics = self.validate(val_loader, epoch)
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}")
            print(f"宏F1: {val_metrics['macro_f1']:.4f}, Top-1: {val_metrics['top1_accuracy']:.4f}, Top-{self.config.topk}: {val_metrics[f'top{self.config.topk}_accuracy']:.4f}")

            # 记录验证历史
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_accuracy)
            self.train_history['val_cls_loss'].append(val_metrics['avg_cls_loss'])
            self.train_history['val_boundary_loss'].append(val_metrics['avg_boundary_loss'])
            self.train_history['val_macro_f1'].append(val_metrics['macro_f1'])
            self.train_history['val_macro_precision'].append(val_metrics['macro_precision'])
            self.train_history['val_macro_recall'].append(val_metrics['macro_recall'])
            self.train_history['val_top1_acc'].append(val_metrics['top1_accuracy'])
            self.train_history[f'val_top{self.config.topk}_acc'].append(val_metrics[f'top{self.config.topk}_accuracy'])

            self.scheduler.step()

            # 判断是否为最佳模型（基于验证准确率）
            is_best = val_accuracy > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_accuracy
                self.best_val_loss = val_loss
                self.best_val_topk = val_metrics[f'top{self.config.topk}_accuracy']
                self.best_epoch = epoch
                self.patience_counter = 0

                # 保存最佳模型的state_dict
                self.best_model_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                # 保存最佳模型（权重和完整checkpoint）
                self.save_checkpoint(epoch, val_loss, val_accuracy, val_metrics, is_best=True)
                # 额外保存一份纯权重
                best_weight_path = os.path.join(self.output_dir, "models", "best_model_weights.pth")
                torch.save(self.best_model_state_dict, best_weight_path)
            else:
                self.patience_counter += 1
                # 非最佳模型也定期保存checkpoint
                if epoch % 10 == 0:
                    self.save_checkpoint(epoch, val_loss, val_accuracy, val_metrics, is_best=False)

            # 早停判断
            if self.patience_counter >= self.patience:
                print(f"早停触发! {self.patience}个epoch验证准确率无改善")
                break

            # 定期保存训练曲线
            if epoch % 10 == 0 or epoch == self.config.num_epochs:
                self.plot_training_history(epoch)

            # 记录日志（不含测试指标）
            self.save_training_log(epoch, train_loss, train_acc, train_cls_loss, train_boundary_loss,
                                   val_loss, val_accuracy, val_metrics)

        print(f"\n{'='*40}\n训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"最佳验证准确率: {self.best_val_accuracy:.4f}")
        print(f"最佳Top-{self.config.topk}准确率: {self.best_val_topk:.4f}")
        print(f"最佳Epoch: {self.best_epoch}")

        # 最终保存模型和曲线
        self.save_final_model()
        self.save_final_curves()

        # ========== 训练结束后，使用最佳模型进行一次测试 ==========
        if self.best_model_state_dict is not None:
            print("\n加载最佳模型进行最终测试...")
            self.model.load_state_dict(self.best_model_state_dict)
            self.model = self.model.to(self.device)
            test_metrics, test_percent_cm = self.evaluate(test_loader, split_name='test')
            # 保存测试集百分比混淆矩阵
            self.save_percent_confusion_matrix(test_percent_cm, self.best_epoch, split_name='test_best')
            # 保存详细类别指标到JSON
            test_results = {
                'best_epoch': self.best_epoch,
                'test_metrics': {
                    'loss': test_metrics['avg_loss'],
                    'accuracy': test_metrics['avg_accuracy'],
                    'top1_accuracy': test_metrics['top1_accuracy'],
                    f'top{self.config.topk}_accuracy': test_metrics[f'top{self.config.topk}_accuracy'],
                    'macro_f1': test_metrics['macro_f1'],
                    'macro_precision': test_metrics['macro_precision'],
                    'macro_recall': test_metrics['macro_recall'],
                },
                'class_metrics': test_metrics['class_metrics']
            }
            test_results_path = os.path.join(self.output_dir, "test_results", "final_test_results.json")
            with open(test_results_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
            print(f"最终测试结果已保存至 {test_results_path}")

            # 打印各类别指标
            print("\n========== 各类别测试指标 ==========")
            for class_name, metrics in test_metrics['class_metrics'].items():
                print(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, Support={metrics['support']}")
            print(f"\n总体指标: Accuracy={test_metrics['avg_accuracy']:.4f}, Macro F1={test_metrics['macro_f1']:.4f}")
        else:
            print("警告: 未找到最佳模型，跳过测试。")

    def save_checkpoint(self, epoch, val_loss, val_accuracy, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'metrics': metrics,
            'config': self.config.__dict__
        }
        if is_best:
            best_path = os.path.join(self.output_dir, "models", "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型, 验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}")
        elif epoch % 10 == 0:
            ckpt_path = os.path.join(self.output_dir, "models", f"checkpoint_epoch_{epoch:03d}.pth")
            torch.save(checkpoint, ckpt_path)

    def save_training_log(self, epoch, train_loss, train_accuracy, train_cls_loss, train_boundary_loss,
                          val_loss=None, val_accuracy=None, metrics=None):
        log_entry = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_accuracy': float(train_accuracy),
            'train_cls_loss': float(train_cls_loss),
            'train_boundary_loss': float(train_boundary_loss),
            'val_loss': float(val_loss) if val_loss is not None else None,
            'val_accuracy': float(val_accuracy) if val_accuracy is not None else None,
            'learning_rate': float(self.optimizer.param_groups[0]['lr']),
            'timestamp': datetime.now().isoformat()
        }
        if metrics:
            log_entry.update({
                'macro_f1': metrics.get('macro_f1'),
                'macro_precision': metrics.get('macro_precision'),
                'macro_recall': metrics.get('macro_recall'),
                'top1_accuracy': metrics.get('top1_accuracy'),
                f'top{self.config.topk}_accuracy': metrics.get(f'top{self.config.topk}_accuracy')
            })

        log_file = os.path.join(self.output_dir, "logs", "training_log.json")
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

    def plot_training_history(self, epoch):
        """绘制训练曲线（仅训练和验证指标）"""
        if len(self.train_history['train_loss']) < 2:
            return

        epochs = range(1, len(self.train_history['train_loss']) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=16)

        # Loss
        ax = axes[0,0]
        ax.plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss')
        if 'val_loss' in self.train_history and self.train_history['val_loss']:
            val_epochs = range(1, len(self.train_history['val_loss']) + 1)
            ax.plot(val_epochs, self.train_history['val_loss'], 'r-', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # Accuracy
        ax = axes[0,1]
        ax.plot(epochs, self.train_history['train_acc'], 'b-', label='Train Acc')
        if 'val_acc' in self.train_history and self.train_history['val_acc']:
            val_epochs = range(1, len(self.train_history['val_acc']) + 1)
            ax.plot(val_epochs, self.train_history['val_acc'], 'r-', label='Val Acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Curves')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # Macro F1
        ax = axes[1,0]
        if 'val_macro_f1' in self.train_history and self.train_history['val_macro_f1']:
            val_epochs = range(1, len(self.train_history['val_macro_f1']) + 1)
            ax.plot(val_epochs, self.train_history['val_macro_f1'], 'r-', label='Val Macro F1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Macro F1')
        ax.set_title('Macro F1 Curve')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # Learning Rate
        ax = axes[1,1]
        ax.plot(epochs, self.train_history['learning_rate'], 'm-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "visualizations", f"training_curves_epoch_{epoch:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_final_curves(self):
        """最终保存训练曲线"""
        if len(self.train_history['train_loss']) == 0:
            return
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Final Training Progress', fontsize=16)

        # Loss
        ax = axes[0,0]
        ax.plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss')
        if 'val_loss' in self.train_history and self.train_history['val_loss']:
            val_epochs = range(1, len(self.train_history['val_loss']) + 1)
            ax.plot(val_epochs, self.train_history['val_loss'], 'r-', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # Accuracy
        ax = axes[0,1]
        ax.plot(epochs, self.train_history['train_acc'], 'b-', label='Train Acc')
        if 'val_acc' in self.train_history and self.train_history['val_acc']:
            val_epochs = range(1, len(self.train_history['val_acc']) + 1)
            ax.plot(val_epochs, self.train_history['val_acc'], 'r-', label='Val Acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Curves')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # Macro F1
        ax = axes[1,0]
        if 'val_macro_f1' in self.train_history and self.train_history['val_macro_f1']:
            val_epochs = range(1, len(self.train_history['val_macro_f1']) + 1)
            ax.plot(val_epochs, self.train_history['val_macro_f1'], 'r-', label='Val Macro F1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Macro F1')
        ax.set_title('Macro F1 Curve')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # Learning Rate
        ax = axes[1,1]
        ax.plot(epochs, self.train_history['learning_rate'], 'm-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        final_path = os.path.join(self.output_dir, "visualizations", "final_training_curves.png")
        plt.savefig(final_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"最终训练曲线已保存至 {final_path}")

    def save_final_model(self):
        final_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
            'train_history': dict(self.train_history),
        }
        final_path = os.path.join(self.output_dir, "models", "final_model.pth")
        torch.save(final_checkpoint, final_path)
        print(f"最终模型保存到: {final_path}")


def main():
    from configs.stgcn_config import STGCNConfig
    config = STGCNConfig()

    # 检查数据划分文件
    split_file = config.data_split_json
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"数据划分文件不存在: {split_file}，请先准备好。")

    print(f"\n配置信息:")
    print(f"设备: {config.device}")
    print(f"批次大小: {config.batch_size}")
    print(f"学习率: {config.learning_rate}")
    print(f"Epoch数: {config.num_epochs}")
    print(f"输出根目录: {config.output_dir}")
    print(f"数据增强: {'启用' if config.use_augmentation else '禁用'} (仅训练集)")

    # 创建三个数据集
    train_dataset = BehaviorDataset(config, split='train')
    val_dataset   = BehaviorDataset(config, split='val')
    test_dataset  = BehaviorDataset(config, split='test')

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    trainer = Trainer(config)
    trainer.train(train_loader, val_loader, test_loader)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
