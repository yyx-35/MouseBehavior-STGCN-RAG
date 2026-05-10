"""优化版：延迟加载 + 模型卸载 + Qwen子进程GPU运行 + 详细日志"""
import os
import sys
import cv2
import re
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon, LineString
import json
from collections import defaultdict, Counter
import warnings
import multiprocessing as mp
import time
import traceback
import logging

# ======================== 日志配置（仅输出到终端） ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 设置多进程启动方式
if sys.platform == 'win32':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

warnings.filterwarnings('ignore')

from ultralytics import YOLO
from models.stgcn import EnhancedBehaviorSegmentationModel

# ======================== 配置参数 ========================
class PipelineConfig:
    hidden_dim = 256
    num_classes = 9
    num_features = 2
    num_frames = 75
    num_joints = 6
    dropout = 0.3
    behavior_names = ['walk', 'stop', 'drinking', 'feeding', 'sniffing',
                      'unsupported rearing', 'supported rearing', 'grooming', 'digging']
    behavior_names_cn = ['行走', '静止', '喝水', '进食', '嗅探', '不支撑直立', '支撑直立', '梳理', '挖洞']
    time_embedding_dim = 128
    env_feature_dim = 175
    env_embedding_dim = 128
    num_attention_heads = 16
    use_spatial_attention = True
    spatial_attn_reduction = 16
    tcn_layers = 4
    tcn_kernel_size = 5
    device = torch.device('cpu')
    env_json_path = r"\path\to\frame.json"
    model_weight_path = r"\path\to\pth"
    output_dir = r"./behavior_predictions"
    keypoint_batch_size = 8
    window_stride = 30
    yolo_model_path = r"\path\to\yolo\pt"
    qwen_model_name = "/path/to/qwen/Qwen2-1.5B-Instruct"
    qwen_cache_dir = "/path/to/models/qwen"
    use_modelscope = False
    use_rag = True
    rag_persist_dir = "/path/to/rag_db"
    rag_collection_name = "mouse_behavior_literature"


# ======================== Qwen 子进程 ========================
def _qwen_subprocess(pipe_conn, model_name, cache_dir, use_modelscope):
    # 子进程独立日志
    sub_logger = logging.getLogger("QwenSub")
    sub_logger.setLevel(logging.INFO)
    if not sub_logger.handlers:
        sub_logger.addHandler(logging.StreamHandler(sys.stderr))
    sub_logger.propagate = False

    # ========== 设置 CUDA_VISIBLE_DEVICES ==========
    # 避免继承父进程可能设置的空值或无效值
    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        sub_logger.info("已设置 CUDA_VISIBLE_DEVICES=0")
    else:
        sub_logger.info(f"继承 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    # ==========  GPU 可用性检测 ==========
    use_cuda = False
    device = torch.device('cpu')
    try:
        # 1. 检查基本可用性
        if torch.cuda.is_available():
            # 2. 获取设备数量
            num_devices = torch.cuda.device_count()
            sub_logger.info(f"检测到 {num_devices} 个 CUDA 设备")
            if num_devices > 0:
                # 3. 尝试创建一个张量在 GPU 上
                test_tensor = torch.tensor([1.0]).cuda()
                sub_logger.info(f"GPU 张量创建成功，设备: {test_tensor.device}")
                # 4. 尝试获取设备名称
                try:
                    device_name = torch.cuda.get_device_name(0)
                    sub_logger.info(f"GPU 名称: {device_name}")
                except Exception as e:
                    sub_logger.warning(f"无法获取 GPU 名称（不影响使用）: {e}")
                use_cuda = True
                device = torch.device('cuda')
                # 清理测试张量
                del test_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                sub_logger.warning("CUDA 可用但设备数为 0")
        else:
            sub_logger.info("CUDA 不可用")
    except Exception as e:
        sub_logger.error(f"GPU 初始化测试失败: {e}")
        sub_logger.info("将使用 CPU 运行模型")

    if use_cuda:
        sub_logger.info("最终决定: 使用 GPU")
        # 打印显存信息
        try:
            free, total = torch.cuda.mem_get_info(0)
            sub_logger.info(f"GPU 显存: 空闲 {free/1024**3:.2f} GB / 总计 {total/1024**3:.2f} GB")
        except:
            pass
    else:
        sub_logger.info("最终决定: 使用 CPU")

    # ========== 加载模型 ==========
    try:
        if use_modelscope:
            try:
                from modelscope import AutoTokenizer, AutoModelForCausalLM, snapshot_download
                sub_logger.info(f"从 ModelScope 加载模型: {model_name}")
                model_dir = snapshot_download(model_name, cache_dir=cache_dir)
                tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                if use_cuda:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    ).to(device)
            except Exception as e:
                sub_logger.error(f"ModelScope 失败: {e}，尝试 HuggingFace")
                from transformers import AutoTokenizer, AutoModelForCausalLM
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
                if use_cuda:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, cache_dir=cache_dir,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, cache_dir=cache_dir,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    ).to(device)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
            if use_cuda:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, cache_dir=cache_dir,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, cache_dir=cache_dir,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to(device)

        model.eval()
        if use_cuda:
            sub_logger.info(f"模型加载后 GPU 显存占用: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        sub_logger.info("Qwen 模型加载完成，等待请求...")
        pipe_conn.send("READY")
    except Exception as e:
        error_msg = f"子进程初始化失败: {str(e)}\n{traceback.format_exc()}"
        sub_logger.error(error_msg)
        pipe_conn.send(f"ERROR:{error_msg}")
        pipe_conn.close()
        return

    # ========== 请求处理循环 ==========
    while True:
        try:
            data = pipe_conn.recv()
            if data is None:
                break
            prompt, max_new_tokens, temperature = data
            sub_logger.info(f"收到生成请求，prompt长度: {len(prompt)}, max_new_tokens={max_new_tokens}, temp={temperature}")
            start_time = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].lstrip()
            elapsed = time.time() - start_time
            sub_logger.info(f"生成完成，耗时 {elapsed:.2f}s，输出长度: {len(response)} 字符")
            pipe_conn.send(response)
        except (EOFError, BrokenPipeError):
            sub_logger.info("主进程断开连接，退出")
            break
        except Exception as e:
            sub_logger.error(f"生成失败: {str(e)}\n{traceback.format_exc()}")
            try:
                pipe_conn.send(f"[错误] 生成失败: {str(e)}")
            except:
                break
    sub_logger.info("子进程退出")


class QwenSubprocessManager:
    def __init__(self, model_name: str, cache_dir: str, use_modelscope: bool = True):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_modelscope = use_modelscope
        self.parent_conn = None
        self.process = None
        self._start()

    def _start(self):
        try:
            self.parent_conn, child_conn = mp.Pipe()
            self.process = mp.Process(
                target=_qwen_subprocess,
                args=(child_conn, self.model_name, self.cache_dir, self.use_modelscope)
            )
            self.process.start()
            logger.info(f"Qwen 子进程已启动，PID: {self.process.pid}")
            if self.parent_conn.poll(180):
                msg = self.parent_conn.recv()
                if msg.startswith("ERROR:"):
                    raise RuntimeError(msg[6:])
                elif msg == "READY":
                    logger.info("Qwen 子进程已就绪（GPU）")
                else:
                    raise RuntimeError(f"未知信号: {msg}")
            else:
                raise TimeoutError("子进程启动超时（180秒）")
        except Exception as e:
            logger.error(f"启动 Qwen 子进程失败: {str(e)}\n{traceback.format_exc()}")
            self.shutdown()
            raise RuntimeError(f"无法启动 Qwen 子进程: {str(e)}")

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def generate_response(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        if not self.is_alive():
            logger.error("Qwen 子进程已死亡，无法处理请求")
            return "问答模型未就绪（子进程已退出），请重新分析视频"

        try:
            # 记录请求前的状态
            if torch.cuda.is_available():
                logger.info(f"请求前GPU显存: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
            logger.info(f"发送生成请求，prompt长度: {len(prompt)}")
            self.parent_conn.send((prompt, max_new_tokens, temperature))
            if self.parent_conn.poll(120):  # 2分钟超时
                response = self.parent_conn.recv()
                logger.info(f"收到响应，长度: {len(response)}")
                return response
            else:
                logger.error("生成超时（120秒）")
                return "生成超时，请重试"
        except Exception as e:
            logger.error(f"问答通信失败: {str(e)}\n{traceback.format_exc()}")
            return f"通信失败: {str(e)}"

    def shutdown(self):
        if self.parent_conn is not None:
            try:
                self.parent_conn.send(None)
                self.parent_conn.close()
            except:
                pass
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
        self.parent_conn = None
        self.process = None
        logger.info("Qwen 子进程已关闭")


# ======================== 时间提取器 ========================
class VideoTimeExtractor:
    def __init__(self):
        self._ocr = None
        self.last_dt_obj = None

    @property
    def ocr(self):
        if self._ocr is None:
            try:
                os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(use_angle_cls=False, lang='ch', show_log=False)
                logger.info("OCR 模型加载成功")
            except Exception as e:
                logger.error(f"OCR 加载失败: {e}")
                raise
        return self._ocr

    def get_frame_at_index(self, video_path: str, frame_index: int) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def extract_datetime_from_text(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        pattern = r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})'
        match = re.search(pattern, text)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def extract_from_video(self, video_path: str) -> float:
        for frame_idx in range(3):
            frame = self.get_frame_at_index(video_path, frame_idx)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            crop_img = frame[0:h//10, 0:w//3]
            try:
                result = self.ocr.ocr(crop_img, cls=False)
                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0] if isinstance(line[1], tuple) else line[1]
                        d, t = self.extract_datetime_from_text(text)
                        if d and t:
                            dt_obj = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S")
                            seconds = dt_obj.hour * 3600 + dt_obj.minute * 60 + dt_obj.second
                            self.last_dt_obj = dt_obj
                            return float(seconds)
            except Exception as e:
                logger.warning(f"OCR 识别帧 {frame_idx} 失败: {e}")
                continue
        if self.last_dt_obj is not None:
            new_dt = self.last_dt_obj + timedelta(seconds=5)
            seconds = new_dt.hour * 3600 + new_dt.minute * 60 + new_dt.second
            self.last_dt_obj = new_dt
            return float(seconds)
        logger.warning(f"视频 {Path(video_path).name} 时间提取失败，使用默认值0")
        return 0.0


# ======================== 关键点检测器 ========================
class VideoKeypointDetector:
    def __init__(self, model_path: str, batch_size: int = 8):
        try:
            self.model = YOLO(model_path)
            self.batch_size = batch_size
            self.model.to('cpu')
            logger.info(f"YOLO 关键点模型加载成功: {model_path}")
        except Exception as e:
            logger.error(f"YOLO 模型加载失败: {e}")
            raise

    def process_video(self, video_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_buffer = []
        all_keypoints = []
        all_centers = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_buffer.append(frame)
            frame_idx += 1
            if len(frame_buffer) == self.batch_size or frame_idx == total_frames:
                try:
                    results = self.model(frame_buffer, verbose=False, device='cpu')
                except Exception as e:
                    logger.error(f"YOLO 推理失败: {e}")
                    break
                for result in results:
                    kps = np.zeros((6, 2), dtype=np.float32)
                    center = np.zeros(2, dtype=np.float32)
                    if result.keypoints is not None and result.boxes is not None:
                        keypoints = result.keypoints.xy.cpu().numpy()
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        if len(boxes) > 0:
                            best_idx = np.argmax(confs)
                            box = boxes[best_idx]
                            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                            if keypoints is not None and len(keypoints) > best_idx:
                                kps = keypoints[best_idx][:, :2]
                    all_keypoints.append(kps)
                    all_centers.append(center)
                frame_buffer = []
        cap.release()
        if len(all_keypoints) == 0:
            logger.error("未检测到任何关键点")
            return np.array([]), np.array([]), fps
        logger.info(f"关键点检测完成: {len(all_keypoints)} 帧")
        return np.stack(all_keypoints, axis=0), np.stack(all_centers, axis=0), fps


# ======================== 环境特征计算器 ========================
class EnvironmentFeatureComputer:
    def __init__(self, env_json_path: str):
        try:
            with open(env_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.polygons = {}
            for shape in data['shapes']:
                label = shape['label']
                points = shape['points']
                if points[0] != points[-1]:
                    points.append(points[0])
                self.polygons[label] = Polygon(points)
            logger.info(f"环境特征加载成功，包含: {list(self.polygons.keys())}")
        except Exception as e:
            logger.error(f"环境特征加载失败: {e}")
            raise

    def sample_points_on_polygon(self, polygon: Polygon, num_points: int) -> List[Point]:
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
        return [boundary.interpolate(d) for d in distances]

    def compute_features_for_frame(self, keypoints_frame: np.ndarray, bbox_center: np.ndarray) -> np.ndarray:
        mouse_points = [Point(kp[0], kp[1]) for kp in keypoints_frame] + [Point(bbox_center[0], bbox_center[1])]
        obj_points = {}
        if '食槽' in self.polygons and not self.polygons['食槽'].is_empty:
            poly = self.polygons['食槽']
            coords = list(poly.exterior.coords)[:-1]
            if len(coords) >= 4:
                corners = [Point(coords[0]), Point(coords[1]), Point(coords[2]), Point(coords[3])]
            else:
                corners = self.sample_points_on_polygon(poly, 4)
            center = Point(poly.centroid.x, poly.centroid.y)
            uniform = self.sample_points_on_polygon(poly, 5)
            obj_points['食槽'] = corners + [center] + uniform
        else:
            obj_points['食槽'] = [Point(0, 0)] * 10
        if '饮水器' in self.polygons and not self.polygons['饮水器'].is_empty:
            poly = self.polygons['饮水器']
            coords = list(poly.exterior.coords)[:-1]
            if len(coords) >= 4:
                corners = [Point(coords[0]), Point(coords[1]), Point(coords[2]), Point(coords[3])]
            else:
                corners = self.sample_points_on_polygon(poly, 4)
            center = Point(poly.centroid.x, poly.centroid.y)
            uniform = self.sample_points_on_polygon(poly, 2)
            obj_points['饮水器'] = corners + [center] + uniform
        else:
            obj_points['饮水器'] = [Point(0, 0)] * 7
        if '基座' in self.polygons and not self.polygons['基座'].is_empty:
            poly = self.polygons['基座']
            coords = list(poly.exterior.coords)[:-1]
            if len(coords) >= 4:
                corners = [Point(coords[0]), Point(coords[1]), Point(coords[2]), Point(coords[3])]
            else:
                corners = self.sample_points_on_polygon(poly, 4)
            center = Point(poly.centroid.x, poly.centroid.y)
            uniform = self.sample_points_on_polygon(poly, 3)
            obj_points['基座'] = corners + [center] + uniform
        else:
            obj_points['基座'] = [Point(0, 0)] * 8
        features = []
        for obj_name in ['食槽', '饮水器', '基座']:
            for mouse_pt in mouse_points:
                for obj_pt in obj_points[obj_name]:
                    features.append(mouse_pt.distance(obj_pt))
        return np.array(features, dtype=np.float32)


# ======================== 行为识别模型 ========================
class BehaviorInferenceModel:
    def __init__(self, config: PipelineConfig):
        self.config = config
        from types import SimpleNamespace
        model_config = SimpleNamespace(
            num_frames=config.num_frames,
            num_joints=config.num_joints,
            num_features=config.num_features,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            behavior_names=config.behavior_names,
            env_feature_dim=config.env_feature_dim,
            env_embedding_dim=config.env_embedding_dim,
            time_embedding_dim=config.time_embedding_dim,
            dropout=config.dropout,
            num_attention_heads=config.num_attention_heads,
            use_spatial_attention=config.use_spatial_attention,
            spatial_attn_reduction=config.spatial_attn_reduction,
            tcn_layers=config.tcn_layers,
            tcn_kernel_size=config.tcn_kernel_size,
            device=config.device,
            grad_clip=1.0, learning_rate=0.001, patience=10,
            val_interval=1, num_epochs=100, max_boundaries=5,
            boundary_weight=0.1, temporal_weight=0.05, focal_gamma=0.0
        )
        try:
            self.model = EnhancedBehaviorSegmentationModel(model_config).to(config.device)
            checkpoint = torch.load(config.model_weight_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict, strict=True)
            self.model.eval()
            logger.info(f"行为识别模型加载成功: {config.model_weight_path}")
        except Exception as e:
            logger.error(f"行为识别模型加载失败: {e}")
            raise

    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
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

    def predict_window(self, keypoints_window: np.ndarray, time_seconds: float,
                       env_features_window: np.ndarray) -> np.ndarray:
        T = keypoints_window.shape[0]
        assert T == self.config.num_frames
        normalized = self.normalize_keypoints(keypoints_window)
        time_array = np.full(self.config.num_frames, time_seconds, dtype=np.float32)
        keypoints_tensor = torch.tensor(normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        time_tensor = torch.tensor(time_array, dtype=torch.float32).unsqueeze(0)
        env_tensor = torch.tensor(env_features_window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            predictions = self.model(
                keypoints_tensor.to(self.config.device),
                time_tensor.to(self.config.device),
                env_tensor.to(self.config.device),
                target_length=self.config.num_frames
            )
        return predictions['frame_predictions'][0].cpu().numpy()

    def predict_sliding_window(self, keypoints: np.ndarray, env_features: np.ndarray,
                               fps: float, start_time_seconds: float) -> np.ndarray:
        T = keypoints.shape[0]
        W = self.config.num_frames
        stride = self.config.window_stride
        frame_votes = defaultdict(list)
        for start in range(0, T, stride):
            end = start + W
            if end > T:
                break
            window_kps = keypoints[start:end]
            window_env = env_features[start:end]
            window_time = start / fps
            preds = self.predict_window(window_kps, window_time, window_env)
            for offset, label in enumerate(preds):
                frame_votes[start + offset].append(label)
        final_labels = np.zeros(T, dtype=np.int64)
        for f_idx, votes in frame_votes.items():
            if votes:
                counter = Counter(votes)
                final_labels[f_idx] = counter.most_common(1)[0][0]
        for f_idx in range(T):
            if f_idx not in frame_votes:
                nearest = min(frame_votes.keys(), key=lambda x: abs(x - f_idx))
                final_labels[f_idx] = final_labels[nearest]
        return final_labels


# ======================== 行为语义分析 ========================
class BehaviorAnalyzer:
    def __init__(self, behavior_names: List[str], behavior_names_cn: List[str]):
        self.behavior_names = behavior_names
        self.behavior_names_cn = behavior_names_cn
        self.en_to_cn = {en: cn for en, cn in zip(behavior_names, behavior_names_cn)}

    def labels_to_segments(self, labels: np.ndarray, fps: float, start_time_seconds: float) -> List[Dict]:
        segments = []
        if len(labels) == 0:
            return segments
        current_label = labels[0]
        start_idx = 0
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                end_idx = i - 1
                segments.append({
                    'behavior': self.behavior_names[current_label],
                    'label_id': int(current_label),
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'start_time': start_idx / fps,
                    'end_time': end_idx / fps,
                    'duration_sec': (end_idx - start_idx + 1) / fps
                })
                current_label = labels[i]
                start_idx = i
        end_idx = len(labels) - 1
        segments.append({
            'behavior': self.behavior_names[current_label],
            'label_id': int(current_label),
            'start_frame': start_idx,
            'end_frame': end_idx,
            'start_time': start_idx / fps,
            'end_time': end_idx / fps,
            'duration_sec': (end_idx - start_idx + 1) / fps
        })
        return segments

    def compute_behavior_stats(self, segments: List[Dict]) -> Dict:
        total_duration = sum(seg['duration_sec'] for seg in segments)
        stats = {}
        for behavior in self.behavior_names:
            total = sum(seg['duration_sec'] for seg in segments if seg['behavior'] == behavior)
            stats[behavior] = {
                'duration_sec': total,
                'percentage': total / total_duration * 100 if total_duration > 0 else 0
            }
        return stats

    def generate_5s_windows(self, segments: List[Dict], fps: float, start_time: float, end_time: float) -> List[Dict]:
        windows = []
        current = start_time
        while current < end_time:
            window_end = min(current + 5.0, end_time)
            window_behaviors = []
            for seg in segments:
                seg_start = seg['start_time']
                seg_end = seg['end_time']
                if seg_end >= current and seg_start <= window_end:
                    overlap_start = max(seg_start, current)
                    overlap_end = min(seg_end, window_end)
                    duration = overlap_end - overlap_start
                    if duration > 0:
                        window_behaviors.append({
                            'behavior': seg['behavior'],
                            'duration': duration,
                            'start_rel': overlap_start - current,
                            'end_rel': overlap_end - current
                        })
            window_behaviors.sort(key=lambda x: x['start_rel'])
            merged = []
            for b in window_behaviors:
                if merged and merged[-1]['behavior'] == b['behavior']:
                    merged[-1]['duration'] += b['duration']
                    merged[-1]['end_rel'] = b['end_rel']
                else:
                    merged.append(b)
            windows.append({
                'window_start': current,
                'window_end': window_end,
                'behaviors': merged
            })
            current = window_end
        return windows

    def find_behavior_change_points(self, segments: List[Dict]) -> List[Dict]:
        changes = []
        for i in range(1, len(segments)):
            prev = segments[i-1]
            curr = segments[i]
            changes.append({
                'time': curr['start_time'],
                'from_behavior': prev['behavior'],
                'to_behavior': curr['behavior'],
                'from_duration': prev['duration_sec'],
                'to_duration': curr['duration_sec']
            })
        return changes

    def _format_abs_time(self, start_datetime: Optional[datetime], offset_seconds: float) -> str:
        if start_datetime is not None:
            try:
                abs_time = start_datetime + timedelta(seconds=offset_seconds)
                return abs_time.strftime('%H:%M:%S')
            except Exception:
                pass
        return f"{offset_seconds:.1f}s"

    def generate_optimized_prompt(self, segments: List[Dict], stats: Dict,
                                  windows: List[Dict], changes: List[Dict],
                                  video_name: str, start_datetime: Optional[datetime],
                                  fps: float, total_duration: float) -> str:
        lines = []
        lines.append(f"小鼠行为分析报告 - {video_name}")
        if start_datetime:
            lines.append(f"视频开始绝对时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"视频总时长: {total_duration:.1f} 秒")
        lines.append(f"帧率: {fps:.2f} fps\n")

        lines.append("行为类别（共9种，ID-中文名称）:")
        for idx, cn in enumerate(self.behavior_names_cn):
            lines.append(f"  {idx} - {cn}")
        lines.append("")

        lines.append("行为统计占比:")
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['duration_sec'], reverse=True)
        for beh, data in sorted_stats:
            if data['duration_sec'] > 0.1:
                cn = self.en_to_cn.get(beh, beh)
                lines.append(f"  {cn} ({beh}): {data['duration_sec']:.1f} 秒 ({data['percentage']:.1f}%)")
        lines.append("")

        lines.append("行为变化点（绝对时间）:")
        for ch in changes:
            time_str = self._format_abs_time(start_datetime, ch['time'])
            from_cn = self.en_to_cn.get(ch['from_behavior'], ch['from_behavior'])
            to_cn = self.en_to_cn.get(ch['to_behavior'], ch['to_behavior'])
            lines.append(f"  {time_str}: 从 {from_cn} 变为 {to_cn} (前一行为持续 {ch['from_duration']:.1f}s)")
        lines.append("")

        lines.append("每5秒窗口内的行为变化（绝对时间）:")
        for win in windows:
            start_str = self._format_abs_time(start_datetime, win['window_start'])
            end_str = self._format_abs_time(start_datetime, win['window_end'])
            lines.append(f"  [{start_str} - {end_str}]")
            if not win['behaviors']:
                lines.append("    无行为数据")
            else:
                for beh in win['behaviors']:
                    cn = self.en_to_cn.get(beh['behavior'], beh['behavior'])
                    lines.append(f"    {cn} 持续 {beh['duration']:.1f} 秒")
        lines.append("")

        lines.append("完整行为片段序列:")
        for seg in segments[:30]:
            start_str = self._format_abs_time(start_datetime, seg['start_time'])
            end_str = self._format_abs_time(start_datetime, seg['end_time'])
            cn = self.en_to_cn.get(seg['behavior'], seg['behavior'])
            lines.append(f"  {cn} {start_str} - {end_str} (持续 {seg['duration_sec']:.1f}s)")
        if len(segments) > 30:
            lines.append(f"  ... 共 {len(segments)} 个片段")
        return "\n".join(lines)


# ======================== RAG 检索器 ========================
class RAGRetriever:
    def __init__(self, persist_dir: str, collection_name: str):
        try:
            import chromadb
            from chromadb.config import Settings
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"RAG 检索器已加载，知识库包含 {self.collection.count()} 个文档片段")
        except Exception as e:
            logger.error(f"RAG 加载失败: {e}")
            raise

    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        results = self.collection.query(query_texts=[query], n_results=k)
        docs = []
        if results['documents'] and results['documents'][0]:
            for i, text in enumerate(results['documents'][0]):
                docs.append({
                    "text": text,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None
                })
        return docs


# ======================== 主 Pipeline ========================
class EnhancedVideoBehaviorPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        logger.info("初始化视频行为识别 Pipeline（模型将按需加载）...")
        try:
            self._env_computer = EnvironmentFeatureComputer(config.env_json_path)
            self.behavior_analyzer = BehaviorAnalyzer(config.behavior_names, config.behavior_names_cn)
        except Exception as e:
            logger.error(f"初始化轻量组件失败: {e}")
            raise
        self._keypoint_detector = None
        self._behavior_model = None
        self._time_extractor = None
        self._rag_retriever = None
        self._llm_manager = None

    @property
    def keypoint_detector(self):
        if self._keypoint_detector is None:
            self._keypoint_detector = VideoKeypointDetector(
                self.config.yolo_model_path, self.config.keypoint_batch_size
            )
        return self._keypoint_detector

    @property
    def behavior_model(self):
        if self._behavior_model is None:
            self._behavior_model = BehaviorInferenceModel(self.config)
        return self._behavior_model

    @property
    def time_extractor(self):
        if self._time_extractor is None:
            self._time_extractor = VideoTimeExtractor()
        return self._time_extractor

    def unload_behavior_models(self):
        self._keypoint_detector = None
        self._behavior_model = None
        self._time_extractor = None
        import gc
        gc.collect()
        logger.info("行为识别模型已卸载，内存已释放")

    def process_video(self, video_path: str, output_name: Optional[str] = None) -> Dict:
        video_path = Path(video_path)
        output_name = output_name or video_path.stem

        logger.info(f"\n处理视频: {video_path}")
        # 1. 时间提取
        logger.info("  步骤1/5: 提取视频起始时间...")
        start_seconds = self.time_extractor.extract_from_video(str(video_path))
        if hasattr(self.time_extractor, 'last_dt_obj') and self.time_extractor.last_dt_obj:
            base_date = self.time_extractor.last_dt_obj.date()
        else:
            base_date = datetime.now().date()
        start_datetime = datetime.combine(base_date, datetime.min.time()) + timedelta(seconds=start_seconds)
        logger.info(f"    起始时间: {start_seconds} 秒 ({start_datetime})")

        # 2. 关键点检测
        logger.info("  步骤2/5: 检测关键点和帧率...")
        keypoints, centers, fps = self.keypoint_detector.process_video(str(video_path))
        if keypoints.shape[0] == 0:
            logger.error("未检测到任何关键点，跳过")
            return {}
        logger.info(f"    关键点帧数: {keypoints.shape[0]}, 帧率: {fps:.2f}")

        # 3. 环境特征计算
        logger.info("  步骤3/5: 计算环境特征...")
        T = keypoints.shape[0]
        env_feats = []
        for t in range(T):
            env_feat = self._env_computer.compute_features_for_frame(keypoints[t], centers[t])
            env_feats.append(env_feat)
        env_features = np.stack(env_feats, axis=0)

        # 4. 行为预测
        logger.info("  步骤4/5: 滑动窗口行为预测...")
        pred_labels = self.behavior_model.predict_sliding_window(keypoints, env_features, fps, start_seconds)
        logger.info(f"    预测完成，得到 {len(pred_labels)} 帧标签")

        # 5. 行为语义分析
        logger.info("  步骤5/5: 行为语义分析...")
        segments = self.behavior_analyzer.labels_to_segments(pred_labels, fps, start_seconds)
        stats = self.behavior_analyzer.compute_behavior_stats(segments)
        total_duration = len(pred_labels) / fps
        windows = self.behavior_analyzer.generate_5s_windows(segments, fps, 0.0, total_duration)
        changes = self.behavior_analyzer.find_behavior_change_points(segments)
        report_text = self.behavior_analyzer.generate_optimized_prompt(
            segments, stats, windows, changes, video_path.name, start_datetime, fps, total_duration
        )

        # 保存结果
        report_path = Path(self.config.output_dir) / "reports" / f"{output_name}_report.txt"
        os.makedirs(report_path.parent, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"    文本报告已保存: {report_path}")

        txt_path = Path(self.config.output_dir) / "txts" / f"{output_name}_labels.txt"
        os.makedirs(txt_path.parent, exist_ok=True)
        np.savetxt(txt_path, pred_labels, fmt='%d')

        json_path = Path(self.config.output_dir) / "segments" / f"{output_name}_segments.json"
        os.makedirs(json_path.parent, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=2, default=str)

        return {
            "labels": pred_labels,
            "segments": segments,
            "stats": stats,
            "report": report_text,
            "fps": fps,
            "start_time": start_seconds,
            "start_datetime": start_datetime,
            "total_duration": total_duration,
            "windows": windows,
            "changes": changes
        }

    def ask_question(self, question: str, video_context: Optional[Dict] = None) -> str:
        literature_context = ""
        if self.config.use_rag:
            if self._rag_retriever is None:
                try:
                    self._rag_retriever = RAGRetriever(self.config.rag_persist_dir, self.config.rag_collection_name)
                except Exception as e:
                    logger.error(f"RAG 知识库加载失败: {e}")
                    literature_context = "（文献知识库不可用）"
            if self._rag_retriever:
                try:
                    retrieved = self._rag_retriever.similarity_search(question, k=3)
                    if retrieved:
                        literature_context = "相关文献知识：\n" + "\n".join([f"- {doc['text']}" for doc in retrieved])
                    else:
                        literature_context = "（未检索到相关文献）"
                except Exception as e:
                    logger.error(f"RAG 检索失败: {e}")
                    literature_context = "（文献检索出错）"

        video_report = ""
        if video_context and "report" in video_context:
            video_report = f"视频行为分析报告,报告数据均为5秒内小鼠行为分布和帧级行为预测结果：\n{video_context['report']}"

        prompt = f"""你是一名小鼠行为分析专家。小鼠的行为限定为以下九种：
            行走（walk）、静止（stop）、饮水（drinking）、饮食（feeding）、
            嗅探（sniffing）、原地站立（unsupported rearing）、
            靠墙站立（supported rearing）、理毛（grooming）、挖洞（digging）。
            以下是与你任务相关的专业知识，请优先吸取其中与当前行为高度
            相关的内容，忽略不适用部分；同时结合科学的动物行为学常识
            进行分析。
            {literature_context}
            以下是小鼠的视频行为分析报告，包含各行为的累计时长、时间
            分布及行为切换序列，是回答问题时的重要量化依据。
            {video_report}
            用户问题：{question}
            请综合上述文献知识与行为报告，给出专业、准确的回答。要求：
            1. 分析中必须使用上述九种行为的标准名称；
            2. 引用行为报告中的具体数据时，应确保准确无误；
            3. 结合文献知识进行推理时，应明确指出所依据的知识来源。
        """

        if self._llm_manager is None:
            try:
                logger.info("正在启动 Qwen 子进程（GPU）...")
                self._llm_manager = QwenSubprocessManager(
                    self.config.qwen_model_name,
                    self.config.qwen_cache_dir,
                    self.config.use_modelscope
                )
            except Exception as e:
                logger.error(f"启动 Qwen 子进程失败: {e}")
                return f"问答模型启动失败: {str(e)}。请检查 GPU 内存（建议 6GB+）及依赖是否完整。"
        response = self._llm_manager.generate_response(prompt)
        return response

    def shutdown_llm(self):
        if self._llm_manager is not None:
            self._llm_manager.shutdown()
            self._llm_manager = None
            logger.info("Qwen 子进程已关闭，GPU 资源已释放")


if __name__ == "__main__":
    mp.freeze_support()
    cfg = PipelineConfig()
    cfg.env_json_path = r"\path\to\json"
    cfg.model_weight_path = r"\path\to\pth"
    cfg.yolo_model_path = r"\path\to\yolo\pt"
    cfg.output_dir = r"./behavior_predictions"
    cfg.rag_persist_dir = r"./rag_db"
    cfg.rag_collection_name = "mouse_behavior_literature"
    cfg.use_rag = True
    cfg.use_modelscope = True
    pipeline = EnhancedVideoBehaviorPipeline(cfg)
    video_path = r"\path\to\vedio"
    result = pipeline.process_video(video_path)
    if result:
        print(result['report'])
    ans = pipeline.ask_question("小鼠在什么时候最活跃？", result)
    print(ans)
    pipeline.shutdown_llm()