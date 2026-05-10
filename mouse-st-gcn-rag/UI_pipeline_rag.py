import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import traceback
import cv2
import numpy as np
import torch
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QTextEdit, QSplitter, QProgressBar, QMessageBox, QFrame, QLabel, QStatusBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QImage

try:
    from pipeline_RAG import EnhancedVideoBehaviorPipeline, PipelineConfig
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    print(f"[ERROR] 无法导入 pipeline_RAG: {e}")


class AnalysisThread(QThread):
    finished = pyqtSignal(dict, object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, video_path, config):
        super().__init__()
        self.video_path = video_path
        self.config = config

    def run(self):
        try:
            self.progress.emit("🚀 正在加载AI模型（YOLO、行为识别、OCR）...")
            pipeline = EnhancedVideoBehaviorPipeline(self.config)
            self.progress.emit(f"📹 开始分析视频: {Path(self.video_path).name}")
            result = pipeline.process_video(self.video_path)
            pipeline.unload_behavior_models()
            self.finished.emit(result, pipeline)
        except Exception as e:
            error_msg = f"❌ 分析失败:\n{str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
        finally:
            import gc
            gc.collect()


class QuestionThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, pipeline, question, analysis_result):
        super().__init__()
        self.pipeline = pipeline
        self.question = question
        self.analysis_result = analysis_result

    def run(self):
        try:
            if self.pipeline._llm_manager is None:
                self.progress.emit("⏳ 正在启动 Qwen 模型子进程（首次加载约需几十秒）...")
            answer = self.pipeline.ask_question(self.question, self.analysis_result)
            self.finished.emit(answer)
        except Exception as e:
            self.error.emit(f"问答失败: {str(e)}")


class BehaviorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🐭 小鼠行为分析系统 | 智能问答 (GPU子进程版)")
        self.setGeometry(100, 80, 1400, 850)

        self.video_path = None
        self.cap = None
        self.timer = None
        self.analysis_result = None
        self.analysis_thread = None
        self.question_thread = None
        self.pipeline = None
        self.playing = False

        self._init_ui()
        self._apply_style()
        self._connect_signals()

        if not PIPELINE_AVAILABLE:
            QMessageBox.critical(
                self, "依赖缺失",
                "无法导入 pipeline_RAG 模块，请确保文件在相同目录。"
            )
            self.btn_analyze.setEnabled(False)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # 顶部工具栏
        top_bar = QHBoxLayout()
        self.btn_open = QPushButton("📂 打开视频")
        self.btn_analyze = QPushButton("🚀 开始分析")
        self.btn_analyze.setEnabled(False)
        self.status_label = QLabel("⚡ 就绪")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedWidth(200)

        top_bar.addWidget(self.btn_open)
        top_bar.addWidget(self.btn_analyze)
        top_bar.addStretch()
        top_bar.addWidget(self.status_label)
        top_bar.addWidget(self.progress_bar)

        # 视频播放区
        video_frame = QFrame()
        video_frame.setFrameShape(QFrame.StyledPanel)
        video_layout = QVBoxLayout(video_frame)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(400)
        self.video_label.setStyleSheet("background-color: black;")
        video_layout.addWidget(self.video_label)

        control_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_play)
        control_layout.addWidget(self.btn_pause)
        control_layout.addWidget(self.btn_stop)
        control_layout.addStretch()
        video_layout.addLayout(control_layout)

        # 报告区
        report_frame = QFrame()
        report_frame.setFrameShape(QFrame.StyledPanel)
        report_layout = QVBoxLayout(report_frame)
        report_title = QLabel("📊 行为分析报告")
        report_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText("点击「开始分析」后，详细报告将显示在这里...")
        report_layout.addWidget(report_title)
        report_layout.addWidget(self.report_text)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_layout.addWidget(video_frame, 2)
        left_layout.addWidget(report_frame, 3)

        # 右侧对话区
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        chat_title = QLabel("💬 智能问答 (基于当前分析结果)")
        chat_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("问题与答案会显示在这里...")
        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("输入你的问题，例如：小鼠在哪个时间段活动最频繁？")
        self.chat_input.setMaximumHeight(80)
        self.btn_send = QPushButton("✈ 发送问题")
        self.btn_send.setEnabled(False)
        right_layout.addWidget(chat_title)
        right_layout.addWidget(self.chat_display)
        right_layout.addWidget(self.chat_input)
        right_layout.addWidget(self.btn_send)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([850, 550])

        main_layout.addLayout(top_bar)
        main_layout.addWidget(splitter, 1)

        self.status_bar = QStatusBar()
        self.status_bar.showMessage("就绪 | 请打开视频文件")
        main_layout.addWidget(self.status_bar)

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget { background-color: #f5f7fb; font-family: "Microsoft YaHei"; font-size: 13px; }
            QFrame { background-color: white; border-radius: 12px; border: 1px solid #e2e8f0; }
            QPushButton { background-color: #2c7da0; color: white; border: none; border-radius: 8px; padding: 6px 14px; font-weight: bold; }
            QPushButton:hover { background-color: #1f5e7e; }
            QPushButton:disabled { background-color: #b9d3e8; color: #e0e0e0; }
            QTextEdit { border: 1px solid #cbd5e1; border-radius: 8px; background-color: white; padding: 6px; }
            QProgressBar { border-radius: 5px; text-align: center; background-color: #eef2f6; }
            QProgressBar::chunk { background-color: #2c7da0; border-radius: 5px; }
            QStatusBar { background-color: #f1f5f9; border-radius: 8px; }
        """)

    def _connect_signals(self):
        self.btn_open.clicked.connect(self.open_video)
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_play.clicked.connect(self.play_video)
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_send.clicked.connect(self.send_question)

    # --------------------------- 视频播放 ---------------------------
    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*.*)"
        )
        if not file_path:
            return

        self.stop_video()
        self.video_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开视频文件:\n{file_path}")
            return

        self.update_frame()
        self.btn_analyze.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("📹 视频已加载")
        self.status_bar.showMessage(f"已加载: {Path(file_path).name}")

        if self.pipeline:
            self.pipeline.shutdown_llm()
        self.analysis_result = None
        self.pipeline = None
        self.report_text.clear()
        self.chat_display.clear()
        self.btn_send.setEnabled(False)

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.playing = False

    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
        else:
            if self.playing:
                self.stop_video()

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def next_frame(self):
        if self.cap is None or not self.playing:
            return
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
        else:
            self.stop_video()
            self.status_bar.showMessage("播放结束", 2000)

    def play_video(self):
        if self.cap is None:
            return
        if not self.playing:
            self.playing = True
            if self.timer is not None:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                interval = int(1000 / fps) if fps > 0 else 40
                self.timer.start(interval)
            self.btn_play.setText("⏸ 播放中")
            self.status_bar.showMessage("播放中", 1000)

    def pause_video(self):
        if self.playing:
            self.playing = False
            if self.timer is not None:
                self.timer.stop()
            self.btn_play.setText("▶ 播放")
            self.status_bar.showMessage("已暂停", 1000)

    def stop_video(self):
        if self.playing:
            self.playing = False
        if self.timer is not None:
            self.timer.stop()
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.update_frame()
        self.btn_play.setText("▶ 播放")
        self.status_bar.showMessage("已停止", 1000)

    # --------------------------- 分析流程 ---------------------------
    def start_analysis(self):
        if not self.video_path:
            QMessageBox.warning(self, "提示", "请先打开视频文件")
            return

        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.terminate()
            self.analysis_thread.wait(2000)

        self.btn_analyze.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("⏳ 分析中...")
        self.report_text.setText("正在后台加载模型并分析视频，请稍候...\n（模型将按需加载，分析完成后会释放内存）")
        self.status_bar.showMessage("分析任务已启动...")

        config = PipelineConfig()
        config.device = torch.device('cpu')
        self.analysis_thread = AnalysisThread(self.video_path, config)
        self.analysis_thread.progress.connect(self._on_analysis_progress)
        self.analysis_thread.finished.connect(self._on_analysis_finished)
        self.analysis_thread.error.connect(self._on_analysis_error)
        self.analysis_thread.start()

    def _on_analysis_progress(self, msg):
        self.report_text.append(msg)
        self.status_bar.showMessage(msg, 3000)

    def _on_analysis_finished(self, result, pipeline):
        self.analysis_result = result
        self.pipeline = pipeline
        self.report_text.setText(result.get('report', '无报告内容'))
        self.btn_analyze.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("✅ 分析完成")
        self.status_bar.showMessage("分析完成，现在可以提问了（首次提问会启动 Qwen 子进程，请稍候）")
        self.chat_display.append("🤖 系统: 分析已完成！你可以提问了。注意首次提问会启动问答模型子进程（约需几十秒）。")
        self.analysis_thread = None

    def _on_analysis_error(self, error_msg):
        QMessageBox.critical(self, "分析失败", error_msg)
        self.report_text.setText(f"分析失败:\n{error_msg}")
        self.btn_analyze.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.btn_send.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ 分析出错")
        self.analysis_thread = None

    # --------------------------- 问答系统 ---------------------------
    def send_question(self):
        question = self.chat_input.toPlainText().strip()
        if not question:
            return
        if not self.analysis_result:
            self.chat_display.append("🧑 你: " + question)
            self.chat_display.append("🤖 系统: 请先完成视频分析（点击「开始分析」）后再提问。\n")
            return

        if self.question_thread and self.question_thread.isRunning():
            self.chat_display.append("🤖 系统: 请等待上一个问题回答完成。\n")
            return

        self.chat_display.append(f"🧑 你: {question}")
        self.chat_input.clear()
        self.btn_send.setEnabled(False)

        if self.pipeline is None:
            self.chat_display.append("🤖 系统: 分析 pipeline 未就绪，请重新分析视频。\n")
            self.btn_send.setEnabled(True)
            return

        self.question_thread = QuestionThread(self.pipeline, question, self.analysis_result)
        self.question_thread.progress.connect(self._on_question_progress)
        self.question_thread.finished.connect(self._on_question_finished)
        self.question_thread.error.connect(self._on_question_error)
        self.question_thread.start()

    def _on_question_progress(self, msg):
        self.chat_display.append(f"🤖 系统: {msg}")

    def _on_question_finished(self, answer):
        self.chat_display.append(f"🤖 AI: {answer}\n")
        self.btn_send.setEnabled(True)
        self.question_thread = None

    def _on_question_error(self, error_msg):
        self.chat_display.append(f"🤖 错误: {error_msg}\n")
        self.btn_send.setEnabled(True)
        self.question_thread = None

    # --------------------------- 关闭清理 ---------------------------
    def closeEvent(self, event):
        self.stop_video()
        if self.cap is not None:
            self.cap.release()
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.terminate()
            self.analysis_thread.wait(2000)
        if self.pipeline:
            self.pipeline.shutdown_llm()
            self.pipeline.unload_behavior_models()
            self.pipeline = None
        event.accept()


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    window = BehaviorApp()
    window.show()
    sys.exit(app.exec_())