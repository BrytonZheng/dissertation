import re
import sys
import os
import cv2
import numpy as np
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QMessageBox, QCheckBox, QTextEdit, QPlainTextEdit
)
from PyQt6.QtGui import QPixmap, QMouseEvent, QPainter, QPen, QColor, QTextCharFormat
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QUrl

import json_pixel_to_world
from collision_detect import CollisionDetect

from datetime import datetime


class QTextEditLogger:
    LEVELS = {"DEBUG": 1, "INFO": 2, "ERROR": 3}

    def __init__(self, text_edit: QPlainTextEdit, level: str = "DEBUG"):
        self.text_edit = text_edit
        self.buffer = ""
        self.level = self.LEVELS.get(level.upper(), 1)  # 默认 DEBUG

    def write(self, message):
        self.buffer += message
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._append(line, QColor("black"))

    def flush(self):
        if self.buffer:
            self._append(self.buffer, QColor("black"))
            self.buffer = ""

    def _append(self, line, color: QColor):
        if not line.strip():
            return
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted_line = f"[{timestamp}] {line}"

        cursor = self.text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(color)
        cursor.setCharFormat(fmt)
        cursor.insertText(formatted_line + "\n")

        self.text_edit.setTextCursor(cursor)

    def _should_log(self, level_name: str) -> bool:
        return self.LEVELS[level_name] >= self.level

    def debug(self, message: str):
        if self._should_log("DEBUG"):
            self._append(message.strip(), QColor("blue"))

    def info(self, message: str):
        if self._should_log("INFO"):
            self._append(message.strip(), QColor("green"))

    def error(self, message: str):
        if self._should_log("ERROR"):
            self._append(message.strip(), QColor("red"))


class ClickableImageLabel(QLabel):
    pointClicked = pyqtSignal(tuple)  # 发射原图坐标 (x, y)
    hoverMoved = pyqtSignal(tuple)  # 发射悬停位置原图坐标
    warningMessage = pyqtSignal(str)  # 用于提示不允许操作的信息

    def __init__(self, parent = None, logger = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.points = []
        self.original_points = []
        self._original_pixmap = None
        self._scaled_pixmap = None
        self._scale_ratio_x = 1.0
        self._scale_ratio_y = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self.locked = False
        self.setMouseTracking(True)
        self.logger = logger

    def setPixmap(self, pixmap: QPixmap):
        self._original_pixmap = pixmap

        # 等比缩放到 QLabel 尺寸
        self._scaled_pixmap = pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        super().setPixmap(self._scaled_pixmap)

        # 缩放比
        self._scale_ratio_x = self._scaled_pixmap.width() / self._original_pixmap.width()
        self._scale_ratio_y = self._scaled_pixmap.height() / self._original_pixmap.height()

        # 计算居中偏移
        self._offset_x = (self.width() - self._scaled_pixmap.width()) / 2
        self._offset_y = (self.height() - self._scaled_pixmap.height()) / 2

        self.points.clear()
        self.original_points.clear()
        self.locked = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if self._scaled_pixmap is not None and not self.locked:
            x = event.position().x()
            y = event.position().y()
            if not (self._offset_x <= x <= self.width() - self._offset_x and
                    self._offset_y <= y <= self.height() - self._offset_y):
                return
            # 原图坐标转换
            raw_x = (x - self._offset_x) / self._scale_ratio_x
            raw_y = (y - self._offset_y) / self._scale_ratio_y
            if raw_y < self._original_pixmap.height() / 2:
                self.warningMessage.emit("⚠️ 点在图像上半部分，已忽略")
                return

            pt = QPointF(x, y)
            self.points.append(pt)
            self.original_points.append((raw_x, raw_y))
            # print(f"点击点: 显示图 ({x:.2f}, {y:.2f}) -> 原图坐标 ({raw_x:.2f}, {raw_y:.2f})")
            self.pointClicked.emit((raw_x, raw_y))
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.points:
            return

        painter = QPainter(self)
        pen = QPen(Qt.GlobalColor.cyan, 2)
        painter.setPen(pen)

        for i, pt in enumerate(self.points):
            painter.drawEllipse(pt, 3, 3)
            if i > 0:
                painter.drawLine(self.points[i - 1], pt)
        if self.locked and len(self.points) > 2:
            painter.drawLine(self.points[-1], self.points[0])  # 首尾连接

    def clear_points(self):
        self.points.clear()
        self.original_points.clear()
        self.locked = False  # 解锁
        self.update()

    def get_original_points(self):
        return self.original_points

    def lock_points(self):
        if len(self.points) >= 3:
            self.locked = True
            self.update()
            self.logger.info("✅ 已锁定多边形")
        else:
            self.logger.info("⚠️ 至少需要 3 个点才能锁定")

    def mouseMoveEvent(self, event):
        x, y = event.position().x(), event.position().y()

        # 判断是否在缩放图像区域内
        if (self._offset_x <= x <= self.width() - self._offset_x and
                self._offset_y <= y <= self.height() - self._offset_y):
            raw_x = (x - self._offset_x) / self._scale_ratio_x
            raw_y = (y - self._offset_y) / self._scale_ratio_y
            self.hoverMoved.emit((raw_x, raw_y))


class ClickableVideoWidget(QVideoWidget):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class MainWindow(QWidget):
    # TODO list: 多车合并输出图片，支持选择车辆图片输出视频，支持预测轨迹碰撞颜色变化
    def __init__(self):
        super().__init__()
        self.setWindowTitle("道路施工场景下轨迹预测系统")

        # 日志重定向
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.logger = QTextEditLogger(self.log_output)
        sys.stdout = self.logger
        sys.stderr = self.logger

        # 初始化输入框
        self.data_path_input = QLineEdit()
        self.mat_path_input = QLineEdit()
        self.pic_dir_input = QLineEdit()
        self.camera_params_input = QLineEdit()
        self.save_path_input = QLineEdit()
        self.save_pic_path_input = QLineEdit()
        self.epoch_input = QLineEdit("9")
        self.draw_img_checkbox = QCheckBox("保存预测数据图像")
        self.draw_pic_checkbox = QCheckBox("保存预测原图渲染")
        self.draw_pic_checkbox.setEnabled(False)
        self.draw_img_checkbox.stateChanged.connect(self.update_draw_pic_checkbox_state)

        # 用于展示图片的 QLabel
        self.image_label = ClickableImageLabel("图像预览", self.logger)
        self.image_label.pointClicked.connect(self.show_coord)
        self.image_label.setFixedSize(450, 375)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.hoverMoved.connect(self.show_hover_coord)
        self.image_label.warningMessage.connect(self.show_warning)
        self.coord_label = QLabel("坐标：(x, y)")

        # 视频播放
        self.video_widget = ClickableVideoWidget()
        self.video_widget.setFixedSize(450, 375)
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.video_widget.clicked.connect(self.toggle_video_playback)

        self.init_ui()

    def init_ui(self):
        main_window_layout = QHBoxLayout()

        # 数据和预测参数
        args_layout = QVBoxLayout()
        # args相关
        args_layout.addLayout(
            self.create_file_selector("选择数据文件夹", self.data_path_input, QFileDialog.getExistingDirectory))
        args_layout.addLayout(
            self.create_file_selector("选择 .mat 文件", self.mat_path_input, QFileDialog.getOpenFileName))
        args_layout.addLayout(
            self.create_file_selector("选择图像目录", self.pic_dir_input, QFileDialog.getExistingDirectory))
        args_layout.addLayout(
            self.create_file_selector("选择相机参数文件", self.camera_params_input, QFileDialog.getOpenFileName))
        args_layout.addLayout(
            self.create_file_selector("保存预测数据路径", self.save_path_input, QFileDialog.getExistingDirectory))
        args_layout.addLayout(
            self.create_file_selector("保存图片路径", self.save_pic_path_input, QFileDialog.getExistingDirectory))
        args_layout.addWidget(self.draw_img_checkbox)
        args_layout.addWidget(self.draw_pic_checkbox)
        # Epoch 输入
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("Epoch"))
        epoch_layout.addWidget(self.epoch_input)
        args_layout.addLayout(epoch_layout)

        predict_button = QPushButton("预测")
        predict_button.clicked.connect(self.call_predict)
        args_layout.addWidget(predict_button)

        # 图片预览功能
        image_preview_layout = QVBoxLayout()
        clear_button = QPushButton("清空选择的所有点")
        clear_button.clicked.connect(self.image_label.clear_points)
        lock_button = QPushButton("锁定点")
        lock_button.clicked.connect(self.image_label.lock_points)

        image_preview_layout.addWidget(self.image_label)
        image_preview_layout.addWidget(self.coord_label)
        image_preview_layout.addWidget(clear_button)
        image_preview_layout.addWidget(lock_button)

        # 视频播放
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_widget)
        replay_button = QPushButton("重放视频")
        replay_button.clicked.connect(self.replay_video)
        video_layout.addWidget(replay_button)

        main_window_layout.addLayout(args_layout)
        # main_window_layout.addWidget(self.text_edit)
        main_window_layout.addLayout(image_preview_layout)
        main_window_layout.addLayout(video_layout)

        display_layout = QVBoxLayout()
        display_layout.addLayout(main_window_layout)
        display_layout.addWidget(self.log_output)
        self.setLayout(display_layout)

    def create_file_selector(self, label_text, line_edit, dialog_func):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        layout.addWidget(line_edit)
        browse_button = QPushButton("浏览")
        browse_button.clicked.connect(lambda: self.browse_file(label_text, line_edit, dialog_func))
        layout.addWidget(browse_button)
        return layout

    def browse_file(self, label_text, line_edit, dialog_func):
        result = dialog_func(self, "选择文件或目录")
        if isinstance(result, tuple):  # 如果是文件选择
            result = result[0]
        if result:
            line_edit.setText(result)
            if label_text == "选择图像目录":
                self.display_sample_image(result)
            elif label_text == "选择数据文件夹":
                self.detect_data_files(result)

    def update_draw_pic_checkbox_state(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.draw_pic_checkbox.setEnabled(is_checked)
        if not is_checked:
            self.draw_pic_checkbox.setChecked(False)

    def show_coord(self, point):
        x, y = point
        self.coord_label.setText(f"坐标：({x:.1f}, {y:.1f})")

    def display_sample_image(self, dir_path):
        # 获取任意一张图片
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        for fname in os.listdir(dir_path):
            if os.path.splitext(fname)[1].lower() in supported_formats:
                image_path = os.path.join(dir_path, fname)
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    self.image_label.setPixmap(pixmap)
                    return
        self.image_label.setText("未找到支持的图片")
        self.logger.error("未找到支持的图片")

    def detect_data_files(self, dir_path):
        # firstly check .mat and DumpSettings.json exist
        camera_dir = os.path.join(dir_path, 'DumpSettings.json')
        if not os.path.exists(os.path.join(dir_path, 'DumpSettings.json')):
            self.logger.error("⚠️ 数据文件夹不存在 DumpSettings.json 文件")
            return

        pic_dir = os.path.join(dir_path, 'output/Colorbox')

        mat_dir = os.path.join(dir_path, json_pixel_to_world.CONST_MAT_NAME)
        if not os.path.exists(mat_dir):
            json_pixel_to_world.main(dir_path)

        self.mat_path_input.setText(mat_dir)
        self.pic_dir_input.setText(pic_dir)
        self.camera_params_input.setText(camera_dir)

        # 触发预览
        self.display_sample_image(pic_dir)

    def show_hover_coord(self, point):
        x, y = point
        self.coord_label.setText(f"悬停坐标：({x:.1f}, {y:.1f})")

    def show_warning(self, message):
        self.coord_label.setText(message)

    def sorted_image_files(self, directory):
        def extract_number(filename):
            match = re.match(r"(\d+)\.(png|jpg|jpeg|bmp)", filename)
            return int(match.group(1)) if match else float('inf')

        supported_ext = ('.png', '.jpg', '.jpeg', '.bmp')
        files = [f for f in os.listdir(directory) if f.lower().endswith(supported_ext)]
        files.sort(key = extract_number)
        return [os.path.join(directory, f) for f in files]

    def generate_video_with_polygon(self, image_dir, points, output_video_path = "output.mp4"):
        image_files = self.sorted_image_files(image_dir)
        if not image_files:
            QMessageBox.warning(self, "警告", "没有图片")
            return

        polygon_pts = np.array([[int(x), int(y)] for x, y in points], np.int32).reshape((-1, 1, 2))

        first_img = cv2.imread(image_files[0])
        height, width, _ = first_img.shape
        fps = 5

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for fname in image_files:
            img = cv2.imread(fname)
            if img is None:
                continue
            if self.draw_pic_checkbox.isChecked():
                cv2.polylines(img, [polygon_pts], isClosed = True, color = (255, 255, 0), thickness = 2)
            out.write(img)
        out.release()

        # 设置视频播放
        self.media_player.setSource(QUrl.fromLocalFile(os.path.abspath(output_video_path)))
        self.media_player.play()
        self.current_video_path = output_video_path  # 可选：记录当前视频路径用于重放

    def replay_video(self):
        if self.media_player.source().isLocalFile():
            self.media_player.stop()
            self.media_player.play()

    def toggle_video_playback(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def call_predict(self):
        try:
            args = {
                "data_args": {
                    "dir": self.mat_path_input.text(),
                    "pic_dir": self.pic_dir_input.text(),
                    "camera_params_dir": self.camera_params_input.text(),
                },
                "output_args": {
                    "draw_img": self.draw_img_checkbox.isChecked(),
                    "draw_pic": self.draw_pic_checkbox.isChecked(),
                    "draw_all_pic": False,
                    "save_path": self.save_path_input.text(),
                    "save_pic_path": self.save_pic_path_input.text(),
                },
                "epoch": int(self.epoch_input.text()),
                "multi_model": False,
            }

            detect = CollisionDetect(args)
            detect.detect(self.image_label.original_points)

            # 放视频
            if self.draw_img_checkbox.isChecked():
                save_path = self.save_pic_path_input.text() if self.draw_pic_checkbox.isChecked() else self.save_path_input.text()
                self.generate_video_with_polygon(
                    image_dir = os.path.join(save_path, "1-1"),
                    points = self.image_label.get_original_points(),
                    output_video_path = os.path.join(save_path, "output.mp4")
                )
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
