import sys
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import os


# ── Camera Thread ─────────────────────────────────────────────────────────────
class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, int, int)

    def __init__(self, model, confidence=0.5):
        super().__init__()
        self.model = model
        self.confidence = confidence
        self.running = False

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(frame, conf=self.confidence, verbose=False)
            annotated = results[0].plot(line_width=2, font_size=0.6)
            boxes = results[0].boxes
            names = self.model.names
            detections = [names[int(c)] for c in boxes.cls] if boxes else []
            helmet = detections.count("Helmet")
            no_helmet = detections.count("No_Helmet")
            self.frame_ready.emit(annotated, helmet, no_helmet)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ── Stat Card ─────────────────────────────────────────────────────────────────
class StatCard(QFrame):
    def __init__(self, value, label, accent="#ffffff"):
        super().__init__()
        self.setObjectName("stat_card")
        self.setFixedHeight(96)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 16, 22, 12)
        layout.setSpacing(4)
        self.val = QLabel(value)
        self.val.setStyleSheet(
            f"font-size: 34px; font-weight: 800; color: {accent}; letter-spacing: -1px; background: transparent;"
        )
        self.lbl = QLabel(label)
        self.lbl.setStyleSheet(
            "font-size: 9px; color: #3a3a3a; letter-spacing: 3px; background: transparent;"
        )
        layout.addWidget(self.val)
        layout.addWidget(self.lbl)

    def set_value(self, v):
        self.val.setText(f"{v:02d}")


# ── Main Window ───────────────────────────────────────────────────────────────
class HelmGuard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HelmGuard — Helmet Detection System")
        self.setMinimumSize(1280, 820)
        self.resize(1400, 860)
        self.camera_thread = None

        try:
            self.model = YOLO("best.pt")
        except Exception:
            self.model = None

        self._build_ui()
        self.setStyleSheet(self._stylesheet())

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ───────────────────────────────────────────────────────────
        header = QWidget()
        header.setObjectName("header")
        header.setFixedHeight(68)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(36, 0, 36, 0)

        # Brand
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setObjectName("brand_dot")

        brand_name = QLabel("HelmGuard")
        brand_name.setObjectName("brand_name")

        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setObjectName("v_line")
        vline.setFixedHeight(16)

        brand_sub = QLabel("Helmet Compliance Detection")
        brand_sub.setObjectName("brand_sub")

        brand = QHBoxLayout()
        brand.setSpacing(14)
        brand.addWidget(dot)
        brand.addWidget(brand_name)
        brand.addWidget(vline)
        brand.addWidget(brand_sub)
        brand.addStretch()

        # Status
        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("status_offline")
        self.status_text = QLabel("OFFLINE")
        self.status_text.setObjectName("status_text_off")
        status = QHBoxLayout()
        status.setSpacing(8)
        status.addWidget(self.status_dot)
        status.addWidget(self.status_text)

        # THD Logo
        self.logo_frame = QFrame()
        self.logo_frame.setObjectName("logo_frame")
        self.logo_frame.setFixedSize(180, 50)
        logo_inner = QVBoxLayout(self.logo_frame)
        logo_inner.setContentsMargins(14, 6, 14, 6)
        logo_inner.setSpacing(2)

        logo_loaded = False
        for ext in ["png", "jpg", "jpeg"]:
            path = f"assets/thd_logo.{ext}"
            if os.path.exists(path):
                logo_lbl = QLabel()
                pix = QPixmap(path).scaled(148, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logo_lbl.setPixmap(pix)
                logo_lbl.setAlignment(Qt.AlignCenter)
                logo_inner.addWidget(logo_lbl)
                logo_loaded = True
                break

        if not logo_loaded:
            t1 = QLabel("T H D")
            t1.setObjectName("thd_t1")
            t1.setAlignment(Qt.AlignCenter)
            t2 = QLabel("TECHNISCHE HOCHSCHULE DEGGENDORF")
            t2.setObjectName("thd_t2")
            t2.setAlignment(Qt.AlignCenter)
            logo_inner.addWidget(t1)
            logo_inner.addWidget(t2)

        h_layout.addLayout(brand)
        h_layout.addStretch()
        h_layout.addLayout(status)
        h_layout.addSpacing(28)
        h_layout.addWidget(self.logo_frame)

        h_sep = QFrame()
        h_sep.setFrameShape(QFrame.HLine)
        h_sep.setObjectName("h_line")

        # ── Body ─────────────────────────────────────────────────────────────
        body = QWidget()
        body.setObjectName("body")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # ── Sidebar ───────────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(280)
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(24, 32, 24, 32)
        sb.setSpacing(0)

        def section(text):
            l = QLabel(text)
            l.setObjectName("section_label")
            return l

        sb.addWidget(section("LIVE METRICS"))
        sb.addSpacing(14)

        self.card_total = StatCard("00", "TOTAL DETECTED", "#ffffff")
        self.card_compliant = StatCard("00", "COMPLIANT", "#4ade80")
        self.card_violation = StatCard("00", "VIOLATIONS", "#f87171")

        sb.addWidget(self.card_total)
        sb.addSpacing(8)
        sb.addWidget(self.card_compliant)
        sb.addSpacing(8)
        sb.addWidget(self.card_violation)
        sb.addSpacing(28)

        sb.addWidget(section("SYSTEM STATUS"))
        sb.addSpacing(14)

        self.alert_box = QLabel("System ready.\nPress START to begin.")
        self.alert_box.setObjectName("alert_idle")
        self.alert_box.setWordWrap(True)
        self.alert_box.setFixedHeight(76)
        self.alert_box.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        sb.addWidget(self.alert_box)
        sb.addSpacing(28)

        sb.addWidget(section("MODEL INFO"))
        sb.addSpacing(14)

        for k, v in [
            ("Architecture", "YOLOv8s"),
            ("mAP@0.5", "84.4%"),
            ("Precision", "88.5%"),
            ("Recall", "82.1%"),
            ("Speed", "~75 FPS"),
            ("Classes", "Helmet / No_Helmet"),
        ]:
            row = QHBoxLayout()
            kl = QLabel(k)
            kl.setObjectName("info_key")
            vl = QLabel(v)
            vl.setObjectName("info_val")
            vl.setAlignment(Qt.AlignRight)
            row.addWidget(kl)
            row.addWidget(vl)
            sb.addLayout(row)
            sb.addSpacing(7)

        sb.addStretch()

        sb.addWidget(section("CONTROLS"))
        sb.addSpacing(14)

        self.btn_start = QPushButton("▶   START MONITORING")
        self.btn_start.setObjectName("btn_primary")
        self.btn_start.setFixedHeight(46)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_camera)

        self.btn_stop = QPushButton("⏹   STOP")
        self.btn_stop.setObjectName("btn_secondary")
        self.btn_stop.setFixedHeight(46)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.clicked.connect(self.stop_camera)

        sb.addWidget(self.btn_start)
        sb.addSpacing(8)
        sb.addWidget(self.btn_stop)

        sb_sep = QFrame()
        sb_sep.setFrameShape(QFrame.VLine)
        sb_sep.setObjectName("h_line")

        # ── Feed Area ─────────────────────────────────────────────────────────
        feed_area = QWidget()
        feed_area.setObjectName("feed_area")
        feed_layout = QVBoxLayout(feed_area)
        feed_layout.setContentsMargins(32, 28, 32, 28)
        feed_layout.setSpacing(0)

        feed_header = QHBoxLayout()
        feed_lbl = QLabel("LIVE FEED")
        feed_lbl.setObjectName("section_label")
        self.fps_label = QLabel("")
        self.fps_label.setObjectName("fps_label")
        feed_header.addWidget(feed_lbl)
        feed_header.addStretch()
        feed_header.addWidget(self.fps_label)

        self.feed = QLabel()
        self.feed.setObjectName("camera_feed")
        self.feed.setAlignment(Qt.AlignCenter)
        self.feed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.feed.setText("NO SIGNAL")

        feed_layout.addLayout(feed_header)
        feed_layout.addSpacing(12)
        feed_layout.addWidget(self.feed)

        body_layout.addWidget(sidebar)
        body_layout.addWidget(sb_sep)
        body_layout.addWidget(feed_area)

        # ── Footer ────────────────────────────────────────────────────────────
        f_sep = QFrame()
        f_sep.setFrameShape(QFrame.HLine)
        f_sep.setObjectName("h_line")

        footer = QWidget()
        footer.setObjectName("footer")
        footer.setFixedHeight(36)
        f_layout = QHBoxLayout(footer)
        f_layout.setContentsMargins(36, 0, 36, 0)
        fl = QLabel("HELMGUARD  ·  YOLOV8  ·  HELMET COMPLIANCE MONITORING")
        fl.setObjectName("footer_text")
        fr = QLabel("TH DEGGENDORF  ·  MSS-M-2  ·  SS26")
        fr.setObjectName("footer_text")
        fr.setAlignment(Qt.AlignRight)
        f_layout.addWidget(fl)
        f_layout.addWidget(fr)

        root.addWidget(header)
        root.addWidget(h_sep)
        root.addWidget(body, 1)
        root.addWidget(f_sep)
        root.addWidget(footer)

    def _stylesheet(self):
        return """
            QMainWindow, QWidget { background: #f0f0f0; color: #111111; }

            #header    { background: #ffffff; }
            #sidebar   { background: #f8f8f8; }
            #feed_area { background: #f0f0f0; }
            #footer    { background: #ffffff; }
            #body      { background: #f0f0f0; }

            #brand_dot {
                background: #111111; border-radius: 5px;
                min-width:10px; max-width:10px; min-height:10px; max-height:10px;
            }
            #brand_name { font-size: 16px; font-weight: 800; color: #111111; letter-spacing: -0.3px; }
            #brand_sub  { font-size: 11px; color: #aaaaaa; letter-spacing: 0.5px; }
            #v_line     { color: #e0e0e0; background: #e0e0e0; border: none; max-width: 1px; }
            #h_line     { color: #e0e0e0; background: #e0e0e0; border: none; max-height: 1px; }

            #status_offline { font-size: 11px; color: #cccccc; }
            #status_online  { font-size: 11px; color: #16a34a; }
            #status_text_off { font-size: 9px; color: #cccccc; letter-spacing: 2px; }
            #status_text_on  { font-size: 9px; color: #16a34a; letter-spacing: 2px; }

            #logo_frame {
                border: 1px solid #e0e0e0; border-radius: 8px; background: #ffffff;
            }
            #thd_t1 { font-size: 15px; font-weight: 800; color: #333333; letter-spacing: 6px; background: transparent; }
            #thd_t2 { font-size: 6px; color: #aaaaaa; letter-spacing: 1.5px; background: transparent; }

            #section_label { font-size: 9px; color: #aaaaaa; letter-spacing: 3px; }

            #stat_card {
                background: #ffffff;
                border: 1px solid #e8e8e8;
                border-radius: 10px;
            }

            #alert_idle {
                background: #f8f8f8; border: 1px solid #e8e8e8;
                border-radius: 8px; padding: 12px;
                color: #999999; font-size: 11px; line-height: 1.5;
            }
            #alert_compliant {
                background: #f0fdf4; border: 1px solid #bbf7d0;
                border-radius: 8px; padding: 12px;
                color: #15803d; font-size: 11px; font-weight: 600; line-height: 1.5;
            }
            #alert_violation {
                background: #fff1f2; border: 1px solid #fecdd3;
                border-radius: 8px; padding: 12px;
                color: #be123c; font-size: 11px; font-weight: 600; line-height: 1.5;
            }

            #info_key { font-size: 10px; color: #aaaaaa; }
            #info_val { font-size: 10px; color: #555555; }

            #btn_primary {
                background: #111111; color: #ffffff; border: none;
                border-radius: 8px; font-size: 11px; font-weight: 800; letter-spacing: 1.5px;
            }
            #btn_primary:hover    { background: #333333; }
            #btn_primary:disabled { background: #e0e0e0; color: #aaaaaa; }

            #btn_secondary {
                background: transparent; color: #aaaaaa;
                border: 1px solid #dddddd; border-radius: 8px;
                font-size: 11px; font-weight: 600; letter-spacing: 1.5px;
            }
            #btn_secondary:hover    { border-color: #999999; color: #555555; }
            #btn_secondary:disabled { color: #dddddd; border-color: #eeeeee; }

            #camera_feed {
                background: #e8e8e8; border: 1px solid #d8d8d8;
                border-radius: 12px; color: #cccccc;
                font-size: 11px; letter-spacing: 4px;
            }
            #fps_label   { font-size: 9px; color: #cccccc; letter-spacing: 2px; }
            #footer_text { font-size: 8px; color: #cccccc; letter-spacing: 1.5px; }
        """

    def _refresh(self, widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def start_camera(self):
        if self.model is None:
            self.alert_box.setText("⚠ MODEL NOT FOUND\nPlace best.pt in app folder.")
            self.alert_box.setObjectName("alert_violation")
            self._refresh(self.alert_box)
            return

        self.camera_thread = CameraThread(self.model, confidence=0.5)
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.status_dot.setObjectName("status_online")
        self.status_text.setObjectName("status_text_on")
        self.status_text.setText("LIVE")
        self._refresh(self.status_dot)
        self._refresh(self.status_text)

    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

        self.feed.setPixmap(QPixmap())
        self.feed.setText("NO SIGNAL")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        self.card_total.set_value(0)
        self.card_compliant.set_value(0)
        self.card_violation.set_value(0)

        self.alert_box.setText("System ready.\nPress START to begin.")
        self.alert_box.setObjectName("alert_idle")
        self._refresh(self.alert_box)

        self.status_dot.setObjectName("status_offline")
        self.status_text.setObjectName("status_text_off")
        self.status_text.setText("OFFLINE")
        self._refresh(self.status_dot)
        self._refresh(self.status_text)
        self.fps_label.setText("")

    def update_frame(self, frame, helmet, no_helmet):
        total = helmet + no_helmet
        self.card_total.set_value(total)
        self.card_compliant.set_value(helmet)
        self.card_violation.set_value(no_helmet)

        if total > 0:
            if no_helmet > 0:
                self.alert_box.setText(f"🚨 VIOLATION DETECTED\n{no_helmet} worker(s) without helmet.")
                self.alert_box.setObjectName("alert_violation")
            else:
                self.alert_box.setText(f"✓ ALL COMPLIANT\n{helmet} worker(s) wearing helmets.")
                self.alert_box.setObjectName("alert_compliant")
        else:
            self.alert_box.setText("Monitoring active.\nNo workers detected.")
            self.alert_box.setObjectName("alert_idle")
        self._refresh(self.alert_box)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.feed.width(), self.feed.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.feed.setPixmap(pix)

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = HelmGuard()
    window.show()
    sys.exit(app.exec_())
