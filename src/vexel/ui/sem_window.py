"""SEM module window (under development: roadmap 0.0.5, stages M1-M7).

For now it opens an image and shows what Vexel can already tell about it: which SEM
vendor's metadata it carries. Calibration, measurement and analysis arrive from M1 on.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QFileDialog, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from vexel.ui.widgets import heading, muted

VENDORS = {"thermo_fisher": "Thermo Fisher（FEI）", "zeiss": "Zeiss", "tescan": "TESCAN", "hitachi_jeol": "Hitachi 或 JEOL"}


class SemWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vexel · SEM（开发中）")
        self.resize(1100, 760)
        central = QWidget()
        v = QVBoxLayout(central)
        v.addWidget(heading("SEM 模块（开发中）"))
        v.addWidget(muted("下一阶段将提供：读取仪器元数据并自动标定、识别并裁剪信息栏、长度/面积/角度/周长/计数测量、"
                          "图像处理、线宽与粗糙度等尺寸计量、颗粒与尖端分割、倾转 2.5D 重建。"))  # fmt: skip
        self.open_btn = QPushButton("打开 SEM 图片…")
        self.open_btn.clicked.connect(self._choose)
        v.addWidget(self.open_btn)
        self.info = muted("")
        v.addWidget(self.info)
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.view, 1)
        self.setCentralWidget(central)
        self.path: Path | None = None

    def _choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开 SEM 图片", "", "图像 (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
        if path:
            self.open_path(path)

    def open_path(self, path: str | Path) -> bool:
        import imageio.v3 as iio
        import numpy as np

        from vexel.packs.sem.detect import sidecar_vendor, tiff_info

        path = Path(path)
        try:
            img = np.asarray(iio.imread(path))
        except Exception as e:  # noqa: BLE001 - shown to the person
            self.info.setText(f"无法读取：{type(e).__name__}: {e}")
            return False
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img[..., :3].mean(-1)
        elif img.ndim > 2:
            img = img[0]
        lo, hi = float(np.percentile(img, 0.5)), float(np.percentile(img, 99.5))
        gray = np.clip((img - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        gray = np.ascontiguousarray(gray)
        qimg = QImage(gray.data, gray.shape[1], gray.shape[0], gray.strides[0], QImage.Format.Format_Grayscale8).copy()
        self.view.setPixmap(QPixmap.fromImage(qimg).scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio))
        vendor = tiff_info(path)[1] if path.suffix.lower() in (".tif", ".tiff") else None
        vendor = vendor or sidecar_vendor(path)
        found = f"检测到 {VENDORS.get(vendor, vendor)} 的仪器元数据，下一阶段将据此自动标定。" if vendor else "没有找到仪器元数据；下一阶段将支持从信息栏识别比例尺或手动标定。"
        self.info.setText(f"{path.name}：{img.shape[1]}×{img.shape[0]} 像素。{found}")
        self.path = path
        return True
