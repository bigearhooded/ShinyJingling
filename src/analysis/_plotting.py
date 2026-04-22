"""Shared matplotlib config — load Noto CJK fonts so Chinese labels render."""
from __future__ import annotations

import os
import matplotlib.font_manager as _fm
import matplotlib.pyplot as _plt

_CJK_DIR = "./fonts"


def setup():
    for fn in ("NotoSansCJKsc-Regular.otf", "NotoSansCJKsc-Bold.otf"):
        path = os.path.join(_CJK_DIR, fn)
        if os.path.exists(path):
            _fm.fontManager.addfont(path)
    _plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    _plt.rcParams["axes.unicode_minus"] = False
    _plt.rcParams["figure.dpi"] = 110


setup()
