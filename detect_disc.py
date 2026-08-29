"""
Script 2 / 3 —— 视盘检测 (传统图像处理)

视盘 (optic disc) 是眼底图中最亮的圆形区域。本脚本用绿通道最亮区域定位视盘中心，
输出与 CFP 图同名的灰度掩膜 (视盘=255, 其余=0)，供 get_biomarker.py 使用。

用法:
    python detect_disc.py                    # 处理训练集 CFP (默认)
    python detect_disc.py <CFP目录> <输出目录>  # 处理任意目录

注意: 这是快速验证用的传统方法。跑完请务必抽查几张，确认白色圆圈落在视盘上
(眼底图里那个亮色圆盘、血管发散的中心)，而不是别的亮斑。
"""

import sys
import numpy as np
import cv2
from pathlib import Path


def detect_disc(cfp_path, out_path, dd_frac=0.18):
    img = cv2.imread(str(cfp_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f'  [skip] cannot read {cfp_path.name}')
        return
    h, w = img.shape[:2]

    # 视盘在绿通道对比更好 (红通道容易饱和)
    green = img[:, :, 1].astype(np.float32)

    # 高斯模糊抑制血管细节，突出大面积亮区
    blur = cv2.GaussianBlur(green, (0, 0), sigmaX=max(h * 0.01, 1))

    # 最亮点作为视盘中心
    _, _, _, max_loc = cv2.minMaxLoc(blur)
    cx, cy = max_loc

    # 视盘直径经验值: 约占图像宽度的 dd_frac
    dd = int(w * dd_frac)
    radius = max(dd // 2, 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    cv2.imwrite(str(out_path), mask)
    print(f'  {cfp_path.name}: center=({cx},{cy}), dd={dd}')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        cfp_dir = Path(sys.argv[1])
        out_dir = Path(sys.argv[2])
    else:
        cfp_dir = Path('./data/training/images')
        out_dir = Path('./data/training/disc_traditional')

    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(cfp_dir.glob('*.png'))
    if not files:
        print(f'No .png files found in {cfp_dir}')
        sys.exit(1)

    for f in files:
        detect_disc(f, out_dir / f.name)
    print(f'\nDone. {len(files)} disc masks -> {out_dir}')
