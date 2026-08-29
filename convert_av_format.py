"""
Script 1 / 3 —— AV 标注格式转换

把 GAVE2 的 AV 标注 (R=动脉, G=交叉, B=静脉) 转成 get_biomarker.py 期望的格式:
    动脉 -> 黄色 (255,255,0)  满足脚本的 "G亮 且 B暗"
    静脉 -> 青色 (0,255,255)  满足脚本的 "G亮 且 R暗"
    交叉 -> 绿色 (0,255,0)    G亮、R暗、B暗 (与脚本逻辑一致)

用法:
    python convert_av_format.py                      # 转换真实标注 (默认)
    python convert_av_format.py <输入目录> <输出目录>   # 转换任意目录
"""

import sys
import numpy as np
from pathlib import Path
from skimage import io


def convert(src_path, dst_path):
    gt = io.imread(src_path)
    gt = gt[:, :, :3]  # 丢掉 alpha 通道
    r, g, b = gt[:, :, 0], gt[:, :, 1], gt[:, :, 2]

    is_artery = (r > 127) & (g < 127) & (b < 127)   # 原红 -> 动脉
    is_vein = (b > 127) & (r < 127) & (g < 127)     # 原蓝 -> 静脉
    is_cross = (g > 127) & (r < 127) & (b < 127)    # 原绿 -> 交叉

    out = np.zeros_like(gt)
    out[is_artery] = [255, 255, 0]   # 黄
    out[is_vein] = [0, 255, 255]     # 青
    out[is_cross] = [0, 255, 0]      # 绿
    io.imsave(dst_path, out.astype(np.uint8))


if __name__ == '__main__':
    if len(sys.argv) == 3:
        src_dir = Path(sys.argv[1])
        dst_dir = Path(sys.argv[2])
    else:
        src_dir = Path('./data/training/av')
        dst_dir = Path('./data/training/av_yellowcyan')

    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob('*.png'))
    if not files:
        print(f'No .png files found in {src_dir}')
        sys.exit(1)

    for f in files:
        convert(f, dst_dir / f.name)
        print(f'  converted {f.name}')
    print(f'\nDone. {len(files)} files -> {dst_dir}')
