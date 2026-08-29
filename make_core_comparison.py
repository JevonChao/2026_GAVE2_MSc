"""
核心定性对比图 (带放大框) —— 模仿 RRWNet 论文风格

三列: GT | Task1 (CFP only) | Attention
每张主图上画若干彩色框标出局部区域, 底部把各框区域放大并排对比,
突出 attention 相对单模态 baseline 的动脉恢复 (artery-suppression 的缓解)。

坐标系: 框坐标在 task1 预测图 (992x1504) 上量得。
GT (1024x1536) 会先 resize 到预测图尺寸, 使三列对齐。

用法: 改 CASE / BOXES / 路径后 python make_core_comparison.py
"""

import cv2
import numpy as np
from pathlib import Path

# ============ CONFIG ============
CASE = 'g_008'

# 三列: (标签, 路径, 是否为GT需要resize)
COLUMNS = [
    ('Ground truth',   f'./data/training/av/{CASE}.png',                    True),
    ('Single-modality (CFP)', f'./predictions/task1_train/colored_{CASE}.png', False),
    ('Attention (ours)', f'./predictions/attention_train/colored_{CASE}.png', False),
]

# 目标统一尺寸 (预测图尺寸, 宽 x 高)
TARGET_W, TARGET_H = 1504, 992

# 放大框: (x, y, w, h, 颜色BGR)  —— 坐标在 task1 预测图上量得
BOXES = [
    (1158, 666, 184, 278, (0, 200, 0)),    # 框1 绿
    (289,  7,   162, 250, (255, 100, 0)),  # 框2 蓝
    (649,  686, 267, 151, (0, 220, 220)),  # 框3 黄
]

BOX_THICK = 5          # 主图框线粗细
ZOOM_W = 360           # 放大区宽度(每个放大块缩放到此宽)
GAP = 10               # 间距
LABEL_H = 44           # 顶部列标签高度
BG = (255, 255, 255)   # 白底
# ================================


def load_col(path, is_gt):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return np.full((TARGET_H, TARGET_W, 3), 200, dtype=np.uint8)
    if is_gt:
        img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
    elif img.shape[1] != TARGET_W or img.shape[0] != TARGET_H:
        img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    return img


def draw_boxes(img):
    out = img.copy()
    for (x, y, w, h, color) in BOXES:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, BOX_THICK)
    return out


def crop_zoom(img, box, target_w):
    x, y, w, h, color = box
    crop = img[y:y+h, x:x+w].copy()
    scale = target_w / w
    zoom = cv2.resize(crop, (target_w, int(h * scale)), interpolation=cv2.INTER_NEAREST)
    # 给放大块加同色边框
    zoom = cv2.copyMakeBorder(zoom, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)
    return zoom


def add_top_label(panel, text):
    w = panel.shape[1]
    label = np.full((LABEL_H, w, 3), 255, dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(label, text, (max((w - tw)//2, 4), 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([label, panel])


def main():
    # 1) 载入三列, 画主图框
    cols_raw = []
    for label, path, is_gt in COLUMNS:
        if not Path(path).exists():
            print(f'[skip] {label}: {path} not found')
            return
        cols_raw.append(load_col(path, is_gt))

    disp_w = 620   # 每列显示宽度(主图与放大块统一)

    # 主图: 画框 + 缩到 disp_w + 顶部标签
    def to_disp(im, w=disp_w):
        h = int(im.shape[0] * w / im.shape[1])
        return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    mains = []
    for raw, (lbl, _, _) in zip(cols_raw, COLUMNS):
        m = to_disp(draw_boxes(raw))
        mains.append(m)

    gap_col = np.full((mains[0].shape[0], GAP, 3), 255, dtype=np.uint8)
    main_row = mains[0]
    for m in mains[1:]:
        main_row = np.hstack([main_row, gap_col, m])

    # 2) 每个框: 三列放大, 每个放大块宽度 = disp_w (与主图列对齐), 保持比例
    zoom_rows = []
    for box in BOXES:
        x, y, w, h, color = box
        # 该框放大块的统一高度(按框比例, 缩到 disp_w 宽)
        zh = int(h * disp_w / w)
        zooms = []
        for raw in cols_raw:
            crop = raw[y:y+h, x:x+w].copy()
            z = cv2.resize(crop, (disp_w, zh), interpolation=cv2.INTER_NEAREST)
            z = cv2.copyMakeBorder(z, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=color)
            zooms.append(z)
        gz = np.full((zooms[0].shape[0], GAP, 3), 255, dtype=np.uint8)
        row = zooms[0]
        for z in zooms[1:]:
            row = np.hstack([row, gz, z])
        zoom_rows.append(row)

    # 3) 垂直拼接, 所有行居中对齐到最大宽度
    total_w = max([main_row.shape[1]] + [r.shape[1] for r in zoom_rows])
    def pad_center(im):
        if im.shape[1] < total_w:
            extra = total_w - im.shape[1]
            left = extra // 2
            right = extra - left
            return cv2.copyMakeBorder(im, 0, 0, left, right, cv2.BORDER_CONSTANT, value=BG)
        return im
    blocks = [pad_center(main_row)]
    gap_row = np.full((GAP*2, total_w, 3), 255, dtype=np.uint8)
    for r in zoom_rows:
        blocks.append(gap_row)
        blocks.append(pad_center(r))
    final = np.vstack(blocks)

    out = f'fig_core_comparison_{CASE}.png'
    cv2.imwrite(out, final)
    print(f'Saved: {out}  ({final.shape[1]}x{final.shape[0]})')


if __name__ == '__main__':
    main()