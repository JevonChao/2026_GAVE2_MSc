"""
Script 3 / 3 —— 生物标志物结果对比

把 get_biomarker.py 算出的结果和数据集真值 (data/training/biomarker) 对比，
计算每个指标的平均绝对误差 (MAE) 和对称平均绝对百分比误差 (SMAPE)。
这两个指标是 GAVE2 挑战赛官方用于 Task3 评估的指标。

用法:
    python compare_biomarker.py <算出的结果目录> <真值目录>

例:
    python compare_biomarker.py ./results/biomarker_gt_check ./data/training/biomarker
"""

import sys

from pathlib import Path


METRICS = [
    'CRAE', 'CRVE', 'AVR',
    'artery_density', 'vein_density',
    'artery_fractal_dimension', 'vein_fractal_dimension',
]


def parse_txt(path):
    """把一个生物标志物 txt 解析成 {指标名: 值}。无法解析的值记为 None。"""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            key = parts[0]
            try:
                out[key] = float(parts[1])
            except (IndexError, ValueError):
                out[key] = None
    return out


def smape(pred, true):
    denom = abs(pred) + abs(true)
    if denom == 0:
        return 0.0
    return 2.0 * abs(pred - true) / denom * 100.0


def main():
    if len(sys.argv) < 3:
        print('Usage: python compare_biomarker.py <pred_dir> <gt_dir> [--holdout]')
        sys.exit(1)

    pred_dir = Path(sys.argv[1])
    gt_dir = Path(sys.argv[2])
    only_holdout = '--holdout' in sys.argv

    pred_files = sorted(pred_dir.glob('*.txt'))
    if only_holdout:
        holdout = {f'g_{i:03d}' for i in range(1, 13)}
        pred_files = [f for f in pred_files if f.stem in holdout]
        print(f'(held-out only: {len(pred_files)} files)')

    if not pred_files:
        print(f'No .txt files found in {pred_dir}')
        sys.exit(1)

    # 累加每个指标的误差
    abs_err = {m: [] for m in METRICS}
    smape_err = {m: [] for m in METRICS}
    n_used = 0
    skipped = []

    for pf in pred_files:
        gf = gt_dir / pf.name
        if not gf.exists():
            skipped.append(pf.stem + ' (no gt)')
            continue
        pred = parse_txt(pf)
        true = parse_txt(gf)
        used = False
        for m in METRICS:
            pv, tv = pred.get(m), true.get(m)
            if pv is None or tv is None:
                continue
            abs_err[m].append(abs(pv - tv))
            smape_err[m].append(smape(pv, tv))
            used = True
        if used:
            n_used += 1

    print('=' * 68)
    print(f'Compared {n_used} cases')
    print(f'  pred: {pred_dir}')
    print(f'  gt  : {gt_dir}')
    print('=' * 68)
    print(f'{"Metric":<28}{"MAE":>12}{"SMAPE (%)":>14}{"n":>6}')
    print('-' * 68)
    for m in METRICS:
        if abs_err[m]:
            mae = sum(abs_err[m]) / len(abs_err[m])
            sm = sum(smape_err[m]) / len(smape_err[m])
            print(f'{m:<28}{mae:>12.4f}{sm:>14.2f}{len(abs_err[m]):>6}')
        else:
            print(f'{m:<28}{"--":>12}{"--":>14}{0:>6}')
    print('=' * 68)

    if skipped:
        print(f'\nSkipped {len(skipped)} files:')
        for s in skipped[:10]:
            print(f'  {s}')


if __name__ == '__main__':
    main()
