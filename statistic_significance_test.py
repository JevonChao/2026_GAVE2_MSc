"""
统计显著性检验 —— Wilcoxon signed-rank test

比较两个方法在相同 held-out 图像上的 per-image 分数，对每个指标做配对
Wilcoxon signed-rank 检验（与 RRWNet 原论文一致），输出 p 值和均值差。

前提: evaluate.py 用 --csv 导出的 per-image 文件，每行一张图，
列名形如 Artery_dice / Artery_iou / Vessel_dice / ... （由 evaluate.py 决定）。

用法:
    python significance_test.py <方法A.csv> <方法B.csv> [--name-a 名称] [--name-b 名称]

例:
    python significance_test.py results/task1_fold0.csv results/task2_attention.csv ^
        --name-a "Task1 (CFP)" --name-b "Attention"
"""

import sys
import csv
import argparse

try:
    from scipy.stats import wilcoxon
except ImportError:
    print('scipy is required. Install with: conda install -c conda-forge scipy')
    sys.exit(1)


def load_csv(path):
    """读成 {image: {metric: value}}。"""
    rows = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = r.get('image') or r.get('Image') or r.get('name')
            if key is None:
                # 没有 image 列则用行号
                key = str(len(rows))
            vals = {}
            for k, v in r.items():
                if k in ('image', 'Image', 'name'):
                    continue
                try:
                    vals[k] = float(v)
                except (TypeError, ValueError):
                    pass
            rows[key] = vals
    return rows


def main():
    ap = argparse.ArgumentParser(description='Wilcoxon signed-rank test between two methods')
    ap.add_argument('csv_a')
    ap.add_argument('csv_b')
    ap.add_argument('--name-a', default='A')
    ap.add_argument('--name-b', default='B')
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()

    A = load_csv(args.csv_a)
    B = load_csv(args.csv_b)

    common = sorted(set(A) & set(B))
    if not common:
        print('No overlapping images between the two files.')
        sys.exit(1)

    # 找两边都有的指标列
    metrics = sorted(set(next(iter(A.values())).keys()) &
                     set(next(iter(B.values())).keys()))

    print('=' * 74)
    print(f'Wilcoxon signed-rank test:  {args.name_b}  vs  {args.name_a}')
    print(f'paired images: {len(common)}   alpha = {args.alpha}')
    print('=' * 74)
    print(f'{"Metric":<22}{"mean " + args.name_a:>14}{"mean " + args.name_b:>14}'
          f'{"p-value":>12}{"sig":>6}')
    print('-' * 74)

    for m in metrics:
        a_vals, b_vals = [], []
        for img in common:
            if m in A[img] and m in B[img]:
                a_vals.append(A[img][m])
                b_vals.append(B[img][m])
        if len(a_vals) < 2:
            continue
        mean_a = sum(a_vals) / len(a_vals)
        mean_b = sum(b_vals) / len(b_vals)
        # 若所有差都为 0，wilcoxon 会报错
        diffs = [b - a for a, b in zip(a_vals, b_vals)]
        if all(d == 0 for d in diffs):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(b_vals, a_vals)
            except ValueError:
                p = float('nan')
        sig = '*' if (p == p and p < args.alpha) else ''
        print(f'{m:<22}{mean_a:>14.4f}{mean_b:>14.4f}{p:>12.4f}{sig:>6}')

    print('=' * 74)
    print('* = statistically significant at the chosen alpha (two-sided).')
    print('Note: with only ~12 paired images the test has limited power;')
    print('report exact p-values and treat borderline results with caution.')


if __name__ == '__main__':
    main()
