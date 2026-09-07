"""
Paired significance testing for the GAVE2 experiments.

Compares two or more methods on the same held-out images. For every shared
numeric column it reports:

  * a two-sided Wilcoxon signed-rank test (exact when n is small)
  * the matched-pairs rank-biserial correlation r, the standard effect size
    for the Wilcoxon test
  * a percentile bootstrap confidence interval on the mean paired difference
  * Holm-Bonferroni adjusted p-values across the whole family of tests

The p-value alone says little at n = 12, because the two-sided exact test
cannot go below 2 / 2**n. The effect size and the interval carry the
information about how large the difference actually is.

Inputs are the per-image CSVs written by evaluate.py --csv, or the per-case
CSV written by compare_biomarker.py --csv. Rows are matched on the 'image'
column, so the two files may list images in a different order.

Usage
-----
Two methods, every shared metric:

    python significance_test.py \
        --a results/single.csv  --name-a "Single-modality" \
        --b results/attention.csv --name-b "Attention"

Several comparisons at once, with Holm correction across all of them, and
only the columns that matter:

    python significance_test.py \
        --pair results/single.csv "Single-modality" results/additive.csv  "Additive" \
        --pair results/single.csv "Single-modality" results/weighted.csv  "Weighted" \
        --pair results/single.csv "Single-modality" results/attention.csv "Attention" \
        --pair results/additive.csv "Additive"     results/attention.csv "Attention" \
        --pair results/weighted.csv "Weighted"     results/attention.csv "Attention" \
        --metrics Artery_dice Vein_dice Vessel_dice \
                  Artery_recall Vein_recall Vessel_recall \
        --markdown tables/significance.md

Note on direction: the reported difference is always B minus A, so a positive
value means the second method scored higher. For error columns such as SMAPE,
lower is better, so a negative difference is the improvement.
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import numpy as np
    from scipy.stats import wilcoxon, rankdata
except ImportError:
    print('numpy and scipy are required. '
          'Install with: conda install -c conda-forge numpy scipy')
    sys.exit(1)


ID_KEYS = ('image', 'Image', 'name', 'case', 'id')


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_csv(path):
    """Read a per-image CSV into {image_id: {column: float}}."""
    rows = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            key = next((r[k] for k in ID_KEYS if k in r and r[k]), None)
            if key is None:
                key = str(len(rows))
            vals = {}
            for k, v in r.items():
                if k in ID_KEYS:
                    continue
                try:
                    vals[k] = float(v)
                except (TypeError, ValueError):
                    pass
            rows[key] = vals
    if not rows:
        raise ValueError(f'No usable rows in {path}')
    return rows


def paired_values(A, B, metric):
    """Return the two aligned arrays for images present in both files."""
    common = sorted(set(A) & set(B))
    a, b = [], []
    for k in common:
        if metric in A[k] and metric in B[k]:
            a.append(A[k][metric])
            b.append(B[k][metric])
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def rank_biserial(diff):
    """
    Matched-pairs rank-biserial correlation.

    r = (W+ - W-) / (W+ + W-), computed on the ranks of the absolute
    non-zero differences. Ranges from -1 to +1; the sign follows the
    direction of the differences and |r| gives the strength.
    """
    d = diff[diff != 0]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    total = w_pos + w_neg
    return 0.0 if total == 0 else float((w_pos - w_neg) / total)


def bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap interval for the mean paired difference."""
    if diff.size == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def min_attainable_p(n):
    """Smallest p the two-sided exact signed-rank test can return at n pairs."""
    return 2.0 / (2 ** n) if n > 0 else float('nan')


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def run_test(a, b, n_boot, seed):
    """One paired comparison. Returns a result dict, or None if unusable."""
    n = a.size
    if n < 3:
        return None
    diff = b - a
    if np.allclose(diff, 0):
        return dict(n=n, mean_a=a.mean(), mean_b=b.mean(), delta=0.0,
                    p=1.0, r=0.0, ci=(0.0, 0.0), zero_ties=n)

    method = 'exact' if n <= 25 else 'approx'
    try:
        stat, p = wilcoxon(a, b, zero_method='wilcox',
                           alternative='two-sided', method=method)
    except (ValueError, TypeError):
        stat, p = wilcoxon(a, b, zero_method='wilcox',
                           alternative='two-sided')

    return dict(n=n, mean_a=float(a.mean()), mean_b=float(b.mean()),
                delta=float(diff.mean()), p=float(p),
                r=rank_biserial(diff),
                ci=bootstrap_ci(diff, n_boot=n_boot, seed=seed),
                zero_ties=int((diff == 0).sum()))


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def main():
    ap = argparse.ArgumentParser(
        description='Paired Wilcoxon tests with effect sizes and bootstrap CIs',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pair', nargs=4, action='append', metavar=('CSV_A', 'NAME_A', 'CSV_B', 'NAME_B'),
                    help='One comparison; repeat for several')
    ap.add_argument('--a', help='Shorthand for a single comparison: file A')
    ap.add_argument('--b', help='Shorthand for a single comparison: file B')
    ap.add_argument('--name-a', default='A')
    ap.add_argument('--name-b', default='B')
    ap.add_argument('--metrics', nargs='*', default=None,
                    help='Columns to test (default: every shared numeric column)')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-holm', action='store_true',
                    help='Skip the Holm-Bonferroni correction')
    ap.add_argument('--markdown', default=None,
                    help='Also write a Markdown table to this path')
    args = ap.parse_args()

    pairs = list(args.pair or [])
    if args.a and args.b:
        pairs.append([args.a, args.name_a, args.b, args.name_b])
    if not pairs:
        ap.error('Give --a/--b, or at least one --pair')

    results = []
    for csv_a, name_a, csv_b, name_b in pairs:
        A, B = load_csv(csv_a), load_csv(csv_b)
        shared = sorted(set().union(*(v.keys() for v in A.values())) &
                        set().union(*(v.keys() for v in B.values())))
        metrics = args.metrics if args.metrics else shared
        missing = [m for m in metrics if m not in shared]
        for m in missing:
            print(f'  [skip] {name_a} vs {name_b}: column "{m}" not in both files')
        for m in [x for x in metrics if x in shared]:
            a, b = paired_values(A, B, m)
            res = run_test(a, b, args.n_boot, args.seed)
            if res is None:
                print(f'  [skip] {m}: fewer than 3 paired images')
                continue
            res.update(metric=m, name_a=name_a, name_b=name_b)
            results.append(res)

    if not results:
        print('Nothing to test.')
        return

    if args.no_holm:
        for r in results:
            r['p_adj'] = r['p']
    else:
        for r, pa in zip(results, holm([r['p'] for r in results])):
            r['p_adj'] = float(pa)

    n_pairs = results[0]['n']
    floor = min_attainable_p(n_pairs)
    line = '=' * 108
    print('\n' + line)
    print(f'Paired Wilcoxon signed-rank tests (two-sided)   '
          f'n = {n_pairs} images   {len(results)} tests')
    print(f'Smallest attainable p at n = {n_pairs}: {floor:.6f}'
          + ('' if args.no_holm else '   |   p_adj = Holm-Bonferroni'))
    print(line)
    print(f'{"Comparison":<34}{"Metric":<22}{"mean A":>9}{"mean B":>9}'
          f'{"delta":>9}{"95% CI":>19}{"r":>7}{"p":>9}{"p_adj":>9}  sig')
    print('-' * 108)

    for r in results:
        cmp_name = f'{r["name_b"]} vs {r["name_a"]}'
        ci = f'[{r["ci"][0]:+.3f}, {r["ci"][1]:+.3f}]'
        at_floor = ' <' if abs(r['p'] - floor) < 1e-9 else '  '
        print(f'{cmp_name:<34}{r["metric"]:<22}'
              f'{r["mean_a"]:>9.4f}{r["mean_b"]:>9.4f}{r["delta"]:>+9.4f}'
              f'{ci:>19}{r["r"]:>+7.2f}{r["p"]:>9.4f}{at_floor}'
              f'{r["p_adj"]:>7.4f}  {stars(r["p_adj"])}')
    print(line)
    print('delta = mean(B) - mean(A); positive means B scored higher.')
    print('r     = matched-pairs rank-biserial correlation.')
    print('"<"   = p is at the floor of the exact test, not a coincidence.')
    print(line)

    if args.markdown:
        out = Path(args.markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            f.write(f'Paired Wilcoxon signed-rank test (two-sided), n = {n_pairs}. '
                    f'Effect size r is the matched-pairs rank-biserial correlation; '
                    f'the interval is a percentile bootstrap on the mean difference. '
                    f'p-values are Holm-Bonferroni adjusted across '
                    f'{len(results)} tests.\n\n')
            f.write('| Comparison | Metric | Mean A | Mean B | Difference | '
                    '95% CI | r | p |\n')
            f.write('| --- | --- | --- | --- | --- | --- | --- | --- |\n')
            for r in results:
                f.write(f'| {r["name_b"]} vs {r["name_a"]} | {r["metric"]} | '
                        f'{r["mean_a"]:.3f} | {r["mean_b"]:.3f} | '
                        f'{r["delta"]:+.3f} | '
                        f'[{r["ci"][0]:+.3f}, {r["ci"][1]:+.3f}] | '
                        f'{r["r"]:+.2f} | {r["p_adj"]:.4f} {stars(r["p_adj"])} |\n')
        print(f'Markdown table written to {out}')


if __name__ == '__main__':
    main()
