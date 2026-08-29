"""
get_predictions_new.py
======================

Replacement for get_predictions.py.

Produces four separate, semantically unambiguous renderings per image plus a
per-image pixel statistics table:

  prob/          raw 3-channel sigmoid output (A, VT, V)  -- unchanged, feeds evaluate.py
  av_main/       MAIN FIGURE. Hard argmax assignment: every vessel pixel is
                 red (artery) or blue (vein). No green. Structurally identical
                 to the ground truth, so "should be blue but is red" is
                 directly readable without cross-column comparison.
  error/         ERROR MAP (needs --gt-path). Explicit failure types:
                   light grey  correct classification
                   orange      artery misclassified as vein   (A -> V)
                   cyan        vein misclassified as artery   (V -> A)
                   dark grey   vessel missed entirely         (false negative)
                   purple      spurious vessel                (false positive, optional)
                   white       annotated A/V crossing, excluded from scoring
  uncertain/     SUPPLEMENTARY FIGURE. Threshold rule (the old behaviour):
                   red artery, blue vein, green = detected but unclassified,
                   magenta = both channels fire (predicted crossing).
                 Note green and magenta are now distinct, unlike the old script.
  av_biomarker/  yellow/cyan format consumed by get_biomarker.py.

  pixel_stats.csv   per-image counts + the undecided-pixel fraction that turns
                    "the green went away" into a number you can cite.

Ground-truth colour convention (GAVE2 / README):
    R = artery, G = artery-vein crossing, B = vein.
Crossing pixels are ambiguous by construction and are excluded from the error
map and from the classification-accuracy statistics.

IMPORTANT reproducibility note
------------------------------
--biomarker-assign defaults to `threshold`, i.e. the biomarker masks are built
exactly as the old script built them. Do NOT switch it to `argmax` unless you
intend to recompute all of Task 3, because it changes vessel areas and hence
CRAE/CRVE/density/fractal dimension.

Example
-------
python get_predictions_new.py \
    --weights runs/cmrrwnet_attention/generator_best.pth \
    --images-path data/test/CFP --a-path data/test/FFA_A --av-path data/test/FFA_AV \
    --masks-path data/test/masks --gt-path data/test/av \
    --save-path preds/cmrrwnet_attention \
    --in_channels 5 --fusion attention --refine
"""

from pathlib import Path
import argparse
import csv
import sys

import numpy as np
import torch
from skimage import io
from torchvision import utils as vutils

sys.path.append('./train')

from preprocessing import enhance_image          # noqa: F401  (kept for parity)
from models import RRWNet, CMRRWNet
from utils import pad_images_unet, to_torch_tensors


# --------------------------------------------------------------------------
# Palettes.  Single source of truth -- quote these RGB values in the figure
# caption so the reader never has to guess what a colour means.
# --------------------------------------------------------------------------
C_ARTERY      = (255,   0,   0)
C_VEIN        = (  0,   0, 255)
C_UNDECIDED   = (  0, 255,   0)     # supplementary figure only
C_PRED_CROSS  = (255,   0, 255)     # supplementary figure only

E_CORRECT     = (150, 150, 150)
E_A_AS_V      = (255, 140,   0)
E_V_AS_A      = (  0, 200, 255)
E_MISSED      = ( 60,  60,  60)
E_SPURIOUS    = (128,   0, 128)
E_CROSSING    = (255, 255, 255)


def save_rgb(path, rgb_uint8):
    io.imsave(str(path), rgb_uint8, check_contrast=False)


def paint(canvas, mask, colour):
    """Write `colour` into `canvas` (H,W,3 uint8) wherever `mask` is True."""
    if mask.any():
        canvas[mask] = colour


def read_mask(mask_fn, shape):
    """Load the FOV mask as a boolean array of the given (H, W)."""
    m = io.imread(mask_fn)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != shape:
        raise ValueError(
            f'FOV mask {mask_fn} has shape {m.shape}, expected {shape}. '
            'Masks must be at the native image resolution -- do not resize here, '
            'resizing silently shifts every downstream coordinate.')
    return m > 127


def read_gt(gt_fn, shape):
    """Return (artery, crossing, vein, vessel) boolean arrays from an AV label."""
    gt = io.imread(gt_fn)
    if gt.ndim != 3 or gt.shape[2] < 3:
        raise ValueError(f'Ground truth {gt_fn} is not an RGB/RGBA image')
    gt = gt[..., :3]
    if gt.shape[:2] != shape:
        raise ValueError(
            f'Ground truth {gt_fn} has shape {gt.shape[:2]}, expected {shape}')
    artery   = gt[..., 0] > 127
    crossing = gt[..., 1] > 127
    vein     = gt[..., 2] > 127
    vessel   = artery | crossing | vein
    return artery, crossing, vein, vessel


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------
def render_main(is_artery_am, is_vein_am, shape):
    """MAIN FIGURE: hard red/blue, no third class."""
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    paint(canvas, is_artery_am, C_ARTERY)
    paint(canvas, is_vein_am,   C_VEIN)
    return canvas


def render_gt_main(gt_artery, gt_crossing, gt_vein, shape):
    """Ground truth rendered in the same palette as the main figure.

    Crossing pixels are drawn white and are excluded from scoring, so the
    reader is told explicitly that they are not counted rather than being
    left to wonder what the third colour means.
    """
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    paint(canvas, gt_artery & ~gt_crossing, C_ARTERY)
    paint(canvas, gt_vein   & ~gt_crossing, C_VEIN)
    paint(canvas, gt_crossing,              E_CROSSING)
    return canvas


def render_uncertain(is_artery_th, is_vein_th, is_undecided, is_cross_pred, shape):
    """SUPPLEMENTARY FIGURE: threshold rule, four visually distinct classes."""
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    paint(canvas, is_artery_th,  C_ARTERY)
    paint(canvas, is_vein_th,    C_VEIN)
    paint(canvas, is_undecided,  C_UNDECIDED)
    paint(canvas, is_cross_pred, C_PRED_CROSS)   # drawn last, overrides A and V
    return canvas


def render_error(gt_artery, gt_crossing, gt_vein, gt_vessel,
                 is_artery_am, is_vein_am, is_vessel, shape, show_fp=True):
    """ERROR MAP built on the argmax assignment."""
    canvas = np.zeros((*shape, 3), dtype=np.uint8)

    gt_a = gt_artery & ~gt_crossing
    gt_v = gt_vein   & ~gt_crossing

    correct  = (gt_a & is_artery_am) | (gt_v & is_vein_am)
    a_as_v   = gt_a & is_vein_am
    v_as_a   = gt_v & is_artery_am
    missed   = (gt_a | gt_v) & ~is_vessel
    spurious = is_vessel & ~gt_vessel

    paint(canvas, correct,   E_CORRECT)
    paint(canvas, missed,    E_MISSED)
    if show_fp:
        paint(canvas, spurious, E_SPURIOUS)
    paint(canvas, a_as_v,    E_A_AS_V)
    paint(canvas, v_as_a,    E_V_AS_A)
    paint(canvas, gt_crossing, E_CROSSING)       # drawn last: excluded region

    counts = dict(
        correct=int(correct.sum()),
        a_as_v=int(a_as_v.sum()),
        v_as_a=int(v_as_a.sum()),
        missed=int(missed.sum()),
        spurious=int(spurious.sum()),
        gt_crossing=int(gt_crossing.sum()),
    )
    return canvas, counts


def render_biomarker(is_artery, is_vein, is_cross, shape):
    """Yellow / cyan / green encoding expected by get_biomarker.py.

    extract_av_masks() there does:  artery = G & ~B ,  vein = G & ~R
    so artery must be yellow (R+G), vein cyan (G+B), crossing pure green.
    """
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    canvas[is_artery] = (255, 255,   0)
    canvas[is_vein]   = (  0, 255, 255)
    canvas[is_cross]  = (  0, 255,   0)
    return canvas


# --------------------------------------------------------------------------
def build_model(args):
    if args.in_channels == 3:
        return RRWNet(input_ch=args.in_channels, output_ch=3,
                      base_ch=args.base_channels, num_iterations=args.k)
    if args.in_channels == 5:
        return CMRRWNet(input_ch=args.in_channels, output_ch=3,
                        base_ch=args.base_channels, num_iterations=args.k,
                        fusion_mode=args.fusion)
    raise ValueError(f'Unsupported number of input channels: {args.in_channels}')


def parse_args():
    p = argparse.ArgumentParser(description='Get predictions and unambiguous visualisations')
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--images-path', type=str, required=True)
    p.add_argument('--a-path', type=str, default='')
    p.add_argument('--av-path', type=str, default='')
    p.add_argument('--masks-path', type=str, required=True)
    p.add_argument('--gt-path', type=str, default='',
                   help='AV ground-truth directory. Omit for the challenge test set; '
                        'error maps and accuracy statistics are then skipped.')
    p.add_argument('--save-path', type=str, required=True)
    p.add_argument('--in_channels', type=int, default=3)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--refine', action='store_true')
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--base_channels', type=int, default=64)
    p.add_argument('--fusion', type=str, default='add',
                   choices=['add', 'weighted', 'attention'])
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--biomarker-assign', type=str, default='threshold',
                   choices=['threshold', 'argmax'],
                   help="How the biomarker masks assign A/V. Keep 'threshold' to "
                        "reproduce existing Task 3 numbers.")
    p.add_argument('--no-fp', action='store_true',
                   help='Do not colour false-positive vessel pixels in the error map.')
    return p.parse_args()


def main():
    args = parse_args()
    thr = args.threshold

    model = build_model(args)
    print(f'Loading model from {args.weights}')
    if torch.cuda.is_available():
        print(f'Using GPU: {args.gpu_id}')
        model.eval()
        model.cuda()
        model.load_state_dict(torch.load(args.weights, map_location=f'cuda:{args.gpu_id}'),
                              strict=False)
    else:
        model.eval()
        model.cpu()
        model.load_state_dict(torch.load(args.weights, map_location='cpu'), strict=True)

    save_path = Path(args.save_path)
    dirs = {k: save_path / k for k in
            ('prob', 'av_main', 'error', 'uncertain', 'av_biomarker', 'gt_main')}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    image_fns = sorted(Path(args.images_path).glob('*.png'))
    mask_fns  = sorted(Path(args.masks_path).glob('*.png'))
    a_fns     = sorted(Path(args.a_path).glob('*.png'))  if args.a_path  else []
    av_fns    = sorted(Path(args.av_path).glob('*.png')) if args.av_path else []
    gt_fns    = sorted(Path(args.gt_path).glob('*.png')) if args.gt_path else []
    use_gt    = bool(gt_fns)

    if args.in_channels == 5 and not (a_fns and av_fns):
        raise ValueError('--in_channels 5 requires --a-path and --av-path')

    rows = []
    for idx, image_fn in enumerate(image_fns):
        mask_fn = mask_fns[idx]
        assert mask_fn.stem == image_fn.stem, f'mask mismatch: {mask_fn} vs {image_fn}'
        print(f'  {image_fn.name}')

        img = (io.imread(image_fn) / 255.0)[..., :3]
        mask = io.imread(mask_fn) * 1.0
        if mask.ndim == 3:
            mask = mask[..., 0]

        if args.in_channels == 5:
            a_fn, av_fn = a_fns[idx], av_fns[idx]
            assert a_fn.stem == image_fn.stem and av_fn.stem == image_fn.stem
            r_a  = (io.imread(a_fn)  / 255.0).reshape(*mask.shape, -1)[..., :1]
            r_av = (io.imread(av_fn) / 255.0).reshape(*mask.shape, -1)[..., :1]
            net_in = np.concatenate([img, r_a, r_av], axis=2)
        else:
            net_in = img

        imgs, paddings = pad_images_unet([net_in, mask])
        padded_in, padding = imgs[0], paddings[0]
        padded_mask = np.stack([imgs[1]] * 3, axis=2)

        tensors = to_torch_tensors([padded_in, padded_mask])
        image_tensor, mask_tensor = tensors[0], tensors[1]
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        image_tensor = image_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)

        with torch.no_grad():
            predictions = model.refine(image_tensor) if args.refine else model(image_tensor)
            last_pred = predictions[-1]
            if not args.refine:
                last_pred = torch.sigmoid(last_pred)
            last_pred = last_pred.cpu()
            last_pred[mask_tensor < 0.5] = 0
            # Crop the padding ONCE, here.  Everything below is at native
            # resolution and stays pixel-aligned with the mask, the ground
            # truth and the optic-disc annotation.
            last_pred = last_pred[:, :,
                                  padding[0][0]:padding[0][0] + img.shape[0],
                                  padding[1][0]:padding[1][0] + img.shape[1]]

        vutils.save_image(last_pred, dirs['prob'] / image_fn.name)

        pred = last_pred[0].numpy()
        artery_p, vessel_p, vein_p = pred[0], pred[1], pred[2]
        shape = artery_p.shape
        fov = read_mask(mask_fn, shape)

        # ---- threshold rule (supplementary figure + biomarker masks) ------
        is_vessel      = (vessel_p > thr) & fov
        above_a        = artery_p > thr
        above_v        = vein_p   > thr
        is_cross_pred  = is_vessel & above_a & above_v
        is_artery_th   = is_vessel & above_a & ~above_v
        is_vein_th     = is_vessel & above_v & ~above_a
        is_undecided   = is_vessel & ~above_a & ~above_v

        # ---- argmax rule (main figure + error map) ------------------------
        is_artery_am = is_vessel & (artery_p >= vein_p)
        is_vein_am   = is_vessel & (vein_p   >  artery_p)

        save_rgb(dirs['av_main'] / image_fn.name,
                 render_main(is_artery_am, is_vein_am, shape))
        save_rgb(dirs['uncertain'] / image_fn.name,
                 render_uncertain(is_artery_th, is_vein_th,
                                  is_undecided, is_cross_pred, shape))

        if args.biomarker_assign == 'threshold':
            bm = render_biomarker(is_artery_th, is_vein_th, is_cross_pred, shape)
        else:
            bm = render_biomarker(is_artery_am, is_vein_am,
                                  np.zeros(shape, dtype=bool), shape)
        save_rgb(dirs['av_biomarker'] / image_fn.name, bm)

        n_vessel = int(is_vessel.sum())
        row = {
            'image': image_fn.stem,
            'vessel_px': n_vessel,
            'undecided_px': int(is_undecided.sum()),
            'undecided_frac': (is_undecided.sum() / n_vessel) if n_vessel else 0.0,
            'pred_crossing_px': int(is_cross_pred.sum()),
        }

        if use_gt:
            gt_fn = gt_fns[idx]
            assert gt_fn.stem == image_fn.stem, f'GT mismatch: {gt_fn} vs {image_fn}'
            gt_a, gt_c, gt_v, gt_vessel = read_gt(gt_fn, shape)
            gt_a &= fov; gt_c &= fov; gt_v &= fov; gt_vessel &= fov

            save_rgb(dirs['gt_main'] / image_fn.name,
                     render_gt_main(gt_a, gt_c, gt_v, shape))

            err_img, counts = render_error(gt_a, gt_c, gt_v, gt_vessel,
                                           is_artery_am, is_vein_am, is_vessel,
                                           shape, show_fp=not args.no_fp)
            save_rgb(dirs['error'] / image_fn.name, err_img)

            scored = counts['correct'] + counts['a_as_v'] + counts['v_as_a']
            row.update(counts)
            row['class_acc_argmax'] = (counts['correct'] / scored) if scored else 0.0
            row['scored_px'] = scored

        rows.append(row)

    # ---- write and summarise -------------------------------------------
    csv_fn = save_path / 'pixel_stats.csv'
    with open(csv_fn, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tot_vessel = sum(r['vessel_px'] for r in rows)
    tot_undec  = sum(r['undecided_px'] for r in rows)
    print('\n' + '=' * 58)
    print(f'images                       : {len(rows)}')
    print(f'undecided pixel fraction     : {100 * tot_undec / max(tot_vessel, 1):.2f} % '
          f'(pooled over all vessel pixels)')
    per_img = [r['undecided_frac'] for r in rows]
    print(f'  per-image mean +/- sd      : {100 * np.mean(per_img):.2f} '
          f'+/- {100 * np.std(per_img, ddof=1):.2f} %')
    if use_gt:
        tot_scored = sum(r['scored_px'] for r in rows)
        tot_ok     = sum(r['correct'] for r in rows)
        tot_av     = sum(r['a_as_v'] for r in rows)
        tot_va     = sum(r['v_as_a'] for r in rows)
        print(f'argmax classification acc.   : {100 * tot_ok / max(tot_scored, 1):.2f} %')
        print(f'  artery labelled as vein    : {100 * tot_av / max(tot_scored, 1):.2f} %')
        print(f'  vein labelled as artery    : {100 * tot_va / max(tot_scored, 1):.2f} %')
    print(f'per-image table              : {csv_fn}')
    print('=' * 58)


if __name__ == '__main__':
    main()
