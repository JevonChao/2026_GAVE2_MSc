"""
Standalone evaluation script for GAVE2 AV segmentation.

Loads a trained checkpoint, runs inference, and reports
Dice / IoU / Precision / Recall / Specificity for three targets:
  Artery, Vessel (all), Vein

Works for both:
  Task1 (RRWNet,   --in_channels 3, CFP only)
  Task2 (CMRRWNet, --in_channels 5, CFP + FFA)

The official validation set has no public labels, so evaluation is done on the
fold-0 held-out split of the training set (g_001..g_012), which the model never
saw during training. Use --only-fold-val to enable this (recommended).

Usage (Task1):
  python evaluate.py `
    --weights "./__training/Journal_paper/GAVE_pair/4_folds/RRWNet_5it_lr1e-04_RRLoss-BCE3Loss_bc32/0/generator_best.pth" `
    --images-path "./data/training/images" `
    --masks-path "./data/training/masks" `
    --gt-path "./data/training/av" `
    --in_channels 3 --base_channels 32 --only-fold-val

Usage (Task2):
  python evaluate.py `
    --weights "./__training/Journal_paper/GAVE_pair/4_folds/CMRRWNet_5it_lr1e-04_RRLoss-BCE3Loss_bc32/0/generator_best.pth" `
    --images-path "./data/training/images" `
    --a-path "./data/training/FFA_A" `
    --av-path "./data/training/FFA_AV" `
    --masks-path "./data/training/masks" `
    --gt-path "./data/training/av" `
    --in_channels 5 --base_channels 32 --only-fold-val
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from skimage import io

sys.path.append('./train')
from models import RRWNet, CMRRWNet
from transformations import pad_images_unet, to_torch_tensors


# fold 0 held-out split (4-fold split over g_001..g_048)
FOLD0_VAL_IDS = {f'g_{i:03d}' for i in range(1, 13)}   # g_001 .. g_012

CHANNELS = {
    0: 'Artery',
    1: 'Vessel',
    2: 'Vein',
}


def compute_metrics(pred_bin, target_bin, smooth=1e-6):
    """Metrics from two boolean arrays of identical shape."""
    pred = pred_bin.astype(np.float64).ravel()
    target = target_bin.astype(np.float64).ravel()

    TP = float((pred * target).sum())
    FP = float((pred * (1.0 - target)).sum())
    FN = float(((1.0 - pred) * target).sum())
    TN = float(((1.0 - pred) * (1.0 - target)).sum())

    dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    iou = (TP + smooth) / (TP + FP + FN + smooth)
    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    specificity = (TN + smooth) / (TN + FP + smooth)

    return dice, iou, precision, recall, specificity


def load_gray(path):
    """Read an image and return a float32 HxW array in [0, 1] (drops alpha)."""
    arr = io.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32) / 255.0


def load_gt_channels(path):
    """
    Read an RGBA / RGB AV annotation and rebuild the 3 training targets.

    Annotation colours (uint8):
        red   (255,0,0)  -> artery
        green (0,255,0)  -> crossing / uncertain
        blue  (0,0,255)  -> vein
        black (0,0,0)    -> background

    Training targets (see train/data_vessels.py):
        ch0 = artery + crossing
        ch1 = artery + crossing + vein   (all vessels)
        ch2 = vein   + crossing
    """
    gt = io.imread(path)
    if gt.ndim != 3 or gt.shape[2] < 3:
        raise ValueError(f'Ground truth {path} is not an RGB/RGBA image')

    gt = gt[..., :3].astype(np.float32) / 255.0   # drop alpha channel
    r, g, b = gt[..., 0], gt[..., 1], gt[..., 2]

    artery = np.clip(r + g, 0.0, 1.0)
    vessel = np.clip(r + g + b, 0.0, 1.0)
    vein = np.clip(b + g, 0.0, 1.0)

    return np.stack([artery, vessel, vein], axis=0)   # [3, H, W]


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained AV segmentation model (Task1 or Task2)')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to generator_best.pth')
    parser.add_argument('--images-path', type=str, required=True,
                        help='CFP images directory')
    parser.add_argument('--a-path', type=str, default=None,
                        help='FFA_A directory (required when --in_channels 5)')
    parser.add_argument('--av-path', type=str, default=None,
                        help='FFA_AV directory (required when --in_channels 5)')
    parser.add_argument('--masks-path', type=str, required=True,
                        help='ROI masks directory')
    parser.add_argument('--gt-path', type=str, required=True,
                        help='Ground-truth AV annotation directory')
    parser.add_argument('--in_channels', type=int, default=3, choices=[3, 5],
                        help='3 = RRWNet (Task1), 5 = CMRRWNet (Task2)')
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Must match the value used at training time')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of refinement iterations')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarisation threshold on the sigmoid probability')
    parser.add_argument('--only-fold-val', action='store_true',
                        help='Evaluate only the fold-0 held-out split (g_001..g_012)')
    parser.add_argument('--fusion', type=str, default='add',
                        choices=['add', 'weighted', 'attention'],
                        help='Fusion mode for CMRRWNet (must match training)')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--csv', type=str, default=None,
                        help='Optional path to write per-image scores as CSV')
    args = parser.parse_args()

    device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # ---------------- Model ----------------
    if args.in_channels == 3:
        model = RRWNet(input_ch=3, output_ch=3,
                       base_ch=args.base_channels, num_iterations=args.k)
        model_name = 'RRWNet   (Task1: CFP only)'
    else:
        if args.a_path is None or args.av_path is None:
            parser.error('--a-path and --av-path are required when --in_channels 5')
        model = CMRRWNet(input_ch=5, output_ch=3,
                         base_ch=args.base_channels, num_iterations=args.k,
                         fusion_mode=args.fusion)
        model_name = 'CMRRWNet (Task2: CFP + FFA)'

    print(f'Loading {model_name}')
    print(f'  weights : {args.weights}')
    print(f'  device  : {device}')
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # ---------------- File list ----------------
    images_path = Path(args.images_path)
    masks_path = Path(args.masks_path)
    gt_path = Path(args.gt_path)

    image_fns = sorted(images_path.glob('*.png'))
    if not image_fns:
        raise FileNotFoundError(f'No .png images found in {images_path}')

    if args.only_fold_val:
        image_fns = [fn for fn in image_fns if fn.stem in FOLD0_VAL_IDS]
        print(f'  split   : fold-0 held-out ({len(image_fns)} images)')
    else:
        print(f'  split   : every image in {images_path} ({len(image_fns)} images)')
        print('  WARNING : this includes images the model was trained on.')

    if args.in_channels == 5:
        a_path = Path(args.a_path)
        av_path = Path(args.av_path)

    # ---------------- Inference ----------------
    scores = {name: [] for name in CHANNELS.values()}
    rows = []

    print(f'\nthreshold = {args.threshold}\n')

    for image_fn in image_fns:
        stem = image_fn.stem
        mask_fn = masks_path / image_fn.name
        gt_fn = gt_path / image_fn.name

        if not mask_fn.exists():
            print(f'  [skip] {stem}: no mask')
            continue
        if not gt_fn.exists():
            print(f'  [skip] {stem}: no ground truth')
            continue

        img = (io.imread(image_fn).astype(np.float32) / 255.0)[..., :3]
        mask = load_gray(mask_fn)

        if args.in_channels == 5:
            r_a = load_gray(a_path / image_fn.name)[..., np.newaxis]
            r_av = load_gray(av_path / image_fn.name)[..., np.newaxis]
            net_input = np.concatenate([img, r_a, r_av], axis=2)
        else:
            net_input = img

        imgs, paddings = pad_images_unet([net_input], return_paddings=True)
        net_input, padding = imgs[0], paddings[0]

        x = to_torch_tensors([net_input])[0].unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(x)
            logits = predictions[-1]                # [1, 3, H, W] raw logits
            probs = torch.sigmoid(logits)[0]        # [3, H, W]

        probs = probs[:,
                      padding[0][0]:-padding[0][1],
                      padding[1][0]:-padding[1][1]].cpu().numpy()

        gt_stack = load_gt_channels(gt_fn)          # [3, H, W]
        roi = mask > 0.5                            # retinal field of view

        if probs.shape[1:] != gt_stack.shape[1:]:
            raise ValueError(
                f'{stem}: prediction {probs.shape[1:]} != ground truth '
                f'{gt_stack.shape[1:]}')

        row = {'image': stem}
        for ch, name in CHANNELS.items():
            pred_bin = (probs[ch] > args.threshold) & roi
            targ_bin = (gt_stack[ch] > 0.5) & roi
            m = compute_metrics(pred_bin, targ_bin)
            scores[name].append(m)
            row[f'{name}_dice'] = m[0]
            row[f'{name}_iou'] = m[1]
        rows.append(row)

        d = {n: scores[n][-1][0] for n in CHANNELS.values()}
        print(f'  {stem}   dice  '
              f'A={d["Artery"]:.4f}  V={d["Vessel"]:.4f}  Vn={d["Vein"]:.4f}')

    if not rows:
        print('\nNo images were evaluated.')
        return

    # ---------------- Report ----------------
    n = len(rows)
    line = '=' * 80
    print('\n' + line)
    print(f'{model_name}')
    print(f'threshold = {args.threshold}   n = {n} images')
    print(line)
    print(f'{"Target":<10}{"Dice":>11}{"IoU":>11}{"Precision":>13}'
          f'{"Recall":>11}{"Specificity":>15}')
    print('-' * 80)

    for name in CHANNELS.values():
        arr = np.array(scores[name])           # [n, 5]
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        print(f'{name:<10}'
              f'{mean[0]:>11.4f}{mean[1]:>11.4f}{mean[2]:>13.4f}'
              f'{mean[3]:>11.4f}{mean[4]:>15.4f}')
        print(f'{"  ± std":<10}'
              f'{std[0]:>11.4f}{std[1]:>11.4f}{std[2]:>13.4f}'
              f'{std[3]:>11.4f}{std[4]:>15.4f}')
    print(line)

    if args.csv:
        import csv as _csv
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nPer-image scores written to {out}')


if __name__ == '__main__':
    main()