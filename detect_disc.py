"""
Script 2 / 3 - optic disc detection (classical image processing)

The optic disc is the brightest circular region in a fundus photograph. This
script locates its centre from the brightest area of the green channel and
writes a grayscale mask (disc = 255, background = 0) named after the
corresponding CFP image, for use by get_biomarker.py.

Usage:
    python detect_disc.py                          # process the training CFP images (default)
    python detect_disc.py <cfp_dir> <output_dir>   # process an arbitrary directory

Note: this is a classical method intended for quick verification. After
running it, inspect a few outputs to confirm that the white circle falls on
the optic disc (the bright disc from which the vessels radiate) rather than on
some other bright region.
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

    # The disc has better contrast in the green channel (the red channel saturates easily)
    green = img[:, :, 1].astype(np.float32)

    # Gaussian blur suppresses vessel detail and emphasises large bright regions
    blur = cv2.GaussianBlur(green, (0, 0), sigmaX=max(h * 0.01, 1))

    # Take the brightest point as the disc centre
    _, _, _, max_loc = cv2.minMaxLoc(blur)
    cx, cy = max_loc

    # Empirical disc diameter: about dd_frac of the image width
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