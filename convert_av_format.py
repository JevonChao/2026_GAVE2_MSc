"""
Script 1 / 3 - artery and vein label format conversion

Converts the GAVE2 artery and vein labels (R = artery, G = crossing,
B = vein) into the format expected by get_biomarker.py:
    artery   -> yellow (255,255,0)  satisfies "G high and B low"
    vein     -> cyan   (0,255,255)  satisfies "G high and R low"
    crossing -> green  (0,255,0)    G high, R low, B low (consistent with the
                                    logic of that script)

Usage:
    python convert_av_format.py                        # convert the reference labels (default)
    python convert_av_format.py <input_dir> <output_dir>   # convert an arbitrary directory
"""

import sys
import numpy as np
from pathlib import Path
from skimage import io


def convert(src_path, dst_path):
    gt = io.imread(src_path)
    gt = gt[:, :, :3]  
    r, g, b = gt[:, :, 0], gt[:, :, 1], gt[:, :, 2]

    is_artery = (r > 127) & (g < 127) & (b < 127)   # originally red -> artery
    is_vein = (b > 127) & (r < 127) & (g < 127)     # originally blue -> vein
    is_cross = (g > 127) & (r < 127) & (b < 127)    # originally green -> crossing

    out = np.zeros_like(gt)
    out[is_artery] = [255, 255, 0]    
    out[is_vein] = [0, 255, 255]     
    out[is_cross] = [0, 255, 0]    
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