from pathlib import Path
import argparse
import cv2
import torch
from skimage import io
from torchvision import utils as vutils
import numpy as np

from preprocessing import enhance_image
from model import RRWNet, CMRRWNet
from utils import pad_images_unet, to_torch_tensors



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get predictions from a model')
    parser.add_argument('--weights', type=str, default='',
        help='Path to the model weights')
    parser.add_argument('--images-path', type=str, default='',
        help='Path to the images')
    parser.add_argument('--a-path', type=str, default='',
        help='Path to the masks')
    parser.add_argument('--av-path', type=str, default='',
        help='Path to the masks')
    parser.add_argument('--masks-path', type=str, default='',
        help='Path to the masks')
    parser.add_argument('--save-path', type=str, default='',
        help='Path to save the predictions')
    parser.add_argument('--in_channels', type=int, default=3,
        help='Number of input channels for the model.')
    parser.add_argument('--gpu_id', type=int, default=0,
        help='GPU ID to use. ')
    parser.add_argument('--preprocess', action='store_true',
        help='Preprocess the images')
    parser.add_argument('--refine', action='store_true',
        help='Refine the predictions')
    parser.add_argument('--k', type=int, default=5,
        help='Number of iterations for the refinement module. Default=5')
    parser.add_argument('--base_channels', type=int, default=64,
        help='Base channels for the model. Default=64')
    args = parser.parse_args()

    if args.in_channels == 3:
        model = RRWNet(iterations=args.k, input_ch=args.in_channels, base_ch=args.base_channels)
    elif args.in_channels == 5:
        model = CMRRWNet(iterations=args.k, input_ch=args.in_channels, base_ch=args.base_channels)
    else:
        raise ValueError('Unsupported number of input channels: {}'.format(args.in_channels))
    

    print(f'Loading model from {args.weights}')
    if torch.cuda.is_available():
        
        
        gpu_id = args.gpu_id
        print(f"Using GPU: {gpu_id}")
        model.eval()
        model.cuda()
        model.load_state_dict(torch.load(args.weights, map_location=f'cuda:{gpu_id}'), strict=True)
        device = torch.device("cuda")



    else:
        model.eval()
        model.cpu() 
        model.load_state_dict(torch.load(args.weights, map_location='cpu'), strict=True)
        device = torch.device("cpu")

    print(f'Creating save path {args.save_path}')
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    print(f'Getting images, a, av, masks from {args.images_path} , {args.a_path}, {args.av_path}, {args.masks_path}')
    image_fns = sorted(Path(args.images_path).glob('*.png'))
    a_fns = sorted(Path(args.a_path).glob('*.png'))
    av_fns = sorted(Path(args.av_path).glob('*.png'))
    mask_fns = sorted(Path(args.masks_path).glob('*.png'))

    print('Processing images')
    if args.in_channels == 3:
        pairs = list(zip(image_fns, mask_fns))
    else:
        pairs = list(zip(image_fns, mask_fns, a_fns, av_fns))
    
    for items in pairs:
        if args.in_channels == 3:
            image_fn, mask_fn = items
            a_fn, av_fn = None, None
        else:
            image_fn, mask_fn, a_fn, av_fn = items

        print(f'  {image_fn.name}')
        assert Path(mask_fn).stem == Path(image_fn).stem
        if a_fn is not None:
            assert Path(a_fn).stem == Path(image_fn).stem
            assert Path(av_fn).stem == Path(image_fn).stem

        img = (io.imread(image_fn) / 255.0)[..., :3]
        mask = io.imread(mask_fn) * 1.0
    
        if args.in_channels == 5:
            r_a = (io.imread(a_fn) / 255.0)[..., :1]
            r_av = (io.imread(av_fn) / 255.0)[..., :1]
            img_5ch = np.concatenate([img, r_a, r_av], axis=2)
            imgs, paddings = pad_images_unet([img_5ch, mask])
        else:
            imgs, paddings = pad_images_unet([img, mask])

        img = imgs[0]
        padding = paddings[0]
        mask = imgs[1]
        mask = np.stack([mask,] * 3, axis=2)
        mask_padding = paddings[1]
        tensors = to_torch_tensors([img, mask])
        image_tensor = tensors[0]
        mask_tensor = tensors[1]
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        else:
            tensor = image_tensor.cpu()
        image_tensor = image_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)
        with torch.no_grad():
            if args.refine:
                predictions = model.refine(image_tensor)
            else:
                predictions = model(image_tensor)
            last_pred = predictions[-1]
            if not args.refine:
                last_pred = torch.sigmoid(last_pred)
            last_pred[mask_tensor < 0.5] = 0
            last_pred = last_pred[:, :, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]]
            target_fn = save_path / Path(image_fn).name
            vutils.save_image(last_pred, target_fn)

            # 新增：正确的可视化
            pred = last_pred[0]  # 3×H×W

            artery = pred[0]   # 动脉概率
            vessel = pred[1]   # 整体血管概率  ← 关键，之前没用
            vein   = pred[2]   # 静脉概率

            # 第一步：先确定哪些像素是血管
            is_vessel = vessel > 0.5

            # 第二步：在血管像素里，判断是动脉还是静脉
            is_artery = (artery > 0.5) & is_vessel
            is_vein   = (vein   > 0.5) & is_vessel

            # 第三步：是血管，但动静脉都判断不了 → 不确定区域
            is_uncertain = is_vessel & (~is_artery) & (~is_vein)

            # 上色
            H, W = artery.shape
            vis = torch.zeros(3, H, W)
            vis[0][is_artery]    = 1.0   # 动脉 → 红
            vis[2][is_vein]      = 1.0   # 静脉 → 蓝
            vis[1][is_uncertain] = 1.0   # 不确定 → 绿  ← 新增

            # 动静脉都判断为真的像素（交叉点）会同时有红和蓝 → 品红
            # 如果想让交叉点也显示绿色，可以额外处理：
            is_crossing = (artery > 0.5) & (vein > 0.5) & is_vessel
            vis[0][is_crossing] = 0.0
            vis[2][is_crossing] = 0.0
            vis[1][is_crossing] = 1.0    # 交叉点 → 绿

            # 裁 padding
            vis = vis[:, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]]

            # 应用 mask
            H2, W2 = vis.shape[1], vis.shape[2]
            # import cv2
            mask_orig = cv2.imread(str(mask_fn), cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask_orig, (W2, H2), interpolation=cv2.INTER_NEAREST)
            mask_2d = torch.from_numpy(mask_resized).float() / 255.0
            vis[:, mask_2d < 0.5] = 0

            colored_fn = save_path / ("colored_" + Path(image_fn).name)
            vutils.save_image(vis, colored_fn)