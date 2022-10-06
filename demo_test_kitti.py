import argparse
import os
import cv2
import datetime
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
import core.datasets as datasets

DEVICE = 'cuda'


class myKITTI(datasets.KITTI):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(myKITTI, self).__init__(aug_params, sparse=True)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if self.is_test:
            return data
        out_data = [d for d in data]
        out_data.append(self.extra_info[index])
        return out_data


class KITTIOld(datasets.FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI/2012'):
        super(KITTIOld, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'colored_0/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'colored_0/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if self.is_test:
            return data
        out_data = [d for d in data]
        out_data.append(self.extra_info[index])
        return out_data


def load_image(imfile):
    # from torchvision import transforms
    # to_tensor = transforms.ToTensor()
    # img = to_tensor(Image.open(imfile)) * 255
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)
    # return img[None].cuda()


def viz(img, flo, fname=""):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    if fname != "":
        cv2.imwrite(fname, img_flo[:, :, [2, 1, 0]])

    # cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    # cv2.waitKey()
    return torch.from_numpy(flo).permute(2, 0, 1)


def adjust_dim(tar, out_dim=4):
    in_dim = tar.ndim
    assert isinstance(out_dim, int)
    assert isinstance(tar, torch.Tensor)
    num = out_dim - in_dim
    out = tar.clone()
    if num == 0:
        return out
    is_down_dim = num <= 0
    for i in range(abs(num)):
        if is_down_dim:
            assert out.shape[0] == 1
            out = out.squeeze(0)
        else:
            out = out.unsqueeze(0)
    return out


def apply_mask(img_src, mask_src=None, pad_v=255):
    if mask_src is None or img_src is None:
        return None
    in_dim = img_src.ndim
    mask = adjust_dim(mask_src, 3)
    img = adjust_dim(img_src, 4)
    mask_rev = torch.logical_not(mask)
    img_mask = img.clone()
    img_mask = img_mask.permute(0, 2, 3, 1)
    img_mask[mask_rev] = pad_v
    img_mask = img_mask.permute(0, 3, 1, 2)
    img_mask = adjust_dim(img_mask, in_dim)
    return img_mask


def concat_img(im1, im2, cat_dst="h", fname="", mask=None, pad_v=255):
    if mask is not None:
        im2 = apply_mask(im2, mask, pad_v)
        print(im2.ndim)
    img1 = im1.permute(1, 2, 0).cpu().numpy()
    img2 = im2.permute(1, 2, 0).cpu().numpy()
    axis = 0
    if cat_dst == "w":
        axis = 1
    img_cat = np.concatenate([img1, img2], axis=axis)
    if fname != "":
        cv2.imwrite(fname, img_cat[:, :, [2, 1, 0]])

    return torch.from_numpy(img_cat).permute(2, 0, 1)


def warp(x, flo, mask=None):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda or flo.is_cuda:
        if flo.is_cuda:
            grid = grid.cuda()
            x = x.cuda()
        else:
            grid = grid.cuda()
            flo = flo.cuda()

    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x - 255, vgrid, align_corners=True)
    output = output + 255
    out_mask = apply_mask(output, mask, 255)
    # mask = torch.ones(x.size()).to(DEVICE)
    # mask = F.grid_sample(mask, vgrid)

    # mask[mask < 0.999] = 0
    # mask[mask > 0] = 1

    return output, out_mask


def demo(args):
    # out_root = f"./output/flowtest/warp_test2/{base_name}"
    out_root = os.path.join(args.out_path, "flow_test")
    os.makedirs(out_root, exist_ok=True)
    if os.path.basename(args.path) == "2012":
        val_dataset = KITTIOld(split='training', root=args.path)
    else:
        val_dataset = myKITTI(split='training', root=args.path)

    with torch.no_grad():
        # images = glob.glob(os.path.join(args.path, '*.png')) + \
        #          glob.glob(os.path.join(args.path, '*.jpg'))

        # images = sorted(images)
        # print(f"len_img: {len(images)}")
        # images = [images[0], images[-1]]
        # for imfile1, imfile2 in zip(images[:-1], images[1:]):
        for val_id in range(len(val_dataset)):
            # imbase1 = os.path.basename(imfile1)
            # imbase2 = os.path.basename(imfile2)
            # image1 = load_image(imfile1)
            # image2 = load_image(imfile2)
            image1, image2, flow_gt, valid_gt, info = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            flow_gt = flow_gt.unsqueeze(0)
            valid_gt = valid_gt.unsqueeze(0)
            frame_id = info[0]
            imbase1 = frame_id.replace(".png", "")
            imbase2 = frame_id.replace("_10.png", "_11")
            fname_base = os.path.join(out_root, f"{imbase1}_{imbase2}")
            fname_warp = f"{fname_base}_warp.png"
            fname_flo = f"{fname_base}_flo.png"
            fname_flo_bwd = f"{fname_base}_flo_bwd.png"
            fname_flo_mask = f"{fname_base}_flo_mask.png"
            fname_flo_mask_bwd = f"{fname_base}_flo_mask_bwd.png"

            # padder = InputPadder(image1.shape, mode='kitti')
            # image1, image2 = padder.pad(image1, image2)
            val = valid_gt >= 0.5
            # val = valid_gt.view(-1) >= 0.5

            print(image1.shape, flow_gt.shape, val.shape)

            flo_img = viz(image1, flow_gt, fname_flo)
            output, out_mask = warp(image2, flow_gt, val)
            im1_cat = concat_img(image1[0], output[0])
            im2_cat = concat_img(image2[0], out_mask[0])
            flo_cat = concat_img(image1[0], flo_img, "h", fname_flo_mask, val, 0)
            concat_img(im1_cat, im2_cat, "w", fname_warp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='datasets/KITTI/2015', help="dataset path")
    parser.add_argument('--out_path', help="out path")
    args = parser.parse_args()

    demo(args)
