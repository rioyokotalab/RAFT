import argparse
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

DEVICE = 'cuda'


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


def concat_img(im1, im2, fname=""):
    img1 = im1.permute(1, 2, 0).cpu().numpy()
    img2 = im2.permute(1, 2, 0).cpu().numpy()
    img_cat = np.concatenate([img1, img2], axis=0)
    if fname != "":
        cv2.imwrite(fname, img_cat[:, :, [2, 1, 0]])

    return img_cat


def warp(x, flo, fname=""):
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

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to(DEVICE)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    out_root = "./output/flowtest/warp_test"
    os.makedirs(out_root, exist_ok=True)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        print(f"len_img: {len(images)}")
        images = [images[0], images[-1]]
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            imbase1 = os.path.basename(imfile1)
            imbase2 = os.path.basename(imfile2)
            fname = f"{imbase1}_{imbase2}_warp.png"
            fname = os.path.join(out_root, fname)
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            # padder = InputPadder(image1.shape)
            # image1, image2 = padder.pad(image1, image2)

            # flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_low, flow_up = model(image1, image2, test_mode=True)
            flow_b_low, flow_b_up = model(image2, image1, test_mode=True)
            # viz(image1, flow_up)
            # output = warp(image2, flow_up, fname)
            # concat_img(image1[0], output[0], fname)
            rank = 0
            print(f"rank: {rank} orig_im1: {image1.dtype} orig_im2: {image2.dtype}")
            print(f"rank: {rank} orig_im1: {image1.shape}", image1.tolist())
            print(f"rank: {rank} orig_im2: {image2.shape}", image2.tolist())
            print(f"rank: {rank} flow_fwd: {flow_low.shape}", flow_low.tolist())
            print(f"rank: {rank} flow_bwd: {flow_b_low.shape}", flow_b_low.tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision',
                        action='store_true',
                        help='use mixed precision')
    parser.add_argument('--alternate_corr',
                        action='store_true',
                        help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
