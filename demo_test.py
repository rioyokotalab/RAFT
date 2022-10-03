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

DEVICE = 'cuda'

# implement: https://arxiv.org/pdf/1711.07837.pdf
@torch.no_grad()
def forward_backward_consistency(flow_fwd, flow_bwd, alpha_1=0.01, alpha_2=0.5,
                                 is_norm=True, is_cycle_norm=True, is_coord_norm=True,
                                 is_mask_norm=True, is_alpha2_scale=False):
    flow_fwd = flow_fwd.detach()
    flow_bwd = flow_bwd.detach()
    if is_norm:
        flow_fwd_norm = flow_fwd.clone()
        flow_bwd_norm = flow_bwd.clone()
        flow_fwd = denormalize_flow(flow_fwd)
        flow_fwd = denormalize_flow(flow_fwd)
    else:
        flow_fwd_norm = normalize_flow(flow_fwd)
        flow_bwd_norm = normalize_flow(flow_bwd)

    nb, _, ht, wd = flow_fwd.shape
    coords0 = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords0 = torch.stack(coords0[::-1], dim=0).float().repeat(nb, 1, 1, 1)
    coords0 = coords0.to(flow_fwd.device)
    coords0_norm = normalize_coord(coords0)

    if is_coord_norm:
        coords1_norm = coords0_norm + flow_fwd_norm
    else:
        coords1 = coords0 + flow_fwd
        coords1_norm = normalize_coord(coords1)

    mask = (torch.abs(coords1_norm[:, 0]) < 1) & (torch.abs(coords1_norm[:, 1]) < 1)

    if is_cycle_norm:
        flow_bwd_interpolate_norm = grid_sample_flow(flow_bwd_norm, coords1_norm)
        flow_cycle_norm = flow_fwd_norm + flow_bwd_interpolate_norm
        flow_cycle = denormalize_flow(flow_cycle_norm)
        flow_bwd_interpolate = denormalize_flow(flow_bwd_interpolate_norm)
    else:
        flow_bwd_interpolate = grid_sample_flow(flow_bwd, coords1_norm)
        flow_cycle = flow_fwd + flow_bwd_interpolate
        flow_cycle_norm = normalize_flow(flow_cycle)
        flow_bwd_interpolate_norm = normalize_flow(flow_bwd_interpolate)

    if is_mask_norm:
        flow_cycle_tmp = flow_cycle_norm.clone()
        flow_fwd_tmp = flow_fwd_norm.clone()
        flow_bwd_interpolate_tmp = flow_bwd_interpolate_norm.clone()
        h, w = flow_fwd.shape[-2:]
        h, w = torch.tensor(h), torch.tensor(w)
        if is_alpha2_scale:
            alpha_2 = alpha_2 / (torch.sqrt(h**2 + w**2).item())
    else:
        flow_cycle_tmp = flow_cycle.clone()
        flow_fwd_tmp = flow_fwd.clone()
        flow_bwd_interpolate_tmp = flow_bwd_interpolate.clone()

    flow_cycle_abs_norm = (flow_cycle_tmp**2).sum(1)
    eps = alpha_1 * ((flow_fwd_tmp**2).sum(1) + (flow_bwd_interpolate_tmp**2).sum(1)) + alpha_2
    mask = mask & ((flow_cycle_abs_norm - eps) <= 0)

    # mask = mask & ((flow_cycle[:, 0] <= 0) & (flow_cycle[:, 0] <= 0))
    # return coords0, coords1, mask

    # mask_rev = torch.logical_not(mask)
    mask_rev = mask.clone()
    flow_fwd_mask = flow_fwd.clone()
    flow_fwd_mask = flow_fwd_mask.permute(0, 2, 3, 1)
    flow_fwd_mask[mask_rev] = 1.0
    flow_fwd_mask = flow_fwd_mask.permute(0, 3, 1, 2)
    flow_fwd_mask_norm = normalize_flow(flow_fwd_mask)
    return flow_cycle, mask, [flow_fwd_mask, flow_fwd_mask_norm]


@torch.no_grad()
def normalize_coord(coords):
    _, _, ht, wd = coords.shape
    coords_norm = coords.clone()
    coords_norm[:, 0] = 2 * coords_norm[:, 0] / (wd - 1) - 1
    coords_norm[:, 1] = 2 * coords_norm[:, 1] / (ht - 1) - 1
    return coords_norm


@torch.no_grad()
def normalize_flow(flow):
    _, _, ht, wd = flow.shape
    flow_norm = flow.clone()
    flow_norm[:, 0] = 2 * flow_norm[:, 0] / (wd - 1)
    flow_norm[:, 1] = 2 * flow_norm[:, 1] / (ht - 1)
    return flow_norm


def denormalize_flow(flow_norm):
    _, _, ht, wd = flow_norm.shape
    flow = flow_norm.clone()
    flow[:, 0] = flow[:, 0] * (wd - 1) / 2
    flow[:, 1] = flow[:, 1] * (ht - 1) / 2
    return flow


@torch.no_grad()
def grid_sample_flow(flow, coords_norm):
    flow_interpolate = F.grid_sample(flow,
                                     coords_norm.permute(0, 2, 3, 1),
                                     align_corners=True)
    return flow_interpolate


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


def concat_img(im1, im2, fname=""):
    img1 = im1.permute(1, 2, 0).cpu().numpy()
    img2 = im2.permute(1, 2, 0).cpu().numpy()
    img_cat = np.concatenate([img1, img2], axis=0)
    if fname != "":
        cv2.imwrite(fname, img_cat[:, :, [2, 1, 0]])

    return img_cat


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

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x - 255, vgrid, align_corners=True)
    output = output + 255
    if mask is not None:
        mask_rev = torch.logical_not(mask)
        output = output.permute(0, 2, 3, 1)
        output[mask_rev] = 255
        output = output.permute(0, 3, 1, 2)
    # mask = torch.ones(x.size()).to(DEVICE)
    # mask = F.grid_sample(mask, vgrid)

    # mask[mask < 0.999] = 0
    # mask[mask > 0] = 1

    return output


def prepare_out(args):
    out_root = args.out_path
    alpha1, alpha2 = args.alpha
    is_norm = not args.not_calc_norm_flow
    is_cycle_norm = not args.is_not_cycle_norm
    is_coord_norm = not args.is_not_coord_norm
    is_mask_norm = not args.is_not_mask_norm
    is_alpha2_scale = args.is_alpha2_scale
    out_path = os.path.join(out_root, f"alpha1_{alpha1}_alpha2_{alpha2}")
    if is_norm:
        out_path = os.path.join(out_path, "calc_norm_flow")
    else:
        out_path = os.path.join(out_path, "not_calc_norm_flow")
    if is_cycle_norm:
        out_path = os.path.join(out_path, "is_cycle_norm")
    else:
        out_path = os.path.join(out_path, "is_not_cycle_norm")
    if is_coord_norm:
        out_path = os.path.join(out_path, "is_coord_norm")
    else:
        out_path = os.path.join(out_path, "is_not_coord_norm")
    if is_mask_norm:
        out_path = os.path.join(out_path, "is_mask_norm")
    else:
        out_path = os.path.join(out_path, "is_not_mask_norm")
    if is_alpha2_scale:
        out_path = os.path.join(out_path, "is_alpha2_scale")
    else:
        out_path = os.path.join(out_path, "is_not_alpha2_scale")
    base_name = os.path.basename(args.path)
    dt_str = args.date_str
    out_path = os.path.join(out_path, base_name, dt_str)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    # out_root = f"./output/flowtest/warp_test2/{base_name}"
    out_root = prepare_out(args)
    alpha1, alpha2 = args.alpha
    is_norm = not args.not_calc_norm_flow
    is_cycle_norm = not args.is_not_cycle_norm
    is_coord_norm = not args.is_not_coord_norm
    is_mask_norm = not args.is_not_mask_norm
    is_alpha2_scale = args.is_alpha2_scale

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        print(f"len_img: {len(images)}")
        # images = [images[0], images[-1]]
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            imbase1 = os.path.basename(imfile1)
            imbase2 = os.path.basename(imfile2)
            fname = f"{imbase1}_{imbase2}_warp.png"
            fname_warp2 = f"{imbase1}_{imbase2}_warp2.png"
            fname_warp3 = f"{imbase1}_{imbase2}_warp3.png"
            fname_warp4 = f"{imbase1}_{imbase2}_warp4.png"
            fname2 = f"{imbase1}_{imbase2}_flo.png"
            fname3 = f"{imbase1}_{imbase2}_flo_bwd.png"
            fname4 = f"{imbase1}_{imbase2}_flo_cycle.png"
            fname5 = f"{imbase1}_{imbase2}_flo_cycle_bwd.png"
            fname = os.path.join(out_root, fname)
            fname_warp2 = os.path.join(out_root, fname_warp2)
            fname_warp3 = os.path.join(out_root, fname_warp3)
            fname_warp4 = os.path.join(out_root, fname_warp4)
            fname2 = os.path.join(out_root, fname2)
            fname3 = os.path.join(out_root, fname3)
            fname4 = os.path.join(out_root, fname4)
            fname5 = os.path.join(out_root, fname5)
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_low, flow_up = model(image1, image2, test_mode=True)
            flow_low_bwd, flow_up_bwd = model(image2, image1, test_mode=True)
            if is_norm:
                flow_up_norm = normalize_flow(flow_up)
                flow_up_bwd_norm = normalize_flow(flow_up_bwd)
                flow_up_tmp = flow_up_norm.clone()
                flow_up_bwd_tmp = flow_up_bwd_norm.clone()
            else:
                flow_up_tmp = flow_up.clone()
                flow_up_bwd_tmp = flow_up_bwd.clone()
            flow_cycle, mask, flow_fwd_mask = forward_backward_consistency(flow_up_tmp, flow_up_bwd_tmp, alpha1, alpha2,
                                                                           is_norm=is_norm, is_cycle_norm=is_cycle_norm,
                                                                           is_coord_norm=is_coord_norm,
                                                                           is_mask_norm=is_mask_norm,
                                                                           is_alpha2_scale=is_alpha2_scale)
            flow_cycle_bwd, mask_bwd, flow_bwd_mask = forward_backward_consistency(flow_up_bwd_tmp, flow_up_tmp, alpha1, alpha2,
                                                                                   is_norm=is_norm, is_cycle_norm=is_cycle_norm,
                                                                                   is_coord_norm=is_coord_norm,
                                                                                   is_mask_norm=is_mask_norm,
                                                                                   is_alpha2_scale=is_alpha2_scale)
            flow_fwd_mask, flow_fwd_mask_norm = flow_fwd_mask
            flow_bwd_mask, flow_bwd_mask_norm = flow_bwd_mask
            viz(image1, flow_up, fname2)
            viz(image1, flow_up_bwd, fname3)
            flo_cycle_img = viz(image1, flow_cycle, fname4)
            mask2 = ((flo_cycle_img[0] >= 245) & (flo_cycle_img[1] >= 245) & (flo_cycle_img[2] >= 245))
            # white_c = 255 * 3 - 50
            # mask2 = (flo_cycle_img[0].to(torch.int) + flo_cycle_img[1].to(torch.int) + flo_cycle_img[2].to(torch.int)) >= white_c
            mask2 = mask2.unsqueeze(0)
            viz(image1, flow_cycle_bwd, fname5)
            output = warp(image2, flow_up)
            output2 = warp(image2, flow_fwd_mask)
            # output3 = warp(image2, flow_fwd_mask, mask)
            output3 = warp(image2, flow_up, mask)
            output4 = warp(image2, flow_up, mask2)
            concat_img(image1[0], output[0], fname)
            concat_img(image1[0], output2[0], fname_warp2)
            concat_img(image1[0], output3[0], fname_warp3)
            concat_img(image1[0], output4[0], fname_warp4)
            # rank = 0
            # print(f"rank: {rank} orig_im1: {image1.dtype} orig_im2: {image2.dtype}")
            # print(f"rank: {rank} orig_im1: {image1.shape}", image1.tolist())
            # print(f"rank: {rank} orig_im2: {image2.shape}", image2.tolist())
            # print(f"rank: {rank} flow_fwd: {flow_low.shape}", flow_low.tolist())
            # print(f"rank: {rank} flow_bwd: {flow_low_bwd.shape}", flow_low_bwd.tolist())


if __name__ == '__main__':
    dt_now = datetime.datetime.now()
    dt_str = dt_now.strftime("%Y%m%d_%H%M%S")

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
    parser.add_argument('--out_path', help="out path")
    parser.add_argument('--alpha', nargs=2, type=float, default=[0.01, 0.5], help="out path")
    parser.add_argument('--not_calc_norm_flow', action='store_true')
    parser.add_argument('--is_not_cycle_norm', action='store_true')
    parser.add_argument('--is_not_coord_norm', action='store_true')
    parser.add_argument('--is_not_mask_norm', action='store_true')
    parser.add_argument('--is_alpha2_scale', action='store_true')
    parser.add_argument('--date_str', default=dt_str)
    args = parser.parse_args()

    demo(args)
