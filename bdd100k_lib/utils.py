import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.utils import flow_viz
from core.utils.utils import InputPadder


def viz(img, flo, show=False):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    if show:
        cv2.imshow("image", img_flo[:, :, [2, 1, 0]] / 255.0)
        cv2.waitKey()

    return img_flo[:, :, [2, 1, 0]]


def preprocessing_imgs(imgs):
    image1 = imgs[0]
    padder = InputPadder(image1.shape)
    dim = image1.dim()
    if dim == 3:
        imgs = [image[None].cuda().unsqueeze(0) for image in imgs]
    elif dim == 4:
        imgs = [image[None].cuda() for image in imgs]
    else:
        raise NotImplementedError("not supported # of dim of images")
    imgs = padder.pad(*imgs)
    return imgs, padder


@torch.no_grad()
def final_gen_flow(flow_model, imgs, iters=12):
    s_img, e_img, flow_init, mask = gen_flow_correspondence(flow_model, imgs, iters)
    nb = s_img.shape[0]
    mask_rev = torch.logical_not(mask)
    for idx in range(nb):
        flow_init[idx][mask_rev[idx]] = 0
    flow = flow_model(s_img,
                      e_img,
                      iters=iters,
                      flow_init=flow_init,
                      upsample=False,
                      test_mode=True)
    return flow, flow_init


@torch.no_grad()
def gen_flow_correspondence(flow_model, imgs, iters=12):
    flow_model.eval()
    flow_fwds = torch.stack([
        flow_model(img0, img1, iters=iters, upsample=False, test_mode=True)[0]
        for img0, img1 in zip(imgs[:-1], imgs[1:])
    ])
    flow_bwds = torch.stack([
        flow_model(img0, img1, iters=iters, upsample=False, test_mode=True)[0]
        for img0, img1 in zip(imgs[1:][::-1], imgs[:-1][::-1])
    ])
    flow_fwd = concat_flow(flow_fwds)
    flow_bwd = concat_flow(flow_bwds)
    coords0, coords1, mask = forward_backward_consistency(flow_fwd, flow_bwd)
    return imgs[0], imgs[-1], flow_fwd, mask


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


@torch.no_grad()
def grid_sample_flow(flow, coords_norm):
    flow_interpolate = F.grid_sample(flow,
                                     coords_norm.permute(0, 2, 3, 1),
                                     align_corners=True)
    return flow_interpolate


@torch.no_grad()
def concat_flow(flows):
    _, nb, _, ht, wd = flows.shape
    coords0 = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords0 = normalize_coord(
        torch.stack(coords0[::-1], dim=0).float().repeat(nb, 1, 1, 1)).to(flows.device)
    coords1 = coords0.clone()
    for flow in flows:
        flow_interpolate = grid_sample_flow(normalize_flow(flow), coords1)
        coords1 = coords1 + flow_interpolate
    return coords1 - coords0


@torch.no_grad()
def forward_backward_consistency(flow_fwd, flow_bwd, alpha=0.2):
    flow_fwd = flow_fwd.detach()
    flow_bwd = flow_bwd.detach()

    nb, _, ht, wd = flow_fwd.shape
    coords0 = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords0 = normalize_coord(
        torch.stack(coords0[::-1], dim=0).float().repeat(nb, 1, 1,
                                                         1)).to(flow_fwd.device)

    coords1 = coords0 + flow_fwd
    mask = (torch.abs(coords1[:, 0]) < 1) & (torch.abs(coords1[:, 1]) < 1)

    flow_bwd_interpolate = grid_sample_flow(flow_bwd, coords1)
    flow_cycle = flow_fwd + flow_bwd_interpolate

    flow_cycle_norm = (flow_cycle**2).sum(1)
    eps = alpha * ((flow_fwd**2).sum(1) + (flow_bwd_interpolate**2).sum(1))

    mask = mask & ((flow_cycle_norm - eps) <= 0)
    return coords0, coords1, mask
