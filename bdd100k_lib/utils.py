import os.path as osp
import pickle

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.utils import flow_viz, frame_utils, upflow8
from core.utils.utils import InputPadder

FORMAT_SAVE = ["torch_save", "pickle", "png", "flo"]
TORCH_SAVE = 0
PICKLE_SAVE = 1
PNG_SAVE = 2
FLO_SAVE = 3


def viz(img, flo, fname="", show=False):
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

    if fname != "":
        cv2.imwrite(fname, img_flo[:, :, [2, 1, 0]])

    return img_flo[:, :, [2, 1, 0]]


def vizes(imgs, flos, fname="", show=False):
    img_flo_list = []
    for i, (img, flo) in enumerate(zip(imgs, flos)):
        local_fname = fname
        if fname != "":
            local_fname = f"{i}-{fname}"
        local_img_flo = viz(img, flo, fname, local_fname, show)
        img_flo_list.append(local_img_flo)
    return np.array(img_flo_list)


def save_flow(flow,
              out_dir,
              base_name,
              format_save,
              image1,
              padder,
              debug=False,
              is_denormalize=False):
    if is_denormalize:
        flow = denormalize_flow(flow)
    if format_save == FORMAT_SAVE[PICKLE_SAVE]:
        picfile = osp.join(out_dir, "flow-{}.binaryfile".format(base_name))
        if debug:
            print("debug:", picfile)
        with open(picfile, mode="wb") as f:
            pickle.dump(flow, f)
        # with open(picfile, mode="rb") as f:
        #     d = pickle.load(f)
        # print(type(d), d.size(), flow_up.size())
    elif format_save == FORMAT_SAVE[TORCH_SAVE]:
        picfile = osp.join(out_dir, "flow-{}.pth".format(base_name))
        if debug:
            print("debug:", picfile)
        torch.save(flow, picfile)
        # d = torch.load(picfile)
        # print(type(d), d.size(), flow_up.size())
    elif format_save == FORMAT_SAVE[PNG_SAVE]:
        if image1 is None or type(flow) == list:
            return
        hi, hf = image1.shape[2], flow.shape[2]
        if hf == (hi // 8):
            flow = upflow8(flow)
        picfile = osp.join(out_dir, "flow-{}.png".format(base_name))
        if debug:
            print("debug:", picfile)
        flow_save = viz(image1, flow, fname=picfile)
        # cv2.imwrite(picfile, flow_save)
    elif format_save == FORMAT_SAVE[FLO_SAVE]:
        if padder is None or type(flow) == list:
            return
        output_file = osp.join(out_dir, "frame-{}.flo".format(base_name))
        if debug:
            print("debug:", output_file)
        # denormalize flow
        # flow = denormalize_flow(flow)
        flow_save_list = []
        is_one_flow = flow.ndim == 3 or (flow.ndim == 4 and flow.shape[0] == 1)
        if is_one_flow:
            flow_save = flow if flow.ndim == 3 else flow[0]
            flow_save = padder.unpad(flow_save).permute(1, 2, 0).cpu().numpy()
            flow_save_list.append(flow_save)
        elif flow.ndim == 4:
            nb = flow.shape[0]
            if nb > 1:
                for i in range(nb):
                    flow_tmp = flow[i]
                    flow_save = padder.unpad(flow_tmp).permute(1, 2, 0).cpu().numpy()
                    flow_save_list.append(flow_save)
        num_flow = len(flow_save_list)
        if num_flow == 1:
            frame_utils.writeFlow(output_file, flow_save_list[0])
        else:
            for i, flow_save in enumerate(flow_save_list):
                local_base = f"{i}-{base_name}"
                output_file = osp.join(out_dir, "frame-{}.flo".format(local_base))
                frame_utils.writeFlow(output_file, flow_save_list[i])


def preprocessing_imgs(imgs):
    image1 = imgs[0]
    dim = image1.dim()
    device = image1.device
    if dim == 3:
        imgs = [image[None].to(device) for image in imgs]
    elif dim != 4:
        raise NotImplementedError(f"not supported {dim}dims # of dim of images")
    padder = InputPadder(image1.shape)
    imgs = padder.pad(*imgs)
    return imgs, padder


def apply_mask(flow, mask):
    flow_tmp = flow.permute(0, 2, 3, 1).clone()
    mask_rev = torch.logical_not(mask)
    # nb = flow_tmp.shape[0]
    # for idx in range(nb):
    #     flow_tmp[idx][mask_rev[idx]] = 0
    flow_tmp[mask_rev] = 0
    return flow_tmp.permute(0, 3, 1, 2)


@torch.no_grad()
def final_gen_flow(flow_model, imgs, iters=12, up=False, alpha_1=0.01, alpha_2=0.5):
    s_img, e_img, flow_onlycat, mask = gen_flow_correspondence(
        flow_model, imgs, iters, up, alpha_1, alpha_2)
    flow_cat_mask = apply_mask(flow_onlycat, mask)
    s_img, e_img = imgs[0], imgs[-1]
    flow_fwd_init, flow_bwd_init = gen_flows(flow_model, imgs, iters, up)
    # flow_fwd_init = flow_init
    flow_fwd, _ = flow_model(s_img,
                             e_img,
                             iters=iters,
                             flow_init=flow_fwd_init,
                             upsample=False,
                             test_mode=True)
    flow_bwd, _ = flow_model(e_img,
                             s_img,
                             iters=iters,
                             flow_init=flow_bwd_init,
                             upsample=False,
                             test_mode=True)
    flow_fwd = concat_flow(torch.stack([flow_fwd]))
    flow_bwd = concat_flow(torch.stack([flow_bwd]))
    _, _, mask = forward_backward_consistency(flow_fwd, flow_bwd, alpha_1, alpha_2)
    flow_fwd_mask = apply_mask(flow_fwd, mask)
    return flow_fwd_mask, flow_fwd_init, flow_fwd, flow_onlycat, flow_cat_mask


@torch.no_grad()
def gen_flows(flow_model, imgs, iters=12, up=False):
    index = 1 if up else 0
    flow_model.eval()
    flow_fwds = torch.stack([
        flow_model(img0, img1, iters=iters, upsample=False, test_mode=True)[index]
        for img0, img1 in zip(imgs[:-1], imgs[1:])
    ])
    flow_bwds = torch.stack([
        flow_model(img0, img1, iters=iters, upsample=False, test_mode=True)[index]
        for img0, img1 in zip(imgs[1:][::-1], imgs[:-1][::-1])
    ])
    flow_fwd = concat_flow(flow_fwds)
    flow_bwd = concat_flow(flow_bwds)
    return flow_fwd, flow_bwd


@torch.no_grad()
def gen_flow_correspondence(flow_model,
                            imgs,
                            iters=12,
                            up=False,
                            alpha_1=0.01,
                            alpha_2=0.5):
    flow_fwd, flow_bwd = gen_flows(flow_model, imgs, iters, up)
    coords0, coords1, mask = forward_backward_consistency(flow_fwd, flow_bwd, alpha_1,
                                                          alpha_2)
    return imgs[0], imgs[-1], flow_fwd, mask


@torch.no_grad()
def normalize_coord(coords):
    _, _, ht, wd = coords.shape
    coords_norm = coords.clone()
    coords_norm[:, 0] = 2 * coords_norm[:, 0] / (wd - 1) - 1
    coords_norm[:, 1] = 2 * coords_norm[:, 1] / (ht - 1) - 1
    return coords_norm


def denormalize_flow(flow_norm):
    _, _, ht, wd = flow_norm.shape
    flow = flow_norm.clone()
    flow[:, 0] = (flow[:, 0] * (wd - 1)) / 2
    flow[:, 1] = (flow[:, 1] * (ht - 1)) / 2
    return flow


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
    coords0 = torch.stack(coords0[::-1], dim=0).float().repeat(nb, 1, 1, 1)
    coords0 = normalize_coord(coords0)
    coords0 = coords0.to(flows.device)
    coords1 = coords0.clone()
    for flow in flows:
        tmp_flow = normalize_flow(flow)
        flow_interpolate = grid_sample_flow(tmp_flow, coords1)
        coords1 = coords1 + flow_interpolate
    return coords1 - coords0


# implement: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16502/16319
@torch.no_grad()
def forward_backward_consistency(flow_fwd, flow_bwd, alpha_1=0.01, alpha_2=0.5):
    flow_fwd = flow_fwd.detach()
    flow_bwd = flow_bwd.detach()

    nb, _, ht, wd = flow_fwd.shape
    coords0 = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords0 = torch.stack(coords0[::-1], dim=0).float().repeat(nb, 1, 1, 1)
    coords0 = normalize_coord(coords0)
    coords0 = coords0.to(flow_fwd.device)

    coords1 = coords0 + flow_fwd
    mask = (torch.abs(coords1[:, 0]) < 1) & (torch.abs(coords1[:, 1]) < 1)

    flow_bwd_interpolate = grid_sample_flow(flow_bwd, coords1)
    flow_cycle = flow_fwd + flow_bwd_interpolate

    flow_cycle_norm = (flow_cycle**2).sum(1)
    eps = alpha_1 * ((flow_fwd**2).sum(1) + (flow_bwd_interpolate**2).sum(1)) + alpha_2

    mask = mask & ((flow_cycle_norm - eps) <= 0)
    return coords0, coords1, mask
