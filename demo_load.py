import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from core.utils import flow_viz
from core.utils.utils import InputPadder
from core.utils.utils import upflow8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img=None, flo=None, imfile=None, upsample=True):
    if flo is None:
        return
    if upsample:
        # change size to origin size
        flo = upflow8(flo)
    if imfile is None:
        imfile = "test.png"
    if img is not None:
        img = img[0].permute(1, 2, 0).cpu().numpy()

    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # print(img.shape, flo.shape)
    img_flo = flo if img is None else np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow("image", img_flo[:, :, [2, 1, 0]] / 255.0)
    # cv2.waitKey()
    cv2.imwrite(imfile, img_flo[:, :, [2, 1, 0]])


def demo(args):

    result_path = args.result
    os.makedirs(result_path, exist_ok=True)
    flows = glob.glob(os.path.join(args.flow_path, "*.pth"))
    images = flows.copy()
    if args.image_path is not None:
        images = glob.glob(os.path.join(args.image_path, "*.png")) + \
                 glob.glob(os.path.join(args.image_path, "*.jpg"))

    flows = sorted(flows)
    images = sorted(images)
    for flofile, imfile1, imfile2 in zip(flows, images[:-1], images[1:]):
        flow_low = torch.load(flofile, map_location=torch.device(DEVICE))
        image1 = None if args.image_path is None else load_image(imfile1)
        if image1 is not None:
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape)
            image1, _ = padder.pad(image1, image2)
        save_imfile = result_path + "flow-img" + os.path.basename(imfile1)
        viz(image1, flow_low, save_imfile, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="result", help="result path")
    parser.add_argument("--flow-path", default="flow", help="flow load path")
    parser.add_argument("--image-path", default=None, help="images path")
    args = parser.parse_args()

    demo(args)
