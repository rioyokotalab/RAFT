import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import pickle
import time

import sys

sys.path.append('core')

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, imfile):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    # cv2.waitKey()
    cv2.imwrite(imfile, img_flo[:, :, [2, 1, 0]])


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        sequences, pickle_list = [], []
        subset = "all" if args.all else "flow_low"
        if args.root:
            img_path = args.root
            dir_names = sorted(os.listdir(img_path))
            stop = len(dir_names)
            if args.datanum:
                stop = args.datanum + args.start
            selected_dirs = dir_names[args.start:stop]
            for seq in selected_dirs:
                result_path = os.path.join(img_path, seq)
                images = sorted(glob.glob(os.path.join(result_path, "*.jpg")))
                output_path = os.path.join(args.output, result_path.lstrip("/"),
                                           args.format_save, subset)
                os.makedirs(output_path, exist_ok=True)
                sequences.append([images, output_path])
        else:
            result_path = args.path.lstrip("/")
            output_path = os.path.join(args.output, result_path, args.format_save,
                                       subset)
            os.makedirs(output_path, exist_ok=True)
            images = glob.glob(os.path.join(args.path, '*.png')) + \
                glob.glob(os.path.join(args.path, '*.jpg'))
            images = sorted(images)
            sequences.append([images, output_path])

        total_time = 0
        for i, [images, output_path] in enumerate(sequences):
            s_time = time.time()
            for imfile1, imfile2 in zip(images[:-1], images[1:]):
                print("debug:", imfile1, imfile2)
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                if args.all:
                    pickle_list.append(flow_low)
                    continue

                base_name = os.path.splitext(os.path.basename(imfile1))[0]
                if args.format_save == "png":
                    imfile = os.path.join(output_path, "flow-{}.png".format(base_name))
                    print("debug:", imfile)
                    viz(image1, flow_up, imfile)
                elif args.format_save == "pickle":
                    picfile = os.path.join(output_path,
                                           "flow-{}.binaryfile".format(base_name))
                    print("debug:", picfile)
                    with open(picfile, mode="wb") as f:
                        pickle.dump(flow_low, f)
                    # with open(picfile, mode="rb") as f:
                    #     d = pickle.load(f)
                    # print(type(d), d.size(), flow_up.size())
                else:
                    picfile = os.path.join(output_path, "flow-{}.pth".format(base_name))
                    print("debug:", picfile)
                    torch.save(flow_low, picfile)
                    # d = torch.load(picfile)
                    # print(type(d), d.size(), flow_up.size())
            if args.all:
                if args.format_save == "pickle":
                    picfile = os.path.join(output_path, "flow-all.binaryfile")
                    print("debug:", picfile)
                    with open(picfile, mode="wb") as f:
                        pickle.dump(pickle_list, f)
                elif args.format_save == "torch_save":
                    picfile = os.path.join(output_path, "flow-all.pth")
                    print("debug:", picfile)
                    torch.save(pickle_list, picfile)
            cur_time = time.time() - s_time
            total_time += cur_time
            print(i, ": ", cur_time)
        print("total: ", total_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--root',
                        type=str,
                        default=None,
                        help="root of dataset for evaluation")
    parser.add_argument('--subset', type=str, default="train", help="subset name")
    parser.add_argument('--datanum', type=int, default=None, help="# of video")
    parser.add_argument('--start', type=int, default=0, help="start number")
    parser.add_argument('--format-save',
                        default="torch_save",
                        choices=["torch_save", "pickle", "png"])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision',
                        action='store_true',
                        help='use mixed precision')
    parser.add_argument('--alternate_corr',
                        action='store_true',
                        help='use efficent correlation implementation')
    parser.add_argument('--output', default='result', help="output flow image")
    args = parser.parse_args()

    demo(args)
