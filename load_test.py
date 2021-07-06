import argparse
import os
import glob
import torch
import pickle
import time

DEVICE = 'cuda'


def demo(args):
    torch_list = glob.glob(os.path.join(args.path, "*.pth"))
    pickle_list = glob.glob(os.path.join(args.path, "*.binaryfile"))

    torch_list = sorted(torch_list)
    pickle_list = sorted(pickle_list)

    print("torch start: ", len(torch_list))
    s_time = time.time()
    for imfile in torch_list:
        d = torch.load(imfile)
        if type(d) is list:
            print(imfile, type(d), len(d), d[0].size())
        else:
            print(imfile, type(d), d[0].size())
    torch_time = time.time() - s_time
    print("pickle start: ", len(pickle_list))
    s_time = time.time()
    for imfile in pickle_list:
        with open(imfile, mode="rb") as f:
            d = pickle.load(f)
        if type(d) is list:
            print(imfile, type(d), len(d), d[0].size())
        else:
            print(imfile, type(d), d[0].size())
    pickle_time = time.time() - s_time
    print("torch: ", torch_time, "pickle: ", pickle_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="dataset for evaluation")
    args = parser.parse_args()

    demo(args)
