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
    max_tmp = None
    for imfile in torch_list:
        d = torch.load(imfile, map_location="cpu")
        if type(d) is list:
            print(imfile, type(d), len(d), d[0].size())
        else:
            local_max = torch.max(d.reshape(-1)).item()
            if max_tmp is None:
                max_tmp = local_max
            else:
                max_tmp = max(local_max, max_tmp)
            print(imfile, type(d), d[0].size(), local_max)
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
    print("torch: ", torch_time, "pickle: ", pickle_time, "max:", max_tmp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="dataset for evaluation")
    args = parser.parse_args()

    demo(args)
