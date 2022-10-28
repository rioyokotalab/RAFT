import argparse
import os
import glob
import time
import datetime

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from core.raft import RAFT
from core.utils import frame_utils
# from core.utils.utils import InputPadder

# DEVICE = 'cuda'


def change_second_to_humantime(sec, is_split=True):
    td = datetime.timedelta(seconds=sec)
    out_str = str(td)
    days = td.days
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    if is_split:
        out_str = f"{days}day {h}h {m}m {s}s"
    return out_str


def split_range(nlist):
    mpirank = dist.get_rank()
    mpisize = dist.get_world_size()
    begin = 0
    # Increment of splitting
    increment = nlist // mpisize
    # Remainder of splitting
    remainder = nlist % mpisize
    # Increment the begin counter
    begin += mpirank * increment + min(mpirank, remainder)
    end = begin + increment  # Increment the end counter
    if (remainder > mpirank):
        end += 1  # Adjust the end counter for remainder
    return begin, end


def dist_setup():
    master_addr = os.getenv("MASTER_ADDR", default="localhost")
    master_port = os.getenv("MASTER_PORT", default="8888")
    method = "tcp://{}:{}".format(master_addr, master_port)
    rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "-1"))
    local_size = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", "-2"))
    node_rank = int(os.getenv("OMPI_COMM_WORLD_NODE_RANK", "-3"))
    host_port_str = f"host: {master_addr}, port: {master_port}"
    print(
        "rank: {}, world_size: {}, local_rank: {}, local_size: {}, node_rank: {}, {}"
        .format(rank, world_size, local_rank, local_size, node_rank, host_port_str))
    dist.init_process_group("nccl", init_method=method, rank=rank,
                            world_size=world_size)
    print("Rank: {}, Size: {}, Host: {} Port: {}".format(dist.get_rank(),
                                                         dist.get_world_size(),
                                                         master_addr, master_port))
    return local_rank


def print_rank(*args, log_out_root=None):
    rank = dist.get_rank()
    if rank == 0 or log_out_root is None:
        print(f"rank:{rank}", *args)
    if log_out_root is not None:
        log_filename = os.path.join(log_out_root, f"log_{rank}.txt")
        with open(log_filename, "a") as f:
            print(f"rank:{rank}", *args, file=f)


@torch.no_grad()
def calc_optical_flow(imgs, flow_model, up=False):
    i = 1 if up else 0
    flow_model.eval()
    flow_fwds = torch.stack([
        flow_model(img0, img1, upsample=False, test_mode=True)[i]
        for img0, img1 in zip(imgs[:-1], imgs[1:])
    ])
    flow_bwds = torch.stack([
        flow_model(img0, img1, upsample=False, test_mode=True)[i]
        for img0, img1 in zip(imgs[1:][::-1], imgs[:-1][::-1])
    ])
    flow_fwds = flow_fwds.cuda()
    flow_bwds = flow_bwds.cuda()
    return flow_fwds, flow_bwds


def load_image(imfile):
    # from torchvision import transforms
    # to_tensor = transforms.ToTensor()
    # img = to_tensor(Image.open(imfile)) * 255
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    # return img[None].to(DEVICE)
    return img[None].cuda()


def load_images(imfiles):
    return [load_image(imfile) for imfile in imfiles]


def adjust_dim(tar, out_dim, return_none=True):
    in_dim = tar.ndim
    diff_dim = out_dim - in_dim
    out_tar = tar.clone()
    if diff_dim > 0:
        for _ in range(diff_dim):
            out_tar = out_tar.unsqueeze(0)
    elif diff_dim < 0:
        for _ in range(abs(diff_dim)):
            if return_none and out_tar.shape[0] != 1:
                return None
            # if out_tar.shape[0] != 1, raise error
            out_tar = out_tar.squeeze(0)
    return out_tar


def save_flow(flo, fname):
    base_name = os.path.basename(fname)
    ext = os.path.splitext(base_name)[-1]
    if ext in [".flo", ".png", ".jpg"]:
        flo_np = flo.permute(1, 2, 0).cpu().numpy()
    if ext == ".pth":
        torch.save(flo, fname)
    elif ext == ".flo":
        frame_utils.writeFlow(fname, flo_np)
    elif ext == ".png" or ext == ".jpg":
        frame_utils.writeFlowKITTI(fname, flo_np)
    else:
        raise NotImplementedError(f"{ext} is not supported!!")


def save_flows(flos, fnames):
    is_seperate = isinstance(fnames, (tuple, list))
    is_str = isinstance(fnames, str)
    assert is_seperate or is_str
    if is_seperate:
        base_name = os.path.basename(fnames[0])
    else:
        base_name = os.path.basename(fnames)
    ext = os.path.splitext(base_name)[-1]
    if is_seperate or ext == ".flo" or ext == ".png" or ext == ".jpg":
        for flo, fname in zip(flos, fnames):
            l_flo = adjust_dim(flo, 3)
            if l_flo is None:
                raise Exception(f"save error, flo.shape:{flo.shape}, {fname}")
            save_flow(l_flo, fname)
    elif ext == ".pth":
        save_flow(flos, fnames)
    else:
        raise NotImplementedError(f"{ext} and {fnames} is not supported!!")


def save_imfiles_optical_flow(model, video_name, imfiles, out_fwd_path, out_bwd_path, args):
    up = args.up
    ext = args.save_type
    is_split_file = args.split_file
    imbasefiles = [os.path.basename(imfile) for imfile in imfiles]
    l_images = load_images(imfiles)
    # image1 = l_images[0]
    # padder = InputPadder(image1.shape, mode=mode)
    # l_images = padder.pad(*l_images)
    print_rank(l_images[0].device, log_out_root=args.out_path)

    l_s_time = time.perf_counter()
    flow_fwd, flow_bwd = calc_optical_flow(l_images, model, up=up)
    l_m_time = time.perf_counter()
    l_exec_m_time = l_m_time - l_s_time

    num, nb, c, h, w = flow_fwd.shape
    if is_split_file:
        fnames_fwd, fnames_bwd = [], []
        for i in range(num):
            imbase = os.path.splitext(imbasefiles[i])[0]
            filename = f"{imbase}{ext}"
            fnames_fwd.append(os.path.join(out_fwd_path, filename))
            fnames_bwd.append(os.path.join(out_bwd_path, filename))
    else:
        fnames_fwd = out_fwd_path.rstrip("/") + ext
        fnames_bwd = out_bwd_path.rstrip("/") + ext
        err_msg = "not supprted batch calc optical flow, "
        err_msg += f"{flow_fwd.shape}, {video_name}"
        assert nb == 1, err_msg
        flow_fwd = flow_fwd.reshape(num, c, h, w)
        flow_bwd = flow_bwd.reshape(num, c, h, w)

    save_flows(flow_fwd, fnames_fwd)
    save_flows(flow_bwd, fnames_bwd)
    l_e_time = time.perf_counter()
    l_exec_time = l_e_time - l_m_time
    prin_str = f"{video_name} {imbasefiles}"
    print_rank(prin_str, "calc optical flow time (s):", l_exec_m_time, log_out_root=args.out_path)
    print_rank(video_name, "save optical flow time (s):", l_exec_time, log_out_root=args.out_path)


@torch.no_grad()
def demo_one_video(model, data_root, video_name, out_root_fwd, out_root_bwd, args):
    is_split_file = args.split_file
    path = os.path.join(data_root, video_name)
    out_fwd_path = os.path.join(out_root_fwd, video_name)
    out_bwd_path = os.path.join(out_root_bwd, video_name)
    if is_split_file:
        os.makedirs(out_fwd_path, exist_ok=True)
        os.makedirs(out_bwd_path, exist_ok=True)
    images = glob.glob(os.path.join(path, '*.png')) + \
        glob.glob(os.path.join(path, '*.jpg'))

    images = sorted(images)
    len_img = len(images)
    print_rank(f"len_img: {len_img}", video_name, path, log_out_root=args.out_path)
    s_time = time.perf_counter()
    if is_split_file:
        for i, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            imfiles = [imfile1, imfile2]
            save_imfiles_optical_flow(model, video_name, imfiles, out_fwd_path, out_bwd_path, args)
    else:
        imfiles = images[:]
        save_imfiles_optical_flow(model, video_name, imfiles, out_fwd_path, out_bwd_path, args)
    e_time = time.perf_counter()
    exec_time = e_time - s_time
    h_exec_time = change_second_to_humantime(exec_time)
    print_str = f"{video_name}, total calc, save optical flow time (s): {exec_time}\n"
    print_str += f"total calc, save optical flow time: {h_exec_time}"
    print_rank(print_str, log_out_root=args.out_path)


def demo(args):
    # support all flow save type is only pth type
    if not args.split_file:
        assert args.save_type == ".pth"

    # setup gpu device
    local_rank = dist_setup()
    torch.cuda.set_device(local_rank)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    # model.to(DEVICE)
    model = model.cuda()
    model.eval()

    # input data path
    data_root = args.path

    # output data path
    out_root = args.out_root
    data_name = args.data_name
    if out_root is None or out_root == "":
        out_root = os.path.dirname(args.path)
    if data_name is None or data_name == "":
        data_name = "flow"
    save_type = args.save_type[1:]
    out_root = os.path.join(out_root, data_name, save_type)
    os.makedirs(out_root, exist_ok=True)
    # setting for rank log dir
    args.out_path = out_root

    # use subset
    subset = args.subset
    if subset is not None and subset != "":
        # input data path
        data_root = os.path.join(data_root, subset)
        # output data path
        out_root = os.path.join(out_root, subset)

    # output data path
    out_root_fwd = os.path.join(out_root, "forward")
    out_root_bwd = os.path.join(out_root, "backward")
    os.makedirs(out_root_fwd, exist_ok=True)
    os.makedirs(out_root_bwd, exist_ok=True)

    # check exist dir for input data root
    if not os.path.isdir(args.path):
        raise FileNotFoundError(f"{args.path} is not exist dir!!")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"{data_root} is not exist dir!!")

    # get video list
    video_names = [os.path.basename(data_root)]
    images = glob.glob(os.path.join(data_root, '*.png')) + \
        glob.glob(os.path.join(data_root, '*.jpg'))
    if len(images) <= 0:
        video_names = sorted(os.listdir(data_root))
    else:
        data_root = os.path.dirname(data_root)

    # adjust video num specified by args.start_idx, args.make_num
    start_idx = args.start_idx
    make_num = len(video_names) if args.make_num is None else args.make_num
    end_idx = make_num + start_idx
    video_names = video_names[start_idx:end_idx]

    # setting for rank log dir
    args.out_path = os.path.join(args.out_path, f"{start_idx}_{end_idx}")
    os.makedirs(args.out_path, exist_ok=True)

    # assign video list per gpu(process)
    num_video = len(video_names)
    s_idx, e_idx = split_range(num_video)
    gpu = dist.get_world_size()
    print_rank("num video:", num_video, log_out_root=args.out_path)
    l_video_names = video_names[s_idx:e_idx]
    s_time = time.perf_counter()
    print_rank("process video num", len(l_video_names), "gpu:", gpu, log_out_root=args.out_path)
    for video_name in l_video_names:
        demo_one_video(model, data_root, video_name, out_root_fwd, out_root_bwd, args)
    e_time = time.perf_counter()
    exec_time = e_time - s_time
    h_exec_time = change_second_to_humantime(exec_time)
    print_rank("all calc, save optical flow time (s):", exec_time, log_out_root=args.out_path)
    print_rank("all calc, save optical flow time:", h_exec_time, log_out_root=args.out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset root path",
                        default="/datasets/bdd100k_root/bdd100k/images")
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision',
                        action='store_true',
                        help='use mixed precision')
    parser.add_argument('--alternate_corr',
                        action='store_true',
                        help='use efficent correlation implementation')
    parser.add_argument('--out_root', type=str, default=None, help="out root path")
    parser.add_argument('--data_name', type=str, default="flow")
    parser.add_argument('--save_type', type=str, default=".pth",
                        choices=[".pth", ".flo", ".png", ".jpg"])
    parser.add_argument('--up', action='store_true', help='save flow up')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--make_num', type=int, default=None)
    parser.add_argument('--split_file', action='store_true',
                        help='if True, save 1file/1flow, else 1file/all flow on video')
    args = parser.parse_args()

    demo(args)
    # torch.backend.cudnn.benchmark = False
