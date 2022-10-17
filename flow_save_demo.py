import argparse
import os
import glob
import time
import datetime

# import cv2
import numpy as np
import torch
import torch.distributed as dist
# import torch.nn.functional as F
from PIL import Image

from core.raft import RAFT
# from core.utils import flow_viz
from core.utils import frame_utils
# from core.utils.utils import InputPadder
from core.utils.utils import upflow8

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


def save_flow(flo, fname, up=False):
    base_name = os.path.basename(fname)
    ext = os.path.splitext(base_name)[-1]
    flo_np = flo.permute(1, 2, 0).cpu().numpy()
    read_flo, val = None, None
    if ext == ".pth":
        torch.save(flo, fname)
        read_flo = torch.load(fname, map_location=torch.device('cpu'))
    elif ext == ".flo":
        frame_utils.writeFlow(fname, flo_np)
        read_flo = frame_utils.readFlow(fname)
        read_flo = torch.from_numpy(read_flo).permute(2, 0, 1)
    elif ext == ".png" or ext == ".jpg":
        # flo = flow_viz.flow_to_image(flo_np)
        frame_utils.writeFlowKITTI(fname, flo_np)
        read_flo, val = frame_utils.readFlowKITTI(fname)
        read_flo = torch.from_numpy(read_flo).permute(2, 0, 1)
        # flo_pil = Image.fromarray(flo)
        # dir_name = os.path.dirname(fname)
        # imbase = os.path.basename(fname)
        # pil_root = os.path.join(dir_name, "pil")
        # cv2_root = os.path.join(dir_name, "cv2")
        # os.makedirs(pil_root, exist_ok=True)
        # os.makedirs(cv2_root, exist_ok=True)
        # flo_pil.save(os.path.join(pil_root, imbase))
        # cv2.imwrite(os.path.join(cv2_root, imbase), flo[:, :, [2, 1, 0]])
    else:
        raise NotImplementedError(f"{ext} is not supported!!")

    if up:
        in_dim = read_flo.ndim
        read_flo = adjust_dim(read_flo, 4)
        read_flo = upflow8(read_flo)
        read_flo = adjust_dim(read_flo, in_dim)
    read_flo = read_flo.cuda()
    return read_flo, val


def save_flows(flos, fnames, up=False):
    is_seperate = isinstance(fnames, (tuple, list))
    base_name = os.path.basename(fnames[0])
    ext = os.path.splitext(base_name)[-1]
    read_flo, val = flos.clone(), None
    if is_seperate or ext == ".flo" or ext == ".png" or ext == ".jpg":
        for flo, fname in zip(flos, fnames):
            l_flo = adjust_dim(flo, 3)
            if l_flo is None:
                raise Exception(f"save error, flo.shape:{flo.shape}, {fname}")
            read_flo, val = save_flow(l_flo, fname, up)
    elif ext == ".pth":
        read_flo, val = save_flow(flos, fnames, up)

    return read_flo, val


def prepare_out(path, args, name=""):
    out_root = args.out_path
    base_name = os.path.basename(path)
    dt_str = args.date_str
    out_path = os.path.join(out_root, args.save_type[1:])
    if name != "":
        out_path = os.path.join(out_path, name)
    out_path = os.path.join(out_path, base_name, dt_str)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def save_imfiles_optical_flow(model, video_name, imfiles, out_fwd_path, out_bwd_path, args):
    # debug
    imfiles, load_flows = imfiles
    out_fwd_path, debug_out_root_fwd = out_fwd_path
    out_bwd_path, debug_out_root_bwd = out_bwd_path
    debug_out_root_orig_fwd, debug_out_root_read_fwd = debug_out_root_fwd
    debug_out_root_orig_bwd, debug_out_root_read_bwd = debug_out_root_bwd

    up = args.up
    ext = args.save_type
    imbasefiles = [os.path.basename(imfile) for imfile in imfiles]
    l_images = load_images(imfiles)
    load_flow = [torch.load(load_flow, map_location=torch.device('cpu')) for load_flow in load_flows]
    load_flow = [load_flow.cuda() for load_flow in load_flow]
    # image1 = l_images[0]
    # padder = InputPadder(image1.shape, mode=mode)
    # l_images = padder.pad(*l_images)

    l_s_time = time.perf_counter()
    flow_up, flow_up_bwd = calc_optical_flow(l_images, model, up=True)
    flow_low, flow_low_bwd = calc_optical_flow(l_images, model, up=False)
    save_flow_fwd = flow_low if up else flow_up
    save_flow_bwd = flow_low_bwd if up else flow_up_bwd
    l_m_time = time.perf_counter()
    l_exec_m_time = l_m_time - l_s_time
    print_rank(video_name, "calc optical flow time (s):", l_exec_m_time)
    # print_rank(imbasefiles, flow_up.shape, flow_up, flow_up_bwd)
    # print_rank(imbasefiles, flow_low.shape, flow_low, flow_low_bwd)
    # print_rank(imbasefiles, save_flow_fwd.shape, save_flow_fwd, save_flow_bwd)

    check_load_flow_low = load_flow[0] == flow_low[0]
    print_rank(load_flow[0].shape)
    flow_up_low_fwd = upflow8(flow_low[0])
    flow_up_low_bwd = upflow8(flow_low_bwd[0])
    load_flow_up_low = upflow8(load_flow[0])
    check_flow = flow_up_low_fwd == flow_up[0]
    check_flow_bwd = flow_up_low_bwd == flow_up_bwd[0]
    check_load_flow = load_flow_up_low == flow_up[0]
    print_rank(torch.min(check_flow), check_flow.float().mean(), flow_up_low_fwd.shape, flow_up[0].shape)
    print_rank(torch.min(check_flow_bwd), check_flow_bwd.float().mean(), flow_up_low_bwd.shape, flow_up_bwd[0].shape)
    print_rank(torch.min(check_load_flow), check_load_flow.float().mean(), load_flow_up_low.shape, flow_up[0].shape)
    print_rank(torch.min(check_load_flow_low), check_load_flow_low.float().mean(), load_flow[0].shape, flow_low[0].shape)

    num, fnames_fwd, fnames_bwd = flow_up.shape[0], [], []
    debug_orig_fwd, debug_read_fwd = [], []
    debug_orig_bwd, debug_read_bwd = [], []
    for i in range(num):
        imbase = os.path.splitext(imbasefiles[i])[0]
        filename = f"{imbase}{ext}"
        filename_log = f"{imbase}.txt"
        fnames_fwd.append(os.path.join(out_fwd_path, filename))
        fnames_bwd.append(os.path.join(out_bwd_path, filename))
        debug_orig_fwd.append(os.path.join(debug_out_root_orig_fwd, filename_log))
        debug_orig_bwd.append(os.path.join(debug_out_root_orig_bwd, filename_log))
        debug_read_fwd.append(os.path.join(debug_out_root_read_fwd, filename_log))
        debug_read_bwd.append(os.path.join(debug_out_root_read_bwd, filename_log))

    read_flo, _ = save_flows(save_flow_fwd, fnames_fwd, up)
    read_flo_bwd, _ = save_flows(save_flow_bwd, fnames_bwd, up)
    print_rank(read_flo.shape, flow_up.shape, flow_up[0][0].shape)
    check_mask_fwd = read_flo == flow_up[0][0]
    check_mask_bwd = read_flo_bwd == flow_up_bwd[0][0]
    diff_fwd = read_flo - flow_up[0][0]
    diff_bwd = read_flo - flow_up[0][0]
    print_rank(torch.min(check_mask_fwd), check_mask_fwd.float().mean())
    print_rank(torch.min(check_mask_bwd), check_mask_bwd.float().mean())
    print_rank(diff_fwd.reshape(-1), torch.abs(diff_fwd).sum(), torch.abs(diff_bwd).sum())
    with open(debug_orig_fwd[0], "w") as f:
        print(flow_up[0][0].shape, file=f)
        tmp_list = flow_up[0][0].reshape(-1).tolist()
        for t in tmp_list:
            print(t, file=f)
    with open(debug_orig_bwd[0], "w") as f:
        print(flow_up_bwd[0][0].shape, file=f)
        tmp_list = flow_up_bwd[0][0].reshape(-1).tolist()
        for t in tmp_list:
            print(t, file=f)
    with open(debug_read_fwd[0], "w") as f:
        print(read_flo.shape, file=f)
        tmp_list = read_flo.reshape(-1).tolist()
        for t in tmp_list:
            print(t, file=f)
    with open(debug_read_bwd[0], "w") as f:
        print(read_flo_bwd.shape, file=f)
        tmp_list = read_flo_bwd.reshape(-1).tolist()
        for t in tmp_list:
            print(t, file=f)


@torch.no_grad()
def demo_one_video(model, data_root, video_name, args):
    path = os.path.join(data_root, video_name)
    # flow_root = "./flow/train/torch_save/forward"
    flow_path = os.path.join(flow_root, video_name)
    out_fwd_path = prepare_out(path, args, "forward")
    out_bwd_path = prepare_out(path, args, "backward")
    debug_out_root_orig_fwd = prepare_out(path, args, "forward/orig")
    debug_out_root_orig_bwd = prepare_out(path, args, "backward/orig")
    debug_out_root_read_fwd = prepare_out(path, args, "forward/read")
    debug_out_root_read_bwd = prepare_out(path, args, "backward/read")
    images = glob.glob(os.path.join(path, '*.png')) + \
        glob.glob(os.path.join(path, '*.jpg'))
    load_flows = glob.glob(os.path.join(flow_path, '*.pth'))

    load_flows = sorted(load_flows)
    images = sorted(images)
    len_img = len(images)
    print_rank(f"len_img: {len_img}", video_name)
    s_time = time.perf_counter()
    for i, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
        imfiles = [[imfile1, imfile2], [load_flows[i]]]
        tmp_out_fwd_path = [out_fwd_path, [debug_out_root_orig_fwd, debug_out_root_read_fwd]]
        tmp_out_bwd_path = [out_bwd_path, [debug_out_root_orig_bwd, debug_out_root_read_bwd]]
        save_imfiles_optical_flow(model, video_name, imfiles, tmp_out_fwd_path, tmp_out_bwd_path, args)
    # imfiles = [images[:], load_flows[:]]
    # out_fwd_path = [out_fwd_path, [debug_out_root_orig_fwd, debug_out_root_read_fwd]]
    # out_bwd_path = [out_bwd_path, [debug_out_root_orig_bwd, debug_out_root_read_bwd]]
    # save_imfiles_optical_flow(model, video_name, imfiles, out_fwd_path, out_bwd_path, args)
    e_time = time.perf_counter()
    exec_time = e_time - s_time
    h_exec_time = change_second_to_humantime(exec_time)
    print_str = f"{video_name}, total calc, save optical flow time (s): {exec_time}\n"
    print_str += f"total calc, save optical flow time: {h_exec_time}"
    print_rank(print_str)


def demo(args):
    # setup gpu device
    local_rank = dist_setup()
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    print(rank, local_rank)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    # model.to(DEVICE)
    model = model.cuda()
    model.eval()

    # input data path
    data_root = args.path

    # check exist dir for input data root
    if not os.path.isdir(args.path):
        raise FileNotFoundError(f"{args.path} is not exist dir!!")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"{data_root} is not exist dir!!")

    video_names = [os.path.basename(data_root)]
    images = glob.glob(os.path.join(data_root, '*.png')) + \
        glob.glob(os.path.join(data_root, '*.jpg'))
    if len(images) <= 0:
        video_names = sorted(os.listdir(data_root))
    else:
        data_root = os.path.dirname(data_root)
    num_video = len(video_names)
    s_idx, e_idx = split_range(num_video)
    gpu = dist.get_world_size()
    print_rank("num video:", len(video_names))
    l_video_names = video_names[s_idx:e_idx]
    s_time = time.perf_counter()
    print_rank("process video num", len(l_video_names), "gpu:", gpu)
    for video_name in video_names:
        demo_one_video(model, data_root, video_name, args)
    e_time = time.perf_counter()
    exec_time = e_time - s_time
    h_exec_time = change_second_to_humantime(exec_time)
    print_rank("all calc, save optical flow time (s):", exec_time)
    print_rank("all calc, save optical flow time:", h_exec_time)


if __name__ == '__main__':
    dt_now = datetime.datetime.now()
    dt_str = dt_now.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset root path",
                        default="/datasets/bdd100k_root/bdd100k/images")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision',
                        action='store_true',
                        help='use mixed precision')
    parser.add_argument('--alternate_corr',
                        action='store_true',
                        help='use efficent correlation implementation')
    parser.add_argument('--out_path', help="out path")
    parser.add_argument('--date_str', default=dt_str)
    parser.add_argument('--save_type', type=str, default=".pth",
                        choices=[".pth", ".flo", ".png", ".jpg"])
    parser.add_argument('--up', action='store_true', help='save flow up')
    args = parser.parse_args()

    demo(args)
