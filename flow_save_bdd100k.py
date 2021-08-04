import argparse
import os
import os.path as osp
import torch
import pickle
import time

from core.raft import RAFT
from core.utils import frame_utils
from core.utils.utils import InputPadder
from core.utils.utils import forward_interpolate
from bdd100k_lib.video_datasets import BDD

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        bdd_dataset = BDD(root=args.root,
                          subset=args.subset,
                          data_start=args.start,
                          debug_load_num=args.datanum,
                          random_sample=args.random)
        output_path = osp.join(args.output, args.subset)
        total_time, local_id = 0, 0
        pickle_list = []

        flow_prev, sequence_prev = None, None
        # for test_id in range(len(bdd_dataset)):
        for test_id, data in enumerate(bdd_dataset):
            s_time = time.time()
            # image1, image2, (sequence, frame) = bdd_dataset[test_id]
            image1, image2, (sequence, frame) = data
            if args.debug:
                print(local_id, test_id, type(image1), image1.shape, sequence, frame)
            if sequence != sequence_prev:
                local_id = 0
                flow_prev = None
                if args.all and sequence_prev is not None:
                    output_dir = osp.join(output_path, sequence_prev, args.format_save,
                                          "all")
                    if not osp.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.format_save == "pickle":
                        picfile = osp.join(output_dir, "flow-all.binaryfile")
                        if args.debug:
                            print("debug:", picfile)
                        with open(picfile, mode="wb") as f:
                            pickle.dump(pickle_list, f)
                    elif args.format_save == "torch_save":
                        picfile = osp.join(output_dir, "flow-all.pth")
                        if args.debug:
                            print("debug:", picfile)
                        torch.save(pickle_list, picfile)
                if not args.localtime_offprint and not args.all_offprint:
                    print(sequence_prev, " total: ", total_time)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1,
                                      image2,
                                      iters=args.iters,
                                      flow_init=flow_prev,
                                      test_mode=True)
            # flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
            flow = padder.unpad(flow_low[0]).permute(1, 2, 0).cpu().numpy()

            if args.warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = osp.join(output_path, sequence)
            output_file = osp.join(output_dir, "frame%04d.flo" % (frame + 1))

            if not osp.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence
            if args.all:
                pickle_list.append(flow_low)
                cur_time = time.time() - s_time
                total_time += cur_time
                local_id += 1
                if not args.all_offprint and not args.time_offprint:
                    print(local_id, ": ", test_id, ": ", cur_time)
                continue

            base_name = ("%04d" % (frame + 1))
            output_dir = osp.join(output_dir, args.format_save)
            if not osp.exists(output_dir):
                os.makedirs(output_dir)
            if args.format_save == "pickle":
                picfile = osp.join(output_dir, "flow-{}.binaryfile".format(base_name))
                if args.debug:
                    print("debug:", picfile)
                with open(picfile, mode="wb") as f:
                    pickle.dump(flow_low, f)
                # with open(picfile, mode="rb") as f:
                #     d = pickle.load(f)
                # print(type(d), d.size(), flow_up.size())
            else:
                picfile = osp.join(output_dir, "flow-{}.pth".format(base_name))
                if args.debug:
                    print("debug:", picfile)
                torch.save(flow_low, picfile)
                # d = torch.load(picfile)
                # print(type(d), d.size(), flow_up.size())
            cur_time = time.time() - s_time
            total_time += cur_time
            local_id += 1
            if not args.all_offprint and not args.time_offprint:
                print(local_id, ": ", test_id, ": ", cur_time)
            # end for

        if args.all and sequence_prev is not None:
            output_dir = osp.join(output_path, sequence_prev, args.format_save, "all")
            if not osp.exists(output_dir):
                os.makedirs(output_dir)
            if args.format_save == "pickle":
                picfile = osp.join(output_dir, "flow-all.binaryfile")
                if args.debug:
                    print("debug:", picfile)
                with open(picfile, mode="wb") as f:
                    pickle.dump(pickle_list, f)
            elif args.format_save == "torch_save":
                picfile = osp.join(output_dir, "flow-all.pth")
                if args.debug:
                    print("debug:", picfile)
                torch.save(pickle_list, picfile)
        if not args.localtime_offprint and not args.all_offprint:
            print(sequence_prev, " total: ", total_time)
        if not args.all_offprint:
            print("total: ", total_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="result", help="output flow image")
    parser.add_argument("--root",
                        type=str,
                        default=None,
                        help="root of dataset for evaluation")
    parser.add_argument("--subset", type=str, default="train", help="subset name")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--all-offprint", action="store_true")
    parser.add_argument("--localtime-offprint", action="store_true")
    parser.add_argument("--time-offprint", action="store_true")
    parser.add_argument("--datanum", type=int, default=None, help="# of video")
    parser.add_argument("--random", action="store_true", help="random load")
    parser.add_argument("--start", type=int, default=0, help="start number")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--format-save",
                        default="torch_save",
                        choices=["torch_save", "pickle", "png"])
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument("--mixed_precision",
                        action="store_true",
                        help="use mixed precision")
    parser.add_argument("--alternate_corr",
                        action="store_true",
                        help="use efficent correlation implementation")
    parser.add_argument("--iters", type=int, default=20, help="iteration of flow")
    parser.add_argument("--warm-start", action="store_true", help="consider prev flow")
    args = parser.parse_args()

    demo(args)
