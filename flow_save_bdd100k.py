import argparse
import os
import os.path as osp
import time

import torch

from core.raft import RAFT
from core.utils.utils import InputPadder
from core.utils.utils import forward_interpolate
from bdd100k_lib.video_datasets import BDD
from bdd100k_lib.utils import save_flow, FORMAT_SAVE

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
        output_path = osp.join(args.output, args.subset, args.format_save)
        fwd_str, bwd_str = "forward", "backward"
        fwd_outdir = osp.join(output_path, fwd_str)
        # bwd_outdir = osp.join(output_path, bwd_str)
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
                    output_dir = osp.join(output_path, sequence_prev, "all")
                    if not osp.exists(output_dir):
                        os.makedirs(output_dir)
                    save_flow(pickle_list, output_dir, "all", args.format_save, None,
                              None, args.debug)
                if not args.localtime_offprint and not args.all_offprint:
                    print(sequence_prev, " total: ", total_time)
                pickle_list = []

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1,
                                      image2,
                                      iters=args.iters,
                                      flow_init=flow_prev,
                                      test_mode=True)
            # flow_bwd_low, flow_bwd_pr = model(image2,
            #                                   image1,
            #                                   iters=args.iters,
            #                                   flow_init=None,
            #                                   test_mode=True)

            if args.warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = osp.join(fwd_outdir, sequence)
            # bwd_output_dir = osp.join(fwd_outdir, sequence)

            if not osp.exists(output_dir):
                os.makedirs(output_dir)

            sequence_prev = sequence
            if args.all:
                pickle_list.append(flow_low)
                # pickle_list.append(flow_pr)
                cur_time = time.time() - s_time
                total_time += cur_time
                local_id += 1
                if not args.all_offprint and not args.time_offprint:
                    print(local_id, ": ", test_id, ": ", cur_time)
                continue

            base_name = ("%04d" % (frame + 1))
            save_flow(flow_low, output_dir, base_name, args.format_save, image1, padder,
                      args.debug)
            # save_flow(flow_pr, output_dir, base_name, args.format_save, image1,
            #         padder, args.debug)
            cur_time = time.time() - s_time
            total_time += cur_time
            local_id += 1
            if not args.all_offprint and not args.time_offprint:
                print(local_id, ": ", test_id, ": ", cur_time)
            # end for

        if args.all and sequence_prev is not None:
            output_dir = osp.join(output_path, sequence_prev, "all")
            if not osp.exists(output_dir):
                os.makedirs(output_dir)
            save_flow(pickle_list, output_dir, "all", args.format_save, None, None,
                      args.debug)
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
    parser.add_argument("--format-save", default="torch_save", choices=FORMAT_SAVE)
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
