import argparse
import os
import os.path as osp
import time

import torch

from core.raft import RAFT
from bdd100k_lib.video_datasets import BDDVideo as BDD
from bdd100k_lib.utils import final_gen_flow, preprocessing_imgs
from bdd100k_lib.utils import gen_flow_correspondence
from bdd100k_lib.utils import save_flow, FORMAT_SAVE


def demo(args):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(device)
    model.eval()

    with torch.no_grad():
        bdd_dataset = BDD(root=args.root,
                          subset=args.subset,
                          data_start=args.start,
                          num_frames=args.n_frames,
                          normalize=args.normalize,
                          debug_load_num=args.datanum,
                          debug_mode=args.debug)
        output_path = osp.join(args.output, args.subset, args.format_save)
        total_time, num_frames = 0, args.n_frames

        for video_id, data in enumerate(bdd_dataset):
            sequences = data
            sequence = bdd_dataset.sequence_names[video_id]
            print(video_id, sequence, type(sequences), len(sequences))
            all_num_frames = len(sequences)
            e_idx = all_num_frames - num_frames + 1
            for s_frame in range(e_idx):
                s_time = time.perf_counter()
                data = bdd_dataset.get_imgs(video_id, s_frame, num_frames)
                images = data
                if args.debug:
                    images, info = data
                    print(video_id, s_frame, bdd_dataset.get_info(info))
                images = [d.to(device, non_blocking=True) for d in images]
                images, padder = preprocessing_imgs(images)

                if args.no_mask:
                    _, _, flow_init, _ = gen_flow_correspondence(
                        model, images, args.iters, args.alpha_1, args.alpha_2)
                    print(video_id, s_frame, images[0].size(), flow_init.size())
                else:
                    flow, flow_init, flow_init_mask = final_gen_flow(
                        model, images, args.iters, args.alpha_1, args.alpha_2)
                    print(video_id, s_frame, images[0].size(), flow.size(),
                          flow_init.size())

                output_dir = osp.join(output_path, sequence)

                if not osp.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                base_name = ("%04d" % (s_frame + 1))

                save_flow(flow_init, output_dir, f"onlycat-{base_name}",
                          args.format_save, images[0], padder, args.debug)
                if not args.no_mask:
                    save_flow(flow, output_dir, f"cat-mask-{base_name}",
                              args.format_save, images[0], padder, args.debug)
                    save_flow(flow_init_mask, output_dir, f"cat-nomask-{base_name}",
                              args.format_save, images[0], padder, args.debug)
                cur_time = time.perf_counter() - s_time
                total_time += cur_time
                if not args.all_offprint and not args.time_offprint:
                    print(s_frame, ": ", video_id, ": ", cur_time)
                # end for

        if not args.all_offprint:
            print("total: ", total_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="result", help="output flow image")
    parser.add_argument("--root",
                        type=str,
                        default="/path/to/bdd100k",
                        help="root of dataset for evaluation")
    parser.add_argument("--subset", type=str, default="train", help="subset name")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--all-offprint", action="store_true")
    parser.add_argument("--time-offprint", action="store_true")
    parser.add_argument("--datanum", type=int, default=None, help="# of video")
    parser.add_argument("--start", type=int, default=0, help="start number")
    parser.add_argument("--format-save", default="torch_save", choices=FORMAT_SAVE)
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument("--mixed_precision",
                        action="store_true",
                        help="use mixed precision")
    parser.add_argument("--alternate_corr",
                        action="store_true",
                        help="use efficent correlation implementation")
    parser.add_argument("--iters", type=int, default=12, help="iteration of flow")
    parser.add_argument("--alpha_1",
                        type=float,
                        default=0.01,
                        help="cycle consistency coefficient 1")
    parser.add_argument("--alpha_2",
                        type=float,
                        default=0.5,
                        help="cycle consistency coefficient 2")
    parser.add_argument("--no_mask", action="store_true", help="not mask process")
    parser.add_argument("--warm-start", action="store_true", help="consider prev flow")
    parser.add_argument("--normalize", action="store_true", help="normalize data")
    parser.add_argument("--n_frames",
                        type=int,
                        default=15,
                        help="# of frames per video")
    args = parser.parse_args()

    demo(args)
