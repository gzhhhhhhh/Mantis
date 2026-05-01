import argparse
import os
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="cfgs/finetune_scan_hardest_mantis.yaml",
        help="yaml config file",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--sync_bn",
        action="store_true",
        default=False,
        help="whether to use sync bn",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="inference",
        help="experiment name",
    )
    parser.add_argument(
        "--ckpts",
        type=str,
        default=None,
        help="checkpoint path used for inference",
    )
    parser.add_argument(
        "--vote_times",
        type=int,
        default=10,
        help="number of augmentations averaged in each voting pass",
    )
    parser.add_argument(
        "--test_vote_repeat",
        type=int,
        default=1,
        help="number of repeated voting passes during --test",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="run inference/evaluation mode",
    )
    parser.add_argument(
        "--part_seg_model",
        action="store_true",
        default=False,
        help="run part segmentation inference on ShapeNetPart",
    )

    args = parser.parse_args()

    if not args.test:
        raise ValueError("The inference-only release requires --test.")

    if args.ckpts is None:
        raise ValueError("--ckpts is required for inference.")

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    args.exp_name = "test_" + args.exp_name
    args.experiment_path = os.path.join(
        "./experiments",
        Path(args.config).stem,
        Path(args.config).parent.stem,
        args.exp_name,
    )
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args


def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print("Create experiment path successfully at %s" % args.experiment_path)
