import os
import time

import torch

from tools.runner_finetune import test_net as classification_test_net
from tools.runner_partseg import test_net as partseg_test_net
from utils import dist_utils, misc, parser
from utils.config import get_config, log_args_to_file, log_config_to_file
from utils.logger import get_root_logger


INFERENCE_ONLY_MESSAGE = (
    "This public release only supports inference/evaluation. "
    "Use --test with a checkpoint to run classification or part-segmentation."
)


def main():
    args = parser.get_args()
    if not args.test:
        raise RuntimeError(INFERENCE_ONLY_MESSAGE)

    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    if args.launcher == "none":
        args.distributed = False
        world_size = 1
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(args.experiment_path, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    config = get_config(args, logger=logger)

    if config.dataset.get("test") is None:
        raise RuntimeError("The selected config must define dataset.test for inference.")

    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.test.others.bs = config.total_bs // world_size
    else:
        config.dataset.test.others.bs = config.total_bs

    log_args_to_file(args, "args", logger=logger)
    log_config_to_file(config, "config", logger=logger)
    logger.info(f"Distributed inference: {args.distributed}")

    if args.seed is not None:
        logger.info(
            f"Set random seed to {args.seed}, deterministic: {args.deterministic}"
        )
        misc.set_random_seed(
            args.seed + args.local_rank,
            deterministic=args.deterministic,
        )

    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    if args.part_seg_model:
        partseg_test_net(args, config)
    else:
        classification_test_net(args, config)


if __name__ == "__main__":
    main()
