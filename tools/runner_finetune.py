import numpy as np
import torch
from pointnet2_ops import pointnet2_utils
from torchvision import transforms

from datasets import data_transforms
from tools import builder
from utils import dist_utils, misc
from utils.logger import get_logger, print_log


def unwrap_logits(model_out):
    return model_out["logits"] if isinstance(model_out, dict) else model_out


def run_net(*args, **kwargs):
    raise RuntimeError(
        "Training code has been removed from the inference-only release."
    )


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log("Tester start ... ", logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(
        config.model,
        default_args={"loss": config.get("loss", None)},
    )

    builder.load_model(base_model, args.ckpts, logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


def test(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()

    test_pred = []
    test_label = []
    npoints = config.npoints

    with torch.no_grad():
        for _, _, data in test_dataloader:
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits = unwrap_logits(base_model(points))
            target = label.view(-1)
            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.0
        print_log("[TEST] acc = %.4f" % acc, logger=logger)

        gb = 1024.0 * 1024.0 * 1024.0
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        print_log("[GPU Mem] MEM = %.3f GB" % gpu_memory, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

        print_log("[TEST_VOTE]", logger=logger)
        vote_accs = []
        for repeat_idx in range(1, args.test_vote_repeat + 1):
            this_acc = test_vote(
                base_model,
                test_dataloader,
                1,
                None,
                args,
                config,
                logger=logger,
                times=args.vote_times,
            )
            this_acc = float(this_acc)
            vote_accs.append(this_acc)
            print_log(
                "[TEST_VOTE_repeat %d] acc = %.4f" % (repeat_idx, this_acc),
                logger=logger,
            )
        vote_acc_mean = float(np.mean(vote_accs))
        vote_acc_std = float(np.std(vote_accs))
        print_log(
            "[TEST_VOTE] mean acc = %.4f, std = %.4f"
            % (vote_acc_mean, vote_acc_std),
            logger=logger,
        )


def get_point_all(npoints):
    if npoints == 1024:
        return 1200
    if npoints == 2048:
        return 2400
    if npoints == 4096:
        return 4800
    if npoints == 8192:
        return 8192
    raise NotImplementedError()


def build_test_transforms(config):
    if config.dataset.train._base_.NAME == "ModelNet":
        return transforms.Compose(
            [
                data_transforms.PointcloudScaleAndTranslate(),
            ]
        )
    return transforms.Compose(
        [
            data_transforms.PointcloudRotate(),
        ]
    )


def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    test_transforms = build_test_transforms(config)
    base_model.eval()

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for _, _, data in test_dataloader:
            points_raw = data[0].cuda()
            label = data[1].cuda()
            point_all = min(get_point_all(npoints), points_raw.size(1))

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)
            local_pred = []

            for _ in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(
                    points_raw.transpose(1, 2).contiguous(),
                    fps_idx,
                ).transpose(1, 2).contiguous()

                points = test_transforms(points)

                logits = unwrap_logits(base_model(points))
                target = label.view(-1)
                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)

            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.0

        if args.distributed:
            torch.cuda.synchronize()

        gb = 1024.0 * 1024.0 * 1024.0
        gpu_memory_vote = torch.cuda.max_memory_allocated() / gb
        print_log("[Vote GPU Mem] MEM = %.3f GB" % gpu_memory_vote, logger=logger)

    if val_writer is not None:
        val_writer.add_scalar("Metric/ACC_vote", acc, epoch)

    return acc
