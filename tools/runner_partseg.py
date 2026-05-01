import torch
import torch.distributed as dist
import torch.nn as nn

from tools import builder
from tools.runner_finetune import unwrap_logits
from utils.logger import get_logger, print_log


class PartSegMetric:
    def __init__(self, metrics=None):
        if isinstance(metrics, PartSegMetric):
            self.accuracy = metrics.accuracy
            self.class_avg_accuracy = metrics.class_avg_accuracy
            self.class_avg_iou = metrics.class_avg_iou
            self.instance_avg_iou = metrics.instance_avg_iou
        elif isinstance(metrics, dict):
            self.accuracy = float(metrics.get("accuracy", 0.0))
            self.class_avg_accuracy = float(metrics.get("class_avg_accuracy", 0.0))
            self.class_avg_iou = float(metrics.get("class_avg_iou", 0.0))
            self.instance_avg_iou = float(metrics.get("instance_avg_iou", 0.0))
        else:
            self.accuracy = 0.0
            self.class_avg_accuracy = 0.0
            self.class_avg_iou = 0.0
            self.instance_avg_iou = 0.0


def run_net(*args, **kwargs):
    raise RuntimeError(
        "Training code has been removed from the inference-only release."
    )


def reduce_sum_tensor(tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def predict_valid_parts(logits, class_labels, dataset):
    batch_size, num_points, _ = logits.shape
    pred = torch.empty(batch_size, num_points, dtype=torch.long, device=logits.device)
    class_labels = class_labels.view(-1).long()
    for index in range(batch_size):
        category_name = dataset.class_idx_to_cat[int(class_labels[index].item())]
        valid_parts = dataset.seg_classes[category_name]
        valid_part_tensor = torch.tensor(
            valid_parts,
            device=logits.device,
            dtype=torch.long,
        )
        restricted_logits = logits[index, :, valid_parts]
        pred[index] = restricted_logits.argmax(dim=-1) + valid_part_tensor[0]
    return pred


def calculate_shape_iou(pred, target, class_labels, dataset):
    batch_size = pred.shape[0]
    category_iou_sum = torch.zeros(dataset.num_classes, device=pred.device)
    category_count = torch.zeros(dataset.num_classes, device=pred.device)
    instance_iou_sum = torch.zeros(1, device=pred.device)
    instance_count = torch.zeros(1, device=pred.device)

    for index in range(batch_size):
        class_index = int(class_labels[index].item())
        category_name = dataset.class_idx_to_cat[class_index]
        valid_parts = dataset.seg_classes[category_name]
        part_ious = []
        for part_id in valid_parts:
            pred_mask = pred[index] == part_id
            target_mask = target[index] == part_id
            union = torch.sum(pred_mask | target_mask).float()
            if union.item() == 0:
                part_ious.append(torch.tensor(1.0, device=pred.device))
            else:
                intersection = torch.sum(pred_mask & target_mask).float()
                part_ious.append(intersection / union)

        shape_iou = torch.stack(part_ious).mean()
        category_iou_sum[class_index] += shape_iou
        category_count[class_index] += 1
        instance_iou_sum += shape_iou
        instance_count += 1

    return category_iou_sum, category_count, instance_iou_sum, instance_count


@torch.no_grad()
def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=None):
    base_model.eval()
    dataset = test_dataloader.dataset
    device = (
        torch.device("cuda", args.local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    total_correct = torch.zeros(1, device=device)
    total_seen = torch.zeros(1, device=device)
    total_seen_class = torch.zeros(dataset.num_parts, device=device)
    total_correct_class = torch.zeros(dataset.num_parts, device=device)
    category_iou_sum = torch.zeros(dataset.num_classes, device=device)
    category_count = torch.zeros(dataset.num_classes, device=device)
    instance_iou_sum = torch.zeros(1, device=device)
    instance_count = torch.zeros(1, device=device)

    for _, _, data in test_dataloader:
        points = data[0].cuda(non_blocking=True)
        class_labels = data[1].cuda(non_blocking=True)
        seg_labels = data[2].cuda(non_blocking=True)

        logits = unwrap_logits(base_model(points, class_labels))
        pred = predict_valid_parts(logits, class_labels, dataset)

        total_correct += torch.sum(pred == seg_labels).float()
        total_seen += float(seg_labels.numel())

        for part_id in range(dataset.num_parts):
            target_mask = seg_labels == part_id
            total_seen_class[part_id] += torch.sum(target_mask).float()
            total_correct_class[part_id] += torch.sum((pred == part_id) & target_mask).float()

        (
            batch_category_iou_sum,
            batch_category_count,
            batch_instance_iou_sum,
            batch_instance_count,
        ) = calculate_shape_iou(
            pred,
            seg_labels,
            class_labels,
            dataset,
        )
        category_iou_sum += batch_category_iou_sum
        category_count += batch_category_count
        instance_iou_sum += batch_instance_iou_sum
        instance_count += batch_instance_count

    reduce_sum_tensor(total_correct)
    reduce_sum_tensor(total_seen)
    reduce_sum_tensor(total_seen_class)
    reduce_sum_tensor(total_correct_class)
    reduce_sum_tensor(category_iou_sum)
    reduce_sum_tensor(category_count)
    reduce_sum_tensor(instance_iou_sum)
    reduce_sum_tensor(instance_count)

    accuracy = (total_correct / total_seen.clamp(min=1.0)).item()
    class_acc = (total_correct_class / total_seen_class.clamp(min=1.0)).mean().item()
    category_ious = category_iou_sum / category_count.clamp(min=1.0)
    valid_category_mask = category_count > 0
    class_avg_iou = (
        category_ious[valid_category_mask].mean().item()
        if torch.any(valid_category_mask)
        else 0.0
    )
    instance_avg_iou = (instance_iou_sum / instance_count.clamp(min=1.0)).item()

    for class_index in range(dataset.num_classes):
        if category_count[class_index].item() == 0:
            continue
        category_name = dataset.class_idx_to_cat[class_index]
        print_log(
            "eval mIoU of %s %f"
            % (
                category_name + " " * max(1, 14 - len(category_name)),
                category_ious[class_index].item(),
            ),
            logger=logger,
        )

    print_log(
        "[Validation] EPOCH: %d acc = %.6f class_avg_acc = %.6f class_avg_iou = %.6f instance_avg_iou = %.6f"
        % (epoch, accuracy, class_acc, class_avg_iou, instance_avg_iou),
        logger=logger,
    )

    if val_writer is not None:
        val_writer.add_scalar("Metric/ACC", accuracy, epoch)
        val_writer.add_scalar("Metric/ClassAvgACC", class_acc, epoch)
        val_writer.add_scalar("Metric/ClassAvgIoU", class_avg_iou, epoch)
        val_writer.add_scalar("Metric/InstanceAvgIoU", instance_avg_iou, epoch)

    return PartSegMetric(
        {
            "accuracy": accuracy,
            "class_avg_accuracy": class_acc,
            "class_avg_iou": class_avg_iou,
            "instance_avg_iou": instance_avg_iou,
        }
    )


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log("Part segmentation tester start ...", logger=logger)

    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(
        config.model,
        default_args={"loss": config.get("loss", None)},
    )
    builder.load_model(base_model, args.ckpts, logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
        )
    else:
        base_model = nn.DataParallel(base_model).cuda()

    metrics = validate(base_model, test_dataloader, 0, None, args, config, logger=logger)
    print_log(
        "[Test] acc = %.6f class_avg_acc = %.6f class_avg_iou = %.6f instance_avg_iou = %.6f"
        % (
            metrics.accuracy,
            metrics.class_avg_accuracy,
            metrics.class_avg_iou,
            metrics.instance_avg_iou,
        ),
        logger=logger,
    )
