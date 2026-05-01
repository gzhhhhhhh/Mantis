import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .Mamba3D import _extract_base_ckpt, Encoder, Group, Mamba3DBlock
from .build_fn import MODELS
from .peft import DSCDHead, MambaSAA, build_loss_cfg, build_peft_cfg, freeze_module, unfreeze_module
from .z_order import get_z_values
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import print_log


def square_distance(src, dst):
    batch_size, src_points, _ = src.shape
    _, dst_points, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(batch_size, src_points, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(batch_size, 1, dst_points)
    return dist


def index_points(points, idx):
    device = points.device
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        batch_size, num_points, _ = xyz1.shape
        _, num_centers, _ = xyz2.shape

        if num_centers == 1:
            interpolated_points = points2.repeat(1, num_points, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(batch_size, num_points, 3, 1),
                dim=2,
            )

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class Mamba3DEncoderForSegmentation(nn.Module):
    def __init__(
        self,
        fetch_idx,
        k_group_size=8,
        embed_dim=768,
        depth=4,
        drop_path_rate=0.,
        num_group=128,
        num_heads=6,
        bimamba_type='v2',
    ):
        super().__init__()
        self.fetch_idx = tuple(sorted(set(fetch_idx)))
        self.blocks = nn.ModuleList(
            [
                Mamba3DBlock(
                    dim=embed_dim,
                    k_group_size=k_group_size,
                    drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                    num_group=num_group,
                    num_heads=num_heads,
                    bimamba_type=bimamba_type,
                )
                for i in range(depth)
            ]
        )

    def forward(self, center, x, pos, num_prefix_tokens=1):
        feature_list = []
        for layer_idx, block in enumerate(self.blocks):
            x = block(center, x + pos, num_prefix_tokens=num_prefix_tokens)
            if layer_idx in self.fetch_idx:
                feature_list.append(x)

        if not feature_list:
            raise RuntimeError('No segmentation feature maps were collected. Check fetch_idx.')
        return feature_list


class Mamba3DPartSegBase(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.loss_cfg = build_loss_cfg(getattr(config, 'loss', None))

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.num_part = getattr(config, 'num_part', getattr(config, 'cls_dim', 50))
        self.num_classes = getattr(config, 'num_classes', 16)
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.ordering = config.ordering
        self.k_group_size = config.center_local_k
        self.bimamba_type = config.bimamba_type
        self.fetch_idx = tuple(sorted(set(getattr(config, 'fetch_idx', [self.depth - 1]))))
        self.label_smooth = getattr(config, 'label_smooth', 0.0)

        if not self.fetch_idx:
            raise ValueError('fetch_idx must contain at least one layer index.')
        if self.fetch_idx[0] < 0 or self.fetch_idx[-1] >= self.depth:
            raise ValueError(f'fetch_idx must be within [0, {self.depth - 1}], got {self.fetch_idx}')

        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, self.trans_dim),
        )
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        dpr = [value.item() for value in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = Mamba3DEncoderForSegmentation(
            fetch_idx=self.fetch_idx,
            embed_dim=self.trans_dim,
            k_group_size=self.k_group_size,
            depth=self.depth,
            drop_path_rate=dpr,
            num_group=self.num_group,
            num_heads=self.num_heads,
            bimamba_type=self.bimamba_type,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        self.fused_feature_dim = self.trans_dim * len(self.fetch_idx)
        self.label_embed_dim = getattr(config, 'label_embed_dim', 64)
        fp_mlp = list(getattr(config, 'fp_mlp', [self.trans_dim * 4, 1024]))
        seg_channels = list(getattr(config, 'seg_channels', [512, 256]))
        self.seg_dropout = getattr(config, 'seg_dropout', 0.5)

        self.seg_label_conv = nn.Sequential(
            nn.Conv1d(self.num_classes, self.label_embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.label_embed_dim),
            nn.LeakyReLU(0.2),
        )
        self.seg_feature_propagation = PointNetFeaturePropagation(
            in_channel=self.fused_feature_dim + 3,
            mlp=fp_mlp,
        )

        seg_input_dim = fp_mlp[-1] + self.fused_feature_dim * 2 + self.label_embed_dim
        self.seg_conv1 = nn.Conv1d(seg_input_dim, seg_channels[0], 1)
        self.seg_bn1 = nn.BatchNorm1d(seg_channels[0])
        self.seg_dp1 = nn.Dropout(self.seg_dropout)
        self.seg_conv2 = nn.Conv1d(seg_channels[0], seg_channels[1], 1)
        self.seg_bn2 = nn.BatchNorm1d(seg_channels[1])
        self.seg_conv3 = nn.Conv1d(seg_channels[1], self.num_part, 1)
        self.seg_relu = nn.ReLU(inplace=True)

        self.loss_ce = nn.CrossEntropyLoss(label_smoothing=self.label_smooth)
        self.enable_saa = False
        self.enable_dscd = False
        self.dscd_head = None

        self._init_segmentation_modules()

    def _init_segmentation_modules(self):
        modules = [
            self.pos_embed,
            self.seg_label_conv,
            self.seg_feature_propagation,
            self.seg_conv1,
            self.seg_bn1,
            self.seg_conv2,
            self.seg_bn2,
            self.seg_conv3,
        ]
        for module in modules:
            if isinstance(module, nn.Module):
                module.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv1d):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

    def get_profile_inputs(self, npoints, device):
        dummy_points = torch.randn(1, npoints, 3, device=device)
        dummy_labels = torch.zeros(1, dtype=torch.long, device=device)
        return dummy_points, dummy_labels

    def get_pooled_feature_dim(self):
        return self.fused_feature_dim * 2

    def configure_trainable(self):
        for parameter in self.parameters():
            parameter.requires_grad = True
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def _group_points(self, pts):
        coords = pts[:, :, :3].contiguous()
        neighborhood, center = self.group_divider(coords)
        return coords, neighborhood, center

    def _encode_groups(self, neighborhood):
        return self.encoder(neighborhood)

    def _encode_positions(self, center):
        return self.pos_embed(center)

    def _compose_backbone_inputs(self, group_input_tokens, group_pos):
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, group_pos), dim=1)
        return x, pos

    def _forward_backbone_features(self, center, group_input_tokens, group_pos):
        x, pos = self._compose_backbone_inputs(group_input_tokens, group_pos)
        return self.blocks(center, x, pos, num_prefix_tokens=1)

    def _normalize_feature_list(self, feature_list):
        normalized = []
        for feature in feature_list:
            feature = self.norm(feature)
            normalized.append(feature[:, 1:, :].transpose(1, 2).contiguous())
        return normalized

    def _build_pooled_feature(self, point_features):
        x_max = torch.max(point_features, dim=2)[0]
        x_avg = torch.mean(point_features, dim=2)
        return torch.cat([x_max, x_avg], dim=1)

    def _build_class_one_hot(self, class_labels, dtype, device):
        class_labels = class_labels.view(-1).long()
        return F.one_hot(class_labels, num_classes=self.num_classes).to(device=device, dtype=dtype)

    def _decode_segmentation(self, coords, center, feature_list, class_labels):
        batch_size, num_points, _ = coords.shape
        point_features = torch.cat(feature_list, dim=1)
        pooled_feature = self._build_pooled_feature(point_features)

        max_feature = pooled_feature[:, :point_features.shape[1]].unsqueeze(-1).expand(-1, -1, num_points)
        avg_feature = pooled_feature[:, point_features.shape[1]:].unsqueeze(-1).expand(-1, -1, num_points)

        cls_one_hot = self._build_class_one_hot(class_labels, point_features.dtype, point_features.device)
        cls_label_feature = self.seg_label_conv(cls_one_hot.unsqueeze(-1)).expand(-1, -1, num_points)
        global_feature = torch.cat((max_feature, avg_feature, cls_label_feature), dim=1)

        propagated_feature = self.seg_feature_propagation(
            coords.transpose(1, 2),
            center.transpose(1, 2),
            coords.transpose(1, 2),
            point_features,
        )

        fused_feature = torch.cat((propagated_feature, global_feature), dim=1)
        fused_feature = self.seg_relu(self.seg_bn1(self.seg_conv1(fused_feature)))
        fused_feature = self.seg_dp1(fused_feature)
        fused_feature = self.seg_relu(self.seg_bn2(self.seg_conv2(fused_feature)))
        logits = self.seg_conv3(fused_feature).transpose(1, 2).contiguous()
        return logits, pooled_feature

    def _gather_tokens(self, tensor, gather_index):
        return torch.gather(tensor, 1, gather_index.unsqueeze(-1).expand(-1, -1, tensor.size(-1)))

    def _build_order_index(self, center, order):
        if order in (None, 'none', 'native', 'original'):
            return None

        axis_map = {
            'x': (0, False),
            'y': (1, False),
            'z': (2, False),
            'x_rev': (0, True),
            'y_rev': (1, True),
            'z_rev': (2, True),
        }
        if order in axis_map:
            axis, descending = axis_map[order]
            return torch.argsort(center[:, :, axis], dim=1, descending=descending)

        if order in ('z_order', 'z_order_rev'):
            order_index = []
            for batch_center in center.detach().cpu().numpy():
                z_values = torch.from_numpy(get_z_values(batch_center)).to(center.device)
                order_index.append(torch.argsort(z_values, descending=(order == 'z_order_rev')))
            return torch.stack(order_index, dim=0).long()

        raise ValueError(f'Unsupported DSCD order: {order}')

    def _apply_order(self, center, group_input_tokens, group_pos, order):
        order_index = self._build_order_index(center, order)
        if order_index is None:
            return center, group_input_tokens, group_pos
        ordered_center = self._gather_tokens(center, order_index)
        ordered_tokens = self._gather_tokens(group_input_tokens, order_index)
        ordered_pos = self._gather_tokens(group_pos, order_index)
        return ordered_center, ordered_tokens, ordered_pos

    def _forward_single_order_branch(self, center, group_input_tokens, group_pos, coords, class_labels, order):
        branch_center, branch_tokens, branch_pos = self._apply_order(center, group_input_tokens, group_pos, order)
        branch_feature_list = self._forward_backbone_features(branch_center, branch_tokens, branch_pos)
        branch_feature_list = self._normalize_feature_list(branch_feature_list)
        branch_logits, branch_pooled = self._decode_segmentation(
            coords,
            branch_center,
            branch_feature_list,
            class_labels,
        )
        return branch_pooled, branch_logits

    def forward_features(self, pts, class_labels):
        coords, neighborhood, center = self._group_points(pts)
        group_input_tokens = self._encode_groups(neighborhood)
        group_pos = self._encode_positions(center)

        feature_list = self._forward_backbone_features(center, group_input_tokens, group_pos)
        normalized_feature_list = self._normalize_feature_list(feature_list)
        logits, pooled_feature = self._decode_segmentation(coords, center, normalized_feature_list, class_labels)
        return {
            'coords': coords,
            'center': center,
            'group_input_tokens': group_input_tokens,
            'group_pos': group_pos,
            'feature_list': normalized_feature_list,
            'pooled_features': pooled_feature,
            'logits': logits,
        }

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is None:
            print_log('Training from scratch!!!', logger='Mamba3DPartSeg')
            return

        ckpt = torch.load(bert_ckpt_path, map_location='cpu')
        base_ckpt = _extract_base_ckpt(ckpt)
        current_state = self.state_dict()
        compatible_ckpt = {}
        skipped_keys = []
        for key, value in base_ckpt.items():
            if key not in current_state:
                continue
            if current_state[key].shape != value.shape:
                skipped_keys.append(key)
                continue
            compatible_ckpt[key] = value

        incompatible = self.load_state_dict(compatible_ckpt, strict=False)
        if incompatible.missing_keys:
            print_log('missing_keys', logger='Mamba3DPartSeg')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Mamba3DPartSeg',
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Mamba3DPartSeg')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Mamba3DPartSeg',
            )
        if skipped_keys:
            print_log(
                f'[Mamba3DPartSeg] Skipped {len(skipped_keys)} mismatched keys: {skipped_keys[:10]}',
                logger='Mamba3DPartSeg',
            )
        print_log(f'[Mamba3DPartSeg] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba3DPartSeg')

    def get_loss_terms(self, ret, gt):
        logits = ret['logits'] if isinstance(ret, dict) else ret
        flat_logits = logits.reshape(-1, self.num_part)
        flat_gt = gt.reshape(-1).long()
        task_loss = self.loss_ce(flat_logits, flat_gt)
        loss_dict = {'task': task_loss}

        if isinstance(ret, dict):
            aux_losses = ret.get('losses', {})
            for name in ('feat_cons', 'pred_cons', 'ctrl', 'stability'):
                if name in aux_losses:
                    loss_dict[name] = aux_losses[name]

        pred = logits.argmax(dim=-1)
        acc = (pred == gt).float().mean()
        return loss_dict, acc * 100

    def get_loss_weight_map(self):
        return {
            'task': self.loss_cfg['task_weight'],
            'feat_cons': self.loss_cfg['feat_cons_weight'],
            'pred_cons': self.loss_cfg['pred_cons_weight'],
            'ctrl': self.loss_cfg['ctrl_weight'],
            'stability': self.loss_cfg['stability_weight'],
        }

    def combine_loss_terms(self, loss_terms, weight_map=None):
        if weight_map is None:
            weight_map = self.get_loss_weight_map()

        total_loss = None
        for name, loss_value in loss_terms.items():
            weight = float(weight_map.get(name, 0.0 if name != 'task' else 1.0))
            if weight == 0.0:
                continue
            weighted_loss = loss_value * weight
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

        if total_loss is None:
            first_loss = next(iter(loss_terms.values()))
            total_loss = first_loss * 0.0
        return total_loss

    def get_shared_trainable_parameters(self):
        shared_params = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith('seg_') or name.startswith('dscd_head'):
                continue
            shared_params.append(parameter)

        if shared_params:
            return shared_params
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def get_static_loss_acc(self, ret, gt):
        loss_dict, acc = self.get_loss_terms(ret, gt)
        total_loss = self.combine_loss_terms(loss_dict)
        return total_loss, acc, loss_dict

    def get_loss_acc(self, ret, gt):
        return self.get_static_loss_acc(ret, gt)

    def forward(self, pts, class_labels):
        features = self.forward_features(pts, class_labels)
        return features['logits']


@MODELS.register_module()
class Mamba3DPartSeg(Mamba3DPartSegBase):
    pass


@MODELS.register_module()
class Mamba3DPartSegPEFT(Mamba3DPartSegBase):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.peft_cfg = build_peft_cfg(getattr(config, 'peft', None))
        self.enable_saa = self.peft_cfg['enable_saa']
        self.enable_dscd = self.peft_cfg['enable_dscd']

        if self.enable_saa:
            self._inject_saa_mixers()

        if self.enable_dscd:
            dscd_cfg = self.peft_cfg['dscd']
            self.dscd_head = DSCDHead(
                input_dim=self.get_pooled_feature_dim(),
                proj_dim=dscd_cfg['proj_dim'],
                feat_loss=dscd_cfg['feat_loss'],
                temperature=dscd_cfg['temperature'],
            )

        self.configure_trainable()

    def _inject_saa_mixers(self):
        target_layers = self.peft_cfg['saa'].get('target_layers', 'all')
        for layer_idx, block in enumerate(self.blocks.blocks):
            if target_layers != 'all' and layer_idx not in target_layers:
                continue
            block.mixer = MambaSAA.from_mamba(block.mixer, self.peft_cfg['saa'])

    def configure_trainable(self):
        for parameter in self.parameters():
            parameter.requires_grad = True

        if not self.peft_cfg['freeze_backbone']:
            return [name for name, param in self.named_parameters() if param.requires_grad]

        freeze_module(self)
        for module in (
            self.seg_label_conv,
            self.seg_feature_propagation,
            self.seg_conv1,
            self.seg_bn1,
            self.seg_conv2,
            self.seg_bn2,
            self.seg_conv3,
        ):
            unfreeze_module(module)

        if self.enable_dscd:
            unfreeze_module(self.dscd_head)

        if self.enable_saa:
            for block in self.blocks.blocks:
                for name, parameter in block.mixer.named_parameters():
                    if name.startswith('saa_'):
                        parameter.requires_grad = True

        if self.peft_cfg.get('unfreeze_encoder', False):
            unfreeze_module(self.encoder)
        if self.peft_cfg.get('unfreeze_pos_embed', False):
            unfreeze_module(self.pos_embed)
        if self.peft_cfg.get('unfreeze_norms', False):
            unfreeze_module(self.norm)
        if self.peft_cfg.get('unfreeze_prefix_tokens', False):
            self.cls_token.requires_grad = True
            self.cls_pos.requires_grad = True

        unfreeze_last_n_blocks = int(self.peft_cfg.get('unfreeze_last_n_blocks', 0))
        if unfreeze_last_n_blocks > 0:
            for block in self.blocks.blocks[-unfreeze_last_n_blocks:]:
                unfreeze_module(block)

        return [name for name, param in self.named_parameters() if param.requires_grad]

    def freeze_backbone_for_peft(self):
        return self.configure_trainable()

    def forward(self, pts, class_labels):
        features = self.forward_features(pts, class_labels)
        main_logits = features['logits']

        if not self.training:
            return main_logits

        losses = {}
        aux = {}
        if self.enable_dscd and self.dscd_head is not None:
            orders = self.peft_cfg['dscd']['orders']
            if len(orders) != 2:
                raise ValueError('DSCD currently expects exactly two serialization orders.')

            feat_a, logits_a = self._forward_single_order_branch(
                features['center'],
                features['group_input_tokens'],
                features['group_pos'],
                features['coords'],
                class_labels,
                orders[0],
            )
            feat_b, logits_b = self._forward_single_order_branch(
                features['center'],
                features['group_input_tokens'],
                features['group_pos'],
                features['coords'],
                class_labels,
                orders[1],
            )
            dscd_ret = self.dscd_head(feat_a, feat_b, logits_a, logits_b)
            losses['feat_cons'] = dscd_ret['feat_cons']
            losses['pred_cons'] = dscd_ret['pred_cons']
            aux.update(
                {
                    'feat_a': feat_a,
                    'feat_b': feat_b,
                    'logits_a': logits_a,
                    'logits_b': logits_b,
                    'z_a': dscd_ret['z_a'],
                    'z_b': dscd_ret['z_b'],
                }
            )

        return {
            'logits': main_logits,
            'losses': losses,
            'aux': aux,
        }


@MODELS.register_module()
class MantisPartSeg(Mamba3DPartSegPEFT):
    pass
