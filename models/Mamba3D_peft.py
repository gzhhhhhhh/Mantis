import torch

from .Mamba3D import Mamba3D
from .build_fn import MODELS
from .peft import DSCDHead, MambaSAA, build_loss_cfg, build_peft_cfg, freeze_module, unfreeze_module
from .z_order import get_z_values


@MODELS.register_module()
class Mamba3DPEFT(Mamba3D):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.peft_cfg = build_peft_cfg(getattr(config, 'peft', None))
        self.loss_cfg = build_loss_cfg(getattr(config, 'loss', None))
        self.enable_saa = self.peft_cfg['enable_saa']
        self.enable_dscd = self.peft_cfg['enable_dscd']

        if self.enable_saa:
            self._inject_saa_mixers()

        self.dscd_head = None
        if self.enable_dscd:
            dscd_cfg = self.peft_cfg['dscd']
            self.dscd_head = DSCDHead(
                input_dim=self.get_pooled_feature_dim(),
                proj_dim=dscd_cfg['proj_dim'],
                feat_loss=dscd_cfg['feat_loss'],
                temperature=dscd_cfg['temperature'],
            )

        self.configure_trainable()

    def get_pooled_feature_dim(self):
        return self.trans_dim * 2

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
        unfreeze_module(self.cls_head_finetune)
        if self.enable_dscd:
            unfreeze_module(self.dscd_head)
        if self.enable_saa:
            for block in self.blocks.blocks:
                for name, parameter in block.mixer.named_parameters():
                    if name.startswith('saa_'):
                        parameter.requires_grad = True

        return [name for name, param in self.named_parameters() if param.requires_grad]

    def freeze_backbone_for_peft(self):
        return self.configure_trainable()

    def _group_points(self, pts):
        return self.group_divider(pts)

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

    def _forward_backbone(self, center, x, pos):
        x = self.blocks(center, x, pos)
        return self.norm(x)

    def _pool_features(self, x):
        return torch.cat([x[:, 0], x[:, 1:].max(1)[0] + x[:, 1:].mean(1)[0]], dim=-1)

    def _classify_features(self, pooled_features):
        return self.cls_head_finetune(pooled_features)

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
            centers_cpu = center.detach().cpu().numpy()
            for batch_center in centers_cpu:
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

    def forward_features(self, pts):
        neighborhood, center = self._group_points(pts)
        group_input_tokens = self._encode_groups(neighborhood)
        group_pos = self._encode_positions(center)
        x, pos = self._compose_backbone_inputs(group_input_tokens, group_pos)
        hidden_states = self._forward_backbone(center, x, pos)
        pooled_features = self._pool_features(hidden_states)
        return {
            'center': center,
            'group_input_tokens': group_input_tokens,
            'group_pos': group_pos,
            'hidden_states': hidden_states,
            'pooled_features': pooled_features,
        }

    def _forward_single_order_branch(self, center, group_input_tokens, group_pos, order):
        branch_center, branch_tokens, branch_pos = self._apply_order(center, group_input_tokens, group_pos, order)
        backbone_tokens, backbone_pos = self._compose_backbone_inputs(branch_tokens, branch_pos)
        hidden_states = self._forward_backbone(branch_center, backbone_tokens, backbone_pos)
        branch_features = self._pool_features(hidden_states)
        branch_logits = self._classify_features(branch_features)
        return branch_features, branch_logits

    def get_loss_terms(self, ret, gt):
        logits = ret['logits'] if isinstance(ret, dict) else ret
        task_loss = self.loss_ce(logits, gt.long())
        loss_dict = {'task': task_loss}

        if isinstance(ret, dict):
            aux_losses = ret.get('losses', {})
            for name in ('feat_cons', 'pred_cons', 'ctrl', 'stability'):
                if name in aux_losses:
                    loss_dict[name] = aux_losses[name]

        pred = logits.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
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
            if name.startswith('cls_head_finetune') or name.startswith('dscd_head'):
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

    def forward(self, pts):
        features = self.forward_features(pts)
        main_logits = self._classify_features(features['pooled_features'])

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
                orders[0],
            )
            feat_b, logits_b = self._forward_single_order_branch(
                features['center'],
                features['group_input_tokens'],
                features['group_pos'],
                orders[1],
            )
            dscd_ret = self.dscd_head(feat_a, feat_b, logits_a, logits_b)
            losses['feat_cons'] = dscd_ret['feat_cons']
            losses['pred_cons'] = dscd_ret['pred_cons']
            aux.update({
                'feat_a': feat_a,
                'feat_b': feat_b,
                'logits_a': logits_a,
                'logits_b': logits_b,
                'z_a': dscd_ret['z_a'],
                'z_b': dscd_ret['z_b'],
            })

        return {
            'logits': main_logits,
            'losses': losses,
            'aux': aux,
        }


@MODELS.register_module()
class Mantis(Mamba3DPEFT):
    pass
