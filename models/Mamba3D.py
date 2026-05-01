import torch

import torch.nn as nn

import torch.nn.functional as F

import timm

from timm.models.layers import DropPath, trunc_normal_



import numpy as np

from .build_fn import MODELS

from .idpt_prompt_modules import build_prompt_generator

from utils import misc

from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

from utils.logger import *

import random

from knn_cuda import KNN





from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath



from .bimamba_ssm.modules.mamba_simple import Mamba

import random



try:

    from .bimamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

except ImportError:

    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None





import math

from models.z_order import *





def _extract_base_ckpt(ckpt):

    if ckpt.get("base_model") is not None:

        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt["base_model"].items()}

    elif ckpt.get("model") is not None:

        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}

    else:

        raise RuntimeError("mismatch of ckpt weight")



    remapped_ckpt = {}

    for key, value in base_ckpt.items():

        if key.startswith("MAE_encoder."):

            key = key[len("MAE_encoder."):]

        elif key.startswith("base_model."):

            key = key[len("base_model."):]

        remapped_ckpt[key] = value

    return remapped_ckpt





class Encoder(nn.Module):

    def __init__(self, encoder_channel):

        super().__init__()

        self.encoder_channel = encoder_channel

        self.first_conv = nn.Sequential(

            nn.Conv1d(3, 128, 1),

            nn.BatchNorm1d(128),

            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, 1)

        )

        self.second_conv = nn.Sequential(

            nn.Conv1d(512, 512, 1),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Conv1d(512, self.encoder_channel, 1)

        )



    def forward(self, point_groups):

        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''

        bs, g, n , _ = point_groups.shape

        point_groups = point_groups.reshape(bs * g, n, 3)



        feature = self.first_conv(point_groups.transpose(2,1))

        feature_global = torch.max(feature,dim=2,keepdim=True)[0]

        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)

        feature = self.second_conv(feature)

        feature_global = torch.max(feature, dim=2, keepdim=False)[0]

        return feature_global.reshape(bs, g, self.encoder_channel)





class Group(nn.Module):

    def __init__(self, num_group, group_size):

        super().__init__()

        self.num_group = num_group

        self.group_size = group_size

        self.knn = KNN(k=self.group_size, transpose_mode=True)



    def forward(self, xyz):

        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''

        batch_size, num_points, _ = xyz.shape



        center = misc.fps(xyz, self.num_group)



        _, idx = self.knn(xyz, center)

        assert idx.size(1) == self.num_group

        assert idx.size(2) == self.group_size

        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]

        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()



        neighborhood = neighborhood - center.unsqueeze(2)





        return neighborhood, center





class GroupFeature(nn.Module):

    def __init__(self, group_size):

        super().__init__()

        self.group_size = group_size

        self.knn = KNN(k=self.group_size, transpose_mode=True)



    def forward(self, xyz, feat):

        '''
            input:
                xyz: B N 3
                feat: B N C
            ---------------------------
            output:
                neighborhood: B N K 3
                feature: B N K C
        '''

        batch_size, num_points, _ = xyz.shape

        C = feat.shape[-1]



        center = xyz



        _, idx = self.knn(xyz, xyz)

        assert idx.size(1) == num_points

        assert idx.size(2) == self.group_size

        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]

        neighborhood = neighborhood.view(batch_size, num_points, self.group_size, 3).contiguous()

        neighborhood_feat = feat.contiguous().view(-1, C)[idx, :]

        assert neighborhood_feat.shape[-1] == feat.shape[-1]

        neighborhood_feat = neighborhood_feat.view(batch_size, num_points, self.group_size, feat.shape[-1]).contiguous()



        neighborhood = neighborhood - center.unsqueeze(2)



        return neighborhood, neighborhood_feat





class Sine(nn.Module):

    def __init__(self, w0 = 30.):

        super().__init__()

        self.w0 = w0

    def forward(self, x):

        return torch.sin(self.w0 * x)





class K_Norm(nn.Module):

    def __init__(self, out_dim, k_group_size, alpha, beta):

        super().__init__()

        self.group_feat = GroupFeature(k_group_size)

        self.affine_alpha_feat = nn.Parameter(torch.ones([1, 1, 1, out_dim]))

        self.affine_beta_feat = nn.Parameter(torch.zeros([1, 1, 1, out_dim]))



    def forward(self, lc_xyz, lc_x):



        knn_xyz, knn_x = self.group_feat(lc_xyz, lc_x)





        mean_x = lc_x.unsqueeze(dim=-2)

        std_x = torch.std(knn_x - mean_x)



        mean_xyz = lc_xyz.unsqueeze(dim=-2)

        std_xyz = torch.std(knn_xyz - mean_xyz)



        knn_x = (knn_x - mean_x) / (std_x + 1e-5)

        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)



        B, G, K, C = knn_x.shape





        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)





        knn_x = self.affine_alpha_feat * knn_x + self.affine_beta_feat





        knn_x_w = knn_x.permute(0, 3, 1, 2)



        return knn_x_w





class MaxPool(nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, knn_x_w):



        lc_x = knn_x_w.max(-1)[0]

        return lc_x





class Pooling(nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, knn_x_w):



        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)[0]

        return lc_x





class K_Pool(nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, knn_x_w):



        e_x = torch.exp(knn_x_w)

        up = (knn_x_w * e_x).mean(-1)

        down = e_x.mean(-1)

        lc_x = torch.div(up, down)



        return lc_x





class Post_ShareMLP(nn.Module):

    def __init__(self, in_dim, out_dim, permute=True):

        super().__init__()

        self.share_mlp = torch.nn.Conv1d(in_dim, out_dim, 1)

        self.permute = permute



    def forward(self, x):



        if self.permute:

            return self.share_mlp(x).permute(0, 2, 1)

        else:

            return self.share_mlp(x)





class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):

        super().__init__()

        out_features = out_features or in_features

        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()

        self.fc2 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)



    def forward(self, x):

        x = self.fc1(x)

        x = self.act(x)

        x = self.drop(x)

        x = self.fc2(x)

        x = self.drop(x)

        return x





class LNPBlock(nn.Module):

    def __init__(self, lga_out_dim, k_group_size, alpha, beta, mlp_in_dim, mlp_out_dim, num_group=128, act_layer=nn.SiLU, drop_path=0., norm_layer=nn.LayerNorm,):

        super().__init__()

        '''
        lga_out_dim: 2C
        mlp_in_dim: 2C
        mlp_out_dim: C
        x --->  (lga -> pool -> mlp -> act) --> x

        '''

        self.num_group = num_group

        self.lga_out_dim = lga_out_dim



        self.lga = K_Norm(self.lga_out_dim, k_group_size, alpha, beta)

        self.kpool = K_Pool()

        self.mlp = Post_ShareMLP(mlp_in_dim, mlp_out_dim)

        self.pre_norm_ft = norm_layer(self.lga_out_dim)



        self.act = act_layer()





    def forward(self, center, feat, num_prefix_tokens=1):



        _, total_tokens, _ = feat.shape

        if not 0 < num_prefix_tokens < total_tokens:

            raise ValueError(

                f"num_prefix_tokens must be in [1, {total_tokens - 1}], got {num_prefix_tokens}"

            )



        special_tokens = feat[:, :num_prefix_tokens, :]

        feat = feat[:, num_prefix_tokens:, :]

        if feat.shape[1] != center.shape[1]:

            raise ValueError(

                f"point token count ({feat.shape[1]}) must match center count ({center.shape[1]})"

            )



        lc_x_w = self.lga(center, feat)



        lc_x_w = self.kpool(lc_x_w)





        lc_x_w = self.pre_norm_ft(lc_x_w.permute(0, 2, 1))

        lc_x = self.mlp(lc_x_w.permute(0, 2, 1))



        lc_x = self.act(lc_x)



        lc_x = torch.cat((special_tokens, lc_x), dim=1)

        return lc_x





class Mamba3DBlock(nn.Module):

    def __init__(self,

                dim,

                mlp_ratio=4.,

                drop=0.,

                drop_path=0.,

                act_layer=nn.SiLU,

                norm_layer=nn.LayerNorm,

                k_group_size=8,

                alpha=100,

                beta=1000,

                num_group=128,

                num_heads=6,

                bimamba_type="v2",

                ):

        super().__init__()

        self.norm1 = norm_layer(dim)





        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)



        self.num_group = num_group

        self.k_group_size = k_group_size



        self.num_heads = num_heads



        self.lfa = LNPBlock(lga_out_dim=dim*2,

                    k_group_size=self.k_group_size,

                    alpha=alpha,

                    beta=beta,

                    mlp_in_dim=dim*2,

                    mlp_out_dim=dim,

                    num_group=self.num_group,

                    act_layer=act_layer,

                    drop_path=drop_path,



                    norm_layer=norm_layer,

                    )



        self.mixer = Mamba(dim, bimamba_type=bimamba_type)



    def shuffle_x(self, x, shuffle_idx):

        pos = x[:, None, 0, :]

        feat = x[:, 1:, :]

        shuffle_feat = feat[:, shuffle_idx, :]

        x = torch.cat([pos, shuffle_feat], dim=1)

        return x



    def mamba_shuffle(self, x):

        G = x.shape[1] - 1

        shuffle_idx = torch.randperm(G)



        x = self.shuffle_x(x, shuffle_idx)



        x = self.mixer(self.norm2(x))



        x = self.shuffle_x(x, shuffle_idx)

        return x



    def forward(self, center, x, num_prefix_tokens=1):



        x = x + self.drop_path(

            self.lfa(center, self.norm1(x), num_prefix_tokens=num_prefix_tokens)

        )





        x = x + self.drop_path(self.mixer(self.norm2(x)))



        return x





class Mamba3DEncoder(nn.Module):

    def __init__(self, k_group_size=8, embed_dim=768, depth=4, drop_path_rate=0., num_group=128, num_heads=6, bimamba_type="v2",):

        super().__init__()

        self.num_group = num_group

        self.k_group_size = k_group_size

        self.num_heads = num_heads

        self.blocks = nn.ModuleList([

            Mamba3DBlock(

                dim=embed_dim,

                k_group_size = self.k_group_size,

                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,

                num_group=self.num_group,

                num_heads=self.num_heads,

                bimamba_type=bimamba_type,

                )

            for i in range(depth)])



    def forward(self, center, x, pos, num_prefix_tokens=1):

        '''
        INPUT:
            x: patched point cloud and encoded, B G+1 C, 8 128+1=129 384
            pos: positional encoding, B G+1 C, 8 128+1=129 384
        OUTPUT:
            x: x after transformer block, keep dim, B G+1 C, 8 128+1=129 384
        '''



        for _, block in enumerate(self.blocks):

              x = block(center, x + pos, num_prefix_tokens=num_prefix_tokens)

        return x





@MODELS.register_module()

class Mamba3D(nn.Module):

    def __init__(self, config, **kwargs):

        super().__init__()

        self.config = config



        self.trans_dim = config.trans_dim

        self.depth = config.depth

        self.drop_path_rate = config.drop_path_rate

        self.cls_dim = config.cls_dim

        self.num_heads = config.num_heads



        self.group_size = config.group_size

        self.num_group = config.num_group

        self.encoder_dims = config.encoder_dims





        self.encoder = Encoder(encoder_channel=self.encoder_dims)



        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))



        self.pos_embed = nn.Sequential(

            nn.Linear(3, 128),

            nn.SiLU(),

            nn.Linear(128, self.trans_dim)

        )



        self.ordering = config.ordering

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)



        self.k_group_size = config.center_local_k



        self.bimamba_type = config.bimamba_type



        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]



        self.blocks = Mamba3DEncoder(

            embed_dim=self.trans_dim,

            k_group_size=self.k_group_size,

            depth=self.depth,

            drop_path_rate=dpr,

            num_group=self.num_group,

            num_heads=self.num_heads,

            bimamba_type=self.bimamba_type,

        )





        self.norm = nn.LayerNorm(self.trans_dim)



        self.cls_head_finetune = nn.Sequential(

                nn.Linear(self.trans_dim * 2, 256),

                nn.BatchNorm1d(256),

                nn.ReLU(inplace=True),

                nn.Dropout(0.5),

                nn.Linear(256, 256),

                nn.BatchNorm1d(256),

                nn.ReLU(inplace=True),

                nn.Dropout(0.5),

                nn.Linear(256, self.cls_dim)

            )



        self.label_smooth = config.label_smooth

        self.build_loss_func()



        trunc_normal_(self.cls_token, std=.02)

        trunc_normal_(self.cls_pos, std=.02)



    def build_loss_func(self):

        self.loss_ce = nn.CrossEntropyLoss(label_smoothing=self.label_smooth)



    def get_loss_acc(self, ret, gt):

        loss = self.loss_ce(ret, gt.long())

        pred = ret.argmax(-1)

        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100



    def load_model_from_ckpt(self, bert_ckpt_path):

        if bert_ckpt_path is not None:

            ckpt = torch.load(bert_ckpt_path, map_location="cpu")

            base_ckpt = _extract_base_ckpt(ckpt)



            incompatible = self.load_state_dict(base_ckpt, strict=False)



            if incompatible.missing_keys:

                print_log('missing_keys', logger='Transformer')

                print_log(

                    get_missing_parameters_message(incompatible.missing_keys),

                    logger='Transformer'

                )

            if incompatible.unexpected_keys:

                print_log('unexpected_keys', logger='Transformer')

                print_log(

                    get_unexpected_parameters_message(incompatible.unexpected_keys),

                    logger='Transformer'

                )



            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')

        else:

            print_log('Training from scratch!!!', logger='Transformer')

            self.apply(self._init_weights)



    def _init_weights(self, m):

        if isinstance(m, nn.Linear):

            trunc_normal_(m.weight, std=.02)

            if isinstance(m, nn.Linear) and m.bias is not None:

                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):

            nn.init.constant_(m.bias, 0)

            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv1d):

            trunc_normal_(m.weight, std=.02)

            if m.bias is not None:

                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv2d):

            trunc_normal_(m.weight, std=.02)

            if m.bias is not None:

                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):

            if m.bias is not None:

                nn.init.constant_(m.bias, 0)

            if m.weight is not None:

                nn.init.constant_(m.weight, 1.0)



    def forward(self, pts):



        neighborhood, center = self.group_divider(pts)

        group_input_tokens = self.encoder(neighborhood)



        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)

        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)



        pos = self.pos_embed(center)



        x = torch.cat((cls_tokens, group_input_tokens), dim=1)

        pos = torch.cat((cls_pos, pos), dim=1)



        x = self.blocks(center, x, pos)

        x = self.norm(x)

        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0] + x[:, 1:].mean(1)[0]], dim=-1)

        ret = self.cls_head_finetune(concat_f)

        return ret





class Mamba3DEncoderIDPT(nn.Module):

    def __init__(

        self,

        k_group_size=8,

        embed_dim=768,

        depth=4,

        drop_path_rate=0.,

        num_group=128,

        num_heads=6,

        bimamba_type="v2",

    ):

        super().__init__()

        self.blocks = nn.ModuleList([

            Mamba3DBlock(

                dim=embed_dim,

                k_group_size=k_group_size,

                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,

                num_group=num_group,

                num_heads=num_heads,

                bimamba_type=bimamba_type,

            )

            for i in range(depth)

        ])



    def forward(self, center, x, pos, prompt_generator, prompt_layer, prompt_pos):

        prompt_inserted = False

        num_prefix_tokens = 1



        for layer_idx, block in enumerate(self.blocks):

            if not prompt_inserted and layer_idx == prompt_layer:

                hidden_states = x + pos

                prompt = prompt_generator(hidden_states[:, 1:, :].transpose(1, 2), None)

                prompt_pos_token = prompt_pos.expand(prompt.size(0), -1, -1).to(pos.dtype)

                x = torch.cat([x[:, :1, :], prompt, x[:, 1:, :]], dim=1)

                pos = torch.cat([pos[:, :1, :], prompt_pos_token, pos[:, 1:, :]], dim=1)

                num_prefix_tokens = 2

                prompt_inserted = True



            x = block(center, x + pos, num_prefix_tokens=num_prefix_tokens)



        if not prompt_inserted:

            raise RuntimeError("Prompt token was not inserted into Mamba3DEncoderIDPT.")

        return x





@MODELS.register_module()

class Mamba3DIDPT(nn.Module):

    def __init__(self, config, **kwargs):

        super().__init__()

        self.config = config



        self.trans_dim = config.trans_dim

        self.depth = config.depth

        self.drop_path_rate = config.drop_path_rate

        self.cls_dim = config.cls_dim

        self.num_heads = config.num_heads



        self.group_size = config.group_size

        self.num_group = config.num_group

        self.encoder_dims = config.encoder_dims



        self.prompt_layer = config.prompt_layer

        self.prompt_module = config.prompt_module

        self.cls_type = config.cls_type



        if not 0 <= self.prompt_layer < self.depth:

            raise ValueError(f"prompt_layer must be in [0, {self.depth - 1}], got {self.prompt_layer}")

        if self.cls_type not in {"promptonly", "promptcls", "pointcls", "all"}:

            raise ValueError(f"Unsupported cls_type: {self.cls_type}")



        channel_map = {

            "promptonly": 1,

            "promptcls": 2,

            "pointcls": 2,

            "all": 3,

        }

        self.head_channels = channel_map[self.cls_type]



        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.prompt_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))



        self.pos_embed = nn.Sequential(

            nn.Linear(3, 128),

            nn.SiLU(),

            nn.Linear(128, self.trans_dim)

        )



        self.ordering = config.ordering

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.k_group_size = config.center_local_k

        self.bimamba_type = config.bimamba_type



        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        self.blocks = Mamba3DEncoderIDPT(

            embed_dim=self.trans_dim,

            k_group_size=self.k_group_size,

            depth=self.depth,

            drop_path_rate=dpr,

            num_group=self.num_group,

            num_heads=self.num_heads,

            bimamba_type=self.bimamba_type,

        )



        self.norm = nn.LayerNorm(self.trans_dim)

        self.prompt_generator = build_prompt_generator(self.prompt_module, dim=self.trans_dim)

        self.cls_head_finetune = nn.Sequential(

            nn.Linear(self.trans_dim * self.head_channels, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(256, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(256, self.cls_dim)

        )



        self.label_smooth = getattr(config, "label_smooth", 0.0)

        self.build_loss_func()



        self.prompt_generator.apply(self._init_weights)

        self.cls_head_finetune.apply(self._init_weights)

        trunc_normal_(self.cls_token, std=.02)

        trunc_normal_(self.cls_pos, std=.02)

        trunc_normal_(self.prompt_pos, std=.02)



    def build_loss_func(self):

        self.loss_ce = nn.CrossEntropyLoss(label_smoothing=self.label_smooth)



    def get_loss_acc(self, ret, gt):

        loss = self.loss_ce(ret, gt.long())

        pred = ret.argmax(-1)

        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100



    def _init_weights(self, m):

        if isinstance(m, nn.Linear):

            trunc_normal_(m.weight, std=.02)

            if m.bias is not None:

                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):

            nn.init.constant_(m.bias, 0)

            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv1d):

            trunc_normal_(m.weight, std=.02)

            if m.bias is not None:

                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv2d):

            trunc_normal_(m.weight, std=.02)

            if m.bias is not None:

                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):

            if m.bias is not None:

                nn.init.constant_(m.bias, 0)

            if m.weight is not None:

                nn.init.constant_(m.weight, 1.0)



    def freeze_backbone_for_peft(self):

        for param in self.parameters():

            param.requires_grad = False



        self.cls_token.requires_grad = True

        self.cls_pos.requires_grad = True

        self.prompt_pos.requires_grad = True



        for module in (self.prompt_generator, self.cls_head_finetune):

            for param in module.parameters():

                param.requires_grad = True



        return [name for name, param in self.named_parameters() if param.requires_grad]



    def load_model_from_ckpt(self, bert_ckpt_path):

        if bert_ckpt_path is None:

            print_log('Training from scratch!!!', logger='Mamba3DIDPT')

            self.apply(self._init_weights)

            return



        ckpt = torch.load(bert_ckpt_path, map_location="cpu")

        base_ckpt = _extract_base_ckpt(ckpt)

        ignored_prefixes = (

            "cls_head_finetune.",

            "cls_token",

            "cls_pos",

            "prompt_pos",

            "prompt_generator.",

        )

        remapped_ckpt = {

            key: value

            for key, value in base_ckpt.items()

            if not any(key == prefix or key.startswith(prefix) for prefix in ignored_prefixes)

        }



        incompatible = self.load_state_dict(remapped_ckpt, strict=False)



        if incompatible.missing_keys:

            print_log('missing_keys', logger='Mamba3DIDPT')

            print_log(

                get_missing_parameters_message(incompatible.missing_keys),

                logger='Mamba3DIDPT'

            )

        if incompatible.unexpected_keys:

            print_log('unexpected_keys', logger='Mamba3DIDPT')

            print_log(

                get_unexpected_parameters_message(incompatible.unexpected_keys),

                logger='Mamba3DIDPT'

            )



        print_log(f'[Mamba3DIDPT] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba3DIDPT')



    def _build_cls_feature(self, x):

        expected_tokens = self.num_group + 2

        if x.shape[1] != expected_tokens:

            raise RuntimeError(

                f"Mamba3DIDPT expects {expected_tokens} tokens (cls + prompt + points), got shape {tuple(x.shape)}"

            )



        cls_token = x[:, 0, :]

        prompt_token = x[:, 1, :]

        point_tokens = x[:, 2:, :]

        pooled_points = point_tokens.max(1)[0] + point_tokens.mean(1)[0]



        if self.cls_type == "promptonly":

            return prompt_token

        if self.cls_type == "promptcls":

            return torch.cat([cls_token, prompt_token], dim=1)

        if self.cls_type == "pointcls":

            return torch.cat([cls_token, pooled_points], dim=1)

        return torch.cat([cls_token, prompt_token, pooled_points], dim=1)



    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)

        group_input_tokens = self.encoder(neighborhood)



        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)

        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)



        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)

        pos = torch.cat((cls_pos, pos), dim=1)



        x = self.blocks(

            center,

            x,

            pos,

            prompt_generator=self.prompt_generator,

            prompt_layer=self.prompt_layer,

            prompt_pos=self.prompt_pos,

        )

        x = self.norm(x)

        ret = self.cls_head_finetune(self._build_cls_feature(x))

        return ret
