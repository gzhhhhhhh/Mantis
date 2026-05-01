import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from ..bimamba_ssm.modules.mamba_simple import Mamba
from ..bimamba_ssm.ops.selective_scan_interface import selective_scan_fn

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


class MambaSAA(Mamba):
    def __init__(self, *args, saa_cfg=None, **kwargs):
        saa_cfg = saa_cfg or {}
        self.saa_rank = saa_cfg.get('rank', 8)
        self.saa_control_dim = saa_cfg.get('control_dim', 64)
        self.saa_modulate_dt = saa_cfg.get('modulate_dt', True)
        self.saa_modulate_B = saa_cfg.get('modulate_B', True)
        self.saa_modulate_C = saa_cfg.get('modulate_C', True)
        self.saa_modulate_A = saa_cfg.get('modulate_A', False)
        self.saa_disable_fast_path = saa_cfg.get('disable_fast_path', True)
        kwargs['use_fast_path'] = False if self.saa_disable_fast_path else kwargs.get('use_fast_path', True)
        super().__init__(*args, **kwargs)

        self.saa_proj = nn.Linear(self.d_model, self.saa_control_dim)
        self.saa_controller = nn.Linear(self.saa_control_dim, self.saa_rank)

        self.saa_basis_dt = nn.Parameter(torch.zeros(self.saa_rank, self.d_inner)) if self.saa_modulate_dt else None
        self.saa_basis_B = nn.Parameter(torch.zeros(self.saa_rank, self.d_state)) if self.saa_modulate_B else None
        self.saa_basis_C = nn.Parameter(torch.zeros(self.saa_rank, self.d_state)) if self.saa_modulate_C else None

        self.saa_basis_dt_b = nn.Parameter(torch.zeros(self.saa_rank, self.d_inner)) if self.saa_modulate_dt else None
        self.saa_basis_B_b = nn.Parameter(torch.zeros(self.saa_rank, self.d_state)) if self.saa_modulate_B else None
        self.saa_basis_C_b = nn.Parameter(torch.zeros(self.saa_rank, self.d_state)) if self.saa_modulate_C else None

        if self.saa_modulate_A:
            raise NotImplementedError('SAA modulation for A is intentionally deferred to a later phase.')

        self.reset_saa_parameters()

    def reset_saa_parameters(self):
        nn.init.trunc_normal_(self.saa_proj.weight, std=0.02)
        nn.init.zeros_(self.saa_proj.bias)
        nn.init.trunc_normal_(self.saa_controller.weight, std=0.02)
        nn.init.zeros_(self.saa_controller.bias)

        for parameter in (
            self.saa_basis_dt,
            self.saa_basis_B,
            self.saa_basis_C,
            self.saa_basis_dt_b,
            self.saa_basis_B_b,
            self.saa_basis_C_b,
        ):
            if parameter is not None:
                nn.init.zeros_(parameter)

    @classmethod
    def from_mamba(cls, mixer, saa_cfg):
        saa_mixer = cls(
            d_model=mixer.d_model,
            d_state=mixer.d_state,
            d_conv=mixer.d_conv,
            expand=mixer.expand,
            dt_rank=mixer.dt_rank,
            conv_bias=mixer.conv1d.bias is not None,
            bias=mixer.in_proj.bias is not None,
            use_fast_path=mixer.use_fast_path,
            layer_idx=mixer.layer_idx,
            device=mixer.in_proj.weight.device,
            dtype=mixer.in_proj.weight.dtype,
            bimamba_type=mixer.bimamba_type,
            saa_cfg=saa_cfg,
        )
        saa_mixer.load_state_dict(mixer.state_dict(), strict=False)
        return saa_mixer

    def _compute_control(self, hidden_states):
        control = self.saa_proj(hidden_states)
        control = F.silu(control)
        control = torch.sigmoid(self.saa_controller(control))
        return control

    def _apply_token_modulation(self, tensor, control, basis):
        if basis is None:
            return tensor
        delta = torch.einsum('blr,rf->blf', control, basis)
        return tensor + rearrange(delta, 'b l f -> b f l')

    def _run_branch(self, x, z, control, conv1d, x_proj, dt_proj, A_log, D, basis_dt, basis_B, basis_C):
        seqlen = x.shape[-1]
        if causal_conv1d_fn is None:
            x = self.act(conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(conv1d.weight, 'd 1 w -> d w'),
                bias=conv1d.bias,
                activation=self.activation,
            )

        x_dbl = x_proj(rearrange(x, 'b d l -> (b l) d'))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = dt_proj.weight @ dt.t()
        dt = rearrange(dt, 'd (b l) -> b d l', l=seqlen)
        B = rearrange(B, '(b l) dstate -> b dstate l', l=seqlen).contiguous()
        C = rearrange(C, '(b l) dstate -> b dstate l', l=seqlen).contiguous()

        dt = self._apply_token_modulation(dt, control, basis_dt)
        B = self._apply_token_modulation(B, control, basis_B)
        C = self._apply_token_modulation(C, control, basis_C)

        A = -torch.exp(A_log.float())
        return selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            D.float(),
            z=z,
            delta_bias=dt_proj.bias.float(),
            delta_softplus=True,
        )

    def forward(self, hidden_states, inference_params=None):
        if inference_params is not None and inference_params.seqlen_offset > 0:
            return super().forward(hidden_states, inference_params=inference_params)

        _, seqlen, _ = hidden_states.shape
        control = self._compute_control(hidden_states)

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, 'b l d -> d (b l)'),
            'd (b l) -> b d l',
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), 'd -> d 1')

        x, z = xz.chunk(2, dim=1)
        y = self._run_branch(
            x,
            z,
            control,
            self.conv1d,
            self.x_proj,
            self.dt_proj,
            self.A_log,
            self.D,
            self.saa_basis_dt,
            self.saa_basis_B,
            self.saa_basis_C,
        )

        if self.bimamba_type == 'v4':
            y_b = self._run_branch(
                x.flip([-1]),
                z.flip([-1]),
                control.flip([1]),
                self.conv1d_b,
                self.x_proj_b,
                self.dt_proj_b,
                self.A_b_log,
                self.D_b,
                self.saa_basis_dt_b,
                self.saa_basis_B_b,
                self.saa_basis_C_b,
            )
            y = y + y_b.flip([-1])

        y = rearrange(y, 'b d l -> b l d')
        return self.out_proj(y)
