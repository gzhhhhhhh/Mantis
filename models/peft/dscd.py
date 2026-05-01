import torch.nn as nn
import torch.nn.functional as F


class DSCDHead(nn.Module):
    def __init__(self, input_dim, proj_dim=128, feat_loss='cosine', temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.feat_loss = feat_loss
        self.proj_head = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def feature_consistency(self, feat_a, feat_b):
        z_a = self.proj_head(feat_a)
        z_b = self.proj_head(feat_b)
        if self.feat_loss == 'l2':
            loss = F.mse_loss(z_a, z_b)
        elif self.feat_loss == 'cosine':
            loss = 1.0 - F.cosine_similarity(z_a, z_b, dim=-1).mean()
        else:
            raise ValueError(f'Unsupported feature consistency loss: {self.feat_loss}')
        return loss, z_a, z_b

    def prediction_consistency(self, logits_a, logits_b):
        temperature = self.temperature
        log_prob_a = F.log_softmax(logits_a / temperature, dim=-1)
        log_prob_b = F.log_softmax(logits_b / temperature, dim=-1)
        prob_a = log_prob_a.exp()
        prob_b = log_prob_b.exp()


        loss_ab = F.kl_div(log_prob_a, prob_b, reduction='none').sum(dim=-1).mean()
        loss_ba = F.kl_div(log_prob_b, prob_a, reduction='none').sum(dim=-1).mean()
        return 0.5 * (loss_ab + loss_ba) * (temperature ** 2)

    def forward(self, feat_a, feat_b, logits_a, logits_b):
        feat_loss, z_a, z_b = self.feature_consistency(feat_a, feat_b)
        pred_loss = self.prediction_consistency(logits_a, logits_b)
        return {
            'feat_cons': feat_loss,
            'pred_cons': pred_loss,
            'z_a': z_a,
            'z_b': z_b,
        }
