import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(x, k=20, idx=None):
    batch_size, _, num_points = x.size()
    if idx is None:
        idx = knn(x, k=k)

    idx_base = torch.arange(batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).expand(-1, -1, k, -1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class _PromptArgs:
    def __init__(self, k=20, leaky_relu=True):
        self.k = k
        self.leaky_relu = int(leaky_relu)


class DGCNNView(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)
        self.dim = dim

        act_mod = nn.LeakyReLU if self.leaky_relu else nn.ReLU
        act_kwargs = {"negative_slope": 0.2} if self.leaky_relu else {}

        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim)
        self.bn3 = nn.BatchNorm2d(self.dim)
        self.bn5 = nn.BatchNorm1d(self.dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
            self.bn1,
            act_mod(**act_kwargs),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
            self.bn2,
            act_mod(**act_kwargs),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
            self.bn3,
            act_mod(**act_kwargs),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(self.dim * 3, self.dim, kernel_size=1, bias=False),
            self.bn5,
            act_mod(**act_kwargs),
        )

    def forward(self, x, pos=None):
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv5(x)
        return F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(1)


def build_prompt_generator(prompt_module, dim):
    if prompt_module != "dgcnn":
        raise NotImplementedError(f"Unsupported prompt_module: {prompt_module}")
    return DGCNNView(_PromptArgs(), dim=dim)
