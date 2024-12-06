import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm import SynchronizedBatchNorm2d
from models.necks.utils import MLPChannelReducer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

bn_mom = 0.0003

class Cross_AgentAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 shift_size=0, agent_num=16):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(1, dim * 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.shift_size = shift_size

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 4, 4))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 4, 4))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0], 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1]))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))


    def forward(self, input1, input2, guidmap, mask=None):
        """
        Args:
            input (Tensor):B, C, h, w
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x1 = input1.flatten(2).transpose(1, 2)  # B h*w C
        x2 = input2.flatten(2).transpose(1, 2)  # B h*w C
        guidmap = guidmap.flatten(2).transpose(1, 2)  # B h*w C
        b, n, c = x1.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.kv(x1).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        k1, v1 = qkv[0], qkv[1]  # make torchscript happy (cannot use tensor as tuple)
        qkv = self.kv(x2).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        k2, v2 = qkv[0], qkv[1]  # make torchscript happy (cannot use tensor as tuple)
        qg = self.q(guidmap).reshape(b, n, 1, c).permute(2, 0, 1, 3)
        q = qg[0]
        # q, k, v: b, n, c

        agent_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k1 = k1.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v1 = v1.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k2 = k2.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v2 = v2.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        ##
        position_bias1 = nn.functional.interpolate(self.an_bias, size=self.window_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, n).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, n).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2

        agent_attn1 = self.softmax((agent_tokens * self.scale) @ k1.transpose(-2, -1) + position_bias)
        agent_attn1 = self.attn_drop(agent_attn1)
        agent_v1 = agent_attn1 @ v1

        agent_attn2 = self.softmax((agent_tokens * self.scale) @ k2.transpose(-2, -1) + position_bias)
        agent_attn2 = self.attn_drop(agent_attn2)
        agent_v2 = agent_attn2 @ v2
        ##

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)

        out1 = q_attn @ agent_v1
        out2 = q_attn @ agent_v2
        ##

        out1 = out1.transpose(1, 2).reshape(b, n, c)
        v1 = v1.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        out1 = out1 + self.dwc(v1).permute(0, 2, 3, 1).reshape(b, n, c)
        out1 = self.proj(out1)
        out1 = self.proj_drop(out1)
        out1 = out1.transpose(1, 2).view(b, c, h, w)

        out2 = out2.transpose(1, 2).reshape(b, n, c)
        v2 = v2.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        out2 = out2 + self.dwc(v2).permute(0, 2, 3, 1).reshape(b, n, c)
        out2 = self.proj(out2)
        out2 = self.proj_drop(out2)
        out2 = out2.transpose(1, 2).view(b, c, h, w)

        return out1+input1, out2+input2

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class PriorGuideFusionModule(nn.Module):
    def __init__(self, in_dim, out_dim, window_size):
        super(PriorGuideFusionModule, self).__init__()
        self.mlp = MLPChannelReducer(in_channels=in_dim, out_channels=out_dim)
        self.convfd = nn.Sequential(*[
            nn.Conv2d(in_dim, in_dim//2, kernel_size=1, stride=1, padding=0),
            SynchronizedBatchNorm2d(in_dim//2, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
            nn.Conv2d(in_dim//2, out_dim, kernel_size=1, stride=1),
            SynchronizedBatchNorm2d(out_dim, momentum=bn_mom),
            torch.nn.ReLU(inplace=True)
        ])
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.conv2d = nn.Sequential(
            nn.Conv2d(out_dim + out_dim, out_dim, kernel_size=1, stride=1),
            SynchronizedBatchNorm2d(out_dim, momentum=bn_mom),
            torch.nn.ReLU(inplace=True)
        )
        self.chanel_in = in_dim
        self.cross_attn = Cross_AgentAttention(dim=out_dim, window_size=to_2tuple(window_size), num_heads=1)

        self.query_conv = nn.Conv2d(in_channels=out_dim, out_channels=out_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=out_dim, out_channels=out_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        ## 函数内部初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def SelfAtt_guild(self, x, guiding_map):
        m_batchsize, C, height, width = x.size()

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out_f = self.gamma * out + x

        return out_f

    def forward(self, x1, x2):
        # channel sample
        x1 = self.mlp(x1)
        x2 = self.mlp(x2)
        # x1 = self.convfd(x1)
        # x2 = self.convfd(x2)
        #
        cosine_similarity = self.cos(x1, x2)
        cosine_similarity = cosine_similarity.unsqueeze(1)
        guiding_change = F.sigmoid(torch.abs(cosine_similarity-1))

        # change map
        out_change_x1, out_change_x2 = self.cross_attn(x1, x2, guiding_change)

        out_change = torch.cat((out_change_x1, out_change_x2), 1)  # x,y shape(batch_sizxe,channel,w,h), concat at the dim of channel
        out_change = self.conv2d(out_change)

        return out_change
