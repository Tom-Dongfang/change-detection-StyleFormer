import torch
import torch.nn as nn
import torch.nn.functional as F

def momentum_update(old_value, new_value, momentum):
    update = momentum * old_value + (1 - momentum) * new_value
    return update

def calculate_mu_sig(x, eps=1e-6):
    mu = torch.mean(x, dim=(2, 3), keepdim=False)
    var = torch.var(x, dim=(2, 3), unbiased=False)
    sig = torch.sqrt(var + eps)
    return mu, sig

class StyleRepresentation(nn.Module):  # Style Projection module
    def __init__(
            self,
            num_prototype=2,
            channel_size=64,
            batch_size=4,
            gamma=0.9,
            dis_mode='was',
            channel_wise=False
        ):
        super().__init__()
        self.num_prototype = num_prototype
        self.channel_size = channel_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.dis_mode = dis_mode
        self.channel_wise = channel_wise
        style_mu_init = torch.zeros((self.num_prototype, self.channel_size), dtype=torch.float32)
        style_sig_init = torch.ones((self.num_prototype, self.channel_size), dtype=torch.float32)
        self.style_mu = nn.Parameter(style_mu_init, requires_grad=True)
        self.style_sig = nn.Parameter(style_sig_init, requires_grad=True)

    def was_distance(self, cur_mu, cur_sig, proto_mu, proto_sig, batch):
        cur_mu = cur_mu.view(batch, 1, self.channel_size)
        cur_sig = cur_sig.view(batch, 1, self.channel_size)
        proto_mu = proto_mu.view(1, self.num_prototype, self.channel_size)
        proto_sig = proto_sig.view(1, self.num_prototype, self.channel_size)
        distance = torch.pow((cur_mu - proto_mu), 2) + torch.pow(cur_sig, 2) + \
            torch.pow(proto_sig, 2) - 2 * cur_sig * proto_sig
        return distance

    def forward(self, fea):
        batch = fea.size(0)
        cur_mu, cur_sig = calculate_mu_sig(fea)

        proto_mu = self.style_mu
        proto_sig = self.style_sig
        if self.dis_mode == 'was':
            distance = self.was_distance(cur_mu, cur_sig, proto_mu, proto_sig, batch)
        else:  # abs kl others
            raise NotImplementedError('No this distance mode!')

        if not self.channel_wise:
            distance = torch.mean(distance, dim=2, keepdim=False)
        alpha = 1.0 / (1.0 + distance)
        alpha = F.softmax(alpha, dim=1)

        if not self.channel_wise:
            mixed_mu = torch.matmul(alpha, proto_mu)
            mixed_sig = torch.matmul(alpha, proto_sig)
        else:
            raise NotImplementedError('No this distance mode!')

        fea = ((fea - cur_mu.unsqueeze(2).unsqueeze(3)) / cur_sig.unsqueeze(2).unsqueeze(3)) * \
            mixed_sig.unsqueeze(2).unsqueeze(3) + mixed_mu.unsqueeze(2).unsqueeze(3)

        return fea
