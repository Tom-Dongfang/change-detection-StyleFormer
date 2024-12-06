import numpy as np
from torch.optim import lr_scheduler
from torch.nn import init
import torch.fft
import functools

from models.utils import *
from .backbones.vit_comer import ViTCoMer_new
from .necks.prior_guidaat import PriorGuideFusionModule
from .decoders.NestedUNet import NestedUNet_Conc, UNet_Conc

###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)  # 网络参数初始化
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'StyleFormer':
        net = Style_VCFormer(num_band=3, img_size=args.img_size, os=16, use_se='True', output_nc=args.n_class,
                             interaction_indexes=args.interaction_indexes)

    elif args.net_G == 'BDEDN':
        net = BASE_BDEDN(num_band=3, os=16, use_se='True')
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)  # 模型初始化

###############################################################################
# main Functions
###############################################################################

class Style_VCFormer(torch.nn.Module):
    """
    ViT_COMer+Guilding Fusion+Unet
    """
    def __init__(self, num_band, img_size, os, use_se, output_nc, interaction_indexes=None):
        super(Style_VCFormer, self).__init__()
        # style
        # encoder
        self.vitcomer = ViTCoMer_new(pretrain_size=256, conv_inplane=64, n_points=4, deform_num_heads=12,
                 init_values=0., interaction_indexes=interaction_indexes, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=0.5, add_vit_feature=True, use_extra_CTI=True, pretrained=None, with_cp=False,
                 use_CTI_toV=True,
                 use_CTI_toC=True,
                 cnn_feature_interaction=True,
                 dim_ratio=1.0)
        # fusion
        window_size = [64, 32, 16, 8]
        self.Interaction1 = PriorGuideFusionModule(in_dim=768, out_dim=192, window_size=window_size[0])
        self.Interaction2 = PriorGuideFusionModule(in_dim=768, out_dim=192, window_size=window_size[1])
        self.Interaction3 = PriorGuideFusionModule(in_dim=768, out_dim=192, window_size=window_size[2])
        self.Interaction4 = PriorGuideFusionModule(in_dim=768, out_dim=192, window_size=window_size[3])
        # decoder
        # self.unetdecoder = NestedUNet_Conc(out_ch=output_nc, img_size=img_size)
        self.unetdecoder = UNet_Conc(out_ch=output_nc, img_size=img_size)

    def forward(self, x1, x2):
        # encoder: vit_comer
        ef1_1, ef1_2, ef1_3, ef1_4 = self.vitcomer(x1)
        ef2_1, ef2_2, ef2_3, ef2_4 = self.vitcomer(x2)

        # fusion: change prior-guided
        gf1 = self.Interaction1(ef1_1, ef2_1)
        gf2 = self.Interaction2(ef1_2, ef2_2)
        gf3 = self.Interaction3(ef1_3, ef2_3)
        gf4 = self.Interaction4(ef1_4, ef2_4)

        # decoder: UNet++
        y, y1, y2, y3, y4 = self.unetdecoder(gf1, gf2, gf3, gf4)

        return [y, y1, y2, y3, y4]


class Filter_One_Side(nn.Module):
    def __init__(self, C=3, H=600, W=1138):
        super(Filter_One_Side, self).__init__()
        self.C = C
        self.H = H
        self.W = W
        self.learnable_h = self.H
        self.learnable_w = np.floor(self.W/2).astype(int) + 1
        self.register_parameter('conv_invariant', torch.nn.Parameter(torch.rand(self.C, self.learnable_h, self.learnable_w), requires_grad=True))
    def forward(self, feature):
        feature_fft = torch.fft.rfftn(feature, dim=(-2, -1))
        feature_fft = feature_fft + 1e-8
        feature_amp = torch.abs(feature_fft)
        feature_pha = torch.angle(feature_fft)
        feature_amp_invariant = torch.mul(feature_amp, self.conv_invariant)
        feature_fft_invariant = feature_amp_invariant * torch.exp(torch.tensor(1j) * feature_pha)
        feature_invariant = torch.fft.irfftn(feature_fft_invariant, dim=(-2, -1))
        return feature_invariant, feature-feature_invariant

