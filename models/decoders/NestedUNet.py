import torch.nn as nn
import torch

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output

class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x

class NestedUNet_Conc(nn.Module):
    # UNet++ for decoder
    def __init__(self, out_ch=2, img_size=256):
        super(NestedUNet_Conc, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 81     # the initial number of channels of feature map
        filters = [n1, n1, n1, n1]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Up1_0 = up(filters[1])
        self.Up2_0 = up(filters[2])
        self.Up3_0 = up(filters[3])

        self.conv0_1 = conv_block_nested(filters[0] * 1 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 1 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 1 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])

        self.final1 = nn.Sequential(*[
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Conv2d(filters[0], out_ch, kernel_size=1),
                        nn.Upsample(scale_factor=2, mode='bilinear')
                      ])
        self.final2 = nn.Sequential(*[
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Conv2d(filters[0], out_ch, kernel_size=1),
                        nn.Upsample(scale_factor=4, mode='bilinear')
                      ])
        self.final3 = nn.Sequential(*[
                        nn.Upsample(scale_factor=4, mode='bilinear'),
                        nn.Conv2d(filters[0], out_ch, kernel_size=1),
                        nn.Upsample(scale_factor=4, mode='bilinear')
                      ])
        self.final4 = nn.Sequential(*[
                        nn.Upsample(scale_factor=4, mode='bilinear'),
                        nn.Conv2d(filters[0], out_ch, kernel_size=1),
                        nn.Upsample(scale_factor=8, mode='bilinear')
                      ])
        self.conv_final = nn.Sequential(*[
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1),
                        nn.Upsample(scale_factor=2, mode='bilinear')
                      ])
        self.conv_out = torch.nn.Conv2d(out_ch, 1, kernel_size=3, stride=1, padding=1)

        # self.up_out = F.upsample(change, 610, mode='bilinear')
        ## 函数内部初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


    def forward(self, x1_1, x2_1, x3_1, x4_1):

        x1_2 = self.conv0_1(torch.cat([x1_1, self.Up1_0(x2_1)], 1))
        x2_2 = self.conv1_1(torch.cat([x2_1, self.Up2_0(x3_1)], 1))
        x3_2 = self.conv2_1(torch.cat([x3_1, self.Up3_0(x4_1)], 1))

        x1_3 = self.conv0_2(torch.cat([x1_1, x1_2, self.Up1_1(x2_2)], 1))
        x2_3 = self.conv1_2(torch.cat([x2_1, x2_2, self.Up2_1(x3_2)], 1))

        x1_4 = self.conv0_3(torch.cat([x1_1, x1_2, x1_3, self.Up1_2(x2_3)], 1))

        output1 = self.final1(x1_4)
        output2 = self.final2(x2_3)
        output3 = self.final3(x3_2)
        output4 = self.final4(x4_1)
        output = self.conv_final(torch.cat([x1_1, x1_2, x1_3, x1_4], 1))

        output = self.conv_out(output)
        output1 = self.conv_out(output1)
        output2 = self.conv_out(output2)
        output3 = self.conv_out(output3)
        output4 = self.conv_out(output4)

        return output, output1, output2, output3, output4

class UNet_Conc(nn.Module):
    # UNet for decoder
    def __init__(self, out_ch=2, img_size=256):
        super(UNet_Conc, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 81     # the initial number of channels of feature map
        filters = [n1, n1, n1, n1]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Up1_0 = up(filters[1])
        self.Up2_0 = up(filters[2])
        self.Up3_0 = up(filters[3])

        self.conv0_1 = conv_block_nested(filters[0] * 1 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 1 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 1 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])

        self.final1 = nn.Sequential(*[
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Conv2d(filters[0], out_ch, kernel_size=1),
                        nn.Upsample(scale_factor=2, mode='bilinear')
                      ])
        self.final2 = nn.Sequential(*[
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Conv2d(filters[0], out_ch, kernel_size=1),
                        nn.Upsample(scale_factor=4, mode='bilinear')
                      ])
        self.final3 = nn.Sequential(*[
                        nn.Upsample(scale_factor=4, mode='bilinear'),
                        nn.Conv2d(filters[0], out_ch, kernel_size=1),
                        nn.Upsample(scale_factor=4, mode='bilinear')
                      ])
        self.final4 = nn.Sequential(*[
                        nn.Upsample(scale_factor=4, mode='bilinear'),
                        nn.Conv2d(filters[0], out_ch, kernel_size=1),
                        nn.Upsample(scale_factor=8, mode='bilinear')
                      ])
        self.conv_final = nn.Sequential(*[
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1),
                        nn.Upsample(scale_factor=2, mode='bilinear')
                      ])
        self.conv_out = torch.nn.Conv2d(out_ch, 1, kernel_size=3, stride=1, padding=1)

        # self.up_out = F.upsample(change, 610, mode='bilinear')
        ## 函数内部初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


    def forward(self, x1_1, x2_1, x3_1, x4_1):

        x4_2 = x4_1
        x3_2 = self.conv0_1(torch.cat([x3_1, self.Up3_0(x4_2)], 1))
        x2_2 = self.conv0_1(torch.cat([x2_1, self.Up2_0(x3_2)], 1))
        x1_2 = self.conv0_1(torch.cat([x1_1, self.Up1_0(x2_2)], 1))

        output = self.final1(x1_2)
        output = self.conv_out(output)

        return output