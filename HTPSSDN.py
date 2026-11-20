import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class SCRBlock(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 bias=False
                 ):
        super().__init__()
        self.conv_first = nn.Conv2d(op_channel, op_channel, kernel_size=1, bias=bias)
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)
        self.conv_last = nn.Conv2d(op_channel, op_channel, kernel_size=1, bias=bias)

    def forward(self, x):
        res = x
        x = self.conv_first(x)
        x = self.SRU(x)
        x = self.CRU(x)
        x = self.conv_last(x) + res
        return x


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class SALayer(nn.Module):
    def __init__(self, channel):
        super(SALayer, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)
        self.conv21 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)
        self.conv22 = nn.Conv2d(channel, channel, 5, padding=2, bias=True)
        self.conv3 = nn.Conv2d(channel * 2, channel, 3, padding=1, bias=True)
        self.sigmoid1 = nn.Sigmoid()
        self.conv4 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv21(x)
        x2 = self.conv22(x)
        x3 = torch.cat([x1, x2], dim=1)
        x3 = self.conv3(x3)
        x3 = self.sigmoid1(x3)
        y = x * x3   # torch.multiply(x, x3)
        y = self.conv4(y)
        return y


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 3, stride=1, padding=3, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return x * attn

class Atten(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.catt = CALayer(dim)
        self.satt = SALayer(dim)
        self.lkatt = LKA(dim)

    def forward(self, x, mask):
        x = x + mask
        x = self.lkatt(x)
        mask = self.lkatt(mask)

        # x = self.lkatt(self.satt(self.catt(x)))
        # mask = self.lkatt(self.satt(self.catt(mask)))
        return x, mask


def get_HTP(tensor, dim=1):
    avg_1_30 = tensor[:, 0:30, :, :].mean(dim=dim, keepdim=True)
    avg_175_209 = tensor[:, 174:209, :, :].mean(dim=dim, keepdim=True)
    HTP = avg_1_30 - avg_175_209
    return HTP

class HTPSSDN(nn.Module):
    def __init__(self, in_dim=305, out_dim=305, dim=64, num_blocks=3):
        super(HTPSSDN, self).__init__()
        self.blocks = nn.ModuleList([])
        self.blocks1 = nn.ModuleList([])
        self.blocks2 = nn.ModuleList([])
        self.dim = dim
        self.inpConv = nn.Conv2d(in_dim, self.dim, kernel_size=3, padding=1, stride=1)
        self.maskConv = nn.Conv2d(1, self.dim, kernel_size=3, padding=1, stride=1)
        self.Conv2 = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, stride=1)
        self.Conv3 = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, stride=1)
        self.Conv4 = nn.Conv2d(self.dim * 3, self.dim, kernel_size=3, padding=1, stride=1)
        self.Conv5 = nn.Conv2d(self.dim, self.dim, kernel_size=1, padding=0, stride=1)
        self.outConv = nn.Conv2d(self.dim, out_dim, kernel_size=3, padding=1, stride=1)

        self.num_block = num_blocks
        self.attn = Atten(self.dim)

        def weigth_init(m):
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                SCRBlock(self.dim),
                Atten(self.dim),
                SCRBlock(self.dim),
            ]))
        for _ in range(num_blocks):
            self.blocks1.append(nn.ModuleList([
                SCRBlock(self.dim),
                Atten(self.dim),
                SCRBlock(self.dim),
            ]))
        for _ in range(num_blocks):
            self.blocks2.append(nn.ModuleList([
                SCRBlock(self.dim),
                Atten(self.dim),
                SCRBlock(self.dim),
            ]))
        self.blocks.apply(weigth_init)
        self.blocks1.apply(weigth_init)
        self.blocks2.apply(weigth_init)
        self.attn.apply(weigth_init)

    def forward(self, x):
        x_original_features = x
        mask = get_HTP(x)
        mask = self.maskConv(mask)
        x = self.inpConv(x)
        for (scrblock, attn, scrblock1) in self.blocks:
            res = x
            x = scrblock(x)
            x, mask = attn(x, mask)
            x = scrblock1(x)
            x = x + res
        block = x
        x = self.Conv2(x)
        for (scrblock, attn, scrblock1) in self.blocks1:
            res = x
            x = scrblock(x)
            x, mask = attn(x, mask)
            x = scrblock1(x)
            x = x + res
        block1 = x
        x = self.Conv3(x)
        for (scrblock, attn, scrblock1) in self.blocks2:
            res = x
            x = scrblock(x)
            x, mask = attn(x, mask)
            x = scrblock1(x)
            x = x + res
        block2 = x

        bloc = torch.cat([block, block1, block2], dim=1)
        y = self.Conv4(bloc)
        y, mask = self.attn(y, mask)
        y = self.Conv5(y)
        y = self.outConv(y)
        out = y + x_original_features
        return out


from torchsummary import summary
from ptflops import get_model_complexity_info


if __name__ == '__main__':
    net = HTPSSDN()
    device = torch.device('cpu')
    net.to(device)
    # summary(net.cuda(), (305, 64, 64))
    macs, params = get_model_complexity_info(net, (305, 64, 64), as_strings=True, print_per_layer_stat=True)
    print(f'Computational complexity: {macs}, Number of parameters: {params}')

