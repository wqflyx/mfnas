import torch
import torch.nn as nn

OPS = {
    'avg_pool_3x3': lambda C, stride, affine, source, target: nn.AvgPool2d(3, stride=stride, padding=1,
                                                                           count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine, source, target: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine, source, target: Identity() if stride == 1 else FactorizedReduce(C, C,
                                                                                                              source,
                                                                                                              target,
                                                                                                              affine=affine),
    'sep_conv_3x3': lambda C, stride, affine, source, target: SepConv3(C, C, 3, stride, 1, source, target,
                                                                       affine=affine),
    'sep_conv_5x5': lambda C, stride, affine, source, target: SepConv5(C, C, 5, stride, 2, source, target,
                                                                       affine=affine),
    'dil_conv_3x3': lambda C, stride, affine, source, target: DilConv3(C, C, 3, stride, 2, 2, source, target,
                                                                       affine=affine),
    'dil_conv_5x5': lambda C, stride, affine, source, target: DilConv5(C, C, 5, stride, 4, 2, source, target,
                                                                       affine=affine),
}


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv3(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, source, target, affine=True):
        super(DilConv3, self).__init__()
        self.source = source
        self.op_dil_3 = nn.ModuleList()

        for i in range(target):
            if i == source:
                self.op_dil_3 += [nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=C_in, bias=False),
                    nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm2d(C_out, affine=affine),
                )]
            else:
                self.op_dil_3 += [nn.Module()]

    def forward(self, x):
        return self.op_dil_3[self.source](x)


class DilConv5(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, source, target, affine=True):
        super(DilConv5, self).__init__()
        self.source = source
        self.op_dil_5 = nn.ModuleList()

        for i in range(target):
            if i == source:
                self.op_dil_5 += [nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=C_in, bias=False),
                    nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm2d(C_out, affine=affine),
                )]
            else:
                self.op_dil_5 += [nn.Module()]

    def forward(self, x):
        return self.op_dil_5[self.source](x)


class SepConv3(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, source, target, affine=True):
        super(SepConv3, self).__init__()
        self.source = source
        self.op_sep_3 = nn.ModuleList()

        for i in range(target):
            if i == source:
                self.op_sep_3 += [nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in,
                              bias=False),
                    nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm2d(C_in, affine=affine),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
                    nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm2d(C_out, affine=affine),
                )]
            else:
                self.op_sep_3 += [nn.Module()]

    def forward(self, x):
        return self.op_sep_3[self.source](x)


class SepConv5(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, source, target, affine=True):
        super(SepConv5, self).__init__()
        self.source = source
        self.op_sep_5 = nn.ModuleList()

        for i in range(target):
            if i == source:
                self.op_sep_5 += [nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in,
                              bias=False),
                    nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm2d(C_in, affine=affine),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
                    nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm2d(C_out, affine=affine),
                )]
            else:
                self.op_sep_5 += [nn.Module()]

    def forward(self, x):
        return self.op_sep_5[self.source](x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduceOrg(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduceOrg, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, source, target, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)

        self.source = source
        self.conv_1 = nn.ModuleList()
        self.conv_2 = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(target):
            if i == source:
                self.conv_1 += [nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)]
                self.conv_2 += [nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)]
                self.bn += [nn.BatchNorm2d(C_out, affine=affine)]
            else:
                self.conv_1 += [nn.Module()]
                self.conv_2 += [nn.Module()]
                self.bn += [nn.Module()]

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1[self.source](x), self.conv_2[self.source](x[:, :, 1:, 1:])], dim=1)
        out = self.bn[self.source](out)
        return out
