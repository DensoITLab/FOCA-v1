# reference https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/PyramidNet.py
# reference https://github.com/owruby/shake-drop_pytorch/blob/master/models/shakedrop.py
import torch, math
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class ShakeDropFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, training, p_drop, alpha_range):
        if training:
            gate = torch.zeros(1, device=x.device.index).bernoulli_(p_drop)
            ctx.save_for_backward(gate)
            if gate.item() == 0:
                alpha = torch.zeros(x.shape[0], device=x.device.index).uniform_(*alpha_range)
                alpha = alpha.view(alpha.shape[0], 1, 1, 1).expand_as(x)
                return alpha * x
            else:
                return x
        else:
            return p_drop * x

    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0:
            beta = torch.rand(grad_output.shape[0], device=grad_output.device.index)
            beta = Variable(beta.view(beta.shape[0], 1, 1, 1).expand_as(grad_output))
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None


class ShakeDrop(nn.Module):

    def __init__(self, p_drop, alpha_range):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)


class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, in_channels, out_channels, stride, p_shakedrop):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shake_drop = ShakeDrop(p_shakedrop, [-1, 1])
        if stride == 2:
            self.downsample = nn.AvgPool2d(2, ceil_mode=True)
        else:
            self.downsample = None

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.shake_drop(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            map_size = shortcut.shape[2:4]
        else:
            shortcut = x
            map_size = out.shape[2:4]

        batch_size = out.shape[0]
        residual_channel = out.shape[1]
        shortcut_channel = shortcut.shape[1]

        if residual_channel != shortcut_channel:
            padding = Variable(torch.zeros(batch_size, residual_channel - shortcut_channel, map_size[0], map_size[1], device=x.device.index))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out

class BottleneckBlock(nn.Module):
    outchannel_ratio = 4

    def __init__(self, in_channels, out_channels, stride, p_shakedrop):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, (out_channels*1), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d((out_channels*1))
        self.conv3 = nn.Conv2d((out_channels*1), out_channels * self.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels * self.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.shake_drop = ShakeDrop(p_shakedrop, [-1, 1])
        if stride == 2:
            self.downsample = nn.AvgPool2d(2, ceil_mode=True)
        else:
            self.downsample = None

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn4(out)
        out = self.shake_drop(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            map_size = shortcut.shape[2:4]
        else:
            shortcut = x
            map_size = out.shape[2:4]

        batch_size = out.shape[0]
        residual_channel = out.shape[1]
        shortcut_channel = shortcut.shape[1]

        if residual_channel != shortcut_channel:
            padding = Variable(torch.zeros(batch_size, residual_channel - shortcut_channel, map_size[0], map_size[1], device=x.device.index))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out

class Pyramid_ResNet_fet(nn.Module):
        
    def __init__(self, args):
        super(Pyramid_ResNet_fet, self).__init__()
        self.in_channels = 16
        n = int((args.depth - 2) / 9)
        block = BottleneckBlock

        self.addrate = args.factor / (3*n*1.0)
        self.ps_shakedrop = [(1.0-(0.5/(3*n))*(i+1)) for i in range(3*n)]
        self.idx = 0

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        self.out_channels = self.in_channels
        self.layer1 = self._make_layer(block, n, stride=1)
        self.layer2 = self._make_layer(block, n, stride=2)
        self.layer3 = self._make_layer(block, n, stride=2)

        self.numFeatureDim = self.in_channels
        self.bn_final= nn.BatchNorm2d(self.numFeatureDim)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.MLP = self._make_MLP(args)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode=args.init_mode)
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data, mean=0, std=args.init_std)
            if isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight.data, 1)
                m.bias.data.zero_()

    def _make_layer(self, block, block_depth, stride):
        layers = []
        for i in range(block_depth):
            self.out_channels = self.out_channels + self.addrate
            layers.append(block(self.in_channels, int(round(self.out_channels)), stride, self.ps_shakedrop[self.idx]))
            self.in_channels = int(round(self.out_channels) * block.outchannel_ratio)
            self.idx = self.idx + 1
            stride = 1

        return nn.Sequential(*layers)

    def _make_MLP(self, args):
        layers = []
        in_channels = self.numFeatureDim
        out_channels = [args.numNeuron, args.numNeuron]
        for x in out_channels:
            layers += [nn.Linear(in_channels, x, bias=False), nn.BatchNorm1d(x), nn.ReLU()]
            in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.MLP(x)
        return x

class Pyramid_ResNet_clf(nn.Module):
    def __init__(self, args, numClass):
        super(Pyramid_ResNet_clf, self).__init__()
        self.classifier = nn.Linear(args.numNeuron, numClass)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data, mean=0, std=args.init_std)
                m.bias.data.zero_()

    def forward(self, x):
        return self.classifier(x)
