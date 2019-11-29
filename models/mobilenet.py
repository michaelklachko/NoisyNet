import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from quant import QuantMeasure
from plot_histograms import get_layers, plot_layers



def _make_divisible(v, divisor, min_value=None):
    """
    It ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        self.padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, self.padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6(inplace=True)
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        if args.q_a > 0:
            self.quantize = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl / 100, debug=args.debug_quant)

    def forward(self, x):
        input = x
        if args.q_a > 0:# and self.kernel_size == 3
            input = self.quantize(input)
        x = self.conv(x)

        if args.plot:# and self.kernel_size == 3:
            get_layers(self.arrays, input, self.conv1.weight, x, stride=self.stride, layer='conv', basic=args.plot_basic, debug=args.debug)
            """
            with torch.no_grad():
                weight_sums = torch.abs(self.conv.weight).sum((1, 2, 3))
                w_pos = self.conv.weight.clone()
                w_pos[w_pos < 0] = 0
                w_neg = self.conv.weight.clone()
                w_neg[w_neg >= 0] = 0
                pos = F.conv2d(input, w_pos, stride=self.stride, padding=self.padding, groups=self.groups)
                neg = F.conv2d(input, w_neg, stride=self.stride, padding=self.padding, groups=self.groups)
                sep = torch.cat((neg, pos), 0)
                weight_sums_sep = torch.cat((w_pos.sum((1, 2, 3)), w_neg.sum((1, 2, 3))), 0)

                arrays.append([input.half()])
                arrays.append([self.conv.weight.half()])
                arrays.append([x.half()])
                arrays.append([sep.half()])
                
                arrays.append([weight_sums.half()])
                arrays.append([weight_sums_sep.half()])
            """

        if args.merge_bn:
            bias = self.bn.bias.view(1, -1, 1, 1) - self.bn.running_mean.data.view(1, -1, 1, 1) * \
                   self.bn.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn.running_var.data.view(1, -1, 1, 1) + args.eps)
            x = x + bias
        else:
            x = self.bn(x)
            bias = self.bn.bias

        if args.plot:# and self.kernel_size == 3:
            self.arrays.append([bias.half()])

        x = self.relu(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv1 = ConvBNReLU(inp, hidden_dim, kernel_size=1)
        self.conv2 = ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(oup)

        if args.q_a > 0:
            self.quantize1 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl / 100, debug=args.debug_quant)
            self.quantize2 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl / 100, debug=args.debug_quant)
            self.quantize3 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl / 100, debug=args.debug_quant)

    def forward(self, x):
        input = x
        if self.expand_ratio != 1:
            x = self.conv1(x)
        x = self.conv2(x)
        if args.q_a > 0:
            x = self.quantize3(x)
        x = self.conv3(x)

        if args.merge_bn:
            bias = self.bn.bias.view(1, -1, 1, 1) - self.bn.running_mean.data.view(1, -1, 1, 1) * \
                   self.bn.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn.running_var.data.view(1, -1, 1, 1) + args.eps)
            x = x + bias
        else:
            x = self.bn(x)
            #bias = self.bn.bias

        if self.use_res_connect:
            return x + input
        else:
            return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        self.arrays = []
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)

        self.drop1 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(self.last_channel, num_classes)

        if args.bn_out:
            self.bn_out = nn.BatchNorm1d(1000, track_running_stats=args.track_running_stats)

        if args.q_a > 0:
            self.quantize = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl / 100, debug=args.debug_quant)

        #self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes), )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, epoch=0, i=0, acc=0.0):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.drop1(x)
        if args.q_a > 0:
            x = self.quantize(x)
        x = self.fc1(x)

        if args.bn_out:
            x = self.bn_out(x)

        if args.plot:
            names = ['input', 'weights', 'vmm', 'vmm diff', 'bias', 'weight sums', 'weight sums diff']
            #names = ['input', 'weights', 'vmm', 'bias']
            print('\n\nPreparing arrays for plotting:\n')
            layers = []
            layer = []
            print('\n\nlen(arrays) // len(names):', len(self.arrays), len(names), len(self.arrays) // len(names), '\n\n')
            num_layers = len(self.arrays) // len(names)
            for k in range(num_layers):
                print('layer', k, names)
                for j in range(len(names)):
                    #print('\t', names[j])
                    layer.append([self.arrays[len(names)*k+j][0].detach().cpu().numpy()])
                layers.append(layer)
                layer = []

            print('\nPlotting {}\n'.format(names))
            var_ = ''#[np.prod(self.conv1.weight.shape[1:]), np.prod(self.conv2.weight.shape[1:]), np.prod(self.linear1.weight.shape[1:]), np.prod(self.linear2.weight.shape[1:])]
            var_name = ''
            best_acc = 64.2
            plot_layers(num_layers=len(layers), models=['plots/'], epoch=68, i=0, layers=layers,
                        names=names, var=var_name, vars=[var_], pctl=args.pctl, acc=best_acc, tag=args.tag)
            print('\n\nSaved plots to current dir\n\n')
            raise (SystemExit)

        return x

def mobilenet_v2(parameters):
    """MobileNetV2: Inverted Residuals and Linear Bottlenecks https://arxiv.org/abs/1801.04381"""
    global args
    args = parameters
    model = MobileNetV2()
    return model
