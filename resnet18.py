import argparse
import os
import math
import time
import warnings
import numpy as np
from datetime import datetime
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

import utils
from quantized_modules_clean import QuantMeasure
from plot_histograms import get_layers, plot_layers
#from mn import mobilenet_v2

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/data/imagenet/', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='dali: 10, dataparallel: 16')
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--tag', default='', type=str, metavar='PATH', help='tag')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--debug', dest='debug', action='store_true', help='debug')
parser.add_argument('--distort_w_test', dest='distort_w_test', action='store_true', help='distort weights during test')
parser.add_argument('--distort_w_train', dest='distort_w_train', action='store_true', help='distort weights during train')
parser.add_argument('--noise', default=0.1, type=float, help='mult weights by uniform noise with this range +/-')
parser.add_argument('--stochastic', default=0.5, type=float, help='stochastic uniform noise to add before rounding during quantization')
parser.add_argument('--step-after', default=30, type=int, help='reduce LR after this number of epochs')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--num_sims', default=1, type=int, help='number of simulations.')
parser.add_argument('--q_a', default=4, type=int, help='number of bits to quantize layer input')
parser.add_argument('--local_rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--act_max', default=0, type=float, help='clipping threshold for activations')
parser.add_argument('--eps', default=1e-7, type=float, help='epsilon to add to avoid dividing by zero')
parser.add_argument('--grad_clip', default=0, type=float, help='max value of gradients')
parser.add_argument('--q_scale', default=1, type=float, help='scale upper value of quantized tensor by this value')
parser.add_argument('--pctl', default=99.9, type=float, help='percentile to show when plotting')
parser.add_argument('--gpu', default=None, type=str, help='GPU to use, if None use all')

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--pretrained', dest='pretrained', action='store_true')
feature_parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
parser.set_defaults(pretrained=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--debug_quant', dest='debug_quant', action='store_true')
feature_parser.add_argument('--no-debug_quant', dest='debug_quant', action='store_false')
parser.set_defaults(debug_quant=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--dali', dest='dali', action='store_true')
feature_parser.add_argument('--no-dali', dest='dali', action='store_false')
parser.set_defaults(dali=True)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--dali_cpu', dest='dali_cpu', action='store_true')
feature_parser.add_argument('--no-dali_cpu', dest='dali_cpu', action='store_false')
parser.set_defaults(dali_cpu=True)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--merge_bn', dest='merge_bn', action='store_true')
feature_parser.add_argument('--no-merge_bn', dest='merge_bn', action='store_false')
parser.set_defaults(merge_bn=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--plot', dest='plot', action='store_true')
feature_parser.add_argument('--no-plot', dest='plot', action='store_false')
parser.set_defaults(plot=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--print_shapes', dest='print_shapes', action='store_true')
feature_parser.add_argument('--no-print_shapes', dest='print_shapes', action='store_false')
parser.set_defaults(print_shapes=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--plot_basic', dest='plot_basic', action='store_true')
feature_parser.add_argument('--no-plot_basic', dest='plot_basic', action='store_false')
parser.set_defaults(plot_basic=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--calculate_running', dest='calculate_running', action='store_true')
feature_parser.add_argument('--no-calculate_running', dest='calculate_running', action='store_false')
parser.set_defaults(calculate_running=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--plot_normalize', dest='plot_normalize', action='store_true')
feature_parser.add_argument('--no-plot_normalize', dest='plot_normalize', action='store_false')
parser.set_defaults(plot_normalize=False)

warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)

args = parser.parse_args()

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.dali:
    try:
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator
        from nvidia.dali.pipeline import Pipeline
        import nvidia.dali.ops as ops
        import nvidia.dali.types as types
    except ImportError:
        raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


    class HybridTrainPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
            super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
            self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=True)
            #let user decide which pipeline works him bets for RN version he runs
            dali_device = 'cpu' if dali_cpu else 'gpu'
            decoder_device = 'cpu' if dali_cpu else 'mixed'
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
            host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
            self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB, device_memory_padding=device_memory_padding,
                            host_memory_padding=host_memory_padding, random_aspect_ratio=[0.8, 1.25], random_area=[0.1, 1.0], num_attempts=100)
            self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
            #self.cmnp = ops.CropMirrorNormalize(device="gpu", output_dtype=types.FLOAT, output_layout=types.NCHW, crop=(crop, crop),
                        #image_type=types.RGB, mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
            self.cmnp = ops.CropMirrorNormalize(device="gpu", output_dtype=types.FLOAT, output_layout=types.NCHW, crop=(crop, crop),
                        image_type=types.RGB, mean=[0, 0, 0], std=[255, 255, 255])
            self.coin = ops.CoinFlip(probability=0.5)
            print('DALI "{0}" variant'.format(dali_device))

        def define_graph(self):
            rng = self.coin()
            self.jpegs, self.labels = self.input(name="Reader")
            images = self.decode(self.jpegs)
            images = self.res(images)
            output = self.cmnp(images.gpu(), mirror=rng)
            return [output, self.labels]


    class HybridValPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
            super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
            self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=False)
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
            self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
            #self.cmnp = ops.CropMirrorNormalize(device="gpu", output_dtype=types.FLOAT, output_layout=types.NCHW, crop=(crop, crop),
                        #image_type=types.RGB, mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
            self.cmnp = ops.CropMirrorNormalize(device="gpu", output_dtype=types.FLOAT, output_layout=types.NCHW, crop=(crop, crop),
                        image_type=types.RGB, mean=[0, 0, 0], std=[255, 255, 255])

        def define_graph(self):
            self.jpegs, self.labels = self.input(name="Reader")
            images = self.decode(self.jpegs)
            images = self.res(images)
            output = self.cmnp(images)
            return [output, self.labels]


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

        if args.merge_bn:
            bias = self.bn.bias.view(1, -1, 1, 1) - self.bn.running_mean.data.view(1, -1, 1, 1) * \
                   self.bn.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn.running_var.data.view(1, -1, 1, 1) + args.eps)
            x = x + bias
        else:
            x = self.bn(x)
            bias = self.bn.bias

        if args.plot:# and self.kernel_size == 3:
            arrays.append([bias.half()])
            arrays.append([weight_sums.half()])
            arrays.append([weight_sums_sep.half()])
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

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.drop1(x)
        if args.q_a > 0:
            x = self.quantize(x)
        x = self.fc1(x)

        if args.plot:
            names = ['input', 'weights', 'vmm', 'vmm diff', 'bias', 'weight sums', 'weight sums diff']
            #names = ['input', 'weights', 'vmm', 'bias']
            print('\n\nPreparing arrays for plotting:\n')
            layers = []
            layer = []
            print('\n\nlen(arrays) // len(names):', len(arrays), len(names), len(arrays) // len(names), '\n\n')
            num_layers = len(arrays) // len(names)
            for k in range(num_layers):
                print('layer', k, names)
                for j in range(len(names)):
                    #print('\t', names[j])
                    layer.append([arrays[len(names)*k+j][0].detach().cpu().numpy()])
                layers.append(layer)
                layer = []

            print('\nPlotting {}\n'.format(names))
            var_ = ''#[np.prod(self.conv1.weight.shape[1:]), np.prod(self.conv2.weight.shape[1:]), np.prod(self.linear1.weight.shape[1:]), np.prod(self.linear2.weight.shape[1:])]
            var_name = ''
            best_acc = 64.2
            plot_layers(num_layers=len(layers), models=['plotts/'], epoch=68, i=0, layers=layers,
                        names=names, var=var_name, vars=[var_], pctl=args.pctl, acc=best_acc, tag=args.tag)
            print('\n\nSaved plots to current dir\n\n')
            raise (SystemExit)

        return x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.layer1 = []
        self.layer2 = []

        if args.act_max > 0:
            self.relu = nn.Hardtanh(0.0, args.act_max, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if downsample is not None:
            ds_in, ds_out, ds_strides = downsample
            self.ds_strides = ds_strides
            self.conv3 = nn.Conv2d(ds_in, ds_out, kernel_size=1, stride=ds_strides, bias=False)
            self.bn3 = nn.BatchNorm2d(ds_out)
            self.layer3 = []

        if args.q_a > 0:
            self.quantize1 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl/100, debug=args.debug_quant)
            self.quantize2 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl/100, debug=args.debug_quant)

    def forward(self, x):
        '''[[self.input], [self.conv1.weight], [conv1_weight_sums], [conv1_weight_sums_sep], [conv1_weight_sums_blocked],
            [conv1_weight_sums_sep_blocked], [self.conv1_no_bias], [self.conv1_sep], [conv1_blocks], [conv1_sep_blocked]]'''
        if args.q_a > 0:
            x = self.quantize1(x)
        residual = x
        out = self.conv1(x)

        if args.plot:
            get_layers(arrays, x, self.conv1.weight, out, stride=self.stride, layer='conv', basic=args.plot_basic, debug=args.debug)

        if args.print_shapes:
            print('\nblock input:', x.shape)
            print('conv1:', out.shape)

        if args.merge_bn:
            bias = self.bn1.bias.view(1, -1, 1, 1) - self.bn1.running_mean.data.view(1, -1, 1, 1) * \
                   self.bn1.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn1.running_var.data.view(1, -1, 1, 1) + args.eps)
            out += bias
            if args.plot:
                arrays.append([bias.half()])
        else:
            out = self.bn1(out)

        out = self.relu(out)

        if args.q_a > 0:
            out = self.quantize2(out)

        conv2_input = out
        out = self.conv2(out)

        if args.plot:
            get_layers(arrays, conv2_input, self.conv2.weight, out, stride=1, layer='conv', basic=args.plot_basic, debug=args.debug)

        if args.print_shapes:
            print('conv2:', out.shape)

        if args.merge_bn:
            bias = self.bn2.bias.view(1, -1, 1, 1) - self.bn2.running_mean.data.view(1, -1, 1, 1) * \
                   self.bn2.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn2.running_var.data.view(1, -1, 1, 1) + args.eps)
            out += bias
            if args.plot:
                arrays.append([bias.half()])
        else:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.conv3(x)
            if args.print_shapes:
                print('conv3 (shortcut downsampling):', out.shape)
            if args.merge_bn:
                bias = self.bn3.bias.view(1, -1, 1, 1) - self.bn3.running_mean.data.view(1, -1, 1, 1) * \
                       self.bn3.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn3.running_var.data.view(1, -1, 1, 1) + args.eps)
                residual += bias
            else:
                residual = self.bn3(residual)

        out += residual
        if args.print_shapes:
            print('x + shortcut:', out.shape)

        out = self.relu(out)
        return out


class ResNet(nn.Module):

    inplanes = None

    def __init__(self, block, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if args.act_max > 0:
            self.relu = nn.Hardtanh(0.0, args.act_max, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if args.q_a > 0:
            self.quantize1 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl/100, debug=args.debug_quant)
            self.quantize2 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl/100, debug=args.debug_quant)

        self.layer1 = self._make_layer(block, 64)

        self.layer2 = self._make_layer(block, 128, stride=2)
        self.layer3 = self._make_layer(block, 256, stride=2)
        self.layer4 = self._make_layer(block, 512, stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            #downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes), )
            downsample = (self.inplanes, planes, stride)

        blocks = [block(self.inplanes, planes, stride, downsample), block(planes, planes)]
        self.inplanes = planes

        return nn.Sequential(*blocks)

    def forward(self, x, epoch=0, i=0, acc=0.0):
        if args.print_shapes:
            print('RGB input:', x.shape)
        if args.q_a > 0:
            x = self.quantize1(x)

        conv1_input = x

        x = self.conv1(x)
        if args.print_shapes:
            print('first conv:', x.shape)

        if args.plot:
            get_layers(arrays, conv1_input, self.conv1.weight, x, stride=2, padding=3, layer='conv', basic=args.plot_basic, debug=args.debug)

        if args.merge_bn:
            bias = self.bn1.bias.view(1, -1, 1, 1) - self.bn1.running_mean.data.view(1, -1, 1, 1) * \
                   self.bn1.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn1.running_var.data.view(1, -1, 1, 1) + args.eps)
            x += bias
            if args.plot:
                arrays.append([bias.half()])
        else:
            x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)
        if args.print_shapes:
            print('after max pooling:', x.shape)
        x = self.layer1(x)
        if args.print_shapes:
            print('\nDownsampling the input:')
        x = self.layer2(x)
        if args.print_shapes:
            print('\nDownsampling the input:')
        x = self.layer3(x)
        if args.print_shapes:
            print('\nDownsampling the input:')
        x = self.layer4(x)

        x = self.avgpool(x)

        if args.print_shapes:
            print('\nafter avg pooling:', x.shape)
        x = x.view(x.size(0), -1)
        if args.print_shapes:
            print('reshaped:', x.shape)
        if args.q_a > 0:
            x = self.quantize2(x)

        fc_input = x

        x = self.fc(x)

        if args.plot:
            get_layers(arrays, fc_input, self.fc.weight, x, layer='linear', basic=args.plot_basic, debug=args.debug)

        if args.merge_bn and args.plot:
            arrays.append([self.fc.bias.half()])

        if args.print_shapes:
            print('\noutput:', x.shape)

        if args.plot:
            if args.plot_basic:
                names = ['input', 'weights', 'vmm']
            else:
                names = ['input', 'weights', 'vmm', 'vmm diff', 'vmm blocked', 'vmm diff blocked', 'weight sums', 'weight sums diff', 'weight sums blocked', 'weight sums diff blocked']

            if args.merge_bn:
                names.append('bias')

            print('\n\nPreparing arrays for plotting:\n')
            layers = []
            layer = []
            print('\n\nlen(arrays) // len(names):', len(arrays), len(names), len(arrays) // len(names), '\n\n')
            num_layers = len(arrays) // len(names)
            for k in range(num_layers):
                print('layer', k, names)
                for j in range(len(names)):
                    #print('\t', names[j])
                    layer.append([arrays[len(names)*k+j][0].detach().cpu().numpy()])
                layers.append(layer)
                layer = []

            print('\nPlotting {}\n'.format(names))
            var_ = ''#[np.prod(self.conv1.weight.shape[1:]), np.prod(self.conv2.weight.shape[1:]), np.prod(self.linear1.weight.shape[1:]), np.prod(self.linear2.weight.shape[1:])]
            var_name = ''

            plot_layers(num_layers=len(layers), models=['plotts/'], epoch=epoch, i=i, layers=layers,
                        names=names, var=var_name, vars=[var_], pctl=args.pctl, acc=acc, tag=args.tag, normalize=args.plot_normalize)
            print('\n\nSaved plots to current dir\n\n')
            raise (SystemExit)

        return x


def validate(val_loader, model, epoch=0, plot_acc=0.0):
    model.eval()
    te_accs = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if args.dali:
                input = data[0]["data"]
                target = data[0]["label"].squeeze().cuda().long()
                input_var = Variable(input)
                output = model(input_var, epoch=epoch, i=i, acc=plot_acc)
            else:
                images, target = data
                target = target.cuda(non_blocking=True)
                output = model(images, epoch=epoch, i=i, acc=plot_acc)
            if i == 0:
                args.print_shapes = False
            acc = accuracy(output, target)
            te_accs.append(acc)

        mean_acc = np.mean(te_accs)
        print('\n{}\tEpoch {:d}  Validation Accuracy: {:.2f}\n'.format(str(datetime.now())[:-7], epoch, mean_acc))

    return mean_acc


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // args.step_after))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        pred = output.data.max(1)[1]
        acc = pred.eq(target.data).sum().item() * 100.0 / batch_size
        return acc


if args.pretrained or args.resume:
    print("\n\n\tLoading pre-trained {}\n\n".format(args.arch))
else:
    print("\n\n\tTraining {}\n\n".format(args.arch))

if args.arch == 'mobilenet_v2':
    #model = models.mobilenet_v2(pretrained=args.pretrained)
    model = MobileNetV2()
else:
    model = ResNet(BasicBlock)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))

model = torch.nn.DataParallel(model).cuda()
model = model
if args.resume is not None:
    utils.print_model(model, args)
criterion = nn.CrossEntropyLoss(reduction='mean').cuda()
optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

best_acc = 0
arrays = []

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        #model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})\n".format(args.resume, checkpoint['epoch']))
        utils.print_model(model, args, full=False)
        print('\n\n')
        for name, param in model.state_dict().items():
            print(name)
        print('\n\n')
        for saved_name, saved_param in checkpoint['state_dict'].items():
            matched = False
            if args.debug:
                print(saved_name)
            for name, param in model.named_parameters():
                if name == saved_name:
                    #print('1')
                    matched = True
                    if args.debug:
                        print('\tmatched, copying...')
                    param.data = saved_param.data
            if 'running' in saved_name and 'bn' in saved_name:  #batchnorm stats are not in named_parameters
                #print('2')
                matched = True
                #print('\n')
                #if 'bn' not in saved_name:
                #print(saved_name, saved_param)
                if args.debug:
                    print('\tmatched, copying...')
                m = model.state_dict()
                #print('\nmodel.state_dict:')
                #for name, p in m.items():
                    #print(name)
                m.update({saved_name: saved_param})
                model.load_state_dict(m)
            if args.q_a > 0 and ('quantize1' in saved_name or 'quantize2' in saved_name):
                matched = True
                if args.debug:
                    print('\tmatched, copying...')
                m = model.state_dict()
                m.update({saved_name: saved_param})
                model.load_state_dict(m)
            elif not matched:
                #pass
                print('\t\t\t************ Not copying', saved_name)

        print('\n\nCurrent model')
        for name, param in model.state_dict().items():
            print(name)
        print('\n\n')

        print('\n\ncheckpoint:\n\n')
        for name, param in checkpoint['state_dict'].items():
            print(name)
        print('\n\n')

        #model.load_state_dict(checkpoint['state_dict'])

        if args.merge_bn:
            print('\n\nMerging batchnorm into weights...\n\n')
            if args.arch == 'mobilenet_v2':
                for name, param in model.state_dict().items():  #model.named_parameters():
                    if args.debug:
                        print('\n', name)
                        if name == 'module.features.15.conv2.conv.weight':
                            print('\n\nBefore:\n', param[0, :10])
                    num = name.split('.')[2]
                    if num == '0' or num == '18' or 'conv3' in name:
                        bn_prefix = name.split('.')[:3]
                        bn_prefix = '.'.join(bn_prefix)
                        bn_weight = bn_prefix + '.bn.weight'
                        bn_running_var = bn_prefix + '.bn.running_var'
                    elif 'conv1' in name or 'conv2' in name:
                        bn_prefix = name.split('.')[:4]
                        bn_prefix = '.'.join(bn_prefix)
                        bn_weight = bn_prefix + '.bn.weight'
                        bn_running_var = bn_prefix + '.bn.running_var'
                    if 'conv' in name and 'bn' not in name and 'quantize' not in name:
                        if args.debug:
                            print('bn_prefix', bn_prefix)
                            print('bn_weight', bn_weight)
                            print('bn_running_var', bn_running_var)
                            print(param.data.shape)
                            print(model.state_dict()[bn_weight].data.shape)
                            print(model.state_dict()[bn_running_var].data.shape)
                        #print('model.state_dict()[bn_weight]', model.state_dict()[bn_weight].shape)
                        #print('model.state_dict()[bn_running_var]', model.state_dict()[bn_running_var].shape)
                        param.data *= model.state_dict()[bn_weight].data.view(-1, 1, 1, 1) / \
                                      torch.sqrt(model.state_dict()[bn_running_var].data.view(-1, 1, 1, 1) + args.eps)
                        if name == 'module.features.15.conv2.conv.weight' and args.debug:
                            print('\n\nAfter:\n', param[0, :10])#, model.module.features.15.conv2.conv.weight[0, :10])
            else:
                for name, param in model.state_dict().items():  #model.named_parameters():
                    if name == 'module.conv1.weight':
                        if args.debug:
                            print(name)
                            print('\n\nBefore:\n', model.module.conv1.weight[0,0,0])
                        bn_weight = 'module.bn1.weight'
                        bn_running_var = 'module.bn1.running_var'
                    elif 'conv' in name:
                        if args.debug:
                            print(name)
                        bn_prefix = name[:16]
                        if args.debug:
                            print('bn_prefix', bn_prefix)
                        bn_num = name[20]
                        if args.debug:
                            print('bn_num', bn_num)
                        bn_weight = bn_prefix + 'bn' + bn_num + '.weight'
                        if args.debug:
                            print('bn_weight', bn_weight)
                        bn_running_var = bn_prefix + 'bn' + bn_num + '.running_var'
                        if args.debug:
                            print('bn_running_var', bn_running_var)
                    elif 'downsample.0' in name:
                        bn_prefix = name[:16]
                        bn_weight = bn_prefix + 'downsample.1.weight'
                        bn_running_var = bn_prefix + 'bn' + bn_num + '.running_var'
                    if 'conv' in name or 'downsample.0' in name:
                        param.data *= model.state_dict()[bn_weight].data.view(-1, 1, 1, 1) / \
                                      torch.sqrt(model.state_dict()[bn_running_var].data.view(-1, 1, 1, 1) + args.eps)
                    if name == 'module.conv1.weight':
                        if args.debug:
                            print('\n\nAfter:\n', model.module.conv1.weight[0, 0, 0])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')

if args.dali:
    pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=traindir, crop=224, dali_cpu=args.dali_cpu)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=valdir, crop=224, size=256)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

else:
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(256),
                    transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

if args.evaluate:
    if args.arch == 'mobilenet_v2' and args.pretrained:
        b_acc = 0.0
        b_epoch = 0
    else:
        b_acc = checkpoint['best_acc']
        b_epoch = checkpoint['epoch']
    print('\n\nTesting accuracy on validation set (should be {:.2f})...\n'.format(b_acc))
    if args.distort_w_test:
        orig_m = copy.deepcopy(model.state_dict())
        acc_d = []
        vars = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
        for args.noise in vars:
            print('\n\nDistorting weights by {}%\n\n'.format(args.noise * 100))
            te_acc_dists = []
            orig_params = []

            if args.debug:
                print('\n\nbefore:\n{}\n'.format(model.module.conv1.weight.data.detach().cpu().numpy()[0, 0, 0]))

            #for n, p in model.named_parameters():
                #orig_params.append(p.clone())


            for s in range(args.num_sims):
                te_accuracies_dist = []
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if ('conv' in n or 'fc' in n or 'classifier' in n) and 'weight' in n:
                            if False and s == 0 and args.noise == vars[0]:
                                print(n)
                            p_noise = torch.cuda.FloatTensor(p.size()).uniform_(1. - args.noise, 1. + args.noise)
                            if args.debug and n == 'module.conv1.weight':
                                print('\n\np_noise:\n{}\n'.format(p_noise.detach().cpu().numpy()[0, 0, 0]))
                            p.data.mul_(p_noise)
                        elif 'bn' in n:
                            pass
                            #print('\n\n{}\n{}\n'.format(n, p))

                te_acc_d = validate(val_loader, model)
                te_accuracies_dist.append(te_acc_d.item())

                te_acc_dist = np.mean(te_accuracies_dist)
                te_acc_dists.append(te_acc_dist)

                if args.debug:
                    print('after:\n{}\n'.format(model.module.conv1.weight.data.detach().cpu().numpy()[0, 0, 0]))

                #for (n, p), orig_p in zip(model.named_parameters(), orig_params):
                    #p.data = orig_p.clone().data
                model.load_state_dict(orig_m)

                if args.debug:
                    print('restored:\n{}\n'.format(model.module.conv1.weight.data.detach().cpu().numpy()[0, 0, 0]))

                if args.dali:
                    val_loader.reset()

            avg_te_acc_dist = np.mean(te_acc_dists)
            acc_d.append(avg_te_acc_dist)
            print('\nNoise {:4.2f}: acc {:.2f}\n'.format(args.noise, avg_te_acc_dist))
        print('\n\n{}\n{}\n\n\n'.format(vars, acc_d))
        raise(SystemExit)
    else:
        validate(val_loader, model, epoch=b_epoch, plot_acc=b_acc)
        raise(SystemExit)

#for param_group in optimizer.param_groups:
    #print('wd:', param_group['weight_decay'])

for epoch in range(args.start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch, args)
    #for param_group in optimizer.param_groups:
        #param_group['lr'] = args.lr
        #param_group['weight_decay'] = args.weight_decay
    #print(optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['weight_decay'])
    model.train()
    tr_accs = []

    for i, data in enumerate(train_loader):
        if args.dali:
            input = data[0]["data"]
            target = data[0]["label"].squeeze().cuda().long()
            train_loader_len = int(train_loader._size / args.batch_size)
            input_var = Variable(input)
            target_var = Variable(target)
            output = model(input_var, epoch=epoch, i=i)
            loss = criterion(output, target_var)
        else:
            images, target = data
            train_loader_len = len(train_loader)
            target = target.cuda(non_blocking=True)
            output = model(images, epoch=epoch, i=i)
            loss = criterion(output, target)

        if i == 0:
            args.print_shapes = False
        acc = accuracy(output, target)
        tr_accs.append(acc)
        optimizer.zero_grad()
        loss.backward()

        if args.grad_clip > 0:
            for n, p in model.named_parameters():
                #if p.grad.data.max().item() > 1:
                    #print(i, n, p.grad.data.max().item())
                p.grad.data.clamp_(-args.grad_clip, args.grad_clip)

        optimizer.step()

        if i % args.print_freq == 0:
            print('{}  Epoch {:>2d} Batch {:>4d}/{:d} LR {} | {:.2f}'.format(
                str(datetime.now())[:-7], epoch, i, train_loader_len, optimizer.param_groups[0]["lr"], np.mean(tr_accs)))

        if args.calculate_running and i == 0:
            #torch.save({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(),
                        #'best_acc': best_acc, 'optimizer': optimizer.state_dict()}, args.tag + 'chkpt.pth')
            torch.save({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(),
                         'best_acc': 69.1, 'optimizer': optimizer.state_dict()}, args.tag + '.pth')
            raise(SystemExit)

    acc = validate(val_loader, model, epoch=epoch)
    if acc > best_acc:
        best_acc = acc
        if args.distort_w_train:
            tag = args.tag + 'noise_{:.2f}_'.format(args.noise)
        else:
            tag = args.tag
        torch.save({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(), 'best_acc': best_acc, 'optimizer': optimizer.state_dict()}, tag + '.pth')

    if args.dali:
        train_loader.reset()
        val_loader.reset()

print('\n\nBest Accuracy {:.2f}\n\n'.format(best_acc))
