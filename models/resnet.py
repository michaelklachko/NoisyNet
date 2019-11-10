import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from quant import QuantMeasure
from plot_histograms import get_layers, plot_layers
from hardware_model import add_noise_calculate_power, NoisyConv2d, NoisyLinear, QuantMeasure
import scipy.io


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        if args.act_max > 0:
            self.relu = nn.Hardtanh(0.0, args.act_max, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = NoisyConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, num_bits=0, num_bits_weight=args.q_w,
                                 clip=0, noise=args.n_w, test_noise=args.n_w_test, stochastic=args.stochastic, debug=args.debug_noise)
        if args.old_checkpoint:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = NoisyConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, num_bits=0, num_bits_weight=args.q_w,
                                 clip=0, noise=args.n_w, test_noise=args.n_w_test, stochastic=args.stochastic, debug=args.debug_noise)
        if not args.old_checkpoint:
            self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if downsample is not None:
            ds_in, ds_out, ds_strides = downsample
            self.ds_strides = ds_strides
            #self.conv3 = nn.Conv2d(ds_in, ds_out, kernel_size=1, stride=ds_strides, bias=False)
            self.conv3 = NoisyConv2d(ds_in, ds_out, kernel_size=1, stride=ds_strides, bias=False, num_bits=0, num_bits_weight=args.q_w,
                                     clip=0, noise=args.n_w, test_noise=args.n_w_test, stochastic=args.stochastic, debug=args.debug_noise)
            self.bn3 = nn.BatchNorm2d(ds_out)
            self.layer3 = []

        if args.q_a > 0:
            self.quantize1 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl/100, debug=args.debug, inplace=args.q_inplace)
            self.quantize2 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl/100, debug=args.debug, inplace=args.q_inplace)

    def forward(self, x):
        '''[[self.input], [self.conv1.weight], [conv1_weight_sums], [conv1_weight_sums_sep], [conv1_weight_sums_blocked],
            [conv1_weight_sums_sep_blocked], [self.conv1_no_bias], [self.conv1_sep], [conv1_blocks], [conv1_sep_blocked]]'''
        if args.q_a > 0:
            x = self.quantize1(x)

        if args.distort_act:
            with torch.no_grad():
                x_noise = x * torch.cuda.FloatTensor(x.size()).uniform_(-args.noise, args.noise)
                x = x + x_noise

        residual = x
        out = self.conv1(x)

        if args.plot:
            get_layers(arrays, x, self.conv1.weight, out, stride=self.stride, layer='conv', basic=args.plot_basic, debug=args.debug)

        if args.print_shapes:
            print('\nblock input:', list(x.shape))
            print('conv1:', list(out.shape))

        if args.merge_bn:
            bias = self.bn1.bias.view(1, -1, 1, 1) - self.bn1.running_mean.data.view(1, -1, 1, 1) * \
                   self.bn1.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn1.running_var.data.view(1, -1, 1, 1) + args.eps)
            out += bias
            if args.plot:
                arrays.append([bias.half().detach().cpu().numpy()])
        else:
            out = self.bn1(out)

        if args.plot:
            arrays.append([out.half().detach().cpu().numpy()])

        out = self.relu(out)

        if args.q_a > 0:
            out = self.quantize2(out)

        if args.distort_act:
            with torch.no_grad():
                out_noise = out * torch.cuda.FloatTensor(out.size()).uniform_(-args.noise, args.noise)
                out = out + out_noise

        conv2_input = out
        out = self.conv2(out)

        if args.plot:
            get_layers(arrays, conv2_input, self.conv2.weight, out, stride=1, layer='conv', basic=args.plot_basic, debug=args.debug)

        if args.print_shapes:
            print('conv2:', list(out.shape))

        if args.merge_bn:
            bias = self.bn2.bias.view(1, -1, 1, 1) - self.bn2.running_mean.data.view(1, -1, 1, 1) * \
                   self.bn2.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn2.running_var.data.view(1, -1, 1, 1) + args.eps)
            out += bias
            if args.plot:
                arrays.append([bias.half().detach().cpu().numpy()])
        else:
            out = self.bn2(out)

        if args.plot:
            arrays.append([out.half().detach().cpu().numpy()])

        if self.downsample is not None:
            residual = self.conv3(x)
            if args.print_shapes:
                print('conv3 (shortcut downsampling):', list(out.shape))
            if args.merge_bn:
                bias = self.bn3.bias.view(1, -1, 1, 1) - self.bn3.running_mean.data.view(1, -1, 1, 1) * \
                       self.bn3.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn3.running_var.data.view(1, -1, 1, 1) + args.eps)
                residual += bias
            else:
                residual = self.bn3(residual)

        out += residual
        if args.print_shapes:
            print('x + shortcut:', list(out.shape))

        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_classes=1000):
        self.inplanes = 64
        global arrays  # for plotting
        arrays = []
        super(ResNet, self).__init__()
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = NoisyConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, num_bits=0, num_bits_weight=args.q_w,
                                 clip=0, noise=args.n_w, test_noise=args.n_w_test, stochastic=args.stochastic, debug=args.debug_noise)
        self.bn1 = nn.BatchNorm2d(64)
        if args.act_max > 0:
            self.relu = nn.Hardtanh(0.0, args.act_max, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if args.q_a_first > 0:  #when quantizing, if q_a_first is not specified, set it to 6 bits
            self.q_a_first = args.q_a_first
        elif args.q_a > 0:
            self.q_a_first = 6
        elif args.q_a_first == 8:
            self.q_a_first = 0
        else:
            self.q_a_first = 0

        if self.q_a_first > 0:
            self.quantize1 = QuantMeasure(self.q_a_first, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl / 100, debug=args.debug, inplace=args.q_inplace)
        if args.q_a > 0:
            self.quantize2 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl/100, debug=args.debug, inplace=args.q_inplace)

        self.layer1 = self._make_layer(block, 64)
        self.layer2 = self._make_layer(block, 128, stride=2)
        self.layer3 = self._make_layer(block, 256, stride=2)
        self.layer4 = self._make_layer(block, 512, stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512, num_classes)
        self.fc = NoisyLinear(512, num_classes, bias=True, num_bits=0, num_bits_weight=args.q_w,
                                   clip=0, noise=args.n_w, test_noise=args.n_w_test, stochastic=args.stochastic, debug=args.debug_noise)

        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                #print('is instance of nn.Conv2D\n')
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
            print('RGB input:', list(x.shape))
        if self.q_a_first > 0:
            x = self.quantize1(x)

        if False and args.distort_act:
            with torch.no_grad():
                x_noise = x * torch.cuda.FloatTensor(x.size()).uniform_(-args.noise, args.noise)
                x = x + x_noise

        conv1_input = x

        x = self.conv1(x)
        if args.print_shapes:
            print('first conv:', list(x.shape))

        if args.plot:
            get_layers(arrays, conv1_input, self.conv1.weight, x, stride=2, padding=3, layer='conv', basic=args.plot_basic, debug=args.debug)

        if args.merge_bn:
            bias = self.bn1.bias.view(1, -1, 1, 1) - self.bn1.running_mean.data.view(1, -1, 1, 1) * \
                   self.bn1.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn1.running_var.data.view(1, -1, 1, 1) + args.eps)
            x += bias
            if args.plot:
                arrays.append([bias.half().detach().cpu().numpy()])
        else:
            x = self.bn1(x)

        if args.plot:
            arrays.append([x.half().detach().cpu().numpy()])

        x = self.relu(x)
        x = self.maxpool(x)
        if args.print_shapes:
            print('after max pooling:', list(x.shape))
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
            print('\nafter avg pooling:', list(x.shape))
        x = x.view(x.size(0), -1)
        if args.print_shapes:
            print('reshaped:', list(x.shape))
        if args.q_a > 0:
            x = self.quantize2(x)

        if args.distort_act:
            with torch.no_grad():
                #print('\n\nBefore\n{}\n'.format(x[0,:10]))
                x_noise = x * torch.cuda.FloatTensor(x.size()).uniform_(-args.noise, args.noise)
                x = x + x_noise
                #print('After\n{}\n'.format(x[0, :10]))

        fc_input = x

        x = self.fc(x)

        if args.plot:
            get_layers(arrays, fc_input, self.fc.weight, x, layer='linear', basic=args.plot_basic, debug=args.debug)

        if args.merge_bn and args.plot:
            arrays.append([self.fc.bias.half().detach().cpu().numpy()])

        if args.print_shapes:
            print('\noutput:', list(x.shape))

        if args.plot:
            arrays.append([x.half().detach().cpu().numpy()])

        if args.plot:
            if args.plot_basic:
                names = ['input', 'weights', 'vmm']
            else:
                names = ['input', 'weights', 'vmm', 'vmm diff', 'vmm blocked', 'vmm diff blocked', 'weight sums', 'weight sums diff', 'weight sums blocked', 'weight sums diff blocked']
                names = ['input', 'weights', 'vmm', 'source']

            if args.merge_bn:
                names.append('bias')

            names.append('pre-activations')

            print('\n\nPreparing arrays for plotting:\n')

            layers = []
            layer = []
            print('\n\nlen(arrays) // len(names):', len(arrays), len(names), len(arrays) // len(names), '\n\n')
            num_layers = len(arrays) // len(names)
            for k in range(num_layers):
                print('layer', k, names)
                for j in range(len(names)):
                    #print('\t', names[j])
                    layer.append([arrays[len(names)*k+j][0]])
                layers.append(layer)
                layer = []

            var_ = ''
            var_name = ''
            #scipy.io.savemat('chip_plots/r18_first_layer_q4_act_4_acc_{:.2f}.mat'.format(acc), mdict={names[1]: arrays[1], names[2]: arrays[2]})
            #raise(SystemExit)
            plot_layers(num_layers=len(layers), models=['plots/'], epoch=epoch, i=i, layers=layers,
                        names=names, var=var_name, vars=[var_], pctl=args.pctl, acc=acc, tag=args.tag, normalize=args.normalize)
            raise (SystemExit)

        return x

def ResNet18(parameters):
    global args
    args = parameters
    model = ResNet(BasicBlock)
    return model
