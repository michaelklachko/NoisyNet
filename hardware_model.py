import torch
from torch import nn
from torch.autograd.function import InplaceFunction, Function
import random
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from plot_histograms import plot

# random.seed(1)
# torch.manual_seed(1)
# torch.backends.cudnn.deterministic = True


def add_noise_calculate_power(self, args, arrays, input, weights, output, layer_type='conv', i=0, layer_num=0, merged_dac=True):
    if args.distort_act:
        with torch.no_grad():
            noise = output * torch.cuda.FloatTensor(output.size()).uniform_(-args.noise, args.noise)
        return output + noise

    #merged_dac = True
    with torch.no_grad():
        if (args.uniform_ind > 0 and self.training) or (args.uniform_ind > 0 and args.noise_test):
            sigmas = torch.ones_like(output) * args.uniform_ind * torch.max(torch.abs(output))
            noise_distr = Uniform(-sigmas, sigmas)
            noise = noise_distr.sample()

        elif (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
            noise_distr = Uniform(torch.ones_like(output) * args.uniform_dep, torch.ones_like(output) / args.uniform_dep)
            noise = noise_distr.sample()

        elif (args.normal_ind > 0 and self.training) or (args.normal_ind > 0 and args.noise_test):
            sigmas = (torch.ones_like(output) * args.normal_ind * torch.max(torch.abs(output))).pow(2)
            noise_distr = Normal(loc=0, scale=torch.ones_like(output) * args.normal_ind * torch.max(torch.abs(output)))
            noise = noise_distr.sample()

        elif (args.normal_dep > 0 and self.training) or (args.normal_dep > 0 and args.noise_test):
            sigmas = (args.normal_dep * output).pow(2)
            noise_distr = Normal(loc=0, scale=args.normal_dep * output)
            noise = noise_distr.sample()

        else:
            abs_weights = torch.abs(weights)
            input_max = torch.max(input)  # always 1 for RGB input, unless < 5 bits Imagenet.
            if merged_dac:  # merged DAC digital input (for the current chip - first and third layer input):
                w_max = torch.max(abs_weights)
                if layer_type == 'conv':
                    sigmas = F.conv2d(input, abs_weights)
                    dim = (1, 2, 3)
                elif layer_type == 'linear':
                    sigmas = F.linear(input, abs_weights, bias=None)
                    dim = 1

                if i < 20:
                    sample_sums = torch.sum(sigmas, dim=dim)
                    p = 1.0e-6 * 1.2 * args.layer_currents[layer_num] * torch.mean(sample_sums) / (input_max * w_max)

                noise_distr = Normal(loc=0, scale=torch.sqrt(0.1 * (w_max / args.layer_currents[layer_num]) * sigmas))

            else:  # external DAC (for the next gen hardware) or analog input in the current chip (layers 2 and 4)
                abs_w_squared = abs_weights.pow(2) + abs_weights

                if layer_type == 'conv':
                    sigmas_w_squared = F.conv2d(input, abs_w_squared)
                    dim = (1, 2, 3)
                    if i < 20:
                        sigmas = F.conv2d(input, abs_weights)

                elif layer_type == 'linear':
                    sigmas_w_squared = F.linear(input, abs_w_squared, bias=None)
                    dim = 1

                    if i < 20:
                        sigmas = F.linear(input, abs_weights, bias=None)

                if i < 20:
                    sample_sums = torch.sum(sigmas, dim=dim)
                    p = 1.0e-6 * 1.2 * args.layer_currents[layer_num] * torch.mean(sample_sums) / input_max

                noise_distr = Normal(loc=0, scale=torch.sqrt(0.1 * (input_max / args.layer_currents[layer_num]) * sigmas_w_squared))

            noise = noise_distr.sample()

            if i < 20:
                self.power[layer_num].append(p.item())
                self.nsr[layer_num].append(torch.mean(torch.abs(noise) / torch.max(output)).item())
                self.input_sparsity[layer_num].append(input[input > 0].numel() / input.numel())

    if (args.plot or args.write):
        if merged_dac:
            if args.plot_noise:
                arrays += ([sigmas.half()], [noise.half()])

                clipped_range = np.percentile(output.detach().cpu().numpy(), 99) - np.percentile(output.detach().cpu().numpy(), 1)
                if clipped_range == 0:
                    print('\n\n***** np.percentile(output, 99) = np.percentile(output, 1) *****\n\n')
                    raise (SystemExit)
                    # clipped_range = max(np.max(output) / 100., 1)
                nsr = noise / clipped_range
                arrays.append([nsr.half()])

                print('adding sigmas and noise and snr, len(arrays):', len(arrays))
            if args.plot_power:
                arrays.append([(sigmas / (input_max * w_max)).half()])
                print('adding power, len(arrays):', len(arrays))
        else:
            if args.plot_noise:
                arrays += ([sigmas_w_squared.half()], [noise.half()])

                clipped_range = np.percentile(output.detach().cpu().numpy(), 99) - np.percentile(output.detach().cpu().numpy(), 1)
                if clipped_range == 0:
                    print('\n\n***** np.percentile(output, 99) = np.percentile(output, 1) *****\n\n')
                    raise (SystemExit)
                    # clipped_range = max(np.max(output) / 100., 1)
                nsr = noise / clipped_range
                arrays.append([nsr.half()])

            if args.plot_power:
                arrays.append([(sigmas / input_max).half()])

    if (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
        noisy_out = output * noise.cuda()
    else:
        noisy_out = output + noise.cuda()

    return noisy_out


class UniformQuantize(InplaceFunction):
    """modified from https://github.com/eladhoffer/quantized.pytorch/blob/master/models/modules/quantize.py"""

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, stochastic=0.5, inplace=False, debug=False):
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic
        ctx.save_for_backward(input)

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        qmin = 0.
        qmax = 2. ** num_bits - 1.
        scale = (max_value - min_value) / (qmax - qmin)
        scale = max(scale, 1e-6)  # TODO figure out how to set this robustly! causes nans

        with torch.no_grad():
            output.add_(-min_value).div_(scale).add_(qmin)
            if debug:
                print('\nnum_bits {:d} qmin {} qmax {} min_value {} max_value {} actual max value {}'.format(num_bits, qmin, qmax, min_value, max_value,
                                                                                                             input.max()))
                print('\ninitial input\n', input[0, 0])
                print('\nnormalized input\n', output[0, 0])
            if ctx.stochastic > 0:
                noise = output.new(output.shape).uniform_(-ctx.stochastic, ctx.stochastic)
                output.add_(noise)
                if debug:
                    print('\nadding noise (stoch={:.1f})\n{}\n'.format(ctx.stochastic, output[0, 0]))

            output.clamp_(qmin, qmax).round_()  # quantize
            if debug:
                print('\nquantized\n', output[0, 0])

            output.add_(-qmin).mul_(scale).add_(min_value)  # dequantize
        if debug:
            print('\ndenormalized output\n', output[0, 0])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Saturated Straight Through Estimator
        input, = ctx.saved_tensors
        # Should we clone the grad_output???
        grad_output[input > ctx.max_value] = 0
        grad_output[input < ctx.min_value] = 0
        # grad_input = grad_output
        return grad_output, None, None, None, None, None, None


class QuantMeasure(nn.Module):
    """
    https://arxiv.org/abs/1308.3432
    https://arxiv.org/abs/1903.05662
    https://arxiv.org/abs/1903.01061
    https://arxiv.org/abs/1906.03193
    https://github.com/cooooorn/Pytorch-XNOR-Net/blob/master/util/util.py
    https://github.com/jiecaoyu/XNOR-Net-PyTorch/blob/master/ImageNet/networks/util.py
    https://github.com/Wizaron/binary-stochastic-neurons/blob/master/utils.py
    https://github.com/penhunt/full-quantization-DNN/blob/master/nets/quant_uni_type.py
    https://github.com/salu133445/bmusegan/blob/master/musegan/utils/ops.py

    Calculate_running indicates if we want to calculate the given percentile of signals to use as a max_value for quantization range
    if True, we will calculate pctl for several batches (only on training set), and use the average as a running_max, which will became max_value
    if False we will either use self.max_value (if given), or self.running_max (previously calculated)

    Currently, calculate_running param is set in the training code, and NOT passed as an argument - TODO need to fix that

    If using dropout, during training the activations are divided by 1-p. Multiply calculate_running by 1-p during test.
    """

    def __init__(self, num_bits=8, momentum=0.0, stochastic=0.5, min_value=0., max_value=0., scale=1,
                 calculate_running=False, pctl=90., debug=False, inplace=False):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros([]))
        self.momentum = momentum
        self.num_bits = num_bits
        self.stochastic = stochastic
        self.inplace = inplace
        self.debug = debug
        self.max_value = max_value
        self.min_value = min_value
        self.scale = scale
        self.calculate_running = calculate_running
        self.running_list = []
        self.pctl = pctl
        if pctl < 1:
            print('\n\npctl is {} please check!!!\n\n\n'.format(pctl))
            raise(SystemExit)

    def forward(self, input):
        # max_value_their = input.detach().contiguous().view(input.size(0), -1).max(-1)[0].mean()
        with torch.no_grad():
            min_value = self.min_value
            if self.calculate_running:
                if not self.training:
                    print('\n\n\nCalculating running min/max during inference\n\n\n')
                if self.min_value < 0:  # quantizing weights (ReLU is always positive)
                    pctl_pos, _ = torch.kthvalue(input[input > 0].flatten(), int(input[input > 0].numel() * self.pctl / 100.))
                    pctl_neg, _ = torch.kthvalue(torch.abs(input[input < 0]).flatten(), int(input[input < 0].numel() * self.pctl / 100.))
                    self.running_min = -pctl_neg
                    self.running_max = pctl_pos
                    self.calculate_running = False
                    min_value = self.running_min.item()
                    max_value = self.running_max.item()
                else:
                    if 224 in list(input.shape):  # first layer input (Imagenet) needs more precision (at least 6 bits)
                        if self.num_bits == 4:
                            pctl = torch.tensor(0.92)  # args.q_a_first == 4
                        else:
                            pctl = torch.tensor(1.0)
                    else:
                        #print('\nlen(input)', input.numel(), 'pctl', self.pctl, 'self.pctl / 100.', self.pctl / 100., 'int(input.numel() * self.pctl / 100.)',
                              #int(input.numel() * self.pctl / 100.), 'max', input.max().item(), 'min', input.min().item())
                        pctl, _ = torch.kthvalue(input.flatten(), int(input.numel() * self.pctl / 100.))
                    # print('input.shape', input.shape, 'pctl.shape', pctl.shape)
                    # self.running_max = pctl
                    # max_value = pctl
                    max_value = input.max().item()  #use actual max value until done calculating running_max (few iterations)
                    # raise(SystemExit)
                    self.running_list.append(pctl)  # self.running_max)
                    # self.running_max.mul_(self.momentum).add_(max_value * (1 - self.momentum))
                    if self.debug:
                        print('{} gpu {} self.calculate_running {}  max value (pctl/actual) {:.3f}/{:.1f}'.format(
                            list(input.shape), torch.cuda.current_device(), self.calculate_running, pctl.item(), input.max().item()))
            else:
                if self.debug:
                    pctl, _ = torch.kthvalue(input.flatten(), int(input.numel() * self.pctl / 100.))
                    print('{} gpu {} self.calculate_running {}  max value (pctl/actual) {:.3f}/{:.1f}'.format(
                            list(input.shape), torch.cuda.current_device(), self.calculate_running, pctl.item(), input.max().item()))
                if self.min_value < 0 and self.running_min < 0:
                    min_value = self.running_min.item()
                    max_value = self.running_max.item()
                elif self.running_max.item() > 0:
                    max_value = self.running_max.item()
                elif self.max_value > 0:
                    max_value = self.max_value
                else:
                    print('\n\nSetting max_value to input.max\nrunning_max is ', self.running_max.item())
                    max_value = input.max().item()

            if self.training:
                stoch = self.stochastic
            else:
                stoch = 0

        return UniformQuantize().apply(input, self.num_bits, min_value, max_value, stoch, self.inplace, False)


class AddNoise(InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, noise=0, debug=False):
        output = input.clone()
        with torch.no_grad():
            # unoise = output * torch.cuda.FloatTensor(output.size()).uniform_(-noise, noise)
            unoise = output * output.new_empty(output.shape).uniform_(-noise, noise)
            output.add_(unoise)
            if debug:
                print('\n\nAdded {:d}% of noise\n\nBefore:\n{}\nNoise:\n{}\nAfter:\n{}\n\n'.format(int(noise * 100), input[0, 0], noise[0, 0], output[0, 0]))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None


class NoisyConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 num_bits=0, num_bits_weight=0, noise=0.5, test_noise=0, stochastic=True, debug=False):
        super(NoisyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.fms = out_channels
        self.fs = kernel_size
        self.noise = noise
        self.num_bits_weight = num_bits_weight
        if num_bits > 0:
            self.quantize_input = QuantMeasure(self.num_bits, stochastic=stochastic, debug=debug)
        if num_bits_weight > 0:
            self.quantize_weights = QuantMeasure(self.num_bits_weight, min_value=-1.0, max_value=1.0, stochastic=stochastic, debug=debug)
        self.stochastic = stochastic
        self.debug = debug
        self.test_noise = test_noise

    def forward(self, input):
        if self.debug:
            print('\n\nEntering Convolutional Layer with {:d} {:d}x{:d} filters'.format(self.fms, self.fs, self.fs))

        weight = self.weight
        bias = self.bias

        if self.num_bits > 0 and self.num_bits < 8:
            qinput = self.quantize_input(input)
        else:
            qinput = input

        if self.num_bits_weight > 0:
            weight = self.quantize_weights(self.weight)
            # TODO how to quantize biases?
            if self.bias is not None:
                pass
                #print('\n\n\n****************** Quantizing bias, adjust args.pctl! *****************\n\n\n')
                #bias = quantize(self.bias, num_bits=self.num_bits_weight, min_value=-1.0, max_value=1.0, stochastic=self.stochastic)

        elif self.test_noise > 0 and not self.training:  #TODO use no-track_running_stats if using bn, or adjust bn params!
            weight = AddNoise().apply(self.weight, self.test_noise, self.debug)
            if self.bias is not None:
                bias = AddNoise().apply(self.bias, self.test_noise, self.debug)

        elif self.noise > 0 and self.training:
            weight = AddNoise().apply(self.weight, self.noise, self.debug)
            if self.bias is not None:
                bias = AddNoise().apply(self.bias, self.noise, self.debug)

        output = F.conv2d(qinput, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        if self.debug:
            pass
            #raise(SystemExit)
        return output


class NoisyLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False, num_bits=0, num_bits_weight=0, noise=0, test_noise=0, stochastic=True, debug=False):
        super(NoisyLinear, self).__init__(in_features, out_features, bias)
        self.fc_in = in_features
        self.fc_out = out_features
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight
        self.noise = noise
        if num_bits > 0:
            self.quantize_input = QuantMeasure(self.num_bits, stochastic=stochastic, debug=debug)
        if num_bits_weight > 0:
            self.quantize_weights = QuantMeasure(self.num_bits_weight, min_value=-1.0, max_value=1.0, stochastic=stochastic, debug=debug)
        self.stochastic = stochastic
        self.debug = debug
        self.test_noise = test_noise

    def forward(self, input):
        if self.debug:
            print('\n\nEntering Fully connected Layer {:d}x{:d}\n\n'.format(self.fc_in, self.fc_out))

        weight = self.weight
        bias = self.bias

        if self.num_bits > 0 and self.num_bits < 8:
            qinput = self.quantize_input(input)
        else:
            qinput = input

        if self.num_bits_weight > 0 and self.num_bits_weight < 8:
            weight = self.quantize_weights(self.weight)
            # TODO how to quantize biases?
            if self.bias is not None:
                pass
                # print('\n\n\n****************** Quantizing bias, adjust args.pctl! *****************\n\n\n')
                # bias = quantize(self.bias, num_bits=self.num_bits_weight)

        elif self.test_noise > 0 and not self.training:
            weight = AddNoise().apply(self.weight, self.test_noise, self.debug)
            if self.bias is not None:
                bias = AddNoise().apply(self.bias, self.test_noise, self.debug)

        elif self.noise > 0 and self.training:
            if self.debug:
                print('Adding noise to weights in linear layer:', 100.*self.noise, '%')
                print('\n\nBefore:\n{}'.format(self.weight[0, :20]))

            weight = AddNoise().apply(self.weight, self.noise, self.debug)
            if self.debug:
                print('\n\nAfter:\n{}'.format(weight[0, :20]))
            if self.bias is not None:
                bias = AddNoise().apply(self.bias, self.noise, self.debug)
        output = F.linear(qinput, weight, bias)

        return output


def distort_tensor(self, args, input, scale=0, stop=False):   # TODO this is horrible
    with torch.no_grad():
        if args.offset or args.offset_input:
            if args.debug:
                print('\n\ndistorting {}'.format(list(input.shape)))
            if self.generate_offsets:
                distr = Normal(loc=0, scale=scale * torch.ones_like(input))
                if '224' in input.shape:  # TODO fragile
                    self.input_offsets = distr.sample()
                elif stop:
                    self.act2_offsets = distr.sample()
                else:
                    self.act1_offsets = distr.sample()

                offsets = distr.sample()

                if stop:  # last layer, fix generated offsets
                    self.generate_offsets = False

            if '224' in input.shape:  # TODO fragile
                out = input + self.input_offsets
            elif stop:
                out = input + self.act2_offsets
            else:
                out = input + self.act1_offsets

            if args.debug:
                print('\nbefore  {}\noffsets {}\nafter   {}\n'.format(
                    input.flatten().detach().cpu().numpy()[:6], offsets.flatten().detach().cpu().numpy()[:6], out.flatten().detach().cpu().numpy()[:6]))
        else:
            noise = input * torch.cuda.FloatTensor(input.size()).uniform_(-args.noise, args.noise)
            out = input + noise
    return out

'''
class UniformQuantizeOrig(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, stochastic=0.5,
                inplace=False, enforce_true_zero=False, num_chunks=None, out_half=False, debug=False):

        num_chunks = input.shape[0] if num_chunks is None else num_chunks
        if min_value is None or max_value is None:
            B = input.shape[0]
            y = input.view(B // num_chunks, -1)
        if min_value is None:
            min_value = y.min(-1)[0].mean(-1)  # C
        # min_value = float(input.view(input.size(0), -1).min(-1)[0].mean())
        if max_value is None:
            # max_value = float(input.view(input.size(0), -1).max(-1)[0].mean())
            print('\n\ny', y.shape, 'y.max(-1).shape:', y.max(-1).shape, '\n\n', y.max(-1), '\n\n\n')
            max_value = y.max(-1)[0].mean(-1)  # C
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        qmin = 0.
        qmax = 2. ** num_bits - 1.
        if debug:
            print(
                '\nnum_bits {:d} qmin {} qmax {} min_value {} max_value {} actual max value {}'.format(num_bits, qmin, qmax, min_value, max_value, input.max()))
        scale = (max_value - min_value) / (qmax - qmin)

        scale = max(scale, 1e-6)  # TODO figure out how to set this robustly! causes nans
        if debug:
            print('\ninitial input\n', input[0, 0])

        with torch.no_grad():
            if enforce_true_zero:
                initial_zero_point = qmin - min_value / scale
                zero_point = 0.
                # make zero exactly represented
                if initial_zero_point < qmin:
                    zero_point = qmin
                elif initial_zero_point > qmax:
                    zero_point = qmax
                else:
                    zero_point = initial_zero_point
                zero_point = int(zero_point)
                output.div_(scale).add_(zero_point)
            else:
                output.add_(-min_value).div_(scale).add_(qmin)
            if debug:
                print('\nnormalized input\n', output[0, 0])
            if ctx.stochastic > 0:
                noise = output.new(output.shape).uniform_(-ctx.stochastic, ctx.stochastic)
                # print('\nnoise\n', noise[0, 0])
                output.add_(noise)
                if debug:
                    print('\nadding noise (stoch={:.1f})\n{}\n'.format(ctx.stochastic, output[0, 0]))

            output.clamp_(qmin, qmax).round_()  # quantize
            if debug:
                print('\nquantized\n', output[0, 0])

            if enforce_true_zero:
                output.add_(-zero_point).mul_(scale)  # dequantize
            else:
                output.add_(-qmin).mul_(scale).add_(min_value)  # dequantize
            if out_half and num_bits <= 16:
                output = output.half()
        if debug:
            print('\ndenormalized output\n', output[0, 0])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None, None
'''