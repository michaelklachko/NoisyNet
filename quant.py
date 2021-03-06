"""Author Elad Hoffer
https://github.com/eladhoffer/quantized.pytorch/blob/master/models/modules/quantize.py
"""

import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn


class UniformQuantize(InplaceFunction):

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
        scale = max(scale, 1e-6)  #TODO figure out how to set this robustly! causes nans

        with torch.no_grad():
            output.add_(-min_value).div_(scale).add_(qmin)
            if debug:
                print('\nnum_bits {:d} qmin {} qmax {} min_value {} max_value {} actual max value {}'.format(num_bits, qmin, qmax, min_value, max_value, input.max()))
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
        #Saturated Straight Through Estimator
        input, = ctx.saved_tensors
        #Should we clone the grad_output???
        grad_output[input > ctx.max_value] = 0
        grad_output[input < ctx.min_value] = 0
        #grad_input = grad_output
        return grad_output, None, None, None, None, None, None


class QuantMeasure(nn.Module):
    '''
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
    '''

    def __init__(self, num_bits=8, momentum=0.0, stochastic=0.5, min_value=0, max_value=0, scale=1,
                 calculate_running=False, pctl=.999, debug=False, debug_quant=False, inplace=False):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros([]))
        self.momentum = momentum
        self.num_bits = num_bits
        self.stochastic = stochastic
        self.inplace = inplace
        self.debug = debug
        self.debug_quant = debug_quant
        self.max_value = max_value
        self.min_value = min_value
        self.scale = scale
        self.calculate_running = calculate_running
        self.running_list = []
        self.pctl = pctl

    def forward(self, input):
        #max_value_their = input.detach().contiguous().view(input.size(0), -1).max(-1)[0].mean()
        with torch.no_grad():
            if self.calculate_running and self.training:
                if 224 in list(input.shape): #first layer input is special (needs more precision)
                    if self.num_bits == 4:
                        pctl = torch.tensor(0.92)  #args.q_a_first == 4
                    else:
                        pctl = torch.tensor(1.0)
                else:
                    pctl, _ = torch.kthvalue(input.view(-1), int(input.numel() * self.pctl))
                #print('input.shape', input.shape, 'pctl.shape', pctl.shape)
                #self.running_max = pctl
                max_value = input.max().item()  #self.running_max
                self.running_list.append(pctl)  #self.running_max)
                #self.running_max.mul_(self.momentum).add_(max_value * (1 - self.momentum))
                if self.debug:
                    print('{} gpu {} self.calculate_running {}  max value (pctl/running/actual) {:.3f}/{:.1f}/{:.1f}'.format(list(input.shape), torch.cuda.current_device(), self.calculate_running, pctl.item(), input.max().item() * 0.95, input.max().item()))
            else:
                if self.max_value > 0:
                    max_value = self.max_value
                elif self.running_max.item() > 0:
                    max_value = self.running_max.item()
                else:
                    #print('\n\nrunning_max is ', self.running_max.item())
                    max_value = input.max()

                if False and max_value > 1:
                    max_value = max_value * self.scale

            if False and self.debug:  #list(input.shape) == [input.shape[0], 512] and torch.cuda.current_device() == 1:
                print('{} gpu {}  max value (pctl/running/actual) {:.1f}/{:.1f}/{:.1f}'.format(list(input.shape), torch.cuda.current_device(), self.running_max.item(), input.max().item()*0.95, input.max().item()))

            if self.training:
                stoch = self.stochastic
            else:
                stoch = 0

        return UniformQuantize().apply(input, self.num_bits, float(self.min_value), float(max_value), stoch, self.inplace, self.debug_quant)


class QuantOp(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        #input = input.sign()
        """This is where the quant op goes"""
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

    def backward_(self, grad_output):
        input = self.saved_tensors
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output


class Quantize(nn.Module):
    def __init__(self):
        super(Quantize, self).__init__()

    def forward(self, input):
        input = QuantOp.apply(input)
        return input