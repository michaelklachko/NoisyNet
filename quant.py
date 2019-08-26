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
		grad_input = grad_output
		return grad_input, None, None, None, None, None, None


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
	'''

	def __init__(self, num_bits=8, momentum=0.0, stochastic=0.5, min_value=0, max_value=0, scale=1, show_running=True, calculate_running=False, pctl=.999, debug=False):
		super(QuantMeasure, self).__init__()
		self.register_buffer('running_min', torch.zeros(1))
		self.register_buffer('running_max', torch.zeros(1))
		self.momentum = momentum
		self.num_bits = num_bits
		self.stochastic = stochastic
		self.debug = debug
		self.max_value = max_value
		self.min_value = min_value
		self.scale = scale
		self.show_running = show_running
		self.calculate_running = calculate_running
		self.pctl = pctl
		'''
		if True or torch.cuda.current_device() == 1:
			self.show_running = True
		else:
			self.show_running = False
		'''

	def forward(self, input):
		#max_value_their = input.detach().contiguous().view(input.size(0), -1).max(-1)[0].mean()
		if self.calculate_running:
			pctl, _ = torch.kthvalue(input.view(-1), int(input.numel() * self.pctl))
			max_value = pctl.item()
			self.running_max.mul_(self.momentum).add_(max_value * (1 - self.momentum))
		else:
			#max_value = self.running_max.item()
			max_value = input.max()

		#if max_value > 1:
			#max_value = max_value * self.scale

		if self.debug:  #list(input.shape) == [input.shape[0], 512] and self.show_running:# and torch.cuda.current_device() == 1:
			print('{} gpu {}  max value (pctl/running/actual) {:.1f}/{:.1f}/{:.1f}'.format(list(input.shape), torch.cuda.current_device(), self.running_max.item(), input.max().item()*0.5, input.max().item()))
			#self.show_running = False

		if self.training:
			stoch = self.stochastic
		else:
			stoch = 0

		return UniformQuantize().apply(input, self.num_bits, float(self.min_value), float(max_value), stoch, False, self.debug)
