"""Author Elad Hoffer
https://github.com/eladhoffer/quantized.pytorch/blob/master/models/modules/quantize.py
"""


import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F

'''
class UniformQuantize(InplaceFunction):

	@classmethod
	def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, stochastic=0.5,
				inplace=False, enforce_true_zero=False, num_chunks=None, out_half=False, debug=False):

		num_chunks = input.shape[0] if num_chunks is None else num_chunks
		if min_value is None or max_value is None:
			B = input.shape[0]
			y = input.view(B // num_chunks, -1)
		if min_value is None:
			min_value = y.min(-1)[0].mean(-1)  # C
			#min_value = float(input.view(input.size(0), -1).min(-1)[0].mean())
		if max_value is None:
			#max_value = float(input.view(input.size(0), -1).max(-1)[0].mean())
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
		qmax = 2.**num_bits - 1.
		if debug:
			print('\nnum_bits {:d} qmin {} qmax {} min_value {} max_value {} actual max value {}'.format(num_bits, qmin, qmax, min_value, max_value, input.max()))
		scale = (max_value - min_value) / (qmax - qmin)

		scale = max(scale, 1e-6)   #TODO figure out how to set this robustly! causes nans
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
				#print('\nnoise\n', noise[0, 0])
				output.add_(noise)
				if debug:
					print('\nadding noise (stoch={:.1f})\n{}\n'.format(ctx.stochastic, output[0,0]))

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


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
	out1 = F.conv2d(input.detach(), weight, bias, stride, padding, dilation, groups)
	out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None, stride, padding, dilation, groups)
	return out1 + out2 - out1.detach()


def linear_biprec(input, weight, bias=None):
	out1 = F.linear(input.detach(), weight, bias)
	out2 = F.linear(input, weight.detach(), bias.detach() if bias is not None else None)
	return out1 + out2 - out1.detach()


#def quantize(x, num_bits=8, min_value=None, max_value=None, num_chunks=None, stochastic=0.5, inplace=False, enforce_true_zero=False, out_half=False, debug=False):
	#return UniformQuantize().apply(x, num_bits, min_value, max_value, stochastic, inplace, enforce_true_zero, num_chunks, out_half, debug)

'''
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
		qmax = 2.**num_bits - 1.
		scale = (max_value - min_value) / (qmax - qmin)
		scale = max(scale, 1e-6)   #TODO figure out how to set this robustly! causes nans

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
					print('\nadding noise (stoch={:.1f})\n{}\n'.format(ctx.stochastic, output[0,0]))

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
		print('self.scale', self.scale)
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
			#self.running_max = max_value
			self.running_max.mul_(self.momentum).add_(max_value * (1 - self.momentum))
			print('{} gpu {}  min value {:.1f}  max value (pctl/running/actual) {:.1f}/{:.1f}/{:.1f}'.format(
				list(input.shape), torch.cuda.current_device(), self.min_value, max_value, self.running_max.item(), input.max().item()))
		else:
			max_value = self.running_max.item()
			max_value = input.max()

		if max_value > 1:
			max_value = max_value * self.scale
			#max_value = input.max()

		#print(list(input.shape), self.show_running)
		if False:#list(input.shape) == [input.shape[0], 512] and self.show_running:# and torch.cuda.current_device() == 1:
			#print('{} gpu {}  min value {:.1f}  max value (pctl/running/mean/actual) {:.1f}/{:.1f}/{:.1f}/{:.1f}'.format(
				#list(input.shape), torch.cuda.current_device(), self.min_value, pctl.item(), self.running_max.item(), max_value_their.item(), input.max().item()))
			print('{} gpu {}  min value {:.1f}  max value (pctl/running/actual) {:.1f}/{:.1f}/{:.1f}'.format
				  (list(input.shape), torch.cuda.current_device(), self.min_value, max_value, self.running_max.item(), input.max().item()))
			#self.show_running = False

		'''
		if self.max_value > 0:
			max_value = self.max_value
		else:
			max_value = input.max() * self.scale
		
		if self.min_value < 0:
			min_value = self.min_value
		else:
			min_value = input.min() * self.scale
		'''

		if self.training:
			stoch = self.stochastic
		else:
			stoch = 0

		num_chunks = 16
		enforce_true_zero = False
		inplace = False
		out_half = False
		#return quantize(input, self.num_bits, min_value=float(self.min_value), max_value=float(max_value), num_chunks=16, stochastic=stoch, debug=self.debug)
		#return UniformQuantize().apply(input, self.num_bits, float(self.min_value), float(max_value), stoch, inplace, enforce_true_zero, num_chunks, out_half, self.debug)
		return UniformQuantize().apply(input, self.num_bits, float(self.min_value), float(max_value), stoch, inplace, self.debug)


class QConv2d(nn.Conv2d):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
	             num_bits=8, num_bits_weight=None, biprecision=False, stochastic=0.5, debug=False):
		super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
		self.num_bits = num_bits
		self.fms = out_channels
		self.fs = kernel_size
		self.num_bits_weight = num_bits_weight
		self.quantize_input = QuantMeasure(self.num_bits, stochastic=stochastic, debug=debug)
		self.biprecision = biprecision
		self.stochastic = stochastic
		self.debug = debug

	def forward(self, input):
		if self.debug:
			print('\n\nEntering Convolutional Layer with {:d} {:d}x{:d} filters\n\n'.format(self.fms, self.fs, self.fs))
		if self.num_bits > 0:
			qinput = self.quantize_input(input)
		else:
			qinput = input

		if self.num_bits_weight > 0:
			qweight = quantize(self.weight, num_bits=self.num_bits_weight, min_value=float(self.weight.min()), max_value=float(self.weight.max()), stochastic=self.stochastic)
		else:
			qweight = self.weight
		if self.bias is not None:
			qbias = quantize(self.bias, num_bits=self.num_bits_weight)
		else:
			qbias = None
		if not self.biprecision:
			output = F.conv2d(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)
		else:
			output = conv2d_biprec(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)

		return output


class QLinear(nn.Linear):

	def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=None, biprecision=False, stochastic=0.5, debug=False):
		super(QLinear, self).__init__(in_features, out_features, bias)
		self.fc_in = in_features
		self.fc_out = out_features
		self.num_bits = num_bits
		self.num_bits_weight = num_bits_weight
		self.biprecision = biprecision
		self.quantize_input = QuantMeasure(self.num_bits, stochastic=stochastic, debug=debug)
		self.stochastic = stochastic
		self.debug = debug

	def forward(self, input):
		if self.debug:
			print('\n\nEntering Fully connected Layer {:d}x{:d}\n\n'.format(self.fc_in, self.fc_out))
		if self.num_bits > 0:
			qinput = self.quantize_input(input)
		else:
			qinput = input
		if self.num_bits_weight > 0:
			qweight = quantize(self.weight, num_bits=self.num_bits_weight, min_value=float(self.weight.min()), max_value=float(self.weight.max()), stochastic=self.stochastic)
		else:
			qweight = self.weight

		if self.bias is not None:
			qbias = quantize(self.bias, num_bits=self.num_bits_weight)
		else:
			qbias = None

		if not self.biprecision:
			output = F.linear(qinput, qweight, qbias)
		else:
			output = linear_biprec(qinput, qweight, qbias)
		return output