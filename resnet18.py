import argparse
import os
import math
import time
import warnings
import numpy as np
from datetime import datetime

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
from plot_histograms import plot_layers

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/data/imagenet/', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='dali: 10, dataparallel: 16')
parser.add_argument('--epochs', default=110, type=int, metavar='N', help='number of total epochs to run')
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
parser.add_argument('--print_shapes', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--num_sims', default=1, type=int, help='number of simulations.')
parser.add_argument('--q_a', default=4, type=int, help='number of bits to quantize layer input')
parser.add_argument('--local_rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--act_max', default=0, type=float, help='clipping threshold for activations')
parser.add_argument('--eps', default=1e-7, type=float, help='epsilon to add to avoid dividing by zero')
parser.add_argument('--grad_clip', default=0, type=float, help='max value of gradients')
parser.add_argument('--q_scale', default=1, type=float, help='scale upper value of quantized tensor by this value')
parser.add_argument('--pctl', default=99.9, type=float, help='percentile to show when plotting')

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

warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)

args = parser.parse_args()

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
			self.conv3 = nn.Conv2d(ds_in, ds_out, kernel_size=1, stride=ds_strides, bias=False)
			self.bn3 = nn.BatchNorm2d(ds_out)
			self.layer3 = []

		if args.q_a > 0:
			self.quantize1 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, debug=args.debug_quant)
			self.quantize2 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, debug=args.debug_quant)

	def forward(self, x):
		'''[[self.input], [self.conv1.weight], [conv1_weight_sums], [conv1_weight_sums_sep], [conv1_weight_sums_blocked],
			[conv1_weight_sums_sep_blocked], [self.conv1_no_bias], [self.conv1_sep], [conv1_blocks], [conv1_sep_blocked]]'''
		if args.q_a > 0:
			x = self.quantize1(x)
		residual = x
		out = self.conv1(x)

		if args.plot:
			with torch.no_grad():
				blocks = []
				pos_blocks = []
				neg_blocks = []
				weight_sums_blocked = []
				weight_sums_sep_blocked = []
				block_size = 64
				dim = self.conv1.weight.shape[1]  #weights shape: (fm_out, fm_in, fs, fs)
				num_blocks = max(dim // block_size, 1)  #min 1 block, must be cleanly divisible!
				'''Weight blocking: fm_in is the dimension to split into blocks.  Merge filter size into fm_out, and extract dimx1x1 blocks. 
				Split input (bs, fms, h, v) into blocks of fms dim, and convolve with weight blocks. This could probably be done with grouped convolutions, but meh'''
				f = self.conv1.weight.permute(2, 3, 0, 1).contiguous().view(-1, dim, 1, 1)
				for b in range(num_blocks):
					weight_block = f[:, b * block_size: (b + 1) * block_size, :, :]
					weight_block_pos = weight_block.clone()
					weight_block_neg = weight_block.clone()
					weight_block_pos[weight_block_pos <= 0] = 0
					weight_block_neg[weight_block_neg > 0] = 0
					if b == 0:
						print('\n\nNumber of blocks:', num_blocks, 'weight block shape:', weight_block.shape, '\nweights for single output neuron:', weight_block[0].shape, '\nActual weights (one block):\n', weight_block[0].detach().cpu().numpy().ravel(), '\nWeight block sum(0) shape:',
						      weight_block.sum((1, 2, 3)).shape, '\n\n')
					input_block = x[:, b * block_size: (b + 1) * block_size, :, :]
					blocks.append(F.conv2d(input_block, weight_block, stride=self.stride, padding=1))
					pos_blocks.append(F.conv2d(input_block, weight_block_pos, stride=self.stride, padding=1))
					neg_blocks.append(F.conv2d(input_block, weight_block_neg, stride=self.stride, padding=1))
					weight_sums_blocked.append(torch.abs(weight_block).sum((1, 2, 3)))
					weight_sums_sep_blocked.extend([weight_block_pos.sum((1, 2, 3)), weight_block_neg.sum((1, 2, 3))])

				blocked = torch.cat(blocks, 1)  #conv_out shape: (bs, fms, h, v)
				pos_blocks = torch.cat(pos_blocks, 1)
				neg_blocks = torch.cat(neg_blocks, 1)
				#print('\n\nconv2_pos_blocks:\n', pos_blocks.shape, '\n', pos_blocks[2,2])
				#print('\n\nconv2_neg_blocks:\n', neg_blocks.shape, '\n', neg_blocks[2, 2], '\n\n')
				#raise(SystemExit)
				sep_blocked = torch.cat((pos_blocks, neg_blocks), 0)
				#print('\nblocks shape', blocks.shape, '\n')
				#print(blocks.detach().cpu().numpy()[60, 234, :8, :8])
				weight_sums = torch.abs(self.conv1.weight).sum((1, 2, 3))
				weight_sums_blocked = torch.cat(weight_sums_blocked, 0)
				weight_sums_sep_blocked = torch.cat(weight_sums_sep_blocked, 0)

				w_pos = self.conv1.weight.clone()
				w_pos[w_pos < 0] = 0
				w_neg = self.conv1.weight.clone()
				w_neg[w_neg >= 0] = 0
				pos = F.conv2d(x, w_pos, stride=self.stride, padding=1)
				neg = F.conv2d(x, w_neg, stride=self.stride, padding=1)
				sep = torch.cat((neg, pos), 0)
				weight_sums_sep = torch.cat((w_pos.sum((1, 2, 3)), w_neg.sum((1, 2, 3))), 0)

				arrays.append([x.half()])
				arrays.append([self.conv1.weight.half()])
				arrays.append([out.half()])
				arrays.append([sep.half()])
				arrays.append([blocked.half()])
				arrays.append([sep_blocked.half()])

		if args.print_shapes:
			print('\nblock input:', x.shape)
			print('conv1:', out.shape)

		if args.merge_bn:
			bias = self.bn1.bias.view(1, -1, 1, 1) - self.bn1.running_mean.data.view(1, -1, 1, 1) * \
			       self.bn1.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn1.running_var.data.view(1, -1, 1, 1) + args.eps)
			out += bias
			if args.plot:
				arrays.append([bias.half()])
				arrays.append([weight_sums.half()])
				arrays.append([weight_sums_sep.half()])
				arrays.append([weight_sums_blocked.half()])
				arrays.append([weight_sums_sep_blocked.half()])
		else:
			out = self.bn1(out)

		out = self.relu(out)

		if args.q_a > 0:
			out = self.quantize2(out)

		if args.plot:
			arrays.append([out.half()])

		conv2_input = out
		#print('\n\nBefore: conv2_input:\n', conv2_input[2, 12, 3])
		out = self.conv2(out)
		#print('\nAfter: conv2_input:\n', conv2_input[2, 12, 3])
		#print('\nAfter : out:\n', out[2, 12, 3])
		#raise(SystemExit)

		if args.plot:
			with torch.no_grad():
				blocks = []
				pos_blocks = []
				neg_blocks = []
				weight_sums_blocked = []
				weight_sums_sep_blocked = []
				block_size = 64
				dim = self.conv2.weight.shape[1]  #weights shape: (fm_out, fm_in, fs, fs)
				num_blocks = max(dim // block_size, 1)  #min 1 block, must be cleanly divisible!
				'''Weight blocking: fm_in is the dimension to split into blocks.  Merge filter size into fm_out, and extract dimx1x1 blocks. 
				Split input (bs, fms, h, v) into blocks of fms dim, and convolve with weight blocks. This could probably be done with grouped convolutions, but meh'''
				f = self.conv2.weight.permute(2, 3, 0, 1).contiguous().view(-1, dim, 1, 1)
				for b in range(num_blocks):
					weight_block = f[:, b * block_size: (b + 1) * block_size, :, :]
					weight_block_pos = weight_block.clone()
					weight_block_neg = weight_block.clone()
					weight_block_pos[weight_block_pos <= 0] = 0
					weight_block_neg[weight_block_neg > 0] = 0
					if b == 0:
						print('\n\nNumber of blocks:', num_blocks, 'weight block shape:', weight_block.shape, '\nweights for single output neuron:', weight_block[0].shape, '\nActual weights (one block):\n', weight_block[0].detach().cpu().numpy().ravel(), '\nWeight block sum(0) shape:',
						      weight_block.sum((1, 2, 3)).shape, '\n\n')
					input_block = conv2_input[:, b * block_size: (b + 1) * block_size, :, :]
					blocks.append(F.conv2d(input_block, weight_block, padding=1))
					pos_blocks.append(F.conv2d(input_block, weight_block_pos, padding=1))
					neg_blocks.append(F.conv2d(input_block, weight_block_neg, padding=1))
					weight_sums_blocked.append(torch.abs(weight_block).sum((1, 2, 3)))
					weight_sums_sep_blocked.extend([weight_block_pos.sum((1, 2, 3)), weight_block_neg.sum((1, 2, 3))])

				blocked = torch.cat(blocks, 1)  #conv_out shape: (bs, fms, h, v)
				pos_blocks = torch.cat(pos_blocks, 1)
				neg_blocks = torch.cat(neg_blocks, 1)
				#print('\n\nconv2_pos_blocks:\n', pos_blocks.shape, '\n', pos_blocks[2,2])
				#print('\n\nconv2_neg_blocks:\n', neg_blocks.shape, '\n', neg_blocks[2, 2], '\n\n')
				#raise(SystemExit)
				sep_blocked = torch.cat((pos_blocks, neg_blocks), 0)
				#print('\nblocks shape', blocks.shape, '\n')
				#print(blocks.detach().cpu().numpy()[60, 234, :8, :8])
				weight_sums = torch.abs(self.conv2.weight).sum((1, 2, 3))
				weight_sums_blocked = torch.cat(weight_sums_blocked, 0)
				weight_sums_sep_blocked = torch.cat(weight_sums_sep_blocked, 0)

				w_pos = self.conv2.weight.clone()
				w_pos[w_pos < 0] = 0
				w_neg = self.conv2.weight.clone()
				w_neg[w_neg >= 0] = 0
				pos = F.conv2d(conv2_input, w_pos, padding=1)
				neg = F.conv2d(conv2_input, w_neg, padding=1)
				sep = torch.cat((neg, pos), 0)
				weight_sums_sep = torch.cat((w_pos.sum((1, 2, 3)), w_neg.sum((1, 2, 3))), 0)

				arrays.append([self.conv2.weight.half()])
				arrays.append([out.half()])
				arrays.append([sep.half()])
				arrays.append([blocked.half()])
				arrays.append([sep_blocked.half()])

		if args.print_shapes:
			print('conv2:', out.shape)

		if args.merge_bn:
			bias = self.bn2.bias.view(1, -1, 1, 1) - self.bn2.running_mean.data.view(1, -1, 1, 1) * \
			       self.bn2.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn2.running_var.data.view(1, -1, 1, 1) + args.eps)
			out += bias
			if args.plot:
				arrays.append([bias.half()])
				arrays.append([weight_sums.half()])
				arrays.append([weight_sums_sep.half()])
				arrays.append([weight_sums_blocked.half()])
				arrays.append([weight_sums_sep_blocked.half()])
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
			self.quantize1 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, debug=args.debug_quant)
			self.quantize2 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, debug=args.debug_quant)

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

	def forward(self, x):
		if args.print_shapes:
			print('RGB input:', x.shape)
		if args.q_a > 0:
			x = self.quantize1(x)

		if args.plot:
			arrays.append([x.half()])

		conv1_input = x

		x = self.conv1(x)
		if args.print_shapes:
			print('first conv:', x.shape)

		if args.plot:
			with torch.no_grad():
				blocks = []
				pos_blocks = []
				neg_blocks = []
				weight_sums_blocked = []
				weight_sums_sep_blocked = []
				block_size = 64
				dim = self.conv1.weight.shape[1]  #weights shape: (fm_out, fm_in, fs, fs)
				num_blocks = max(dim // block_size, 1)  #min 1 block, must be cleanly divisible!
				'''Weight blocking: fm_in is the dimension to split into blocks.  Merge filter size into fm_out, and extract dimx1x1 blocks. 
				Split input (bs, fms, h, v) into blocks of fms dim, and convolve with weight blocks. This could probably be done with grouped convolutions, but meh'''
				f = self.conv1.weight.permute(2, 3, 0, 1).contiguous().view(-1, dim, 1, 1)
				for b in range(num_blocks):
					weight_block = f[:, b * block_size: (b + 1) * block_size, :, :]
					weight_block_pos = weight_block.clone()
					weight_block_neg = weight_block.clone()
					weight_block_pos[weight_block_pos <= 0] = 0
					weight_block_neg[weight_block_neg > 0] = 0
					if b == 0:
						print('\n\nNumber of blocks:', num_blocks, 'weight block shape:', weight_block.shape, '\nweights for single output neuron:', weight_block[0].shape, '\nActual weights (one block):\n', weight_block[0].detach().cpu().numpy().ravel(), '\nWeight block sum(0) shape:',
						      weight_block.sum((1, 2, 3)).shape, '\n\n')
					input_block = conv1_input[:, b * block_size: (b + 1) * block_size, :, :]
					blocks.append(F.conv2d(input_block, weight_block, stride=2, padding=3))
					pos_blocks.append(F.conv2d(input_block, weight_block_pos, stride=2, padding=3))
					neg_blocks.append(F.conv2d(input_block, weight_block_neg, stride=2, padding=3))
					weight_sums_blocked.append(torch.abs(weight_block).sum((1, 2, 3)))
					weight_sums_sep_blocked.extend([weight_block_pos.sum((1, 2, 3)), weight_block_neg.sum((1, 2, 3))])

				blocked = torch.cat(blocks, 1)  #conv_out shape: (bs, fms, h, v)
				pos_blocks = torch.cat(pos_blocks, 1)
				neg_blocks = torch.cat(neg_blocks, 1)
				#print('\n\nconv2_pos_blocks:\n', pos_blocks.shape, '\n', pos_blocks[2,2])
				#print('\n\nconv2_neg_blocks:\n', neg_blocks.shape, '\n', neg_blocks[2, 2], '\n\n')
				#raise(SystemExit)
				sep_blocked = torch.cat((pos_blocks, neg_blocks), 0)
				#print('\nblocks shape', blocks.shape, '\n')
				#print(blocks.detach().cpu().numpy()[60, 234, :8, :8])
				weight_sums = torch.abs(self.conv1.weight).sum((1, 2, 3))
				weight_sums_blocked = torch.cat(weight_sums_blocked, 0)
				weight_sums_sep_blocked = torch.cat(weight_sums_sep_blocked, 0)

				w_pos = self.conv1.weight.clone()
				w_pos[w_pos < 0] = 0
				w_neg = self.conv1.weight.clone()
				w_neg[w_neg >= 0] = 0
				pos = F.conv2d(conv1_input, w_pos, stride=2, padding=3)
				neg = F.conv2d(conv1_input, w_neg, stride=2, padding=3)
				sep = torch.cat((neg, pos), 0)
				weight_sums_sep = torch.cat((w_pos.sum((1, 2, 3)), w_neg.sum((1, 2, 3))), 0)

				arrays.append([self.conv1.weight.half()])
				arrays.append([x.half()])
				arrays.append([sep.half()])
				arrays.append([blocked.half()])
				arrays.append([sep_blocked.half()])


		if args.merge_bn:
			bias = self.bn1.bias.view(1, -1, 1, 1) - self.bn1.running_mean.data.view(1, -1, 1, 1) * \
			       self.bn1.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn1.running_var.data.view(1, -1, 1, 1) + args.eps)
			x += bias
			if args.plot:
				arrays.append([bias.half()])
				arrays.append([weight_sums.half()])
				arrays.append([weight_sums_sep.half()])
				arrays.append([weight_sums_blocked.half()])
				arrays.append([weight_sums_sep_blocked.half()])
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

		if args.plot:
			arrays.append([x.half()])

		fc_input = x

		x = self.fc(x)

		if args.plot:
			with torch.no_grad():
				blocks = []
				pos_blocks = []
				neg_blocks = []
				weight_sums_blocked = []
				weight_sums_sep_blocked = []
				block_size = 64
				dim = self.fc.weight.shape[1]  #weights shape: (out, in)
				num_blocks = max(dim // block_size, 1)  #min 1 block, must be cleanly divisible!
				for b in range(num_blocks):
					weight_block = self.fc.weight[:, b * block_size: (b + 1) * block_size]
					weight_block_pos = weight_block.clone()
					weight_block_neg = weight_block.clone()
					weight_block_pos[weight_block_pos <= 0] = 0
					weight_block_neg[weight_block_neg > 0] = 0
					if b == 0:
						print('\n\nNumber of blocks:', num_blocks, 'weight block shape:', weight_block.shape, '\nweights for single output neuron:', weight_block[0].shape, '\nActual weights (one block):\n', weight_block[0].detach().cpu().numpy().ravel(), '\nWeight block sum(0):\n',
						      weight_block.sum(1).shape, '\n\n')
					input_block = fc_input[:, b * block_size: (b + 1) * block_size]
					blocks.append(F.linear(weight_block, input_block))
					pos_blocks.append(F.linear(input_block, weight_block_pos))
					neg_blocks.append(F.linear(input_block, weight_block_neg))
					weight_sums_blocked.append(torch.abs(weight_block).sum(1))
					weight_sums_sep_blocked.extend([weight_block_pos.sum(1), weight_block_neg.sum(1)])

				blocked = torch.cat(blocks, 1)
				#print('\nblocks shape', blocks.shape, '\n')
				#print(blocks.detach().cpu().numpy()[:8, :8])
				pos_blocks = torch.cat(pos_blocks, 1)
				neg_blocks = torch.cat(neg_blocks, 1)
				sep_blocked = torch.cat((pos_blocks, neg_blocks), 0)
				weight_sums = torch.abs(self.fc.weight).sum(1)
				weight_sums_blocked = torch.cat(weight_sums_blocked, 0)
				weight_sums_sep_blocked = torch.cat(weight_sums_sep_blocked, 0)

				w_pos = self.fc.weight.clone()
				w_pos[w_pos < 0] = 0
				w_neg = self.fc.weight.clone()
				w_neg[w_neg >= 0] = 0
				pos = F.linear(fc_input, w_pos)
				neg = F.linear(fc_input, w_neg)
				sep = torch.cat((neg, pos), 0)
				weight_sums_sep = torch.cat((w_pos.sum(1), w_neg.sum(1)), 0)

				arrays.append([self.fc.weight.half()])
				arrays.append([x.half()])
				arrays.append([sep.half()])
				arrays.append([blocked.half()])
				arrays.append([sep_blocked.half()])

				arrays.append([self.fc.bias.half()])
				arrays.append([weight_sums.half()])
				arrays.append([weight_sums_sep.half()])
				arrays.append([weight_sums_blocked.half()])
				arrays.append([weight_sums_sep_blocked.half()])

		if args.print_shapes:
			print('\noutput:', x.shape)

		if args.plot:
			names = ['input', 'weights', 'vmm', 'vmm diff', 'vmm blocked', 'vmm diff blocked', 'bias',
			         'weight sums', 'weight sums diff', 'weight sums blocked', 'weight sums diff blocked']
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

			plot_layers(num_layers=len(layers), models=['plotts/'], epoch=epoch, i=i, layers=layers,
			            names=names, var=var_name, vars=[var_], pctl=args.pctl, acc=best_acc, tag=args.tag)
			print('\n\nSaved plots to current dir\n\n')
			raise (SystemExit)


		return x


def validate(val_loader, model, epoch=0):
	model.eval()
	te_accs = []
	with torch.no_grad():
		for i, data in enumerate(val_loader):
			if args.dali:
				input = data[0]["data"]
				target = data[0]["label"].squeeze().cuda().long()
				input_var = Variable(input)
				output = model(input_var)
			else:
				images, target = data
				target = target.cuda(non_blocking=True)
				output = model(images)
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
	model = models.mobilenet_v2(pretrained=args.pretrained)
else:
	model = ResNet(BasicBlock)
	if args.pretrained:
		model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))

model = torch.nn.DataParallel(model).cuda()
if args.resume is not None:
	utils.print_model(model, args)
criterion = nn.CrossEntropyLoss().cuda()
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
		utils.print_model(model, args)
		for saved_name, saved_param in checkpoint['state_dict'].items():
			matched = False
			if args.debug:
				print(saved_name)
			for name, param in model.named_parameters():
				if name == saved_name:
					matched = True
					if args.debug:
						print('\tmatched, copying...')
					param.data = saved_param.data
			if 'running' in saved_name and 'bn' in saved_name:  #batchnorm stats are not in named_parameters
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
			if args.q_a > 0 and 'running' in saved_name and ('quantize1' in saved_name or 'quantize2' in saved_name):
				matched = True
				if args.debug:
					print('\tmatched, copying...')
				m = model.state_dict()
				m.update({saved_name: saved_param})
				model.load_state_dict(m)
			elif not matched:
				#pass
				print('\t\t\t************ Not copying', saved_name)

		if args.merge_bn:
			print('\n\nMerging batchnorm into weights...\n\n')
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
	print('\n\nTesting accuracy on validation set (should be {:.2f})...\n'.format(checkpoint['best_acc']))
	validate(val_loader, model, epoch=checkpoint['epoch'])
	raise (SystemExit)

for epoch in range(args.start_epoch, args.epochs):
	adjust_learning_rate(optimizer, epoch, args)
	model.train()
	tr_accs = []

	for i, data in enumerate(train_loader):
		if args.dali:
			input = data[0]["data"]
			target = data[0]["label"].squeeze().cuda().long()
			train_loader_len = int(train_loader._size / args.batch_size)
			input_var = Variable(input)
			target_var = Variable(target)
			output = model(input_var)
			loss = criterion(output, target_var)
		else:
			images, target = data
			train_loader_len = len(train_loader)
			target = target.cuda(non_blocking=True)
			output = model(images)
			loss = criterion(output, target)

		if i == 0:
			args.print_shapes = False
		acc = accuracy(output, target)
		tr_accs.append(acc)
		optimizer.zero_grad()
		loss.backward()

		if args.grad_clip > 0:
			for n, p in model.named_parameters():
				p.grad.data.clamp_(-args.grad_clip, args.grad_clip)

		optimizer.step()

		if i % args.print_freq == 0:
			print('{}  Epoch {:>2d} Batch {:>4d}/{:d} LR {} | {:.2f}'.format(
				str(datetime.now())[:-7], epoch, i, train_loader_len, optimizer.param_groups[0]["lr"], np.mean(tr_accs)))

		if False and i == 10:
			torch.save({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(),
			            'best_acc': best_acc, 'optimizer': optimizer.state_dict()}, args.tag + 'chkpt.pth')
			raise(SystemExit)

	acc = validate(val_loader, model, epoch=epoch)
	if acc > best_acc:
		best_acc = acc
		if args.distort_w_train:
			tag = args.tag + 'noise_{:.2f}_'.format(args.noise)
		else:
			tag = args.tag
		torch.save({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(), 'best_acc': best_acc, 'optimizer': optimizer.state_dict()}, tag + 'chkpt.pth')

	if args.dali:
		train_loader.reset()
		val_loader.reset()

print('\n\nBest Accuracy {:.2f}\n\n'.format(best_acc))
