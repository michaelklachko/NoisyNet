import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import random
import os
from datetime import datetime
import argparse
import numpy as np
import math

import utils
from quant_orig import QConv2d, QLinear, QuantMeasure
from plot_histograms import plot_layers


class Net(nn.Module):
	def __init__(self, args=None):
		super(Net, self).__init__()

		self.create_dir = True

		if args.train_act_max:
			self.act_max1 = nn.Parameter(torch.Tensor([0]), requires_grad=True)
			self.act_max2 = nn.Parameter(torch.Tensor([0]), requires_grad=True)
			self.act_max3 = nn.Parameter(torch.Tensor([0]), requires_grad=True)

		if args.train_w_max:
			self.w_max1 = nn.Parameter(torch.Tensor([0]), requires_grad=True)
			self.w_min1 = nn.Parameter(torch.Tensor([0]), requires_grad=True)

		self.pool = nn.MaxPool2d(2, 2)
		self.relu = nn.ReLU()

		self.quantize1 = QuantMeasure(args.q_a1, stochastic=args.stochastic, debug=args.debug_quant)
		self.quantize2 = QuantMeasure(args.q_a2, stochastic=args.stochastic, debug=args.debug_quant)
		self.quantize3 = QuantMeasure(args.q_a3, stochastic=args.stochastic, debug=args.debug_quant)
		self.quantize4 = QuantMeasure(args.q_a4, stochastic=args.stochastic, debug=args.debug_quant)

		if args.q_w1 > 0:
			print('\n\nQuantizing conv1 layer weights to {:d} bits\n\n'.format(args.q_w1))
			self.conv1 = QConv2d(3, args.fm1 * args.width, kernel_size=args.fs, bias=args.use_bias, num_bits=0, num_bits_weight=args.q_w1, biprecision=args.biprecision, stochastic=args.stochastic, debug=args.debug_quant)
		else:
			self.conv1 = nn.Conv2d(3, args.fm1 * args.width, kernel_size=args.fs, bias=args.use_bias)

		if args.q_w2 > 0:
			print('\n\nQuantizing conv2 layer weights to {:d} bits\n\n'.format(args.q_w2))
			self.conv2 = QConv2d(args.fm1 * args.width, args.fm2 * args.width, kernel_size=args.fs, bias=args.use_bias, num_bits=0, num_bits_weight=args.q_w2, biprecision=args.biprecision, stochastic=args.stochastic, debug=args.debug_quant)
		else:
			self.conv2 = nn.Conv2d(args.fm1 * args.width, args.fm2 * args.width, kernel_size=args.fs, bias=args.use_bias)

		if args.q_w3 > 0:
			print('\n\nQuantizing fc1 layer weights to {:d} bits\n\n'.format(args.q_w3))
			self.linear1 = QLinear(args.fm2 * args.width * args.fs * args.fs, args.fc * args.width, bias=args.use_bias, num_bits=0, num_bits_weight=args.q_w3, biprecision=args.biprecision, stochastic=args.stochastic, debug=args.debug_quant)
		else:
			self.linear1 = nn.Linear(args.fm2 * args.width * args.fs * args.fs, args.fc * args.width, bias=args.use_bias)
		if args.q_w4 > 0:
			print('\n\nQuantizing fc2 layer weights to {:d} bits\n\n'.format(args.q_w4))
			self.linear2 = QLinear(args.fc * args.width, 10, bias=args.use_bias, num_bits=0, num_bits_weight=args.q_w4, biprecision=args.biprecision, stochastic=args.stochastic, debug=args.debug_quant)
		else:
			self.linear2 = nn.Linear(args.fc * args.width, 10, bias=args.use_bias)

		if args.batchnorm:
			self.bn1 = nn.BatchNorm2d(args.fm1 * args.width, track_running_stats=args.track_running_stats)
			self.bn2 = nn.BatchNorm2d(args.fm2 * args.width, track_running_stats=args.track_running_stats)
			if args.bn3:
				self.bn3 = nn.BatchNorm1d(args.fc * args.width, track_running_stats=args.track_running_stats)
			if args.bn4:
				self.bn4 = nn.BatchNorm1d(10, track_running_stats=args.track_running_stats)

		if args.weightnorm:
			self.conv1 = nn.utils.weight_norm(nn.Conv2d(3, args.fm1 * args.width, kernel_size=args.fs, bias=args.use_bias))
			self.conv2 = nn.utils.weight_norm(nn.Conv2d(args.fm1 * args.width, args.fm2 * args.width, kernel_size=args.fs, bias=args.use_bias))
			self.linear1 = nn.utils.weight_norm(nn.Linear(args.fm2 * args.width * args.fs * args.fs, args.fc * args.width, bias=args.use_bias))
			self.linear2 = nn.utils.weight_norm(nn.Linear(args.fc * args.width, 10, bias=args.use_bias))

		if args.dropout > 0:
			self.dropout = nn.Dropout(p=args.dropout)

	def forward(self, input, epoch=0, i=0, s=0, acc=0.0):
		'''
		if not self.training and i == 0:
			#pass
			#conv1_out = 200 * conv1_out
			self.conv1.weight.data = 20. * self.conv1.weight.data
		if i == 0 and self.training and epoch != 0:
			#pass
			self.conv1.weight.data = self.conv1.weight.data / 20.
		'''

		if args.q_a1 > 0:
			self.input = self.quantize1(input)
		else:
			self.input = input

		if epoch == 0 and i == 0 and s == 0 and self.training:
			print('\ninput shape:', self.input.shape)

		self.conv1_no_bias = self.conv1(self.input)

		if args.plot or args.write:
			with torch.no_grad():
				'''
				conv1_blocks = []
				for fi in range(args.fs):
					for fk in range(args.fs):
						#print('\nweight shape:', self.conv1.weight[:, :, fi, fk].shape)
						block = F.conv2d(self.input, self.conv1.weight[:, :, fi, fk].view(self.conv1.weight.shape[0], self.conv1.weight.shape[1], 1, 1))
						#print('block shape:', block.shape)
						conv1_blocks.append(block)
				conv1_blocks = torch.cat(conv1_blocks, 1)
				#print('\nconv1_blocks shape', conv1_blocks.shape)
				#print(conv1_blocks.detach().cpu().numpy()[60, 234, :8, :8])
				'''
				conv1_blocks = []
				conv1_pos_blocks = []
				conv1_neg_blocks = []
				conv1_weight_sums_blocked = []
				conv1_weight_sums_sep_blocked = []
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
					input_block = self.input[:, b * block_size: (b + 1) * block_size, :, :]
					conv1_blocks.append(F.conv2d(input_block, weight_block))
					conv1_pos_blocks.append(F.conv2d(input_block, weight_block_pos))
					conv1_neg_blocks.append(F.conv2d(input_block, weight_block_neg))
					conv1_weight_sums_blocked.append(torch.abs(weight_block).sum((1, 2, 3)))
					conv1_weight_sums_sep_blocked.extend([weight_block_pos.sum((1, 2, 3)), weight_block_neg.sum((1, 2, 3))])

				conv1_blocks = torch.cat(conv1_blocks, 1)  #conv_out shape: (bs, fms, h, v)
				conv1_pos_blocks = torch.cat(conv1_pos_blocks, 1)
				conv1_neg_blocks = torch.cat(conv1_neg_blocks, 1)
				#print('\n\nconv2_pos_blocks:\n', conv1_pos_blocks.shape, '\n', conv1_pos_blocks[2,2])
				#print('\n\nconv2_neg_blocks:\n', conv1_neg_blocks.shape, '\n', conv1_neg_blocks[2, 2], '\n\n')
				#raise(SystemExit)
				conv1_sep_blocked = torch.cat((conv1_pos_blocks, conv1_neg_blocks), 0)
				#print('\nconv1_blocks shape', conv1_blocks.shape, '\n')
				#print(conv1_blocks.detach().cpu().numpy()[60, 234, :8, :8])
				conv1_weight_sums = torch.abs(self.conv1.weight).sum((1, 2, 3))
				conv1_weight_sums_blocked = torch.cat(conv1_weight_sums_blocked, 0)
				conv1_weight_sums_sep_blocked = torch.cat(conv1_weight_sums_sep_blocked, 0)

				w_pos = self.conv1.weight.clone()
				w_pos[w_pos < 0] = 0
				w_neg = self.conv1.weight.clone()
				w_neg[w_neg >= 0] = 0
				conv1_pos = F.conv2d(self.input, w_pos)
				conv1_neg = F.conv2d(self.input, w_neg)
				conv1_sep = torch.cat((conv1_neg, conv1_pos), 0)
				conv1_weight_sums_sep = torch.cat((w_pos.sum((1, 2, 3)), w_neg.sum((1, 2, 3))), 0)

		if args.merge_bn:
			self.bias1 = self.bn1.bias.view(1, -1, 1, 1) - self.bn1.running_mean.data.view(1, -1, 1, 1) * self.bn1.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn1.running_var.data.view(1, -1, 1, 1) + 0.0000001)
			self.conv1_ = self.conv1_no_bias + self.bias1
		else:
			self.conv1_ = self.conv1_no_bias
			self.bias1 = torch.Tensor([0])

		if epoch == 0 and i == 0 and s == 0 and self.training:
			print('conv1 out shape:', self.conv1_.shape)

		if args.current1 > 0:
			with torch.no_grad():

				if (args.uniform_ind > 0 and self.training) or (args.uniform_ind > 0 and args.noise_test):
					sigmas1 = torch.ones_like(self.conv1_) * args.uniform_ind * torch.max(torch.abs(self.conv1_))
					noise1_distr = Uniform(-sigmas1, sigmas1)
					self.noise1 = noise1_distr.sample()

				elif (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
					noise1_distr = Uniform(torch.ones_like(self.conv1_) * args.uniform_dep, torch.ones_like(self.conv1_) / args.uniform_dep)
					self.noise1 = noise1_distr.sample()

				elif (args.normal_ind > 0 and self.training) or (args.normal_ind > 0 and args.noise_test):
					sigmas1 = (torch.ones_like(self.conv1_) * args.normal_ind * torch.max(torch.abs(self.conv1_))).pow(2)
					noise1_distr = Normal(loc=0, scale=torch.ones_like(self.conv1_) * args.normal_ind * torch.max(torch.abs(self.conv1_)))
					self.noise1 = noise1_distr.sample()

				elif (args.normal_dep > 0 and self.training) or (args.normal_dep > 0 and args.noise_test):
					sigmas1 = (args.normal_dep * self.conv1_).pow(2)
					noise1_distr = Normal(loc=0, scale=args.normal_dep * self.conv1_)
					self.noise1 = noise1_distr.sample()

				else:
					filter1 = torch.abs(self.conv1.weight)
					x_max1 = 1  #torch.max(self.input) is always 1 for quantized cifar input
					if args.merged_dac:  #merged DAC digital input (for the current chip - first and third layer input):
						sigmas1 = F.conv2d(self.input, filter1)
						w_max1 = torch.max(filter1)
						noise1_distr = Normal(loc=0, scale=torch.sqrt(0.1 * (w_max1 / args.current1) * sigmas1))
						if i < 20:  #calcuate power consumption
							a1 = F.conv2d(self.input, filter1)
							a1_sums = torch.sum(a1, dim=(1, 2, 3))
							self.p1 = 1.0e-6 * 1.2 * args.current1 * torch.mean(a1_sums) / (x_max1 * w_max1)
					else:  #external DAC (for the next gen hardware) or analog input in the current chip (layers 2 and 4)
						f1 = filter1.pow(2) + filter1
						sigmas1 = F.conv2d(self.input, f1)
						noise1_distr = Normal(loc=0, scale=torch.sqrt(0.1 * (x_max1 / args.current1) * sigmas1))
						if i < 20:  #calcuate power consumption
							a1 = F.conv2d(self.input, filter1)
							a1_sums = torch.sum(a1, dim=(1, 2, 3))
							self.p1 = 1.0e-6 * 1.2 * args.current1 * torch.mean(a1_sums) / x_max1

					self.noise1 = noise1_distr.sample()

			if (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
				self.conv1_noisy = self.conv1_ * self.noise1.cuda()
			else:
				self.conv1_noisy = self.conv1_ + self.noise1.cuda()

			conv1_out = self.conv1_noisy
		else:
			conv1_out = self.conv1_
			if i < 20:
				filter1 = torch.abs(self.conv1.weight)
				a1 = F.conv2d(self.input, filter1)
				a1_sums = torch.sum(a1, dim=(1, 2, 3))
				x_max1 = 1  #torch.max(self.input) is always 1 for quantized cifar input
				if args.merged_dac:
					w_max1 = torch.max(torch.abs(self.conv1.weight))
					self.p1 = 1.0e-6 * 1.2 * 30 * torch.mean(a1_sums) / (x_max1 * w_max1)  #50nA corresponds to no noise
				else:
					self.p1 = 1.0e-6 * 1.2 * 30 * torch.mean(a1_sums) / x_max1

		if i == 0 and not self.training:
			pass
		#print('\n\t\t\t\t\tTesting...  Scaling conv1 by 2...\n')

		pool1 = self.pool(conv1_out)

		if args.batchnorm and not args.merge_bn:
			bn1 = self.bn1(pool1)
			self.pool1_out = bn1
		else:
			self.pool1_out = pool1

		self.relu1_ = self.relu(self.pool1_out)
		if epoch == 0 and i == 0 and s == 0 and self.training:
			print('relu1 out shape:', self.relu1_.shape)

		if args.act_max1 > 0:
			if args.train_act_max:
				self.relu1_clipped = torch.where(self.relu1_ > self.act_max1, self.act_max1, self.relu1_)  #fastest
			else:
				self.relu1_clipped = torch.clamp(self.relu1_, max=args.act_max1)
			self.relu1 = self.relu1_clipped
		else:
			self.relu1 = self.relu1_

		self.relu1_.retain_grad()
		if args.train_act_max:
			self.relu1_clipped.retain_grad()
		self.relu1.retain_grad()

		if args.train_w_max:
			self.w_max1.retain_grad()
			self.w_min1.retain_grad()

		if args.dropout_conv > 0:
			self.relu1 = self.dropout(self.relu1)

		if args.q_a2 > 0:
			self.relu1 = self.quantize2(self.relu1)

		self.conv2_no_bias = self.conv2(self.relu1)

		if args.plot or args.write:
			with torch.no_grad():
				'''
				Take 64 input FMs, do 1x1 conv 25 times on them
				f = (128, 64, 5, 5)

				conv2_blocks = []
				for fi in range(args.fs):
					for fk in range(args.fs):
						#print('\nweight shape:', self.conv2.weight[:, :, fi, fk].shape)
						block = F.conv2d(self.relu1, self.conv2.weight[:, :, fi, fk].view(self.conv2.weight.shape[0], self.conv2.weight.shape[1], 1, 1))
						#print('block shape:', block.shape)
						conv2_blocks.append(block)
				conv2_blocks = torch.cat(conv2_blocks, 1)
				#print('\nconv2_blocks shape', conv2_blocks.shape)
				#print(conv2_blocks.detach().cpu().numpy()[60, 1234, :8, :8]
				'''
				conv2_blocks = []
				conv2_pos_blocks = []
				conv2_neg_blocks = []
				conv2_weight_sums_blocked = []
				conv2_weight_sums_sep_blocked = []
				block_size = 64
				dim = self.conv2.weight.shape[1]  #weights shape: (fm_out, fm_in, fs, fs)
				num_blocks = max(dim // block_size, 1)  #min 1 block, must be cleanly divisible!
				f = self.conv2.weight.permute(2, 3, 0, 1).contiguous().view(-1, dim, 1, 1)
				for b in range(num_blocks):
					weight_block = f[:, b * block_size: (b + 1) * block_size, :, :]
					weight_block_pos = weight_block.clone()
					weight_block_neg = weight_block.clone()
					weight_block_pos[weight_block_pos <= 0] = 0
					weight_block_neg[weight_block_neg > 0] = 0
					if b == 0:
						print('\n\nNumber of blocks:', num_blocks, 'weight block shape:', weight_block.shape, '\nweights for single output neuron:', weight_block[0].shape, '\nActual weights (one block):\n', weight_block[0].detach().cpu().numpy().ravel(), '\nWeight block sum(0):\n',
						      weight_block.sum((1, 2, 3)).shape, '\n\n')
					input_block = self.relu1[:, b * block_size: (b + 1) * block_size, :, :]
					conv2_blocks.append(F.conv2d(input_block, weight_block))
					conv2_pos_blocks.append(F.conv2d(input_block, weight_block_pos))
					conv2_neg_blocks.append(F.conv2d(input_block, weight_block_neg))
					conv2_weight_sums_blocked.append(torch.abs(weight_block).sum((1, 2, 3)))
					conv2_weight_sums_sep_blocked.extend([weight_block_pos.sum((1, 2, 3)), weight_block_neg.sum((1, 2, 3))])

				conv2_blocks = torch.cat(conv2_blocks, 1)  #conv_out shape: (bs, fms, h, v)
				#print('\nconv2_blocks shape', conv2_blocks.shape, '\n')
				#print(conv2_blocks.detach().cpu().numpy()[60, 1234, :8, :8])
				conv2_pos_blocks = torch.cat(conv2_pos_blocks, 1)
				conv2_neg_blocks = torch.cat(conv2_neg_blocks, 1)
				conv2_sep_blocked = torch.cat((conv2_pos_blocks, conv2_neg_blocks), 0)
				conv2_weight_sums = torch.abs(self.conv2.weight).sum((1, 2, 3))
				conv2_weight_sums_blocked = torch.cat(conv2_weight_sums_blocked, 0)
				conv2_weight_sums_sep_blocked = torch.cat(conv2_weight_sums_sep_blocked, 0)

				w_pos = self.conv2.weight.clone()
				w_pos[w_pos < 0] = 0
				w_neg = self.conv2.weight.clone()
				w_neg[w_neg >= 0] = 0
				conv2_pos = F.conv2d(self.relu1, w_pos)
				conv2_neg = F.conv2d(self.relu1, w_neg)
				conv2_sep = torch.cat((conv2_neg, conv2_pos), 0)
				conv2_weight_sums_sep = torch.cat((w_pos.sum((1, 2, 3)), w_neg.sum((1, 2, 3))), 0)  #raise(SystemExit)

		if args.merge_bn:
			self.bias2 = self.bn2.bias.view(1, -1, 1, 1) - self.bn2.running_mean.data.view(1, -1, 1, 1) * self.bn2.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn2.running_var.data.view(1, -1, 1, 1) + 0.0000001)
			self.conv2_ = self.conv2_no_bias + self.bias2
		else:
			self.conv2_ = self.conv2_no_bias
			self.bias2 = torch.Tensor([0])

		if epoch == 0 and i == 0 and s == 0 and self.training:
			print('conv2 out shape:', self.conv2_.shape)

		if args.current2 > 0:

			with torch.no_grad():
				filter2 = torch.abs(self.conv2.weight)

				if (args.uniform_ind > 0 and self.training) or (args.uniform_ind > 0 and args.noise_test):
					sigmas2 = torch.ones_like(self.conv2_) * args.uniform_ind * torch.max(torch.abs(self.conv2_))
					noise2_distr = Uniform(-sigmas2, sigmas2)
					self.noise2 = noise2_distr.sample()

				elif (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
					noise2_distr = Uniform(torch.ones_like(self.conv2_) * args.uniform_dep, torch.ones_like(self.conv2_) / args.uniform_dep)
					self.noise2 = noise2_distr.sample()

				elif (args.normal_ind > 0 and self.training) or (args.normal_ind > 0 and args.noise_test):
					sigmas2 = (torch.ones_like(self.conv2_) * args.normal_ind * torch.max(torch.abs(self.conv2_))).pow(2)
					noise2_distr = Normal(loc=0, scale=torch.ones_like(self.conv2_) * args.normal_ind * torch.max(torch.abs(self.conv2_)))
					self.noise2 = noise2_distr.sample()

				elif (args.normal_dep > 0 and self.training) or (args.normal_dep > 0 and args.noise_test):
					sigmas2 = (args.normal_dep * self.conv2_).pow(2)
					noise2_distr = Normal(loc=0, scale=args.normal_dep * self.conv2_)
					self.noise2 = noise2_distr.sample()

				else:  #accurate noise model: this layer always accepts either analog input or external DAC
					f2 = filter2.pow(2) + filter2
					sigmas2 = F.conv2d(self.relu1, f2)
					x_max2 = torch.max(self.relu1)
					noise2_distr = Normal(loc=0, scale=torch.sqrt(0.1 * (x_max2 / args.current2) * sigmas2))
					self.noise2 = noise2_distr.sample()

					if i < 20:
						a2_sums = torch.sum(sigmas2, dim=(1, 2, 3))
						self.p2 = 1.0e-6 * 1.2 * args.current2 * torch.mean(a2_sums) / x_max2

			if (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
				conv2_noisy = self.conv2_ * self.noise2.cuda()
			else:
				conv2_noisy = self.conv2_ + self.noise2
			conv2_out = conv2_noisy
		else:
			conv2_out = self.conv2_
			if i < 20:
				f2 = torch.abs(self.conv2.weight)
				a2 = F.conv2d(self.relu1, f2)
				a2_sums = torch.sum(a2, dim=(1, 2, 3))
				x_max2 = torch.max(self.relu1)
				self.p2 = 30 * 1.0e-6 * 1.2 * torch.mean(a2_sums) / x_max2

		pool2 = self.pool(conv2_out)

		if args.batchnorm and not args.merge_bn:
			bn2 = self.bn2(pool2)
			self.pool2_out = bn2
		else:
			self.pool2_out = pool2

		self.relu2_ = self.relu(self.pool2_out)
		if epoch == 0 and i == 0 and s == 0 and self.training:
			print('relu2 out shape:', self.relu2_.shape)

		if args.act_max2 > 0:
			if args.train_act_max:
				self.relu2_clipped = torch.where(self.relu2_ > self.act_max2, self.act_max2, self.relu2_)
			else:
				self.relu2_clipped = torch.clamp(self.relu2_, max=args.act_max2)
			self.relu2 = self.relu2_clipped
		else:
			self.relu2 = self.relu2_

		self.relu2.retain_grad()

		if args.dropout > 0:
			self.relu2 = self.dropout(self.relu2)

		self.relu2 = self.relu2.view(self.relu2.size(0), -1)
		if epoch == 0 and i == 0 and s == 0 and self.training:
			print('relu2 out shape:', self.relu2.shape)

		if args.q_a3 > 0:
			self.relu2 = self.quantize3(self.relu2)

		self.linear1_no_bias = self.linear1(self.relu2)

		if args.plot or args.write:
			with torch.no_grad():
				linear1_blocks = []  #linear1 shape: (390, 3200)
				linear1_pos_blocks = []
				linear1_neg_blocks = []
				linear1_weight_sums_blocked = []
				linear1_weight_sums_sep_blocked = []
				block_size = 64
				dim = self.linear1.weight.shape[1]  #weights shape: (out, in)
				num_blocks = max(dim // block_size, 1)  #min 1 block, must be cleanly divisible!
				for b in range(num_blocks):
					weight_block = self.linear1.weight[:, b * block_size: (b + 1) * block_size]
					weight_block_pos = weight_block.clone()
					weight_block_neg = weight_block.clone()
					weight_block_pos[weight_block_pos <= 0] = 0
					weight_block_neg[weight_block_neg > 0] = 0
					if b == 0:
						print('\n\nNumber of blocks:', num_blocks, 'weight block shape:', weight_block.shape, '\nweights for single output neuron:', weight_block[0].shape, '\nActual weights (one block):\n', weight_block[0].detach().cpu().numpy().ravel(), '\nWeight block sum(0):\n',
						      weight_block.sum(1).shape, '\n\n')
					input_block = self.relu2[:, b * block_size: (b + 1) * block_size]
					linear1_blocks.append(F.linear(weight_block, input_block))
					linear1_pos_blocks.append(F.linear(input_block, weight_block_pos))
					linear1_neg_blocks.append(F.linear(input_block, weight_block_neg))
					linear1_weight_sums_blocked.append(torch.abs(weight_block).sum(1))
					linear1_weight_sums_sep_blocked.extend([weight_block_pos.sum(1), weight_block_neg.sum(1)])

				linear1_blocks = torch.cat(linear1_blocks, 1)
				#print('\nlinear1_blocks shape', linear1_blocks.shape, '\n')
				#print(linear1_blocks.detach().cpu().numpy()[:8, :8])
				linear1_pos_blocks = torch.cat(linear1_pos_blocks, 1)
				linear1_neg_blocks = torch.cat(linear1_neg_blocks, 1)
				linear1_sep_blocked = torch.cat((linear1_pos_blocks, linear1_neg_blocks), 0)
				linear1_weight_sums = torch.abs(self.linear1.weight).sum(1)
				linear1_weight_sums_blocked = torch.cat(linear1_weight_sums_blocked, 0)
				linear1_weight_sums_sep_blocked = torch.cat(linear1_weight_sums_sep_blocked, 0)

				w_pos = self.linear1.weight.clone()
				w_pos[w_pos < 0] = 0
				w_neg = self.linear1.weight.clone()
				w_neg[w_neg >= 0] = 0
				linear1_pos = F.linear(self.relu2, w_pos)
				linear1_neg = F.linear(self.relu2, w_neg)
				linear1_sep = torch.cat((linear1_neg, linear1_pos), 0)
				linear1_weight_sums_sep = torch.cat((w_pos.sum(1), w_neg.sum(1)), 0)  #raise(SystemExit)

		if args.merge_bn:
			self.bias3 = self.bn3.bias.view(1, -1) - self.bn3.running_mean.data.view(1, -1) * self.bn3.weight.data.view(1, -1) / torch.sqrt(self.bn3.running_var.data.view(1, -1) + 0.0000001)
			self.linear1_ = self.linear1_no_bias + self.bias3
		else:
			self.linear1_ = self.linear1_no_bias
			self.bias3 = torch.Tensor([0])

		if args.current3 > 0:

			with torch.no_grad():

				if (args.uniform_ind > 0 and self.training) or (args.uniform_ind > 0 and args.noise_test):
					sigmas3 = torch.ones_like(self.linear1_) * args.uniform_ind * torch.max(torch.abs(self.linear1_))
					noise3_distr = Uniform(-sigmas3, sigmas3)
					self.noise3 = noise3_distr.sample()

				elif (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
					noise3_distr = Uniform(torch.ones_like(self.linear1_) * args.uniform_dep, torch.ones_like(self.linear1_) / args.uniform_dep)
					self.noise3 = noise3_distr.sample()

				elif (args.normal_ind > 0 and self.training) or (args.normal_ind > 0 and args.noise_test):
					sigmas3 = (torch.ones_like(self.linear1_) * args.normal_ind * torch.max(torch.abs(self.linear1_))).pow(2)
					noise3_distr = Normal(loc=0, scale=torch.ones_like(self.linear1_) * args.normal_ind * torch.max(torch.abs(self.linear1_)))
					self.noise3 = noise3_distr.sample()

				elif (args.normal_dep > 0 and self.training) or (args.normal_dep > 0 and args.noise_test):
					sigmas3 = (args.normal_dep * self.linear1_).pow(2)
					noise3_distr = Normal(loc=0, scale=args.normal_dep * self.linear1_)
					self.noise3 = noise3_distr.sample()

				else:
					filter3 = torch.abs(self.linear1.weight)
					x_max3 = torch.max(self.relu2)
					if args.merged_dac:  #merged DAC digital input (for the current chip - first and third layer input):
						sigmas3 = F.linear(self.relu2, filter3, bias=self.linear1.bias)
						w_max3 = torch.max(filter3)
						noise3_distr = Normal(loc=0, scale=torch.sqrt(0.1 * (w_max3 / args.current3) * sigmas3))
						if i < 20:  #calcuate power consumption
							a3 = F.linear(self.relu2, filter3, bias=self.linear1.bias)
							a3_sums = torch.sum(a3, dim=1)
							self.p3 = 1.0e-6 * 1.2 * args.current3 * torch.mean(a3_sums) / (x_max3 * w_max3)
					else:  #external DAC (for the next gen hardware) or analog input in the current chip (layers 2 and 4)
						f3 = filter3.pow(2) + filter3
						sigmas3 = F.linear(self.relu2, f3, bias=self.linear1.bias)
						noise3_distr = Normal(loc=0, scale=torch.sqrt(0.1 * (x_max3 / args.current3) * sigmas3))
						if i < 20:  #calcuate power consumption
							a3 = F.linear(self.relu2, f3, bias=self.linear1.bias)
							a3_sums = torch.sum(a3, dim=1)
							self.p3 = 1.0e-6 * 1.2 * args.current3 * torch.mean(a3_sums) / x_max3

					self.noise3 = noise3_distr.sample()

			if (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
				linear1_noisy = self.linear1_ * self.noise3.cuda()
			else:
				linear1_noisy = self.linear1_ + self.noise3

			linear1_out = linear1_noisy
		else:
			linear1_out = self.linear1_
			if i < 20:
				f3 = torch.abs(self.linear1.weight)
				a3 = F.linear(self.relu2, f3, bias=self.linear1.bias)
				a3_sums = torch.sum(a3, dim=1)
				x_max3 = torch.max(self.relu2)
				if args.merged_dac:
					w_max3 = torch.max(torch.abs(self.linear1.weight))
					self.p3 = 30 * 1.0e-6 * 1.2 * torch.mean(a3_sums) / (x_max3 * w_max3)
				else:
					self.p3 = 30 * 1.0e-6 * 1.2 * torch.mean(a3_sums) / x_max3

		if args.batchnorm and args.bn3 and not args.merge_bn:
			self.linear1_out = self.bn3(linear1_out)
		else:
			self.linear1_out = linear1_out

		self.relu3_ = self.relu(self.linear1_out)

		if epoch == 0 and i == 0 and s == 0 and self.training:
			print('relu3 out shape:', self.relu3_.shape, '\n')

		if args.act_max3 > 0:
			if args.train_act_max:
				self.relu3_clipped = torch.where(self.relu3_ > self.act_max3, self.act_max3, self.relu3_)
			else:
				self.relu3_clipped = torch.clamp(self.relu3_, max=args.act_max3)
			self.relu3 = self.relu3_clipped
		else:
			self.relu3 = self.relu3_

		self.relu3.retain_grad()

		if args.dropout > 0:
			self.relu3 = self.dropout(self.relu3)

		if args.q_a4 > 0:
			self.relu3 = self.quantize4(self.relu3)

		self.linear2_no_bias = self.linear2(self.relu3)

		if args.plot or args.write:
			with torch.no_grad():
				linear2_blocks = []  #linear1 shape: (390, 3200)
				linear2_pos_blocks = []
				linear2_neg_blocks = []
				linear2_weight_sums_blocked = []
				linear2_weight_sums_sep_blocked = []
				block_size = 64
				dim = self.linear2.weight.shape[1]  #weights shape: (out, in)
				print('\n\nself.linear2.weight.shape', self.linear2.weight.shape)
				num_blocks = max(dim // block_size, 1)  #min 1 block, should be cleanly divisible!
				print('num_blocks', num_blocks)
				print('self.relu3.shape', self.relu3.shape)
				for b in range(num_blocks):
					weight_block = self.linear2.weight[:, b * block_size: (b + 1) * block_size]
					weight_block_pos = weight_block.clone()
					weight_block_neg = weight_block.clone()
					weight_block_pos[weight_block_pos <= 0] = 0
					weight_block_neg[weight_block_neg > 0] = 0
					if b == 0:
						print('\n\nNumber of blocks:', num_blocks, 'weight block shape:', weight_block.shape, '\nweights for single output neuron:', weight_block[0].shape, '\nActual weights (one block):\n', weight_block[0].detach().cpu().numpy().ravel(), '\nWeight block sum(0):\n',
						      weight_block.sum(1).shape, '\n\n')
					input_block = self.relu3[:, b * block_size: (b + 1) * block_size]
					linear2_blocks.append(F.linear(weight_block, input_block))
					linear2_pos_blocks.append(F.linear(input_block, weight_block_pos))
					linear2_neg_blocks.append(F.linear(input_block, weight_block_neg))
					linear2_weight_sums_blocked.append(torch.abs(weight_block).sum(1))
					linear2_weight_sums_sep_blocked.extend([weight_block_pos.sum(1), weight_block_neg.sum(1)])

				linear2_blocks = torch.cat(linear2_blocks, 1)
				#print('\nlinear2_blocks shape', linear2_blocks.shape, '\n')
				#print(linear2_blocks.detach().cpu().numpy()[:8, :8])
				linear2_pos_blocks = torch.cat(linear2_pos_blocks, 1)
				linear2_neg_blocks = torch.cat(linear2_neg_blocks, 1)
				linear2_sep_blocked = torch.cat((linear2_pos_blocks, linear2_neg_blocks), 0)
				linear2_weight_sums = torch.abs(self.linear2.weight).sum(1)
				linear2_weight_sums_blocked = torch.cat(linear2_weight_sums_blocked, 0)
				linear2_weight_sums_sep_blocked = torch.cat(linear2_weight_sums_sep_blocked, 0)

				w_pos = self.linear2.weight.clone()
				w_pos[w_pos < 0] = 0
				w_neg = self.linear2.weight.clone()
				w_neg[w_neg >= 0] = 0
				linear2_pos = F.linear(self.relu3, w_pos)
				linear2_neg = F.linear(self.relu3, w_neg)
				linear2_sep = torch.cat((linear2_neg, linear2_pos), 0)
				linear2_weight_sums_sep = torch.cat((w_pos.sum(1), w_neg.sum(1)), 0)  #raise (SystemExit)

		if args.merge_bn:
			if self.training:
				print('\n\n************ Merging BatchNorm during training! **********\n\n')
			self.bias4 = self.bn4.bias.view(1, -1) - self.bn4.running_mean.data.view(1, -1) * self.bn4.weight.data.view(1, -1) / torch.sqrt(self.bn4.running_var.data.view(1, -1) + 0.0000001)
			self.linear2_ = self.linear2_no_bias + self.bias4
		else:
			self.linear2_ = self.linear2_no_bias
			self.bias4 = torch.Tensor([0])

		if args.current4 > 0:

			with torch.no_grad():
				if (args.uniform_ind > 0 and self.training) or (args.uniform_ind > 0 and args.noise_test):
					sigmas4 = torch.ones_like(self.linear2_) * args.uniform_ind * torch.max(torch.abs(self.linear2_))
					noise4_distr = Uniform(-sigmas4, sigmas4)
					self.noise4 = noise4_distr.sample()

				elif (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
					noise4_distr = Uniform(torch.ones_like(self.linear2_) * args.uniform_dep, torch.ones_like(self.linear2_) / args.uniform_dep)
					self.noise4 = noise4_distr.sample()

				elif (args.normal_ind > 0 and self.training) or (args.normal_ind > 0 and args.noise_test):
					sigmas4 = (torch.ones_like(self.linear2_) * args.normal_ind * torch.max(torch.abs(self.linear2_))).pow(2)
					noise4_distr = Normal(loc=0, scale=torch.ones_like(self.linear2_) * args.normal_ind * torch.max(torch.abs(self.linear2_)))
					self.noise4 = noise4_distr.sample()

				elif (args.normal_dep > 0 and self.training) or (args.normal_dep > 0 and args.noise_test):
					sigmas4 = (args.normal_dep * self.linear2_).pow(2)
					noise4_distr = Normal(loc=0, scale=args.normal_dep * self.linear2_)
					self.noise4 = noise4_distr.sample()

				else:
					filter4 = torch.abs(self.linear2.weight)
					f4 = filter4.pow(2) + filter4
					sigmas4 = F.linear(self.relu3, f4, bias=self.linear2.bias)
					x_max4 = torch.max(self.relu3)
					noise4_distr = Normal(loc=0, scale=torch.sqrt(0.1 * (x_max4 / args.current4) * sigmas4))

					if i < 20:
						a4_sums = torch.sum(sigmas4, dim=1)
						self.p4 = 1.0e-6 * 1.2 * args.current4 * torch.mean(a4_sums) / x_max4

					self.noise4 = noise4_distr.sample()

			if (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
				linear2_noisy = self.linear2_ * self.noise4.cuda()
			else:
				linear2_noisy = self.linear2_ + self.noise4

			linear2_out = linear2_noisy
		else:
			linear2_out = self.linear2_
			if i < 20:
				f4 = torch.abs(self.linear2.weight)
				a4 = F.linear(self.relu3, f4, bias=self.linear2.bias)
				a4_sums = torch.sum(a4, dim=1)
				x_max4 = torch.max(self.relu3)
				self.p4 = 30 * 1.0e-6 * 1.2 * torch.mean(a4_sums) / x_max4

		if args.batchnorm and args.bn4 and not args.merge_bn:
			self.linear2_out = self.bn4(linear2_out)
		else:
			self.linear2_out = linear2_out

		if (args.plot and s == 0 and i == 0 and epoch in [0, 1, 5, 10, 50, 100, 150, 249] and self.training) or args.write or (args.resume is not None and args.plot):

			if self.create_dir:
				utils.saveargs(args)
				self.create_dir = False

			if (epoch == 0 and i == 0) or args.plot:
				print('\n\n\nBatch size', list(self.input.size())[0], '\n\n\n')

			names = ['input', 'weights', 'weight sums', 'weight sums diff', 'weight sums blocked', 'weight sums diff blocked', 'vmm', 'vmm diff', 'vmm blocked', 'vmm diff blocked']
			layer1 = [[self.input], [self.conv1.weight], [conv1_weight_sums], [conv1_weight_sums_sep], [conv1_weight_sums_blocked], [conv1_weight_sums_sep_blocked], [self.conv1_no_bias], [conv1_sep], [conv1_blocks], [conv1_sep_blocked]]
			layer2 = [[self.relu1], [self.conv2.weight], [conv2_weight_sums], [conv2_weight_sums_sep], [conv2_weight_sums_blocked], [conv2_weight_sums_sep_blocked], [self.conv2_no_bias], [conv2_sep], [conv2_blocks], [conv2_sep_blocked]]
			layer3 = [[self.relu2], [self.linear1.weight], [linear1_weight_sums], [linear1_weight_sums_sep], [linear1_weight_sums_blocked], [linear1_weight_sums_sep_blocked], [self.linear1_no_bias], [linear1_sep], [linear1_blocks], [linear1_sep_blocked]]
			layer4 = [[self.relu3], [self.linear2.weight], [linear2_weight_sums], [linear2_weight_sums_sep], [linear2_weight_sums_blocked], [linear2_weight_sums_sep_blocked], [self.linear2_no_bias], [linear2_sep], [linear2_blocks], [linear2_sep_blocked]]
			if args.merge_bn:
				layer1.append([self.bias1])
				layer2.append([self.bias2])
				layer3.append([self.bias3])
				layer4.append([self.bias4])
				names.append('biases')
			layer1.append([self.pool1_out])
			layer2.append([self.pool2_out])
			layer3.append([self.linear1_out])
			layer4.append([self.linear2_out])
			names.append('pre-activation')
			if args.current1 > 0:
				layer1 += ([sigmas1], [self.noise1])
				layer2 += ([sigmas2], [self.noise2])
				layer3 += ([sigmas3], [self.noise3])
				layer4 += ([sigmas4], [self.noise4])
				names.extend(['sigmas', 'noise', 'noise/range'])
			layers = [layer1, layer2, layer3, layer4]

			for layer in layers:  #de-torch arrays
				for array in layer:
					array[0] = array[0].half().detach().cpu().numpy()  #.astype(np.float16)
					print(array[0].type)

			if args.plot:
				figsize = (len(names) * 7, 4 * 6)
				print('\nPlotting {}\n'.format(names))
				var_ = [np.prod(self.conv1.weight.shape[1:]), np.prod(self.conv2.weight.shape[1:]), np.prod(self.linear1.weight.shape[1:]), np.prod(self.linear2.weight.shape[1:])]
				if args.var_name == 'blank':
					var_name = ''
				else:
					var_name = args.var_name

				plot_layers(num_layers=4, models=[args.checkpoint_dir], epoch=epoch, i=i, layers=layers, names=names, var=var_name, vars=[var_], figsize=figsize, acc=acc, tag=args.tag)
				print('\n\nSaved plots to {}\n\n'.format(args.checkpoint_dir))
				if args.resume is not None:
					raise (SystemExit)
				raise (SystemExit)

			if args.write:
				np.save(args.checkpoint_dir + 'layers.npy', np.array(layers))
				print('\n\nnumpy arrays saved to', args.checkpoint_dir, '\n\n')
				raise (SystemExit)

		return self.linear2_out


def noisynet(parameters):
	global args
	args = parameters
	model = Net()
	return model