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

import utils
from quantized_modules_clean import QConv2d, QLinear
from plot_histograms import plot_layers

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Your project title goes here')

parser.add_argument('--dataset', type=str, default='cifar_RGB_4bit.npz', metavar='', help='name of dataset')
parser.add_argument('--precision', type=str, default='full', metavar='', help='float precision: half (16 bits) or full (32 bits)')
parser.add_argument('--resume', type=str, default=None, metavar='', help='full path of models to resume training')
parser.add_argument('--tag', type=str, default='', metavar='', help='string to prepend to args.checkpoint_dir')

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--generate_input', dest='generate_input', action='store_true')
feature_parser.add_argument('--no-generate_input', dest='generate_input', action='store_false')
parser.set_defaults(generate_input=False)  #default is to load entire cifar into RAM

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--use_bias', dest='use_bias', action='store_true')
feature_parser.add_argument('--no-use_bias', dest='use_bias', action='store_false')
parser.set_defaults(use_bias=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--augment', dest='augment', action='store_true')
feature_parser.add_argument('--no-augment', dest='augment', action='store_false')
parser.set_defaults(augment=True)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--normalize', dest='normalize', action='store_true')
feature_parser.add_argument('--no-normalize', dest='normalize', action='store_false')
parser.set_defaults(normalize=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--train_act_max', dest='train_act_max', action='store_true')
feature_parser.add_argument('--no-train_act_max', dest='train_act_max', action='store_false')
parser.set_defaults(train_act_max=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--train_w_max', dest='train_w_max', action='store_true')
feature_parser.add_argument('--no-train_w_max', dest='train_w_max', action='store_false')
parser.set_defaults(train_w_max=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--batchnorm', dest='batchnorm', action='store_true')
feature_parser.add_argument('--no-batchnorm', dest='batchnorm', action='store_false')
parser.set_defaults(batchnorm=True)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--bn3', dest='bn3', action='store_true')
feature_parser.add_argument('--no-bn3', dest='bn3', action='store_false')
parser.set_defaults(bn3=True)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--bn4', dest='bn4', action='store_true')
feature_parser.add_argument('--no-bn4', dest='bn4', action='store_false')
parser.set_defaults(bn4=True)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--biprecision', dest='biprecision', action='store_true')
feature_parser.add_argument('--no-biprecision', dest='biprecision', action='store_false')
parser.set_defaults(biprecision=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--stochastic', dest='stochastic', action='store_true')
feature_parser.add_argument('--no-stochastic', dest='stochastic', action='store_false')
parser.set_defaults(stochastic=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--amsgrad', dest='amsgrad', action='store_true')
feature_parser.add_argument('--no-amsgrad', dest='amsgrad', action='store_false')
parser.set_defaults(amsgrad=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--nesterov', dest='nesterov', action='store_true')
feature_parser.add_argument('--no-nesterov', dest='nesterov', action='store_false')
parser.set_defaults(nesterov=True)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--split', dest='split', action='store_true')
feature_parser.add_argument('--no-split', dest='split', action='store_false')
parser.set_defaults(split=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--distort_w_train', dest='distort_w_train', action='store_true')
feature_parser.add_argument('--no-distort_w_train', dest='distort_w_train', action='store_false')
parser.set_defaults(distort_w_train=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--distort_w_test', dest='distort_w_test', action='store_true')
feature_parser.add_argument('--no-distort_w_test', dest='distort_w_test', action='store_false')
parser.set_defaults(distort_w_test=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--write', dest='write', action='store_true')
feature_parser.add_argument('--no-write', dest='write', action='store_false')
parser.set_defaults(write=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--plot', dest='plot', action='store_true')
feature_parser.add_argument('--no-plot', dest='plot', action='store_false')
parser.set_defaults(plot=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--weightnorm', dest='weightnorm', action='store_true')
feature_parser.add_argument('--no-weightnorm', dest='weightnorm', action='store_false')
parser.set_defaults(weightnorm=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--print_clip', dest='print_clip', action='store_true')
feature_parser.add_argument('--no-print_clip', dest='print_clip', action='store_false')
parser.set_defaults(print_clip=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--track_running_stats', dest='track_running_stats', action='store_true')
feature_parser.add_argument('--no-track_running_stats', dest='track_running_stats', action='store_false')
parser.set_defaults(track_running_stats=True)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--noise_test', dest='noise_test', action='store_true')
feature_parser.add_argument('--no-noise_test', dest='noise_test', action='store_false')
parser.set_defaults(noise_test=False)

parser.add_argument('--current', type=float, default=0.0, metavar='', help='current level in nano Amps, which determines the noise level. 0 disables noise')
parser.add_argument('--current1', type=float, default=0.0, metavar='', help='current level in nano Amps, which determines the noise level. 0 disables noise')
parser.add_argument('--current2', type=float, default=0.0, metavar='', help='current level in nano Amps, which determines the noise level. 0 disables noise')
parser.add_argument('--current3', type=float, default=0.0, metavar='', help='current level in nano Amps, which determines the noise level. 0 disables noise')
parser.add_argument('--current4', type=float, default=0.0, metavar='', help='current level in nano Amps, which determines the noise level. 0 disables noise')
parser.add_argument('--train_current', type=float, default=0.0, metavar='', help='current level in nano Amps, which determines the noise level. 0 disables noise')
parser.add_argument('--test_current', type=float, default=0.0, metavar='', help='current level in nano Amps, which determines the noise level. 0 disables noise')
parser.add_argument('--act_max', type=float, default=0.0, metavar='', help='max value for ReLU act (clipping upper bound)')
parser.add_argument('--act_max1', type=float, default=0.0, metavar='', help='max value for ReLU1 act (clipping upper bound)')
parser.add_argument('--act_max2', type=float, default=0.0, metavar='', help='max value for ReLU2 act (clipping upper bound)')
parser.add_argument('--act_max3', type=float, default=0.0, metavar='', help='max value for ReLU3 act (clipping upper bound)')
parser.add_argument('--w_min1', type=float, default=0.0, metavar='', help='min value for layer 1 weights (clipping lower bound)')
parser.add_argument('--w_max1', type=float, default=0.0, metavar='', help='max value for layer 1 weights (clipping upper bound)')
parser.add_argument('--w_max2', type=float, default=0.0, metavar='', help='max value for layer 2 weights (clipping upper bound)')
parser.add_argument('--w_max3', type=float, default=0.0, metavar='', help='max value for layer 3 weights (clipping upper bound)')
parser.add_argument('--w_max4', type=float, default=0.0, metavar='', help='max value for layer 4 weights (clipping upper bound)')
parser.add_argument('--grad_clip', type=float, default=0.0, metavar='', help='clip gradients if grow beyond this value')
parser.add_argument('--dropout', type=float, default=0.0, metavar='', help='dropout parameter')
parser.add_argument('--dropout_conv', type=float, default=0.0, metavar='', help='dropout parameter')

# ======================== Training Settings =======================================
parser.add_argument('--batch_size', type=int, default=64, metavar='', help='batch size for training')
parser.add_argument('--nepochs', type=int, default=250, metavar='', help='number of epochs to train')
parser.add_argument('--num_sim', type=int, default=1, metavar='', help='number of simulation runs')

# ======================== Hyperparameter Setings ==================================
parser.add_argument('--LR_act_max', type=float, default=0.001, metavar='', help='learning rate for learning act_max clipping threshold')
parser.add_argument('--LR_w_max', type=float, default=0.001, metavar='', help='learning rate for learning w_max clipping threshold')
parser.add_argument('--LR_1', type=float, default=0.0, metavar='', help='learning rate for learning first layer weights')
parser.add_argument('--LR_2', type=float, default=0.0, metavar='', help='learning rate for learning second layer weights')
parser.add_argument('--LR_3', type=float, default=0.0, metavar='', help='learning rate for learning third layer weights')
parser.add_argument('--LR_4', type=float, default=0.0, metavar='', help='learning rate for learning fourth layer weights')
parser.add_argument('--LR', type=float, default=0.0006, metavar='', help='learning rate')
parser.add_argument('--LR_decay', type=float, default=0.95, metavar='', help='learning rate decay')
parser.add_argument('--LR_step_after', type=int, default=100, metavar='', help='multiply learning rate by LR_step after this number of epochs')
parser.add_argument('--LR_step', type=float, default=0.1, metavar='', help='reduce learning rate by this number after LR_step_after number of epochs')
parser.add_argument('--momentum', type=float, default=0.9, metavar='', help='momentum')
parser.add_argument('--optim', type=str, default='Adam', metavar='', help='optimizer type')
parser.add_argument('--LR_scheduler', type=str, default='step', metavar='', help='LR scheduler type')
parser.add_argument('--L2_w_max', type=float, default=0.000, metavar='', help='loss penalty scale to minimize w_max')
parser.add_argument('--L2_act_max', type=float, default=0.000, metavar='', help='loss penalty scale to minimize act_max')
parser.add_argument('--L2', type=float, default=0.000, metavar='', help='weight decay')
parser.add_argument('--L3', type=float, default=0.000, metavar='', help='L2 for param grads')
parser.add_argument('--L3_act', type=float, default=0.000, metavar='', help='L2 for act grads')
parser.add_argument('--L4', type=float, default=0.000, metavar='', help='L2 for param 2nd order grads')
parser.add_argument('--L2_1', type=float, default=0.000, metavar='', help='weight decay for layer 1')
parser.add_argument('--L2_2', type=float, default=0.000, metavar='', help='weight decay for layer 2')
parser.add_argument('--L2_3', type=float, default=0.000, metavar='', help='weight decay for layer 3')
parser.add_argument('--L2_4', type=float, default=0.000, metavar='', help='weight decay for layer 4')
parser.add_argument('--L2_act1', type=float, default=0.000, metavar='', help='weight decay for layer 1')
parser.add_argument('--L2_act2', type=float, default=0.000, metavar='', help='weight decay for layer 1')
parser.add_argument('--L2_act3', type=float, default=0.000, metavar='', help='weight decay for layer 1')
parser.add_argument('--L2_act4', type=float, default=0.000, metavar='', help='weight decay for layer 1')
parser.add_argument('--L2_bn_weight', type=float, default=0.000, metavar='', help='weight decay for batchnorm params')
parser.add_argument('--L2_bn_bias', type=float, default=0.000, metavar='', help='weight decay for batchnorm params')
parser.add_argument('--weight_init', type=str, default='default', metavar='', help='weight initialization (normal, uniform, ortho)')
parser.add_argument('--weight_init_scale', type=float, default=1.0, metavar='', help='weight initialization scaling factor (soft)')
parser.add_argument('--w_scale', type=float, default=1.0, metavar='', help='weight distortion scaling factor')
parser.add_argument('--early_stop_after', type=int, default=60, metavar='', help='number of epochs to tolerate without improvement')
parser.add_argument('--var_name', type=str, default='blank', metavar='', help='variable to test')
parser.add_argument('--q_a', type=int, default=0, metavar='', help='activation quantization bits')
parser.add_argument('--q_w', type=int, default=0, metavar='', help='weight quantization bits')
parser.add_argument('--uniform_ind', type=float, default=0.0, metavar='', help='add random uniform in [-a, a] range to act x, where a is this value')
parser.add_argument('--uniform_dep', type=float, default=0.0, metavar='', help='multiply act x by random uniform in [x/a, ax] range, where a is this value')
parser.add_argument('--normal_ind', type=float, default=0.0, metavar='', help='add random normal with 0 mean and variance = a to each act x where a is this value')
parser.add_argument('--normal_dep', type=float, default=0.0, metavar='', help='add random normal with 0 mean and variance = ax to each act x where a is this value')


args = parser.parse_args()
random.seed(1)
torch.manual_seed(1)


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

		if args.q_a > 0 or args.q_w > 0:
			self.conv1 = QConv2d(3, 65, kernel_size=5, bias=args.use_bias, num_bits=args.q_a, num_bits_weight=args.q_w, biprecision=args.biprecision, stochastic=args.stochastic)
			self.conv2 = QConv2d(65, 120, kernel_size=5, bias=args.use_bias, num_bits=args.q_a, num_bits_weight=args.q_w, biprecision=args.biprecision, stochastic=args.stochastic)
		else:
			self.conv1 = nn.Conv2d(3, 65, kernel_size=5, bias=args.use_bias)
			self.conv2 = nn.Conv2d(65, 120, kernel_size=5, bias=args.use_bias)

		self.pool = nn.MaxPool2d(2, 2)
		self.relu = nn.ReLU()

		if args.q_a > 0 or args.q_w > 0:
			self.linear1 = QLinear(120 * 5 * 5, 390, bias=args.use_bias, num_bits=args.q_a, num_bits_weight=args.q_w, biprecision=args.biprecision, stochastic=args.stochastic)
			self.linear2 = QLinear(390, 10, bias=args.use_bias, num_bits=args.q_a, num_bits_weight=args.q_w, biprecision=args.biprecision, stochastic=args.stochastic)
		else:
			self.linear1 = nn.Linear(120 * 5 * 5, 390, bias=args.use_bias)
			self.linear2 = nn.Linear(390, 10, bias=args.use_bias)

		if args.batchnorm:
			self.bn1 = nn.BatchNorm2d(65, track_running_stats=args.track_running_stats)
			self.bn2 = nn.BatchNorm2d(120, track_running_stats=args.track_running_stats)
			if args.bn3:
				self.bn3 = nn.BatchNorm1d(390, track_running_stats=args.track_running_stats)
			if args.bn4:
				self.bn4 = nn.BatchNorm1d(10, track_running_stats=args.track_running_stats)

		if args.weightnorm:
			self.conv1 = nn.utils.weight_norm(nn.Conv2d(3, 65, kernel_size=5, bias=args.use_bias))
			self.conv2 = nn.utils.weight_norm(nn.Conv2d(65, 120, kernel_size=5, bias=args.use_bias))
			self.linear1 = nn.utils.weight_norm(nn.Linear(120 * 5 * 5, 390, bias=args.use_bias))
			self.linear2 = nn.utils.weight_norm(nn.Linear(390, 10, bias=args.use_bias))

		if args.dropout > 0:
			self.dropout = nn.Dropout(p=args.dropout)

	def forward(self, input, epoch=0, i=0, s=0, acc=0.0):

		self.conv1_ = self.conv1(input)

		if args.current1 > 0:
			with torch.no_grad():
				filter1 = torch.abs(self.conv1.weight)

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
					sigmas1 = F.conv2d(input, filter1)
					w_max1 = torch.max(torch.abs(self.conv1.weight))
					noise1_distr = Normal(loc=0, scale=torch.sqrt((0.1 * w_max1 / args.current1) * sigmas1))

					self.noise1 = noise1_distr.sample()

				if i == 0:
					a1 = F.conv2d(input, filter1)
					a1_sums = torch.sum(a1, dim=(1, 2, 3))
					self.p1 = 1.0e-6 * 1.2 * args.current1 * torch.mean(a1_sums)  # don't need / torch.max(input) because input is in [0, 1]


			if (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
				self.conv1_noisy = self.conv1_ * self.noise1.cuda()
			else:
				self.conv1_noisy = self.conv1_ + self.noise1.cuda()

			conv1_out = self.conv1_noisy
		else:
			conv1_out = self.conv1_

		pool1 = self.pool(conv1_out)

		if args.batchnorm:
			bn1 = self.bn1(pool1)
			pool1_out = bn1
		else:
			pool1_out = pool1

		self.relu1_ = self.relu(pool1_out)

		if args.act_max1 > 0:
			if args.train_act_max:
				self.relu1_clipped = torch.where(self.relu1_ > self.act_max1, self.act_max1, self.relu1_)   #fastest
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

		self.conv2_ = self.conv2(self.relu1)

		if args.current2 > 0:

			with torch.no_grad():
				f2 = torch.abs(self.conv2.weight)

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

				else:
					filter2 = self.conv2.weight.pow(2) + f2
					sigmas2 = F.conv2d(self.relu1, filter2)
					x_max2 = torch.max(self.relu1)
					noise2_distr = Normal(loc=0, scale=torch.sqrt((0.1 / args.current2) * x_max2 * sigmas2))
					self.noise2 = noise2_distr.sample()

				if i == 0:
					a2 = F.conv2d(self.relu1, f2)
					a2_sums = torch.sum(a2, dim=(1, 2, 3))
					self.p2 = 1.0e-6 * 1.2 * args.current2 * torch.mean(a2_sums) / torch.max(self.relu1)


			if (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
				conv2_noisy = self.conv2_ * self.noise2.cuda()
			else:
				conv2_noisy = self.conv2_ + self.noise2
			conv2_out = conv2_noisy
		else:
			conv2_out = self.conv2_

		pool2 = self.pool(conv2_out)

		if args.batchnorm:
			bn2 = self.bn2(pool2)
			pool2_out = bn2
		else:
			pool2_out = pool2

		self.relu2_ = self.relu(pool2_out)

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

		self.linear1_ = self.linear1(self.relu2)

		if args.current3 > 0:

			with torch.no_grad():
				f3 = torch.abs(self.linear1.weight)

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
					filter3 = self.linear1.weight.pow(2) + f3
					sigmas3 = F.linear(self.relu2, filter3, bias=self.linear1.bias)
					x_max3 = torch.max(self.relu2)
					noise3_distr = Normal(loc=0, scale=torch.sqrt((0.1 / args.current3) * x_max3 * sigmas3))
					self.noise3 = noise3_distr.sample()

				if i == 0:
					a3 = F.linear(self.relu2, f3, bias=self.linear1.bias)
					a3_sums = torch.sum(a3, dim=1)
					self.p3 = 1.0e-6 * 1.2 * args.current3 * torch.mean(a3_sums) / torch.max(self.relu2)

			if (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
				linear1_noisy = self.linear1_ * self.noise3.cuda()
			else:
				linear1_noisy = self.linear1_ + self.noise3

			linear1_out = linear1_noisy
		else:
			linear1_out = self.linear1_

		if args.batchnorm and args.bn3:
			linear1_out = self.bn3(linear1_out)

		self.relu3_ = self.relu(linear1_out)

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

		self.linear2_ = self.linear2(self.relu3)

		if args.current4 > 0:

			with torch.no_grad():
				f4 = torch.abs(self.linear2.weight)

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
					filter4 = self.linear2.weight.pow(2) + f4
					sigmas4 = F.linear(self.relu3, filter4, bias=self.linear2.bias)
					x_max4 = torch.max(self.relu3)
					noise4_distr = Normal(loc=0, scale=torch.sqrt((0.1 / args.current4) * x_max4 * sigmas4))
					self.noise4 = noise4_distr.sample()

				if i == 0:
					a4 = F.linear(self.relu3, f4, bias=self.linear2.bias)
					a4_sums = torch.sum(a4, dim=1)
					self.p4 = 1.0e-6 * 1.2 * args.current4 * torch.mean(a4_sums) / torch.max(self.relu3)

			if (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
				linear2_noisy = self.linear2_ * self.noise4.cuda()
			else:
				linear2_noisy = self.linear2_ + self.noise4

			linear2_out = linear2_noisy
		else:
			linear2_out = self.linear2_

		if args.batchnorm and args.bn4:
			linear2_out = self.bn4(linear2_out)

		if (args.write and s == 0 and self.training) or args.plot:
			if (epoch == 0 and i in [0, 10, 100]) or (i == 0 and epoch in [1, 5, 10, 50, 100, 150, 249]) or args.plot:
				if self.create_dir:
					utils.saveargs(args)
					self.create_dir = False

				if args.current1 > 0:
					sigmas1 = sigmas1.detach().cpu().numpy()
					noise1 = self.noise1.detach().cpu().numpy()
				else:
					sigmas1 = np.zeros_like(self.conv1_.detach().cpu().numpy())
					noise1 = np.zeros_like(self.conv1_.detach().cpu().numpy())

				if args.current2 > 0:
					sigmas2 = sigmas2.detach().cpu().numpy()
					noise2 = self.noise2.detach().cpu().numpy()
				else:
					sigmas2 = np.zeros_like(self.conv2_.detach().cpu().numpy())
					noise2 = np.zeros_like(self.conv2_.detach().cpu().numpy())

				if args.current3 > 0:
					sigmas3 = sigmas3.detach().cpu().numpy()
					noise3 = self.noise3.detach().cpu().numpy()
				else:
					sigmas3 = np.zeros_like(self.linear1_.detach().cpu().numpy())
					noise3 = np.zeros_like(self.linear1_.detach().cpu().numpy())

				if args.current4 > 0:
					sigmas4 = sigmas4.detach().cpu().numpy()
					noise4 = self.noise4.detach().cpu().numpy()
				else:
					sigmas4 = np.zeros_like(self.linear2_.detach().cpu().numpy())
					noise4 = np.zeros_like(self.linear2_.detach().cpu().numpy())

				tensors = []
				tensors.append([input.detach().cpu().numpy(), self.conv1.weight.detach().cpu().numpy(), self.conv1_.detach().cpu().numpy(), sigmas1, noise1])
				tensors.append([self.relu1.detach().cpu().numpy(), self.conv2.weight.detach().cpu().numpy(), self.conv2_.detach().cpu().numpy(), sigmas2, noise2])
				tensors.append([self.relu2.detach().cpu().numpy(), self.linear1.weight.detach().cpu().numpy(), self.linear1_.detach().cpu().numpy(), sigmas3, noise3])
				tensors.append([self.relu3.detach().cpu().numpy(), self.linear2.weight.detach().cpu().numpy(), self.linear2_.detach().cpu().numpy(), sigmas4, noise4])

				if args.var_name == 'blank':
					var_ = 0
				else:
					var_ = getattr(args, args.var_name)

				if (epoch == 0 and i == 0) or args.plot:
					print('\n\n\nBatch size', list(input.size())[0], '\n\n\n')

				if args.write and not args.plot:
					plot_layers(layers=4, models=[args.checkpoint_dir], epoch=epoch, i=i, tensors=tensors, var=args.var_name, vars=[var_], figsize=(42, 24), acc=acc, tag=args.tag)
					raise(SystemExit)

				if args.plot:
					np.save(args.checkpoint_dir + 'layer1_input_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), input.detach().cpu().numpy())
					np.save(args.checkpoint_dir + '/layer1_weights_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), self.conv1.weight.detach().cpu().numpy())
					np.save(args.checkpoint_dir + '/layer1_out_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), self.conv1_.detach().cpu().numpy())
					np.save(args.checkpoint_dir + '/layer1_sigmas_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), sigmas1)
					np.save(args.checkpoint_dir + '/layer1_noise_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), noise1)

					np.save(args.checkpoint_dir + 'layer2_input_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), self.relu1.detach().cpu().numpy())
					np.save(args.checkpoint_dir + '/layer2_weights_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), self.conv2.weight.detach().cpu().numpy())
					np.save(args.checkpoint_dir + '/layer2_out_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), self.conv2_.detach().cpu().numpy())
					np.save(args.checkpoint_dir + '/layer2_sigmas_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), sigmas2)
					np.save(args.checkpoint_dir + '/layer2_noise_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), noise2)

					np.save(args.checkpoint_dir + 'layer3_input_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), self.relu2.detach().cpu().numpy())
					np.save(args.checkpoint_dir + '/layer3_weights_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), self.linear1.weight.detach().cpu().numpy())
					np.save(args.checkpoint_dir + '/layer3_out_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), self.linear1_.detach().cpu().numpy())
					np.save(args.checkpoint_dir + '/layer3_sigmas_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), sigmas3)
					np.save(args.checkpoint_dir + '/layer3_noise_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), noise3)

					np.save(args.checkpoint_dir + 'layer4_input_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), self.relu3.detach().cpu().numpy())
					np.save(args.checkpoint_dir + '/layer4_weights_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), self.linear2.weight.detach().cpu().numpy())
					np.save(args.checkpoint_dir + '/layer4_out_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), self.linear2_.detach().cpu().numpy())
					np.save(args.checkpoint_dir + '/layer4_sigmas_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), sigmas4)
					np.save(args.checkpoint_dir + '/layer4_noise_epoch_{:d}_iter_{:d}.npy'.format(epoch, i), noise4)
					print('\n\nnumpy arrays saved to', args.checkpoint_dir, '\n\n')
					raise (SystemExit)

		return linear2_out


np.set_printoptions(precision=3, linewidth=120, suppress=True)

train_inputs, train_labels, test_inputs, test_labels = utils.load_cifar(args)

num_train_batches = 50000 // args.batch_size
num_test_batches = 10000 // args.batch_size

LR_orig = args.LR

if args.LR_1 == 0:
	args.LR_1 = args.LR
if args.LR_2 == 0:
	args.LR_2 = args.LR
if args.LR_3 == 0:
	args.LR_3 = args.LR
if args.LR_4 == 0:
	args.LR_4 = args.LR

currents = {}
if args.var_name == 'current':
	current_vars = [1, 3, 5, 10, 20, 50, 100]
else:
	current_vars = [args.current]

for current in current_vars:
	print('\n\n****************** Current {} ********************\n\n'.format(current))
	currents[current] = []
	args.current = current

	if args.current > 0:
		args.current1 = args.current2 = args.current3 = args.current4 = args.current

	if args.split:
		args.test_current = args.current
		args.train_current = 0
		#args.train_current = current * 0.8
		args.current1 = args.current2 = args.current3 = args.current4 = args.train_current

	if args.distort_w_test or args.distort_w_train:
		margin1 = args.w_scale * 0.1 / args.current1
		margin2 = args.w_scale * 0.1 / args.current2
		margin3 = args.w_scale * 0.1 / args.current3
		margin4 = args.w_scale * 0.1 / args.current4

	results = {}
	power_results = {}
	noise_results = {}

	if args.var_name == 'w_max1':
		var_list = [0.05, 0.1, 0.3, 0.5, 1]
	elif 'act_max' in args.var_name:
		var_list = [0.8, 1.2, 1.5, 2, 2.5, 3, 5, 10, 0]
	elif args.var_name == 'LR':
		var_list = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
	elif args.var_name == 'L2_act_max':
		var_list = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]
	elif args.var_name == 'uniform_ind':
		#var_list = [0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
		var_list = [x/current for x in [0.12, 0.14, 0.16]]
	elif args.var_name == 'uniform_dep':
		var_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  #0.5, 0.6, and 1.5-3.0
	elif args.var_name == 'normal_ind':
		#var_list = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
		var_list = [x/current for x in [0.05, 0.07, 0.09]]
	elif args.var_name == 'normal_dep':
		#var_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
		var_list = [x/current for x in [0.3, 0.4, 0.5]]
	elif args.var_name == 'L2_1':
		var_list = [0.0, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005]
	elif args.var_name == 'L2_2':
		var_list = [0.0, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.001]
	elif args.var_name == 'L3':
		#var_list = [0.1, 0.2, 0.5, 1, 2, 3, 5]  # currents 1,3,5,10,20,50,100:  30, 20, 10, 1, 0.2, 0.05, 0.01
		var_list = [x/args.test_current for x in [10, 20, 50]]
	elif args.var_name == 'L3_act':
		var_list = [500000, 1000000, 2000000, 4000000, 10000000, 20000000]
	elif args.var_name == 'L4':
		var_list = [0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
	elif args.var_name == 'momentum':
		var_list = [0., 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
	elif args.var_name == 'grad_clip':
		var_list = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 0]
	elif args.var_name == 'L2_w_max':
		var_list = [0.1]    #0.1 works fine for current=10 and init_w_max=0.2, no L2, and no act_max: w_min=-0.16, w_max=0.18, Acc 78.72 (epoch 225), power 3.45, noise 0.04 (0.02, 0.03, 0.04, 0.08)
	else:
		var_list = [' ']

	for var in var_list:
		if args.var_name != 'blank':
			print('\n\n********** Setting {} to {} **********\n\n'.format(args.var_name, var))
			setattr(args, args.var_name, var)

		if args.L2 > 0:
			print('\n\nSetting L2 in all layers to {}\n\n'.format(args.L2))
			args.L2_1 = args.L2_2 = args.L2_3 = args.L2_4 = args.L2

		if args.act_max > 0:
			print('\n\nSetting act clipping in all layers to {}\n\n'.format(args.act_max))
			args.act_max1 = args.act_max2 = args.act_max3 = args.act_max

		if args.var_name == "LR":
			args.LR_1 = args.LR_2 = args.LR_3 = args.LR_4 = args.LR

		results[var] = []
		power_results[var] = []
		noise_results[var] = []
		best_accuracies = []
		best_powers = []
		best_noises = []
		nsr1 = nsr2 = nsr3 = nsr4 = 0
		create_dir = True

		if args.var_name != 'blank':
			tag = args.tag + args.var_name + '-' + str(var) + '_'
		else:
			tag = args.tag

		args.checkpoint_dir = os.path.join('results/', tag + 'current-' + str(args.current1) + '-' + str(args.current2) + '-' + str(args.current3) + '-' + str(args.current4) +
			'_L3-' + str(args.L3) + '_L3_act-' + str(args.L3_act) + '_L2-' + str(args.L2_1) + '-' + str(args.L2_2) + '-' + str(args.L2_3) + '-' + str(args.L2_4) +
			'_actmax-' + str(args.act_max1) + '-' + str(args.act_max2) + '-' + str(args.act_max3) +
			'_w_max1-' + str(args.w_max1) + '-' + str(args.w_max2) + '-' + str(args.w_max3) + '-' + str(args.w_max4) + '_bn-' + str(args.batchnorm) + '_LR-' + str(args.LR) + '_' +
			'grad_clip-' + str(args.grad_clip) + '_' +
			datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))

		for s in range(args.num_sim):

			saved = False

			if args.resume is None:

				model = Net(args=args)
				model.p1 = model.p2 = model.p3 = model.p4 = 0
				utils.init_model(model, args, s)

				best_accuracy = 0
				best_epoch = 0
				best_power = 0
				best_nsr = 0
				init_epoch = 0
				te_acc = 0
				best_power_string = ''
				best_noise_string = ''
				if args.train_w_max:
					w_max_values = []
					w_min_values = []
				if args.train_act_max:
					act_max1_values = []
					act_max2_values = []
					act_max3_values = []

			else:
				print('\n\nLoading model from saved checkpoint at\n{}\n\n'.format(args.resume))
				args.checkpoint_dir = '/'.join(args.resume.split('/')[:-1]) + '/'
				model = Net(args=args)
				model.load_state_dict(torch.load(args.resume))
				model = model.cuda()
				model.eval()
				te_accs = []

				model_fname = args.resume.split('/')[-1]
				init_epoch = int(model_fname.split('_')[2])
				init_acc = model_fname.split('_')[-1][:-4]
				print(args.current1, args.current2, args.current3, args.current4)
				random.seed(13)
				torch.manual_seed(13)

				for i in range(10000 // args.batch_size):
					input = test_inputs[i * args.batch_size:(i + 1) * args.batch_size]
					label = test_labels[i * args.batch_size:(i + 1) * args.batch_size]
					output = model(input, init_epoch, i, acc=float(init_acc))
					pred = output.data.max(1)[1]
					te_acc = pred.eq(label.data).cpu().sum().numpy() * 100.0 / args.batch_size
					te_accs.append(te_acc)
				te_acc = np.mean(te_accs)
				if args.current > 0:
					total_power = model.p1 + model.p2 + model.p3 + model.p4
					power_string = '  Power {:.2f}mW ({:.2f} {:.2f} {:.2f} {:.2f})'.format(total_power, model.p1, model.p2, model.p3, model.p4)
				else:
					power_string = ''
				print('\n\nRestored Model Accuracy (epoch {:d}): {:.2f}{}\n\n'.format(init_epoch, te_acc, power_string))
				best_accuracy = te_acc
				best_epoch = init_epoch
				create_dir = False
				init_epoch += 1

				if args.print_clip:
					np.set_printoptions(precision=8, threshold=100000)
					w1 = model.conv1.weight.detach().cpu().numpy().flatten()
					print(w1[:100])
					freqs, vals = np.histogram(w1, bins=100)
					print('\n', freqs, '\n', vals, '\n')

					pos_w = len(w1[w1 > 0.99*args.w_max1])
					neg_w = len(w1[w1 < -0.99*args.w_max1])
					total_w = pos_w + neg_w
					print('\npos saturated weghts: ', pos_w)
					print('\nneg saturated weghts: ', neg_w)
					print('\ntotal saturated:      ', total_w)
					print('\ntotal weights:        ', len(w1))
					fraction = 100. * total_w // len(w1)
					print('\n\nFraction of clipped first layer weights: {:.2f}%\n\n'.format(fraction))

				if args.distort_w_test:
					print('\n\nDistorting weights\n\n')
					model.conv1.weight.data.add_(torch.cuda.FloatTensor(model.conv1.weight.size()).uniform_(-margin1, margin1))
					model.conv2.weight.data.add_(torch.cuda.FloatTensor(model.conv2.weight.size()).uniform_(-margin2, margin2))
					model.linear1.weight.data.add_(torch.cuda.FloatTensor(model.linear1.weight.size()).uniform_(-margin3, margin3))
					model.linear2.weight.data.add_(torch.cuda.FloatTensor(model.linear2.weight.size()).uniform_(-margin4, margin4))

					for i in range(10000 // args.batch_size):
						input = test_inputs[i * args.batch_size:(i + 1) * args.batch_size]
						label = test_labels[i * args.batch_size:(i + 1) * args.batch_size]
						output = model(input, init_epoch, i, acc=init_acc)
						pred = output.data.max(1)[1]
						te_acc = pred.eq(label.data).cpu().sum().numpy() * 100.0 / args.batch_size
						te_accs.append(te_acc)
					te_acc = np.mean(te_accs)
					print('\n\nRestored Model Accuracy after weights distortion {:.2f}\n\n'.format(te_acc))
				raise (SystemExit)

			model = model.cuda()  #do this before constructing optimizer!!

			if s == 0:
				utils.print_model(model, args)

			param_groups = [
				{'params': model.conv1.parameters(), 'weight_decay': args.L2_1, 'lr': args.LR_1},
				{'params': model.conv2.parameters(), 'weight_decay': args.L2_2, 'lr': args.LR_2},
				{'params': model.linear1.parameters(), 'weight_decay': args.L2_3, 'lr': args.LR_3},
				{'params': model.linear2.parameters(), 'weight_decay': args.L2_4, 'lr': args.LR_4}]

			if args.train_act_max:
				param_groups = param_groups + [
					{'params': model.act_max1, 'weight_decay': 0, 'lr': args.LR_act_max},
					{'params': model.act_max2, 'weight_decay': 0, 'lr': args.LR_act_max},
					{'params': model.act_max3, 'weight_decay': 0, 'lr': args.LR_act_max}]

			if args.train_w_max:
				param_groups = param_groups + [
					{'params': model.w_min1, 'weight_decay': 0, 'lr': args.LR_w_max},
					{'params': model.w_max1, 'weight_decay': 0, 'lr': args.LR_w_max}]

			if args.batchnorm:
				param_groups = param_groups + [
					{'params': model.bn1.parameters()},
					{'params': model.bn2.parameters()}]
				if args.bn3:
					param_groups = param_groups + [
						{'params': model.bn3.parameters()}]
				if args.bn4:
					param_groups = param_groups + [
						{'params': model.bn4.parameters()}]


			if args.optim == 'SGD':
				optimizer = torch.optim.SGD(param_groups, lr=args.LR, momentum=args.momentum, nesterov=args.nesterov)
			elif args.optim == 'Adam':
				optimizer = torch.optim.Adam(param_groups, lr=args.LR, amsgrad=args.amsgrad)

			if args.LR_scheduler == 'step':
				scheduler = lr_scheduler.StepLR(optimizer, args.LR_step_after, gamma=args.LR_step)
				#scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 60, 120], gamma=args.LR_step)
			elif args.LR_scheduler == 'exp':
				scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.LR_decay)

			prev_best_acc = 15
			scale_add = 0.2  #used to increase strength of L3 regularizers

			for epoch in range(args.nepochs):
				model.train()
				tr_accuracies = []
				te_accuracies = []
				tr_losses = []
				te_losses = []
				act_grad_norms = []
				scheduler.step()

				rnd_idx = np.random.permutation(len(train_inputs))
				train_inputs = train_inputs[rnd_idx]
				train_labels = train_labels[rnd_idx]

				if args.train_w_max:
					w_max1_grad_sum = 0

				if args.split:
					args.current1 = args.current2 = args.current3 = args.current4 = args.train_current

					if epoch == 0:
						print('*********************** Setting Train Current to', args.train_current, 'currents:', args.current1, args.current2, args.current3, args.current4)

				clip_string = ''

				for i in range(num_train_batches):
					input = train_inputs[i * args.batch_size:(i + 1) * args.batch_size]  #(64, 3, 32, 32)
					label = train_labels[i * args.batch_size:(i + 1) * args.batch_size]

					k = random.randint(0, 8)
					j = random.randint(0, 8)
					input = input[:, :, k:k + 32, j:j + 32]
					if random.random() < 0.5:
						input = torch.flip(input, [3])

					if te_acc > 0:
						acc_ = te_acc
					else:
						acc_ = 10.  #needed to pass to forward

					output = model(input, epoch, i, s, acc=acc_)
					loss = nn.CrossEntropyLoss()(output, label)

					if args.L2_act1 > 0:  #does not help
						loss += args.L2_act1 * (model.conv1_).pow(2).sum()
					if args.L2_act2 > 0:
						loss += args.L2_act2 * (model.conv2_).pow(2).sum()
					if args.L2_act3 > 0:
						loss += args.L2_act3 * (model.linear1_).pow(2).sum()
					if args.L2_act4 > 0:
						loss += args.L2_act4 * (model.linear2_).pow(2).sum()

					if args.train_act_max and args.L2_act_max > 0:
						if args.current1 == 0:
							loss = loss + args.L2_act_max * (model.act_max1 ** 2 + model.act_max2 ** 2 + model.act_max3 ** 2)
						else:
							loss = loss + args.L2_act_max * ((model.act_max1 ** 2) / args.current1 + (model.act_max2 ** 2) / args.current2 + (model.act_max3 ** 2) / args.current3)

					if args.train_w_max and args.L2_w_max > 0:
						loss = loss + args.L2_w_max * (model.w_min1 ** 2 + model.w_max1 ** 2)

					if args.batchnorm:  #haven't tested this properly
						if args.L2_bn_weight > 0:
							loss = loss + args.L2_bn_weight * (torch.sum(model.bn1.weight ** 2) + torch.sum(model.bn2.weight ** 2))
						if args.L2_bn_bias > 0:
							loss = loss + args.L2_bn_bias * (torch.sum(model.bn1.bias ** 2) + torch.sum(model.bn2.bias ** 2))

					optimizer.zero_grad()
					loss.backward(retain_graph=True)

					if args.L3_act > 0:   #L2 penalty for gradient size in respect to activations
						#acts = [model.relu1, model.relu2, model.relu3]
						acts = [model.conv1_, model.conv2_, model.linear1_, model.linear2_]
						acts_grad = torch.autograd.grad(loss, acts, create_graph=True)
						# torch.autograd.grad does not accumuate the gradients into the .grad attributes. It instead returns the gradients as Variable tuples.

						# now compute the 2-norm of the param_grads
						act_grad_norm = 0
						for grad in acts_grad:
							act_grad_norm += grad.pow(2).sum()
						act_grad_norm = args.L3_act * act_grad_norm#.sqrt()

						if i == 500:  #LR increase helps in certain scenarios
							if epoch % 10 == 0:
								scale_add = 0.95 * scale_add
							#args.L3_act = (1 + scale_add) * args.L3_act
							#print('\n\nL3_act grad_norm: {:.4f} loss: {:.4f}  L3_act:  {}\n\n'.format(np.mean(act_grad_norms), np.mean(tr_losses), int(args.L3_act)))

						# take the gradients wrt grad_norm. backward() will accumulate the gradients into the .grad attributes
						act_grad_norm.backward(retain_graph=True)

					if args.L3 > 0:   #L2 penalty for gradient size
						params = [model.conv1.weight, model.conv2.weight, model.linear1.weight, model.linear2.weight]
						param_grads = torch.autograd.grad(loss, params, create_graph=True)
						# torch.autograd.grad does not accumuate the gradients into the .grad attributes. It instead returns the gradients as Variable tuples.

						# now compute the 2-norm of the param_grads
						grad_norm = 0
						grad_sum = 0
						for grad in param_grads:
							#grad_sum += torch.abs(grad).sum()
							grad_norm += grad.pow(2).sum()
						grad_norm = args.L3 * grad_norm#.sqrt()

						#act_grad_norms.append(grad_norm.item())
						if i == 500:
							if epoch % 10 == 0:
								scale_add = 0.8 * scale_add
							#args.L3 = (1 + scale_add) * args.L3
							#print('\n\nL3 grad_norm: {:.4f} loss: {:.4f}  L3:  {}\n\n'.format(np.mean(act_grad_norms), np.mean(tr_losses), int(args.L3)))

						# take the gradients wrt grad_norm. backward() will accumulate the gradients into the .grad attributes
						grad_norm.backward(retain_graph=True)

						if args.L4 > 0:
							grads2 = torch.autograd.grad(grad_sum, params, create_graph=False)
							#grads2 = torch.autograd.grad(grad_norm, params, create_graph=True)
							g2_norm = 0
							for g2 in grads2:
								#g2_norm += g2.norm(p=2)
								g2_norm += g2.pow(2).sum()
							g2_norm = args.L4 * g2_norm

							g2_norm.backward(retain_graph=True)

					if args.L4 > 0 and args.L3 == 0:  #L2 penalty for second order gradient size
						params = [model.conv1.weight, model.conv2.weight, model.linear1.weight, model.linear2.weight]
						grads = torch.autograd.grad(loss, params, create_graph=True)

						grads_sum = 0
						for g in grads:
							#grads_sum += torch.abs(g).sum()
							grads_sum += g.pow(2).sum()
						grads2 = torch.autograd.grad(grads_sum, params, create_graph=True)

						g2_norm = 0
						for g2 in grads2:
							#g2_norm += g2.norm(p=2)
							g2_norm += g2.pow(2).sum()
						g2_norm = args.L4 * g2_norm

						g2_norm.backward(retain_graph=False)

					if args.grad_clip > 0:
						if epoch % 50 == 0 and i == 1110:
							print('\n\nloss: {:.3f}\n\n'.format(loss.item()))
						for n, p in model.named_parameters():
							p.grad.data.clamp_(-args.grad_clip, args.grad_clip)

					if args.train_w_max:
						with torch.no_grad():
							w_max1_grad = torch.sum(model.conv1.weight.grad[model.conv1.weight >= model.w_max1])  #sum of conv1 weight gradients for all weights above threshold
							w_min1_grad = torch.sum(model.conv1.weight.grad[model.conv1.weight <= model.w_min1])
							print('\n\n\nTODO: finish implementing moving avg\n\n\n')
							#w_max1_grad_sum += w_max1_grad
							#w_max1_grad_avg = w_max1_grad_sum / (i + 1)
							#w_max1_grad = 0.8 * w_max1_grad_avg + 0.2 * w_max1_grad

							model.w_min1.data = model.w_min1.data - args.LR_w_max * w_min1_grad
							model.w_max1.data = model.w_max1.data - args.LR_w_max * w_max1_grad

							model.w_min1.data.clamp_(-1, -0.01)
							model.w_max1.data.clamp_(0.01, 1)

							#model.w_max1.grad += w_max1_grad
							#model.w_min1.grad += w_min1_grad

							if args.L2_w_max > 0:
								L2_grad = model.w_max1.grad.data.item()
								if False and model.w_min1.grad is not None:
									model.w_min1.grad.data.clamp_(-1, 1)
									model.w_max1.grad.data.clamp_(-1, 1)

								model.w_min1.data = model.w_min1.data - args.LR_w_max * model.w_min1.grad.data
								model.w_max1.data = model.w_max1.data - args.LR_w_max * model.w_max1.grad.data

								model.w_max1.grad.data[:] = 0
								model.w_min1.grad.data[:] = 0

								#model.w_max1.grad.data += w_max1_grad
								#model.w_min1.grad.data += w_min1_grad
							else:
								L2_grad = 0

					optimizer.step()

					if args.distort_w_train:
						model.conv1.weight.data.add_(torch.cuda.FloatTensor(model.conv1.weight.size()).uniform_(-args.w_scale, args.w_scale))
						model.conv2.weight.data.add_(torch.cuda.FloatTensor(model.conv2.weight.size()).uniform_(-margin2, margin2))
						model.linear1.weight.data.add_(torch.cuda.FloatTensor(model.linear1.weight.size()).uniform_(-margin3, margin3))
						model.linear2.weight.data.add_(torch.cuda.FloatTensor(model.linear2.weight.size()).uniform_(-margin4, margin4))

					if args.w_max1 > 0:
						if args.train_w_max:
							model.conv1.weight.data = torch.where(model.conv1.weight > model.w_max1, model.w_max1, model.conv1.weight)
							model.conv1.weight.data = torch.where(model.conv1.weight < model.w_min1, model.w_min1, model.conv1.weight)
						else:
							model.conv1.weight.data.clamp_(-args.w_max1, args.w_max1)

					if args.w_max2 > 0:
						model.conv2.weight.data.clamp_(-args.w_max2, args.w_max2)

					if args.w_max3 > 0:
						model.linear1.weight.data.clamp_(-args.w_max3, args.w_max3)

					if args.w_max4 > 0:
						model.linear2.weight.data.clamp_(-args.w_max4, args.w_max4)

					pred = output.data.max(1)[1]
					acc = pred.eq(label.data).cpu().sum().numpy() * 100.0 / args.batch_size
					tr_accuracies.append(acc)

				tr_acc = np.mean(tr_accuracies)

				model.eval()
				if args.split:
					args.current1 = args.current2 = args.current3 = args.current4 = args.test_current
					if epoch == 0:
						print('*********************** Setting Test Current to', args.test_current, 'currents:', args.current1, args.current2, args.current3, args.current4)

				orig_params = []
				for p in model.parameters():
					orig_params.append(p.clone())

				if args.distort_w_test:
					model.conv1.weight.data.add_(torch.cuda.FloatTensor(model.conv1.weight.size()).uniform_(-margin1, margin1))
					model.conv2.weight.data.add_(torch.cuda.FloatTensor(model.conv2.weight.size()).uniform_(-margin2, margin2))
					model.linear1.weight.data.add_(torch.cuda.FloatTensor(model.linear1.weight.size()).uniform_(-margin3, margin3))
					model.linear2.weight.data.add_(torch.cuda.FloatTensor(model.linear2.weight.size()).uniform_(-margin4, margin4))

				with torch.no_grad():
					for i in range(num_test_batches):
						input = test_inputs[i * args.batch_size:(i + 1) * args.batch_size]
						label = test_labels[i * args.batch_size:(i + 1) * args.batch_size]
						output = model(input, epoch, i)
						pred = output.data.max(1)[1]
						te_acc = pred.eq(label.data).cpu().sum().numpy() * 100.0 / args.batch_size
						te_accuracies.append(te_acc)

				if args.current1 > 0:  #nsr gets reported only for a single batch!
					nsr1 = torch.mean(torch.abs(model.noise1) / torch.max(model.conv1_)).item()
				if args.current2 > 0:
					nsr2 = torch.mean(torch.abs(model.noise2) / torch.max(model.conv2_)).item()
				if args.current3 > 0:
					nsr3 = torch.mean(torch.abs(model.noise3) / torch.max(model.linear1_)).item()
				if args.current4 > 0:
					nsr4 = torch.mean(torch.abs(model.noise4) / torch.max(model.linear2_)).item()
				avg_nsr = np.mean([nsr1, nsr2, nsr3, nsr4])

				for p, orig_p in zip(model.parameters(), orig_params):
					p.data = orig_p.data

				noise_string = '  avg noise {:.3f} ({:.2f} {:.2f} {:.2f} {:.2f})'.format(avg_nsr, nsr1, nsr2, nsr3, nsr4)

				te_acc = np.mean(te_accuracies)

				total_power = model.p1 + model.p2 + model.p3 + model.p4

				power_string = '  Power {:.2f}mW ({:.2f} {:.2f} {:.2f} {:.2f})'.format(total_power, model.p1, model.p2, model.p3, model.p4)

				if args.train_act_max:
					clip_string = '  act_max {:.2f} {:.2f} {:.2f}'.format(model.act_max1.item(), model.act_max2.item(), model.act_max3.item())
					act_max1_values.append(model.act_max1.item())
					act_max2_values.append(model.act_max2.item())
					act_max3_values.append(model.act_max3.item())
				if args.train_w_max:
					clip_string += '  w_min/max {:.3f} {:.3f}'.format(model.w_min1.item(), model.w_max1.item())
					w_max_values.append(model.w_max1.item())
					w_min_values.append(model.w_min1.item())

				print('{}         Epoch {:>3d}  Train {:.2f}  Test {:.2f}  LR {:.4f}{}{}{}'.format(
					str(datetime.now())[:-7], epoch, tr_acc, te_acc, scheduler.get_lr()[0], clip_string, power_string, noise_string))

				if te_acc > best_accuracy:
					if saved:
						os.remove(args.checkpoint_dir + '/model_epoch_{:d}_acc_{:.2f}.pth'.format(best_epoch, saved_accuracy))

					if epoch > init_epoch + 60:
						if create_dir:
							utils.saveargs(args)
							create_dir = False
						if s == 0:
							saved_accuracy = te_acc
							torch.save(model.state_dict(), args.checkpoint_dir + '/model_epoch_{:d}_acc_{:.2f}.pth'.format(epoch, te_acc))
							best_saved_acc = te_acc
							saved = True

					best_accuracy = te_acc
					best_epoch = epoch
					best_nsr = avg_nsr
					best_power_string = power_string
					best_noise_string = noise_string
					if args.current1 > 0:
						best_power = total_power
					else:
						best_power = 0

				if epoch != 0 and epoch % args.early_stop_after == 0:
					if best_accuracy <= prev_best_acc:
						break
					else:
						prev_best_acc = best_accuracy

			print('\n\nCurrent {}  {} {}  Simulation {:d} Best Accuracy: {:.2f} (epoch {:d}){}{}\n\n'.format(
				args.current1, args.var_name, var, s, best_accuracy, best_epoch, best_power_string, best_noise_string))
			best_accuracies.append(best_accuracy)
			best_powers.append(best_power)
			best_noises.append(best_nsr)

			if args.train_w_max:
				print('\n\nw_max1 values:\n\n')
				for v in w_max_values:
					print('{:.3f}'.format(v), end=', ')
				print('\n\nw_min1 values:\n\n')
				for v in w_min_values:
					print('{:.3f}'.format(v), end=', ')
				print('\n\n')

			if args.train_act_max:
				print('\n\nact_max1 values:\n\n')
				for v in act_max1_values:
					print('{:.3f}'.format(v), end=', ')
				print('\n\nact_max2 values:\n\n')
				for v in act_max2_values:
					print('{:.3f}'.format(v), end=', ')
				print('\n\nact_max3 values:\n\n')
				for v in act_max3_values:
					print('{:.3f}'.format(v), end=', ')
				print('\n\n')

		print('\n\nBest accuracies for current {}  {} {} {} powers {}  noises {}\n\n'.format(
			args.current1, args.var_name, var, ['{:.2f}'.format(x) for x in best_accuracies], ['{:.2f}'.format(y) for y in best_powers], ['{:.3f}'.format(y) for y in best_noises]))

		results[var] += best_accuracies
		power_results[var] += best_powers
		noise_results[var] += best_noises

		fmt = '{} {}  {}  mean {:.2f}  max {:.2f}  min {:.2f}  power {}  mean {:.2f}mW  noise {}  mean {:.3f}'.format(
			args.var_name, str(var), [float('{:.2f}'.format(x)) for x in results[var]], np.mean(results[var]),
			np.max(results[var]), np.min(results[var]), [float('{:.2f}'.format(x)) for x in power_results[var]],
			np.mean(power_results[var]), [float('{:.2f}'.format(x)) for x in noise_results[var]], np.mean(noise_results[var]))
		print(fmt)
		currents[current].append(fmt)
		print('\n\n')
		if not saved or create_dir:
			utils.saveargs(args)
			create_dir = False
		output_file = args.checkpoint_dir + 'results_current_{}_{}.txt'.format(args.current, args.var_name)
		f = open(output_file, 'w')
		for cur in currents:
			print('\nCurrent {}nA:'.format(cur))
			f.write('\nCurrent {}nA\n'.format(cur))
			for res in currents[cur]:
				print(res)
				f.write(res + '\n')
		print('\n\n\n')
		f.close()