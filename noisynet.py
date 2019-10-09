import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.nn.functional as F

import random
import os
from datetime import datetime
import argparse
import numpy as np

import utils
from misc_code.quant_orig import QConv2d, QLinear, QuantMeasure
from plot_histograms import plot_layers, get_layers
from hardware_model import add_noise_calculate_power
from main import merge_batchnorm

#CUDA_LAUNCH_BLOCKING=1

parser = argparse.ArgumentParser(description='Your project title goes here')

#parser.add_argument('--dataset', type=str, default='cifar_RGB_4bit.npz', metavar='', help='name of dataset')
parser.add_argument('--dataset', type=str, default='data/cifar_RGB_4bit.npz', metavar='', help='name of dataset')
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
feature_parser.add_argument('--fp16', dest='fp16', action='store_true')
feature_parser.add_argument('--no-fp16', dest='fp16', action='store_false')
parser.set_defaults(fp16=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--keep_bn_fp32', dest='keep_bn_fp32', action='store_true')
feature_parser.add_argument('--no-keep_bn_fp32', dest='keep_bn_fp32', action='store_false')
parser.set_defaults(keep_bn_fp32=False)

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
feature_parser.add_argument('--amsgrad', dest='amsgrad', action='store_true')
feature_parser.add_argument('--no-amsgrad', dest='amsgrad', action='store_false')
parser.set_defaults(amsgrad=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--debug', dest='debug', action='store_true')
feature_parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--nesterov', dest='nesterov', action='store_true')
feature_parser.add_argument('--no-nesterov', dest='nesterov', action='store_false')
parser.set_defaults(nesterov=True)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--split', dest='split', action='store_true')
feature_parser.add_argument('--no-split', dest='split', action='store_false')
parser.set_defaults(split=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--debug_quant', dest='debug_quant', action='store_true')
feature_parser.add_argument('--no-debug_quant', dest='debug_quant', action='store_false')
parser.set_defaults(debug_quant=False)

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
feature_parser.add_argument('--plot_basic', dest='plot_basic', action='store_true')
feature_parser.add_argument('--no-plot_basic', dest='plot_basic', action='store_false')
parser.set_defaults(plot_basic=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--plot_noise', dest='plot_noise', action='store_true')
feature_parser.add_argument('--no-plot_noise', dest='plot_noise', action='store_false')
parser.set_defaults(plot_noise=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--plot_power', dest='plot_power', action='store_true')
feature_parser.add_argument('--no-plot_power', dest='plot_power', action='store_false')
parser.set_defaults(plot_power=False)

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

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--merged_dac', dest='merged_dac', action='store_true')
feature_parser.add_argument('--no-merged_dac', dest='merged_dac', action='store_false')
parser.set_defaults(merged_dac=True)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--merge_bn', dest='merge_bn', action='store_true')
feature_parser.add_argument('--no-merge_bn', dest='merge_bn', action='store_false')
parser.set_defaults(merge_bn=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--blocked', dest='blocked', action='store_true')
feature_parser.add_argument('--no-blocked', dest='blocked', action='store_false')
parser.set_defaults(blocked=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--print_stats', dest='print_stats', action='store_true')
feature_parser.add_argument('--no-print_stats', dest='print_stats', action='store_false')
parser.set_defaults(print_stats=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--calculate_running', dest='calculate_running', action='store_true')
feature_parser.add_argument('--no-calculate_running', dest='calculate_running', action='store_false')
parser.set_defaults(calculate_running=False)

parser.add_argument('-a', '--arch', metavar='ARCH', default='noisynet')
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
parser.add_argument('--w_max', type=float, default=0.0, metavar='', help='max value for layer 1 weights (clipping upper bound)')
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
parser.add_argument('--num_layers', type=int, default=4, metavar='', help='number of layers')
parser.add_argument('--fs', type=int, default=5, metavar='', help='filter size')
parser.add_argument('--fm1', type=int, default=65, metavar='', help='number of feature maps in the first layer')
parser.add_argument('--fm2', type=int, default=120, metavar='', help='number of feature maps in the first layer')
parser.add_argument('--fm3', type=int, default=256, metavar='', help='number of feature maps in the first layer')
parser.add_argument('--fm4', type=int, default=512, metavar='', help='number of feature maps in the first layer')
parser.add_argument('--fc', type=int, default=390, metavar='', help='size of fully connected layer')
parser.add_argument('--width', type=int, default=1, metavar='', help='expansion multiplier for layer width')

# ======================== Hyperparameter Setings ==================================
parser.add_argument('--LR_act_max', type=float, default=0.001, metavar='', help='learning rate for learning act_max clipping threshold')
parser.add_argument('--LR_w_max', type=float, default=0.001, metavar='', help='learning rate for learning w_max clipping threshold')
parser.add_argument('--LR_1', type=float, default=0.0, metavar='', help='learning rate for learning first layer weights')
parser.add_argument('--LR_2', type=float, default=0.0, metavar='', help='learning rate for learning second layer weights')
parser.add_argument('--LR_3', type=float, default=0.0, metavar='', help='learning rate for learning third layer weights')
parser.add_argument('--LR_4', type=float, default=0.0, metavar='', help='learning rate for learning fourth layer weights')
parser.add_argument('--LR', type=float, default=0.001, metavar='', help='learning rate')
parser.add_argument('--LR_decay', type=float, default=0.95, metavar='', help='learning rate decay')
parser.add_argument('--LR_step_after', type=int, default=100, metavar='', help='multiply learning rate by LR_step after this number of epochs')
parser.add_argument('--LR_max_epoch', type=int, default=10, metavar='', help='for triangle LR schedule (super-convergence) this is the epoch with max LR')
parser.add_argument('--LR_finetune_epochs', type=int, default=20, metavar='', help='for triangle LR schedule (super-convergence), number of epochs to finetune in the end')
parser.add_argument('--LR_step', type=float, default=0.1, metavar='', help='reduce learning rate by this number after LR_step_after number of epochs')
parser.add_argument('--momentum', type=float, default=0.9, metavar='', help='momentum')
parser.add_argument('--optim', type=str, default='Adam', metavar='', help='optimizer type')
parser.add_argument('--LR_scheduler', type=str, default='manual', metavar='', help='LR scheduler type')
parser.add_argument('--L1_1', type=float, default=0.0, metavar='', help='Negative L1 penalty (conv1 layer)')
parser.add_argument('--L1_2', type=float, default=0.0, metavar='', help='Negative L1 penalty (conv2 layer)')
parser.add_argument('--L1_3', type=float, default=0.0, metavar='', help='Negative L1 penalty (linear1 layer)')
parser.add_argument('--L1_4', type=float, default=0.0, metavar='', help='Negative L1 penalty (linear2 layer)')
parser.add_argument('--L1', type=float, default=0.0, metavar='', help='Negative L1 penalty')
parser.add_argument('--L2_w_max', type=float, default=0.000, metavar='', help='loss penalty scale to minimize w_max')
parser.add_argument('--L2_act_max', type=float, default=0.000, metavar='', help='loss penalty scale to minimize act_max')
parser.add_argument('--L2_bn', type=float, default=0.000, metavar='', help='loss penalty scale to minimize bn params (shift and scale)')
parser.add_argument('--L2', type=float, default=0.000, metavar='', help='weight decay')
parser.add_argument('--L3', type=float, default=0.000, metavar='', help='L2 for param grads')
parser.add_argument('--L3_new', type=float, default=0.000, metavar='', help='L2 for param grads')
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
parser.add_argument('--weight_init', type=str, default='default', metavar='', help='weight initialization (normal, uniform, ortho) for conv layers')
parser.add_argument('--weight_init_scale_conv', type=float, default=1.0, metavar='', help='weight initialization scaling factor (soft) for conv layers')
parser.add_argument('--weight_init_scale_fc', type=float, default=1.0, metavar='', help='weight initialization scaling factor (soft) for fc layers')
parser.add_argument('--w_scale', type=float, default=1.0, metavar='', help='weight distortion scaling factor')
parser.add_argument('--early_stop_after', type=int, default=100, metavar='', help='number of epochs to tolerate without improvement')
parser.add_argument('--var_name', type=str, default='', metavar='', help='variable to test')
parser.add_argument('--q_a1', type=int, default=0, metavar='', help='activation quantization bits')
parser.add_argument('--q_w1', type=int, default=0, metavar='', help='weight quantization bits')
parser.add_argument('--q_a2', type=int, default=0, metavar='', help='activation quantization bits')
parser.add_argument('--q_w2', type=int, default=0, metavar='', help='weight quantization bits')
parser.add_argument('--q_a3', type=int, default=0, metavar='', help='activation quantization bits')
parser.add_argument('--q_w3', type=int, default=0, metavar='', help='weight quantization bits')
parser.add_argument('--q_a4', type=int, default=0, metavar='', help='activation quantization bits')
parser.add_argument('--q_w4', type=int, default=0, metavar='', help='weight quantization bits')
parser.add_argument('--stochastic', type=float, default=0.5, metavar='', help='stochastic uniform noise to add before rounding during quantization')
parser.add_argument('--pctl', default=99.98, type=float, help='percentile to show when plotting')
parser.add_argument('--seed', type=int, default=None, metavar='', help='random seed')
parser.add_argument('--uniform_ind', type=float, default=0.0, metavar='', help='add random uniform in [-a, a] range to act x, where a is this value')
parser.add_argument('--uniform_dep', type=float, default=0.0, metavar='', help='multiply act x by random uniform in [x/a, ax] range, where a is this value')
parser.add_argument('--normal_ind', type=float, default=0.0, metavar='', help='add random normal with 0 mean and variance = a to each act x where a is this value')
parser.add_argument('--normal_dep', type=float, default=0.0, metavar='', help='add random normal with 0 mean and variance = ax to each act x where a is this value')
parser.add_argument('--gpu', default=None, type=str, help='GPU to use, if None use all')

args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print('\n\n****** You have chosen to seed training. This will turn on the CUDNN deterministic setting, and training will be SLOW! ******\n\n')
else:
    torch.backends.cudnn.benchmark = True

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


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
        arrays = []

        if args.q_a1 > 0:
            self.input = self.quantize1(input)
        else:
            self.input = input

        if epoch == 0 and i == 0 and s == 0 and self.training:
            print('\ninput shape:', self.input.shape)

        self.conv1_no_bias = self.conv1(self.input)

        if args.plot or args.write:
            get_layers(arrays, self.input, self.conv1.weight, self.conv1_no_bias, stride=1, padding=0, layer='conv', basic=args.plot_basic, debug=args.debug)

        if args.merge_bn:
            self.bias1 = self.bn1.bias.view(1, -1, 1, 1) - self.bn1.running_mean.data.view(1, -1, 1, 1) * self.bn1.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn1.running_var.data.view(1, -1, 1, 1) + 0.0000001)
            self.conv1_ = self.conv1_no_bias + self.bias1
            if args.plot or args.write:
                arrays.append([self.bias1.half()])
        else:
            self.conv1_ = self.conv1_no_bias

        if epoch == 0 and i == 0 and s == 0 and self.training:
            print('conv1 out shape:', self.conv1_.shape)

        if args.current1 > 0:
            conv1_out = add_noise_calculate_power(self, args, arrays, self.input, self.conv1.weight, self.conv1_, layer_type='conv', i=i, layer_num=0, merged_dac=args.merged_dac)
        else:
            conv1_out = self.conv1_

        pool1 = self.pool(conv1_out)

        if args.batchnorm and not args.merge_bn:
            bn1 = self.bn1(pool1)
            self.pool1_out = bn1
        else:
            self.pool1_out = pool1

        if args.plot or args.write:
            arrays.append([self.pool1_out.half()])

        self.relu1_ = self.relu(self.pool1_out)
        if epoch == 0 and i == 0 and s == 0 and self.training:
            print('relu1 out shape:', self.relu1_.shape)

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

        if args.q_a2 > 0:
            self.relu1 = self.quantize2(self.relu1)

        self.conv2_no_bias = self.conv2(self.relu1)

        if args.plot or args.write:
            get_layers(arrays, self.relu1, self.conv2.weight, self.conv2_no_bias, stride=1, padding=0, layer='conv', basic=args.plot_basic, debug=args.debug)

        if args.merge_bn:
            self.bias2 = self.bn2.bias.view(1, -1, 1, 1) - self.bn2.running_mean.data.view(1, -1, 1, 1) * self.bn2.weight.data.view(1, -1, 1, 1) / torch.sqrt(self.bn2.running_var.data.view(1, -1, 1, 1) + 0.0000001)
            self.conv2_ = self.conv2_no_bias + self.bias2
            if args.plot or args.write:
                arrays.append([self.bias2.half()])
        else:
            self.conv2_ = self.conv2_no_bias

        if epoch == 0 and i == 0 and s == 0 and self.training:
            print('conv2 out shape:', self.conv2_.shape)

        if args.current2 > 0:
            conv2_out = add_noise_calculate_power(self, args, arrays, self.relu1, self.conv2.weight, self.conv2_, layer_type='conv', i=i, layer_num=1, merged_dac=False)
        else:
            conv2_out = self.conv2_

        pool2 = self.pool(conv2_out)

        if args.batchnorm and not args.merge_bn:
            bn2 = self.bn2(pool2)
            self.pool2_out = bn2
        else:
            self.pool2_out = pool2

        if args.plot or args.write:
            arrays.append([self.pool2_out.half()])

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
            get_layers(arrays, self.relu2, self.linear1.weight, self.linear1_no_bias, layer='linear', basic=args.plot_basic, debug=args.debug)

        if args.merge_bn:
            self.bias3 = self.bn3.bias.view(1, -1) - self.bn3.running_mean.data.view(1, -1) * self.bn3.weight.data.view(1, -1) / torch.sqrt(self.bn3.running_var.data.view(1, -1) + 0.0000001)
            self.linear1_ = self.linear1_no_bias + self.bias3
            if args.plot or args.write:
                arrays.append([self.bias3.half()])
        else:
            self.linear1_ = self.linear1_no_bias

        if args.current3 > 0:
            linear1_out = add_noise_calculate_power(self, args, arrays, self.relu2, self.linear1.weight, self.linear1_, layer_type='linear', i=i, layer_num=2, merged_dac=args.merged_dac)
        else:
            linear1_out = self.linear1_

        if args.batchnorm and args.bn3 and not args.merge_bn:
            self.linear1_out = self.bn3(linear1_out)
        else:
            self.linear1_out = linear1_out

        if args.plot or args.write:
            arrays.append([self.linear1_out.half()])

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
            get_layers(arrays, self.relu3, self.linear2.weight, self.linear2_no_bias, layer='linear', basic=args.plot_basic, debug=args.debug)

        if args.bn4 and args.merge_bn:
            if self.training:
                print('\n\n************ Merging BatchNorm during training! **********\n\n')
            self.bias4 = self.bn4.bias.view(1, -1) - self.bn4.running_mean.data.view(1, -1) * self.bn4.weight.data.view(1, -1) / torch.sqrt(self.bn4.running_var.data.view(1, -1) + 0.0000001)
            self.linear2_ = self.linear2_no_bias + self.bias4
            if args.plot or args.write:
                arrays.append([self.bias4.half()])
        else:
            self.linear2_ = self.linear2_no_bias
            self.bias4 = torch.Tensor([0])

        if args.current4 > 0:
            linear2_out = add_noise_calculate_power(self, args, arrays, self.relu3, self.linear2.weight, self.linear2_, layer_type='linear', i=i, layer_num=3, merged_dac=False)
        else:
            linear2_out = self.linear2_

        if args.batchnorm and args.bn4 and not args.merge_bn:
            self.linear2_out = self.bn4(linear2_out)
        else:
            self.linear2_out = linear2_out

        if args.plot or args.write:
            arrays.append([self.linear2_out.half()])

        if (args.plot and s == 0 and i == 0 and epoch in [0, 1, 5, 10, 50, 100, 150, 249] and self.training) or args.write or (args.resume is not None and args.plot):

            if self.create_dir:
                utils.saveargs(args)
                self.create_dir = False

            if (epoch == 0 and i == 0) or args.plot:
                print('\n\n\nBatch size', list(self.input.size())[0], '\n\n\n')

            names = ['input', 'weights', 'vmm']

            if args.merge_bn:
                names.append('bias')
                args.tag += '_merged_bn'

            if args.plot_noise:
                names.extend(['sigmas', 'noise', 'noise/range'])
                args.tag += '_noise'

            if args.plot_power:
                names.append('power')
                args.tag += '_power'

            names.append('pre-activation')

            if not args.plot_basic:
                names.extend(['vmm diff', 'vmm blocked', 'vmm diff blocked', 'weight sums', 'weight sums diff', 'weight sums blocked', 'weight sums diff blocked'])
                args.tag += '_blocked_diff'

            print('\n\nPreparing arrays for plotting or writing:\n')
            layers = []
            layer = []
            print('\n\nlen(arrays) // len(names):', len(arrays), len(names), len(arrays) // len(names), '\n\n')
            num_layers = len(arrays) // len(names)
            for k in range(num_layers):
                print('layer', k, names)
                for j in range(len(names)):
                    # print('\t', names[j])
                    layer.append([arrays[len(names) * k + j][0].detach().cpu().numpy()])
                layers.append(layer)
                layer = []

            info = []
            inputs = []
            for n, p in model.named_parameters():
                if 'weight' in n and ('conv' in n or 'linear' in n):
                    inputs.append(np.prod(p.shape[1:]))

            for i in range(len(inputs)):
                temp = []
                temp.append('{:d} inputs '.format(inputs[i]))
                if args.plot_power:
                    temp.append('{:.2f}mW '.format(self.power[i][0]))
                info.append(temp)

            if args.plot:
                print('\nPlotting {}\n'.format(names))
                var_ = ''
                var_name = args.var_name

                plot_layers(num_layers=len(layers), models=[args.checkpoint_dir], epoch=epoch, i=i, layers=layers,
                            names=names, var=var_name, vars=[var_], infos=info, pctl=args.pctl, acc=acc, tag=args.tag, normalize=args.normalize)

            if args.write and not self.training:
                np.save(args.checkpoint_dir + 'layers.npy', np.array(layers))
                print('\n\nnumpy arrays saved to', args.checkpoint_dir + 'layers.npy', '\n\n')
                np.save(args.checkpoint_dir + 'array_names.npy', np.array(names))
                print('array names saved to', args.checkpoint_dir + 'array_names.npy', '\n\n')
                np.save(args.checkpoint_dir + 'input_sizes.npy', np.array(inputs))
                print('input sizes saved to', args.checkpoint_dir + 'input_sizes.npy', '\n\n')
                if args.plot_power:
                    np.save(args.checkpoint_dir + 'layer_power.npy', np.array([x[0] for x in self.power]))
                    print('layers power saved to', args.checkpoint_dir + 'layers_power.npy', '\n\n')

            if (args.plot and args.resume is not None) or args.write:
                raise (SystemExit)

        return self.linear2_out


np.set_printoptions(precision=4, linewidth=120, suppress=True)

train_inputs, train_labels, test_inputs, test_labels = utils.load_cifar(args)

num_train_batches = 50000 // args.batch_size
num_test_batches = 10000 // args.batch_size

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

    args.layer_currents = [args.current1, args.current2, args.current3, args.current4]

    if args.distort_w_test or args.distort_w_train:
        if args.current1 > 0:
            margin1 = args.w_scale * 0.1 / args.current1
            margin2 = args.w_scale * 0.1 / args.current2
            margin3 = args.w_scale * 0.1 / args.current3
            margin4 = args.w_scale * 0.1 / args.current4
        else:
            margin1 = args.w_scale * 0.1
            margin2 = args.w_scale * 0.1
            margin3 = args.w_scale * 0.1
            margin4 = args.w_scale * 0.1

    results = {}
    results_dist = {}
    power_results = {}
    noise_results = {}
    act_sparsity_results = {}
    w_sparsity_results = {}

    if args.var_name == 'w_max1':
        var_list = [0.05, 0.1, 0.3, 0.5, 1]
        #var_list = [0, 2, 4, 8]
        var_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]
    elif args.var_name == 'act_max':#'act_max' in args.var_name:
        #var_list = [0.8, 1.2, 1.5, 2, 2.5, 3, 5, 10, 0]
        var_list = [0, 0.2, 1, 5, 20]
        var_list = [0.25, 1, 2, 4, 10, 0]
    elif args.var_name == 'act_max1':#'act_max' in args.var_name:
        #var_list = [0.8, 1.2, 1.5, 2, 2.5, 3, 5, 10, 0]
        var_list = [0, 0.2, 1, 5, 20]
        var_list = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5]
    elif args.var_name == 'act_max2':#'act_max' in args.var_name:
        #var_list = [0.8, 1.2, 1.5, 2, 2.5, 3, 5, 10, 0]
        var_list = [0, 0.2, 1, 5, 20]
        var_list = [0.5, 1, 2, 3, 4, 5, 10]
    elif args.var_name == 'act_max3':#'act_max' in args.var_name:
        #var_list = [0.8, 1.2, 1.5, 2, 2.5, 3, 5, 10, 0]
        var_list = [0, 0.2, 1, 5, 20]
        var_list = [0.5, 1, 2, 3, 4, 5, 10]
    elif args.var_name == 'LR':
        #var_list = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
        #var_list = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02]
        var_list = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035]
        var_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.04]
        var_list = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]
        var_list = [0.5, 0.6, 0.7, 0.8, 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 7, 10]
        var_list = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01]
    elif args.var_name == 'L2_act_max':
        var_list = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]
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
    elif args.var_name == 'L2':
        var_list = [0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.001]
        var_list = [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.001, 0.0012, 0.0015, 0.002]
        var_list = [0, 0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.001]
        var_list = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.02, 0.03, 0.05]
        var_list = [0.5, 0.6, 0.7, 0.8, 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 7, 10]
        var_list = [0, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
        var_list = [0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    elif args.var_name == 'L1':
        var_list = [1e-6, 2e-6, 5e-6, 1e-5, 3e-5, 2e-5, 5e-5, 0.0001]
    elif args.var_name == 'L2_2':
        var_list = [0.0, 0.00001, 0.00002, 0.00003, 0.00005, 0.0001]#, 0.0002, 0.0003, 0.0005, 0.001]
    elif args.var_name == 'L3':
        #var_list = [0.1, 0.2, 0.5, 1, 2, 3, 5]  # currents 1,3,5,10,20,50,100:  30, 20, 10, 1, 0.2, 0.05, 0.01
        #var_list = [x/args.test_current for x in [10, 20, 50]]
        var_list = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50]
        var_list = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]
        var_list = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02]
        #var_list = [0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08]
        var_list = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03]
        var_list = [0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01]
        var_list = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]#0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01]
        var_list = [0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.5, 1]
    elif args.var_name == 'L3_new':
        var_list = [0, 1, 2, 3, 5, 10, 20, 30]
    elif args.var_name == 'L3_act':
        #var_list = [500000, 1000000, 2000000, 4000000, 10000000, 20000000]
        #var_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
        var_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]
    elif args.var_name == 'L4':
        var_list = [0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    elif args.var_name == 'momentum':
        var_list = [0., 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
    elif args.var_name == 'grad_clip':
        #var_list = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2, 5, 0]
        var_list = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2, 5, 0]
        var_list = [0.005, 0.05, 0.5, 2, 0]
    elif args.var_name == 'dropout':
        var_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        var_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    elif args.var_name == 'width':
        var_list = [1, 2, 4]
    elif args.var_name == 'L2_w_max':
        var_list = [0.1]    #0.1 works fine for current=10 and init_w_max=0.2, no L2, and no act_max: w_min=-0.16, w_max=0.18, Acc 78.72 (epoch 225), power 3.45, noise 0.04 (0.02, 0.03, 0.04, 0.08)
    else:
        var_list = [' ']

    for var in var_list:
        if args.var_name != '':
            print('\n\n********** Setting {} to {} **********\n\n'.format(args.var_name, var))
            setattr(args, args.var_name, var)

        if args.L2 > 0:
            #args.L2_1 = args.L2_2 = args.L2_3 = args.L2_4 = args.L2
            #args.L2_1 = args.L2_2 = args.L2_3 = args.L2_4 = args.L2 * math.sqrt(args.width)
            if args.q_a2 == 1:
                args.L2_1 = args.L2_2 = args.L2_3 = args.L2_4 = args.L2 * args.width
            else:
                args.L2_1 = args.L2_2 = args.L2_3 = args.L2_4 = args.L2
            print('\n\nSetting L2 in all layers to {}\n\n'.format(args.L2_1))

        if args.L1 > 0:
            args.L1_1 = args.L1_2 = args.L1_3 = args.L1_4 = args.L1

        if args.dropout > 0 and args.var_name != 'dropout':
            #pass
            #args.dropout = 0.1 * args.width
            '''
            if args.q_a2 == 0:
                args.dropout = args.width * 4 / 40.
            else:
                args.dropout = args.width * args.q_a2 / 40.
            '''
            print('\n\nSetting dropout in fc layers to {}\n\n'.format(args.dropout))


        if args.act_max > 0:
            print('\n\nSetting act clipping in all layers to {}\n\n'.format(args.act_max))
            args.act_max1 = args.act_max2 = args.act_max3 = args.act_max

        if args.w_max > 0:
            args.w_max1 = args.w_max2 = args.w_max3 = args.w_max4 = args.w_max

        if args.var_name == "LR":
            args.LR_1 = args.LR_2 = args.LR_3 = args.LR_4 = args.LR

        results[var] = []
        results_dist[var] = []
        te_acc_dists = []
        power_results[var] = []
        noise_results[var] = []
        act_sparsity_results[var] = []
        w_sparsity_results[var] = []
        best_accuracies = []
        best_accuracies_dist = []
        best_powers = []
        best_noises = []
        best_act_sparsities = []
        best_w_sparsities = []
        te_acc_dist_string = ''
        avg_te_acc_dist = 0
        create_dir = True

        if args.var_name != '':
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

            best_accuracy = 0
            best_accuracy_dist = 0
            best_epoch = 0
            best_power = 0
            best_nsr = 0
            best_input_sparsity = 0
            avg_w_sparsity = 0
            best_w_sparsity = 0
            init_epoch = 0
            te_acc = 0
            best_power_string = ''
            best_noise_string = ''
            best_input_sparsity_string = ''
            best_w_sparsity_string = ''
            w_input_sparsity_string = ''
            input_sparsity_string = ''
            noise_string = ''
            power_string = ''
            saved = False

            if args.resume is None:

                model = Net(args=args)
                utils.init_model(model, args, s)
                model = model.cuda() #do this before constructing optimizer!!
                if args.fp16:
                    model = model.half()
                    if args.keep_bn_fp32:
                        for layer in model.modules():
                            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                                layer.float()

                if args.train_w_max:
                    print('\n\nSetting w_min1 to {:.2f} and w_max1 to {:.2f}\n\n'.format(model.w_min1.item(), model.w_max1.item()))

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
                model = model.cuda()

                saved_model = torch.load(args.resume)  #ignore unnecessary parameters

                for saved_name, saved_param in saved_model.items():
                    print(saved_name)
                    for name, param in model.named_parameters():
                        if name == saved_name:
                            print('\tmatched, copying...')
                            param.data = saved_param.data
                    if 'running' in saved_name:  #batchnorm stats are not in named_parameters
                        print('\tmatched, copying...')
                        m = model.state_dict()
                        m.update({saved_name: saved_param})
                        model.load_state_dict(m)

                #model.load_state_dict(torch.load(args.resume))

                if args.fp16:
                    model = model.half()
                    if args.keep_bn_fp32:
                        for layer in model.modules():
                            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                                layer.float()

                model.eval()

                if args.merge_bn:
                    merge_batchnorm(model, args)

                model_fname = args.resume.split('/')[-1]
                init_epoch = int(model_fname.split('_')[2])
                init_acc = model_fname.split('_')[-1][:-4]
                print('\n\nCurrents:', args.current1, args.current2, args.current3, args.current4)

                model.power = [[] for _ in range(args.num_layers)]  #[[]]*num_layers won't work!
                model.nsr = [[] for _ in range(args.num_layers)]
                model.input_sparsity = [[] for _ in range(args.num_layers)]
                w_sparsity = []
                te_accs = []

                for i in range(10000 // args.batch_size):
                    input = test_inputs[i * args.batch_size:(i + 1) * args.batch_size]
                    label = test_labels[i * args.batch_size:(i + 1) * args.batch_size]
                    output = model(input, init_epoch, i, acc=float(init_acc))
                    pred = output.data.max(1)[1]
                    te_acc = pred.eq(label.data).cpu().sum().numpy() * 100.0 / args.batch_size
                    te_accs.append(te_acc)

                if args.print_stats:
                    p = []
                    input_sp = []
                    nsr = []
                    for ind in range(args.num_layers):
                        p.append(np.nanmean(model.power[ind]))
                        input_sp.append(np.nanmean(model.input_sparsity[ind]))
                        nsr.append(np.nanmean(model.nsr[ind]))

                    avg_input_sparsity = np.nanmean(input_sp)
                    input_sparsity_string = '  act spars {:.2f} ({:.2f} {:.2f} {:.2f} {:.2f})'.format(avg_input_sparsity, *input_sp)
                    avg_nsr = np.nanmean(nsr)
                    noise_string = '  avg noise {:.3f} ({:.2f} {:.2f} {:.2f} {:.2f})'.format(avg_nsr, *nsr)
                    total_power = np.nansum(p)
                    power_string = '  Power {:.2f}mW ({:.2f} {:.2f} {:.2f} {:.2f})'.format(total_power, *p)

                te_acc = np.mean(te_accs)

                print('\n\nRestored Model Accuracy (epoch {:d}): {:.2f}{}{}{}\n\n'.format(init_epoch, te_acc, power_string, noise_string, input_sparsity_string))
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
                    {'params': model.bn1.parameters(), 'weight_decay': args.L2_bn},
                    {'params': model.bn2.parameters(), 'weight_decay': args.L2_bn}]
                if args.bn3:
                    param_groups = param_groups + [
                        {'params': model.bn3.parameters(), 'weight_decay': args.L2_bn}]
                if args.bn4:
                    param_groups = param_groups + [
                        {'params': model.bn4.parameters(), 'weight_decay': args.L2_bn}]


            if args.optim == 'SGD':
                optimizer = torch.optim.SGD(param_groups, lr=args.LR, momentum=args.momentum, nesterov=args.nesterov)
            elif args.optim == 'Adam':
                optimizer = torch.optim.Adam(param_groups, lr=args.LR, amsgrad=args.amsgrad)
            elif args.optim == 'AdamW':
                optimizer = torch.optim.AdamW(param_groups, lr=args.LR, amsgrad=args.amsgrad)
                if s == 0:
                    print('\n\n*********** Using AdamW **********\n')
                    for param_group in optimizer.param_groups:
                        print('param_group weight decay {} LR {}'.format(param_group["weight_decay"], param_group["lr"]))
                    print('\n')

            if args.LR_scheduler == 'step':
                scheduler = lr_scheduler.StepLR(optimizer, args.LR_step_after, gamma=args.LR_step)
                #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 60, 120], gamma=args.LR_step)
                lr = scheduler.get_lr()[0]
            elif args.LR_scheduler == 'exp':
                scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.LR_decay)
                lr = scheduler.get_lr()[0]
            elif args.LR_scheduler == 'triangle':
                lr_increment = args.LR / ((args.LR_max_epoch + 1) * num_train_batches)
                mom_decrement = args.momentum / ((args.LR_max_epoch + 1) * num_train_batches)
                lr_decrement = (args.LR - 0.05 * args.LR) / ((args.nepochs - args.LR_max_epoch - args.LR_finetune_epochs) * num_train_batches)
                lr_decrement2 = (0.05 * args.LR) / (args.LR_finetune_epochs * num_train_batches)
                mom_increment = (args.LR - 0.05 * args.LR) / (   #TODO!!!
                                (args.nepochs - args.LR_max_epoch - args.LR_finetune_epochs) * num_train_batches)
                mom_increment2 = (0.05 * args.LR) / (args.LR_finetune_epochs * num_train_batches)
                lr = 0
                mom = args.momentum

            prev_best_acc = 15
            scale_add = 0.2  #used to increase strength of L3 regularizers

            grad_norms = []
            act_grad_norms = []
            act_norms = []
            weight_norms = []
            w_sparsity = []
            max_weights = []
            max_acts = []
            max_grads = []
            max_act_grads = []
            best_accuracy_dist_string = ''
            norm_string = ''
            max_string = ''

            for epoch in range(args.nepochs):

                model.power = [[] for _ in range(args.num_layers)]
                model.nsr = [[] for _ in range(args.num_layers)]
                model.input_sparsity = [[] for _ in range(args.num_layers)]

                model.train()
                tr_accuracies = []
                te_accuracies = []
                te_accuracies_dist = []

                if args.LR_scheduler == 'manual':
                    lr = args.LR * args.LR_step ** (epoch // args.LR_step_after)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr #/ args.batch_size
                elif args.LR_scheduler != 'triangle':
                    scheduler.step()
                    lr = scheduler.get_lr()[0]

                rnd_idx = np.random.permutation(len(train_inputs))
                train_inputs = train_inputs[rnd_idx]
                train_labels = train_labels[rnd_idx]

                if args.train_w_max:
                    w_max1_grad_sum = 0

                if args.split:
                    args.current1 = args.current2 = args.current3 = args.current4 = args.train_current
                    args.layer_currents = [args.current1, args.current2, args.current3, args.current4]

                    if epoch == 0:
                        print('*********************** Setting Train Current to', args.train_current, 'currents:', args.current1, args.current2, args.current3, args.current4)

                clip_string = ''

                for i in range(num_train_batches):
                    input = train_inputs[i * args.batch_size:(i + 1) * args.batch_size]  #(64, 3, 32, 32)
                    label = train_labels[i * args.batch_size:(i + 1) * args.batch_size]

                    if args.augment:
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
                    #loss = nn.CrossEntropyLoss(reduction='none')(output, label).sum()
                    loss = nn.CrossEntropyLoss()(output, label)

                    if args.debug and i < 5:
                        utils.print_batchnorm(model, i)

                    if args.LR_scheduler == 'triangle':
                        if epoch <= args.LR_max_epoch:
                            lr +=  lr_increment
                            mom -= mom_decrement
                        elif epoch <= args.nepochs - args.LR_finetune_epochs:
                            lr -= lr_decrement
                            mom += mom_increment
                        else:
                            lr -= lr_decrement2
                            mom += mom_increment2

                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr / args.batch_size
                            param_group['momentum'] = mom

                    if args.L2_act1 > 0:  #does not help
                        loss += args.L2_act1 * (model.conv1_).pow(2).sum()
                    if args.L2_act2 > 0:
                        loss += args.L2_act2 * (model.conv2_).pow(2).sum()
                    if args.L2_act3 > 0:
                        loss += args.L2_act3 * (model.linear1_).pow(2).sum()
                    if args.L2_act4 > 0:
                        loss += args.L2_act4 * (model.linear2_).pow(2).sum()

                    if args.L1_1 > 0:
                        if epoch == 0 and i == 0:
                            print('\n\nApplying L1 loss penalty {} in conv1 layer\n'.format(args.L1_1))
                        loss = loss + args.L1_1 * model.conv1.weight.norm(p=1)

                    if args.L1_2 > 0:
                        if epoch == 0 and i == 0:
                            print('\nApplying L1 loss penalty {} in conv2 layer\n'.format(args.L1_2))
                        loss = loss + args.L1_2 * model.conv2.weight.norm(p=1)

                    if args.L1_3 > 0:
                        if epoch == 0 and i == 0:
                            print('\nApplying L1 loss penalty {} in linear1 layer\n'.format(args.L1_3))
                        loss = loss + args.L1_3 * model.linear1.weight.norm(p=1)

                    if args.L1_4 > 0:
                        if epoch == 0 and i == 0:
                            print('\nApplying L1 loss penalty {} in linear2 layer\n'.format(args.L1_4))
                        loss = loss + args.L1_4 * model.linear2.weight.norm(p=1)


                    if args.train_act_max and args.L2_act_max > 0:
                        if args.current1 == 0:
                            loss = loss + args.L2_act_max * (model.act_max1 ** 2 + model.act_max2 ** 2 + model.act_max3 ** 2)
                        else:
                            loss = loss + args.L2_act_max * ((model.act_max1 ** 2) / args.current2 + (model.act_max2 ** 2) / args.current3 + (model.act_max3 ** 2) / args.current4)

                    if args.train_w_max and args.L2_w_max > 0:
                        if i % 100 == 0:
                            pass
                            #print('w_min1/w_max1: {:.2f}/{:.2f}'.format(model.w_min1.item(), model.w_max1.item()))
                        loss = loss + args.L2_w_max * (model.w_min1 ** 2 + model.w_max1 ** 2)

                    if args.batchnorm:  #haven't tested this properly
                        if args.L2_bn_weight > 0:
                            loss = loss + args.L2_bn_weight * (torch.sum(model.bn1.weight ** 2) + torch.sum(model.bn2.weight ** 2))
                        if args.L2_bn_bias > 0:
                            loss = loss + args.L2_bn_bias * (torch.sum(model.bn1.bias ** 2) + torch.sum(model.bn2.bias ** 2))

                    optimizer.zero_grad()

                    if args.L3_new > 0:  # L2 penalty for gradient size
                        params = [model.conv1.weight, model.conv2.weight, model.linear1.weight, model.linear2.weight]
                        param_grads = torch.autograd.grad(loss, params, create_graph=True, only_inputs=True)
                        # torch.autograd.grad does not accumuate the gradients into the .grad attributes. It instead returns the gradients as Variable tuples.
                        # now compute the 2-norm of the param_grads
                        grad_norm = 0
                        for grad in param_grads:
                            # print('param_grad {}:\n{}\ngrad.pow(2).mean(): {:.4f}'.format(grad.shape, grad[0,0], grad.pow(2).mean().item()))
                            grad_norm += args.L3_new * grad.pow(2).mean()
                        # take the gradients wrt grad_norm. backward() will accumulate the gradients into the .grad attributes
                        # grad_norm.backward(retain_graph=False)  # or like this:
                        # print('loss {:.4f} grad_norm {:.4f}'.format(loss.item(), grad_norm.item()))
                        #print('\nloss before', loss.item())
                        loss = loss + grad_norm
                        #print('\nloss after ', loss.item())

                    if args.L3 > 0 or args.L4 > 0 or args.print_stats:
                        retain_graph = True
                    else:
                        retain_graph = False

                    loss.backward(retain_graph=retain_graph)

                    if args.print_stats and i == 0:
                        for act in [model.conv1_, model.conv2_, model.linear1_, model.linear2_]:
                            act_norms.append(torch.mean(torch.abs(act)).item())
                        for param in [model.conv1.weight, model.conv2.weight, model.linear1.weight, model.linear2.weight]:
                            weight_norms.append(torch.mean(torch.abs(param)).item())

                        if args.L3 == 0:
                            params = [model.conv1.weight, model.conv2.weight, model.linear1.weight, model.linear2.weight]
                            param_grads = torch.autograd.grad(loss, params, create_graph=True)
                            for grad in param_grads:
                                grad_norms.append(torch.mean(torch.abs(grad)).item() * 1000.)

                        if args.L3_act == 0:
                            acts = [model.conv1_, model.conv2_, model.linear1_, model.linear2_]
                            acts_grad = torch.autograd.grad(loss, acts, create_graph=True)
                            for act_grad in acts_grad:
                                act_grad_norms.append(torch.mean(torch.abs(act_grad)).item()*1000.)

                    if args.L3_act > 0:   #L2 penalty for gradient size in respect to activations
                        acts = [model.conv1_, model.conv2_, model.linear1_, model.linear2_]
                        acts_grad = torch.autograd.grad(loss, acts, create_graph=True)
                        # torch.autograd.grad does not accumuate the gradients into the .grad attributes. It instead returns the gradients as Variable tuples.
                        # now compute the 2-norm of the param_grads
                        act_grad_norm = args.L3_act * torch.stack([act_grad.pow(2).sum() for act_grad in acts_grad]).sum()  #.sqrt()

                        if i == 0:
                            act_grad_norms = [torch.mean(torch.abs(act_grad)).item()*1000. for act_grad in acts_grad]
                            #max_act_grads.append(torch.max(torch.abs(act_grad)).item())

                        # take the gradients wrt grad_norm. backward() will accumulate the gradients into the .grad attributes
                        if False and i == 0:
                            print('\n\nconv1 grads before\n{}'.format(model.conv1.weight.grad.detach().cpu().numpy()[0, 0]))
                        act_grad_norm.backward(retain_graph=False)
                        if False and i == 0:
                            print('\n\nconv1 grads after\n{}\n\n'.format(model.conv1.weight.grad.detach().cpu().numpy()[0, 0]))

                    if args.L3 > 0:   #L2 penalty for gradient size
                        params = [model.conv1.weight, model.conv2.weight, model.linear1.weight, model.linear2.weight]
                        param_grads = torch.autograd.grad(loss, params, create_graph=True, only_inputs=True)
                        # torch.autograd.grad does not accumuate the gradients into the .grad attributes. It instead returns the gradients as Variable tuples.
                        # now compute the 2-norm of the param_grads
                        grad_sum = 0
                        grad_norm = 0
                        for grad in param_grads:
                            if args.L4 > 0:
                                grad_sum += grad.pow(2).sum()
                            grad_norm += args.L3 * grad.pow(2).sum()
                            if i == 0:
                                grad_norms.append(torch.mean(torch.abs(grad)).item()*1000.)
                                #max_grads.append(torch.max(torch.abs(grad)).item())

                        # take the gradients wrt grad_norm. backward() will accumulate the gradients into the .grad attributes
                        if args.L4 > 0:
                            retain_graph = True
                        else:
                            retain_graph = False
                        grad_norm.backward(retain_graph=retain_graph)

                        if args.L4 > 0:
                            grads2 = torch.autograd.grad(grad_sum, params, create_graph=False)
                            #grads2 = torch.autograd.grad(grad_norm, params, create_graph=True)
                            g2_norm = 0
                            for g2 in grads2:
                                #g2_norm += g2.norm(p=2)
                                g2_norm += g2.pow(2).sum()
                            g2_norm = args.L4 * g2_norm

                            g2_norm.backward(retain_graph=True)

                    if args.print_stats and i == 0:
                        norm_string = '  weights {}  weight_grads {}  acts {}  act_grads {}'.format(
                            '{:.2f} {:.2f} {:.2f} {:.2f}'.format(*weight_norms), '{:.2f} {:.2f} {:.2f} {:.2f}'.format(*grad_norms),
                            '{:.2f} {:.2f} {:.2f} {:.2f}'.format(*act_norms), '{:.2f} {:.2f} {:.2f} {:.2f}'.format(*act_grad_norms))
                        norm_string_reduced = '  weights {}  acts {}'.format(
                            '{:.2f} {:.2f} {:.2f} {:.2f}'.format(*weight_norms), '{:.2f} {:.2f} {:.2f} {:.2f}'.format(*act_norms))
                        #max_string = 'max weights {}  weight_grads {}  acts {}  act_grads {}'.format('{:.3f} {:.3f} {:.3f} {:.3f}'.format(*max_weights),
                            #'{:.3f} {:.3f} {:.3f} {:.3f}'.format(*max_grads), '{:.3f} {:.3f} {:.3f} {:.3f}'.format(*max_acts), '{:.3f} {:.3f} {:.3f} {:.3f}'.format(*max_act_grads))
                        grad_norms = []
                        act_grad_norms = []
                        act_norms = []
                        weight_norms = []
                        max_weights = []
                        max_acts = []
                        max_grads = []
                        max_act_grads = []

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
                        for n, p in model.named_parameters():
                            p.grad.data.clamp_(-args.grad_clip, args.grad_clip)

                    if args.train_w_max:
                        w_max1_grad = torch.sum(model.conv1.weight.grad[model.conv1.weight >= model.w_max1])  #sum of conv1 weight gradients for all weights above threshold
                        w_min1_grad = torch.sum(model.conv1.weight.grad[model.conv1.weight <= model.w_min1])
                        #TODO: finish implementing moving avg
                        #w_max1_grad_sum += w_max1_grad
                        #w_max1_grad_avg = w_max1_grad_sum / (i + 1)
                        #w_max1_grad = 0.8 * w_max1_grad_avg + 0.2 * w_max1_grad

                        model.w_min1.data = model.w_min1.data - args.LR_w_max * w_min1_grad
                        model.w_max1.data = model.w_max1.data - args.LR_w_max * w_max1_grad

                        #model.w_min1.data.clamp_(-100, -0.01)
                        #model.w_max1.data.clamp_(0.01, 100)

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
                    if False and i == 0:
                        print('\n\n\nWeights before update:\n{}\n{}\n{}\n{}\n'.format(
                                model.conv1.weight.detach().cpu().numpy()[0, 0, :2], model.conv2.weight.detach().cpu().numpy()[0, 0, :2],
                                model.linear1.weight.detach().cpu().numpy()[0, :10], model.linear2.weight.detach().cpu().numpy()[0, :10]))
                    optimizer.step()

                    if False and i == 0:
                        print('\n\n\nWeights after update:\n{}\n{}\n{}\n{}\n'.format(
                                model.conv1.weight.detach().cpu().numpy()[0, 0, :2], model.conv2.weight.detach().cpu().numpy()[0, 0, :2],
                                model.linear1.weight.detach().cpu().numpy()[0, :10], model.linear2.weight.detach().cpu().numpy()[0, :10]))

                    if args.distort_w_train:
                        if False and i == 0:
                            print('\n\n\nWeights before distortion:\n{}\n{}\n{}\n{}\n'.format(
                                model.conv1.weight.detach().cpu().numpy()[0, 0, :2], model.conv2.weight.detach().cpu().numpy()[0, 0, :2],
                                model.linear1.weight.detach().cpu().numpy()[0, :10], model.linear2.weight.detach().cpu().numpy()[0, :10]))
                        #model.conv1.weight.data.add_(torch.cuda.FloatTensor(model.conv1.weight.size()).uniform_(-args.w_scale, args.w_scale))
                        model.conv1.weight.data.add_(torch.cuda.FloatTensor(model.conv1.weight.size()).uniform_(-margin1, margin1))
                        model.conv2.weight.data.add_(torch.cuda.FloatTensor(model.conv2.weight.size()).uniform_(-margin2, margin2))
                        model.linear1.weight.data.add_(torch.cuda.FloatTensor(model.linear1.weight.size()).uniform_(-margin3, margin3))
                        model.linear2.weight.data.add_(torch.cuda.FloatTensor(model.linear2.weight.size()).uniform_(-margin4, margin4))
                        if False and i == 0:
                            print('Weights after distortion:\n{}\n{}\n{}\n{}\n'.format(
                                model.conv1.weight.data.detach().cpu().numpy()[0, 0, :2], model.conv2.weight.data.detach().cpu().numpy()[0, 0, :2],
                                model.linear1.weight.data.detach().cpu().numpy()[0, :10], model.linear2.weight.data.detach().cpu().numpy()[0, :10]))
                        #raise(SystemExit)

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
                    args.layer_currents = [args.current1, args.current2, args.current3, args.current4]
                    if epoch == 0:
                        print('*********************** Setting Test Current to', args.test_current, 'currents:', args.current1, args.current2, args.current3, args.current4)

                #print('\n\n\nWeights after update:\n{}\n{}\n{}\n{}\n'.format(model.conv1.weight.data.detach().cpu().numpy()[0, 0, :2],
                        #model.conv2.weight.data.detach().cpu().numpy()[0, 0, :2], model.linear1.weight.data.detach().cpu().numpy()[0, :10], model.linear2.weight.data.detach().cpu().numpy()[0, :10]))

                with torch.no_grad():
                    for i in range(num_test_batches):
                        input = test_inputs[i * args.batch_size:(i + 1) * args.batch_size]
                        label = test_labels[i * args.batch_size:(i + 1) * args.batch_size]
                        output = model(input, epoch, i)
                        pred = output.data.max(1)[1]
                        te_acc = pred.eq(label.data).cpu().sum().numpy() * 100.0 / args.batch_size
                        te_accuracies.append(te_acc)

                    if args.print_stats:
                        p = []
                        input_sp = []
                        nsr = []
                        for ind in range(args.num_layers):
                            p.append(np.nanmean(model.power[ind]))
                            input_sp.append(np.nanmean(model.input_sparsity[ind]))
                            nsr.append(np.nanmean(model.nsr[ind]))

                        avg_input_sparsity = np.nanmean(input_sp)
                        input_sparsity_string = '  act spars {:.2f} ({:.2f} {:.2f} {:.2f} {:.2f})'.format(avg_input_sparsity, *input_sp)
                        avg_nsr = np.nanmean(nsr)
                        noise_string = '  avg noise {:.3f} ({:.2f} {:.2f} {:.2f} {:.2f})'.format(avg_nsr, *nsr)
                        total_power = np.nansum(p)
                        power_string = '  Power {:.2f}mW ({:.2f} {:.2f} {:.2f} {:.2f})'.format(total_power, *p)

                te_acc = np.mean(te_accuracies)

                if args.distort_w_test:
                    te_acc_dists = []
                    orig_params = []

                    for n, p in model.named_parameters():
                        if False and n == 'conv1.weight':
                            print('\n\nConv1 Weights (p) before:                 {}'.format(p.data.detach().cpu().numpy()[0, 0, :1]))
                            print('Conv1 Weights (p.clone()) before:         {}'.format(p.clone().data.detach().cpu().numpy()[0, 0, :1]))
                            print('Conv1 Weights (conv1.weight.data) before: {}\n\n'.format(model.conv1.weight.data.detach().cpu().numpy()[0, 0, :1]))

                        orig_params.append(p.clone())

                    for _ in range(5):
                        te_accuracies_dist = []
                        with torch.no_grad():
                            model.conv1.weight.data.add_(torch.cuda.FloatTensor(model.conv1.weight.size()).uniform_(-margin1, margin1))
                            model.conv2.weight.data.add_(torch.cuda.FloatTensor(model.conv2.weight.size()).uniform_(-margin2, margin2))
                            model.linear1.weight.data.add_(torch.cuda.FloatTensor(model.linear1.weight.size()).uniform_(-margin3, margin3))
                            model.linear2.weight.data.add_(torch.cuda.FloatTensor(model.linear2.weight.size()).uniform_(-margin4, margin4))

                            bs = 2000
                            for i in range(10000 // bs):
                                input = test_inputs[i * bs:(i + 1) * bs]
                                label = test_labels[i * bs:(i + 1) * bs]
                                output = model(input, epoch, i)
                                pred = output.data.max(1)[1]
                                te_acc_d = pred.eq(label.data).cpu().sum().numpy() * 100.0 / bs
                                te_accuracies_dist.append(te_acc_d)

                        #print('\n', te_accuracies_dist)
                        te_acc_dist = np.mean(te_accuracies_dist)
                        te_acc_dists.append(te_acc_dist)

                        for (n, p), orig_p in zip(model.named_parameters(), orig_params):
                            if False and n == 'conv1.weight':
                                if p is model.conv1.weight:
                                    print('\np is model.conv1.weight\n\n')
                                print('\n\nConv1 Weights (p) before:                 {}'.format(p.data.detach().cpu().numpy()[0, 0, :1]))
                                print('Conv1 Weights (conv1.weight.data) before: {}'.format(model.conv1.weight.data.detach().cpu().numpy()[0, 0, :1]))
                                print('Conv1 Weights (orig_p) before:            {}'.format(orig_p.data.detach().cpu().numpy()[0, 0, :1]))

                            p.data = orig_p.clone().data

                            if False and n == 'conv1.weight':
                                #if p is model.conv1.weight:
                                print('\nConv1 Weights (p) after:                  {}'.format(p.data.detach().cpu().numpy()[0, 0, :1]))
                                print('Conv1 Weights (conv1.weight.data) after:  {}'.format(model.conv1.weight.data.detach().cpu().numpy()[0, 0, :1]))
                                print('Conv1 Weights (orig_p) after:             {}'.format(orig_p.data.detach().cpu().numpy()[0, 0, :1]))

                    avg_te_acc_dist = np.mean(te_acc_dists)
                    te_acc_dist_string = ' ({:.2f})'.format(avg_te_acc_dist)

                if args.train_act_max:
                    clip_string = '  act_max {:.2f} {:.2f} {:.2f}'.format(model.act_max1.item(), model.act_max2.item(), model.act_max3.item())
                    act_max1_values.append(model.act_max1.item())
                    act_max2_values.append(model.act_max2.item())
                    act_max3_values.append(model.act_max3.item())
                if args.train_w_max:
                    clip_string += '  w_min/max {:.3f} {:.3f}'.format(model.w_min1.item(), model.w_max1.item())
                    w_max_values.append(model.w_max1.item())
                    w_min_values.append(model.w_min1.item())

                if args.q_a1 > 0 or args.q_a2 > 0 or args.q_a3 > 0 or args.q_a4 > 0:
                    act_q_string = '  q {:d} {:d} {:d} {:d}'.format(args.q_a1, args.q_a2, args.q_a3, args.q_a4)
                else:
                    act_q_string = ''

                if args.print_stats:
                    for n, p in model.named_parameters():  # calcualte weight sparsity as the smallest 2% of all weights by magnitude
                        if 'weight' in n and ('linear' in n or 'conv' in n):
                            w_sparsity.append(p[torch.abs(p) > 0.02*(p.max() - p.min())].numel() / p.numel())
                    avg_w_sparsity = np.mean(w_sparsity)
                    w_sparsity_string = '  w spars {:.2f} ({:.2f} {:.2f} {:.2f} {:.2f})'.format(avg_w_sparsity, *w_sparsity)

                if args.print_stats:
                    print('{}\tEpoch {:>3d}  Train {:.2f}  Test {:.2f}{}  LR {:.4f}{}{}{}{}{}{}{}'.format(
                        str(datetime.now())[:-7], epoch, tr_acc, te_acc, te_acc_dist_string, lr, clip_string, power_string,
                        noise_string, w_sparsity_string, input_sparsity_string, act_q_string, norm_string_reduced))
                else:
                    print('{}\tEpoch {:>3d}  Train {:.2f}  Test {:.2f}  LR {:.4f}'.format(str(datetime.now())[:-7], epoch, tr_acc, te_acc, lr))

                #print('Epoch {:>3d}  Train {:.2f}  Test {:.2f}  {}\n\t\t\t\t{}\n'.format(epoch, tr_acc, te_acc, norm_string, max_string))
                #print('Epoch {:>3d}  Train {:.2f}  Test {:.2f}  {} {}'.format(epoch, tr_acc, te_acc, norm_string, power_string))

                if te_acc > best_accuracy:
                #if avg_te_acc_dist > best_accuracy_dist:
                    if saved:
                        os.remove(args.checkpoint_dir + '/model_epoch_{:d}_acc_{:.2f}.pth'.format(best_epoch, saved_accuracy))

                    if epoch > init_epoch + 10:
                        if create_dir:
                            utils.saveargs(args)
                            create_dir = False
                        if s == 0:
                            saved_accuracy = te_acc
                            torch.save(model.state_dict(), args.checkpoint_dir + '/model_epoch_{:d}_acc_{:.2f}.pth'.format(epoch, te_acc))
                            best_saved_acc = te_acc
                            saved = True

                    best_accuracy = te_acc
                    best_accuracy_dist = avg_te_acc_dist
                    best_epoch = epoch
                    if args.print_stats:
                        best_nsr = avg_nsr
                        best_input_sparsity = avg_input_sparsity
                        best_w_sparsity = avg_w_sparsity
                        best_power_string = power_string
                        best_noise_string = noise_string
                        best_input_sparsity_string = input_sparsity_string
                        best_w_sparsity_string = w_sparsity_string
                        best_accuracy_dist_string = te_acc_dist_string
                        best_power = total_power

                if epoch != 0 and epoch % args.early_stop_after == 0:
                    if best_accuracy <= prev_best_acc:
                        break
                    else:
                        prev_best_acc = best_accuracy

            print('\n\n{} {}  Best Accuracy: {:.2f} (epoch {})\n\n'.format(args.var_name, var, best_accuracy, best_epoch))
            best_accuracies.append(best_accuracy)
            if args.print_stats:
                print('\n\nCurrent {}  {} {}  Simulation {:d} Best Accuracy: {:.2f}{} (epoch {:d}){}{}{}\n\n'.format(
                    args.current1, args.var_name, var, s, best_accuracy, best_accuracy_dist_string, best_epoch, best_power_string, best_noise_string, best_w_sparsity_string, best_input_sparsity_string))

                best_accuracies_dist.append(best_accuracy_dist)
                best_powers.append(best_power)
                best_noises.append(best_nsr)
                best_act_sparsities.append(best_input_sparsity)
                best_w_sparsities.append(best_w_sparsity)

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


        results[var] += best_accuracies
        if args.print_stats:
            noise_results[var] += best_noises
            power_results[var] += best_powers
            act_sparsity_results[var] += best_act_sparsities
            w_sparsity_results[var] += best_w_sparsities

        if args.distort_w_test:
            print('\n\nBest accuracies for current {}  {} {} {} powers {}  noises {}\n\n'.format(
                args.current1, args.var_name, var, ['{:.2f} ({:.2f})'.format(x, y) for x, y in zip(best_accuracies, best_accuracies_dist)],
                ['{:.2f}'.format(y) for y in best_powers], ['{:.3f}'.format(y) for y in best_noises]))

            results_dist[var] += best_accuracies_dist

            fmt = '{} {}  {} ({}) mean {:.2f} ({:.2f})  max {:.2f}  min {:.2f}  w spars {:.2f}{}  act spars {:.2f}{}  power {:.2f}mW{}  noise {}  mean {:.3f}'.format(
                args.var_name, str(var), [float('{:.2f}'.format(x)) for x in results[var]], [float('{:.2f}'.format(x)) for x in results_dist[var]],
                np.mean(results[var]), np.mean(results_dist[var]), np.max(results[var]), np.min(results[var]), np.mean(w_sparsity_results[var]), best_w_sparsity_string,
                np.mean(act_sparsity_results[var]), best_input_sparsity_string, np.mean(power_results[var]), best_power_string, [float('{:.2f}'.format(x)) for x in noise_results[var]],
                np.mean(noise_results[var]))
        else:
            if args.current1 > 0 or args.current2 > 0 or args.current3 > 0 or args.current4 > 0:
                print('\n\nBest accuracies for current {}  {} {} {} powers {}  noises {}\n\n'.format(
                    args.current1, args.var_name, var, ['{:.2f}'.format(x) for x in best_accuracies],
                    ['{:.2f}'.format(y) for y in best_powers], ['{:.3f}'.format(y) for y in best_noises]))

                fmt = '{} {}  {} mean {:.2f}  max {:.2f}  min {:.2f}  w spars {:.2f}{}  act spars {:.2f}{}  power {:.2f}mW{}  noise {}  mean {:.3f}'.format(
                    args.var_name, str(var), [float('{:.2f}'.format(x)) for x in results[var]], np.mean(results[var]), np.max(results[var]), np.min(results[var]),
                    np.mean(w_sparsity_results[var]), best_w_sparsity_string, np.mean(act_sparsity_results[var]), best_input_sparsity_string,
                    np.mean(power_results[var]), best_power_string, [float('{:.2f}'.format(x)) for x in noise_results[var]], np.mean(noise_results[var]))

            else:
                '''
                print('\n\nBest accuracies for current {}  {} {} {} powers {}\n\n'.format(
                    args.current1, args.var_name, var, ['{:.2f}'.format(x) for x in best_accuracies],
                    ['{:.2f}'.format(y) for y in best_powers]))

                fmt = '{} {}  {} mean {:.2f}  max {:.2f}  min {:.2f}  w spars {:.2f}{}  act spars {:.2f}{}  power {:.2f}mW{}'.format(
                    args.var_name, str(var), [float('{:.2f}'.format(x)) for x in results[var]], np.mean(results[var]), np.max(results[var]),
                    np.min(results[var]), np.mean(w_sparsity_results[var]), best_w_sparsity_string, np.mean(act_sparsity_results[var]),
                    best_input_sparsity_string, np.mean(power_results[var]), best_power_string)
                '''
                print('\n\nBest accuracies for {} {} {}\n\n'.format(args.var_name, var, ['{:.2f}'.format(x) for x in best_accuracies]))

                fmt = '{} {:<8}  {} mean {:>4.2f}  max {:>4.2f}  min {:>4.2f}'.format(
	                args.var_name, str(var), [float('{:.2f}'.format(x)) for x in results[var]], np.mean(results[var]), np.max(results[var]), np.min(results[var]))


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




''' 
orig_params = []
for n, p in model.named_parameters():
    if n == 'conv1.weight':
        print('\n\nConv1 Weights (p) before:                 {}'.format(p.data.detach().cpu().numpy()[0, 0, :1]))
        print('Conv1 Weights (p.clone()) before:         {}'.format(p.clone().data.detach().cpu().numpy()[0, 0, :1]))
        print('Conv1 Weights (conv1.weight.data) before: {}\n\n'.format(model.conv1.weight.data.detach().cpu().numpy()[0, 0, :1]))
    orig_params.append(p.clone())

model.conv1.weight.data.add_(torch.cuda.FloatTensor(model.conv1.weight.size()).uniform_(-0.05, 0.05))

for orig_p, (n, p) in zip(orig_params, model.named_parameters()):
    if n == 'conv1.weight':
        print('\n\nConv1 Weights (p) before:                 {}'.format(p.data.detach().cpu().numpy()[0, 0, :1]))
        print('Conv1 Weights (conv1.weight.data) before: {}'.format(model.conv1.weight.data.detach().cpu().numpy()[0, 0, :1]))
        print('Conv1 Weights (orig_p) before:            {}'.format(orig_p.data.detach().cpu().numpy()[0, 0, :1]))

    p = orig_p.clone()

    if n == 'conv1.weight':
        print('\nConv1 Weights (p) after:                  {}'.format(p.data.detach().cpu().numpy()[0, 0, :1]))
        print('Conv1 Weights (conv1.weight.data) after:  {}'.format(model.conv1.weight.data.detach().cpu().numpy()[0, 0, :1]))
        print('Conv1 Weights (orig_p) after:             {}'.format(orig_p.data.detach().cpu().numpy()[0, 0, :1]))

'''
