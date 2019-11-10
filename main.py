import argparse
import os
import math
import time
import warnings
import numpy as np
from datetime import datetime
import copy
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F

#import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from models.resnet import ResNet18
from models.mobilenet import mobilenet_v2

import utils
from quant import QuantMeasure
#from mn import mobilenet_v2

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', default='/data/imagenet/', metavar='DIR', help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='dali: 10, dataparallel: 16')
    parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
    parser.add_argument('--lr', '--LR', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--L1', type=float, default=0.000, metavar='', help='L1 for params')
    parser.add_argument('--wd', '--L2', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--L3', type=float, default=0.000, metavar='', help='L2 for param grads')
    parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--tag', default='', type=str, metavar='PATH', help='tag')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--debug', dest='debug', action='store_true', help='debug')
    parser.add_argument('--distort_w_test', dest='distort_w_test', action='store_true', help='distort weights during test')
    parser.add_argument('--distort_act', dest='distort_act', action='store_true', help='distort activations')
    parser.add_argument('--distort_act_test', dest='distort_act_test', action='store_true', help='distort activations during test')
    parser.add_argument('--noise', default=0, type=float, help='mult weights by uniform noise with this range +/-')
    parser.add_argument('--stochastic', default=0.5, type=float, help='stochastic uniform noise to add before rounding during quantization')
    parser.add_argument('--step-after', default=30, type=int, help='reduce LR after this number of epochs')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--num_sims', default=1, type=int, help='number of simulations.')
    parser.add_argument('--var_name', default=None, type=str, help='var name for hyperparam search. ')
    parser.add_argument('--q_a', default=4, type=int, help='number of bits to quantize layer input')
    parser.add_argument('--q_a_first', default=0, type=int, help='number of bits to quantize first layer input (RGB dataset)')
    parser.add_argument('--q_w', default=0, type=int, help='number of bits to quantize layer weights')
    parser.add_argument('--n_w', type=float, default=0, metavar='', help='weight noise to add during training (0.05 == 5%)')
    parser.add_argument('--n_w_test', type=float, default=0, metavar='', help='weight noise to add during test')
    parser.add_argument('--local_rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--act_max', default=0, type=float, help='clipping threshold for activations')
    parser.add_argument('--w_max', default=0, type=float, help='clipping threshold for weights')
    parser.add_argument('--eps', default=1e-7, type=float, help='epsilon to add to avoid dividing by zero')
    parser.add_argument('--grad_clip', default=0, type=float, help='max value of gradients')
    parser.add_argument('--q_scale', default=1, type=float, help='scale upper value of quantized tensor by this value')
    parser.add_argument('--pctl', default=99.98, type=float, help='percentile to show when plotting')
    parser.add_argument('--gpu', default=None, type=str, help='GPU to use, if None use all')
    parser.add_argument('--amp_level', default='O1', type=str, help='GPU to use, if None use all')
    parser.add_argument('--loss_scale', default=128.0, type=float, help='when using FP16 precision, scale loss by this value')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--selected_weights', type=float, default=0, metavar='', help='reduce noise for this fraction (%) of weights by selected_weights_noise_scale')
    parser.add_argument('--selection_criteria', type=str, default=0, metavar='', help='how to choose important weights: "weight_magnitude", "grad_magnitude", "combined"')
    parser.add_argument('--selected_weights_noise_scale', type=float, default=0, metavar='', help='multiply noise for selected_weights by this amount')
    parser.add_argument('--debug_noise', dest='debug_noise', action='store_true', help='debug when adding noise to weights')
    parser.add_argument('--old_checkpoint', dest='old_checkpoint', action='store_true', help='use this to load checkpoints from Oct 2, 2019 or earlier')

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    feature_parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.set_defaults(pretrained=False)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--debug_quant', dest='debug_quant', action='store_true')
    feature_parser.add_argument('--no-debug_quant', dest='debug_quant', action='store_false')
    parser.set_defaults(debug_quant=False)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--normalize', dest='normalize', action='store_true')
    feature_parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=False)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--dali', dest='dali', action='store_true')
    feature_parser.add_argument('--no-dali', dest='dali', action='store_false')
    parser.set_defaults(dali=True)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--amp', dest='amp', action='store_true')
    feature_parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.set_defaults(amp=False)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--dali_cpu', dest='dali_cpu', action='store_true')
    feature_parser.add_argument('--no-dali_cpu', dest='dali_cpu', action='store_false')
    parser.set_defaults(dali_cpu=True)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--merge_bn', dest='merge_bn', action='store_true')
    feature_parser.add_argument('--no-merge_bn', dest='merge_bn', action='store_false')
    parser.set_defaults(merge_bn=False)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--fp16', dest='fp16', action='store_true')
    feature_parser.add_argument('--no-fp16', dest='fp16', action='store_false')
    parser.set_defaults(fp16=False)

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
    feature_parser.add_argument('--q_inplace', dest='q_inplace', action='store_true')
    feature_parser.add_argument('--no-q_inplace', dest='q_inplace', action='store_false')
    parser.set_defaults(q_inplace=False)

    warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)

    args = parser.parse_args()
    return args


def load_from_checkpoint(args):
    model, criterion, optimizer = build_model(args)
    if os.path.isfile(args.resume):
        if args.var_name is None:
            print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        best_epoch = checkpoint['epoch']
        #model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.var_name is None:
            print("=> loaded checkpoint '{}' {:.2f} (epoch {})\n".format(args.resume, best_acc, best_epoch))
        if args.debug:
            utils.print_model(model, args, full=True)

        for saved_name, saved_param in checkpoint['state_dict'].items():
            #if saved model used DataParallel, convert this model to DP even if using a single GPU
            if 'module' in saved_name and torch.cuda.device_count() == 1:
                model = torch.nn.DataParallel(model)
                break
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
                if args.debug:
                    print('\tmatched, copying...')
                m = model.state_dict()
                m.update({saved_name: saved_param})
                model.load_state_dict(m)
            if args.q_a > 0 and ('quantize1' in saved_name or 'quantize2' in saved_name):
                matched = True
                if args.debug:
                    print('\tmatched, copying...')
                m = model.state_dict()
                m.update({saved_name: saved_param})
                model.load_state_dict(m)
            if not matched and args.debug:
                print('\t\t\t************ Not copying', saved_name)
        if args.debug:
            print('\n\nCurrent model')
            for name, param in model.state_dict().items():
                print(name)
            print('\n\n')

            print('\n\ncheckpoint:\n\n')
            for name, param in checkpoint['state_dict'].items():
                print(name)
            print('\n\n')
        #model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        raise(SystemExit)

    return model, criterion, optimizer, best_acc, best_epoch, start_epoch


def get_gradients(model, args, val_loader):
    params = []
    grads = []
    criterion = nn.CrossEntropyLoss().cuda()
    for n, p in model.named_parameters():
        if ('conv' in n or 'fc' in n or 'classifier' in n or 'linear' in n) and 'weight' in n:
            grads.append(torch.zeros_like(p))
            params.append(p)
    # accumulate gradients for all (n) batches
    if isinstance(val_loader, tuple):  # TODO do not treat cifar-10 as a special case here!
        inputs, labels = val_loader  # TODO do not use validation set for this!!!
        for i in range(10000 // args.batch_size):
            input = inputs[i * args.batch_size:(i + 1) * args.batch_size]
            label = labels[i * args.batch_size:(i + 1) * args.batch_size]
            output = model(input)
            loss = criterion(output, label)
            batch_grads = torch.autograd.grad(loss, params)  # grads for a single batch
            for bg, grad in zip(batch_grads, grads):  # accumulate grads
                # grad += bg  # TODO verify that abs values work better (typo in the paper??)
                grad += torch.abs(bg)
    else:  # Imagenet
        for i, data in enumerate(val_loader):
            if args.dali:
                input = data[0]["data"]
                target = data[0]["label"].squeeze().cuda().long()
                images = Variable(input)
                target = Variable(target)
            else:
                images, target = data
            if args.fp16 and not args.amp:
                images = images.half()
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            batch_grads = torch.autograd.grad(loss, params)  # grads for a single batch
            for bg, grad in zip(batch_grads, grads):  # accumulate grads
                #grad += bg
                grad += torch.abs(bg)
            if i == 10:
                break
        if args.dali:
            val_loader.reset()

    return grads


def distort_weights(params, args, noise=0.0, selective=False, grads=None, criteria='grad_magnitude'):
    # np.set_printoptions(precision=4, linewidth=200, suppress=True)
    with torch.no_grad():
        if grads is None:
            grads = [0] * len(params)  # ugly placeholder
            pctls_normalized = [0] * len(params)

        if selective:  # select thresholds for weight importance:
            pctls = []
            for p, g in zip(params, grads):
                if criteria == 'grad_magnitude':
                    pctl, _ = torch.kthvalue(torch.abs(g.view(-1)), int(g.numel() * (100 - args.selected_weights) / 100.0))
                elif criteria == 'weight_magnitude':
                    pctl, _ = torch.kthvalue(torch.abs(p.view(-1)), int(p.numel() * (100 - args.selected_weights) / 100.0))
                elif criteria == 'combined':  # first order term of Taylor expansion - product of weight derivative and weight value
                    pctl, _ = torch.kthvalue(torch.abs((g * p).view(-1)), int(g.numel() * (100 - args.selected_weights) / 100.0))
                else:
                    print('\n\nUnknown selection criteria: {}, Exiting...\n\n'.format(criteria))
                    raise(SystemExit)
                pctls.append(pctl)

            # Normalize pctls across layers:
            pctl_norm = torch.norm(torch.tensor(pctls), p=2)
            #pctl_norm2 = torch.sqrt(torch.sum(torch.tensor([t.pow(2) for t in pctls])))
            #print('\n\n\nnorm1, norm2:', pctl_norm1.item(), pctl_norm2.item())
            pctls_normalized = [p/pctl_norm for p in pctls]

        for p, g, pctl in zip(params, grads, pctls_normalized):
            p_noise = p * torch.cuda.FloatTensor(p.size()).uniform_(-noise, noise)
            if selective:  # distort most important weights less
                if criteria == 'grad_magnitude':
                    # distort the weights with top n gradients less than the rest of the weights
                    values = g.data
                elif criteria == 'weight_magnitude':
                    # distort these largest weights less than the rest of the weights
                    values = p.clone().data  # torch.where issues when using same data in assign and condition
                elif criteria == 'combined':  # first order term of Taylor expansion - product of weight derivative and weight value
                    # distort the weights with top n (weight * gradients)  less than the rest of the weights
                    values = g.data * p.clone().data  # torch.where issues when using same data in assign and condition
                else:
                    print('\n\nUnknown selection criteria: {}, Exiting...\n\n'.format(criteria))
                    raise(SystemExit)
                # reduce distortion of selected weights by args.selected_weights_noise_scale
                p.data = torch.where(torch.abs(values) < pctl, p.data + p_noise, p.data + p_noise * args.selected_weights_noise_scale)
            else:
                p.data.add_(p_noise)

        '''
        for p, g in zip(params, grads):
            p_noise = p * torch.cuda.FloatTensor(p.size()).uniform_(-noise, noise)
            if selective:  # distort most important weights less
                if criteria == 'grad_magnitude':
                    pctl, _ = torch.kthvalue(torch.abs(g.view(-1)), int(g.numel() * (100 - args.selected_weights) / 100.0))
                    # distort the weights with top n gradients less than the rest of the weights
                    values = g.data
                elif criteria == 'weight_magnitude':
                    # choose top K largest weights:
                    pctl, _ = torch.kthvalue(torch.abs(p.view(-1)), int(p.numel() * (100 - args.selected_weights) / 100.0))
                    # distort these largest weights less than the rest of the weights
                    values = p.clone().data  # torch.where issues when using same data in assign and condition
                elif criteria == 'combined':  # first order term of Taylor expansion - product of weight derivative and weight value
                    pctl, _ = torch.kthvalue(torch.abs((g * p).view(-1)), int(g.numel() * (100 - args.selected_weights) / 100.0))
                    # distort the weights with top n (weight * gradients)  less than the rest of the weights
                    values = g.data * p.clone().data  # torch.where issues when using same data in assign and condition
                # reduce distortion of selected weights by args.selected_weights_noise_scale
                else:
                    print('\n\nUnknown selection criteria: {}, Exiting...\n\n'.format(criteria))
                    raise(SystemExit)
                # TODO normalize pctls across layers!!!!
                raise (SystemExit)
                p.data = torch.where(torch.abs(values) < pctl, p.data + p_noise, p.data + p_noise * args.selected_weights_noise_scale)
            else:
                p.data.add_(p_noise)
        '''


def test_distortion(model, args, val_loader=None, mode='weights', vars=None):
    model.eval()

    if mode == 'weights':
        orig_m = copy.deepcopy(model.state_dict())
    if mode == 'acts':
        args.distort_act = True

    acc_d = []
    error_bars = []

    if args.noise > 0:
        vars = [args.noise]

    # get weights
    params = []
    for n, p in model.named_parameters():
        if ('conv' in n or 'fc' in n or 'classifier' in n or 'linear' in n) and 'weight' in n:
            #print(n, list(p.shape), p.requires_grad)
            params.append(p)

    if args.selected_weights > 0:
        grads = get_gradients(model, args, val_loader)
    else:
        grads = None

    for noise in vars:
        print('\n\nDistorting {} by {:d}%'.format(mode, int(noise * 100)))
        te_acc_dist = []

        if args.debug:
            print('\n\nbefore:\n{}\n'.format(model.module.conv1.weight.data.detach().cpu().numpy()[0, 0, 0]))

        for s in range(args.num_sims):
            if mode == 'weights':
                distort_weights(params, args, noise=noise, selective=args.selected_weights > 0, grads=grads, criteria=args.selection_criteria)
            if isinstance(val_loader, tuple):   #TODO cifar-10
                inputs, labels = val_loader
                te_accs = []
                for i in range(10000 // args.batch_size):
                    input = inputs[i * args.batch_size:(i + 1) * args.batch_size]
                    label = labels[i * args.batch_size:(i + 1) * args.batch_size]
                    output = model(input)
                    pred = output.data.max(1)[1]
                    te_acc = pred.eq(label.data).cpu().sum().numpy() * 100.0 / args.batch_size
                    te_accs.append(te_acc)
                te_acc_d = np.mean(te_accs)
            else:
                te_acc_d = validate(val_loader, model, args)

            te_acc_dist.append(te_acc_d.item())

            if args.debug:
                print('after:\n{}\n'.format(model.module.conv1.weight.data.detach().cpu().numpy()[0, 0, 0]))

            if mode == 'weights':
                model.load_state_dict(orig_m)

            if args.debug:
                print('restored:\n{}\n'.format(model.module.conv1.weight.data.detach().cpu().numpy()[0, 0, 0]))

        avg_te_acc_dist = np.mean(te_acc_dist)
        error_bars.append(te_acc_dist)
        acc_d.append(avg_te_acc_dist)
        print('\nNoise {:>5.2f}: {}  avg acc {:>5.2f}'.format(noise, [float('{:.2f}'.format(v)) for v in te_acc_dist], avg_te_acc_dist))
    print('\n\n{}\n{}\n\n\n'.format(vars, [float('{0:.2f}'.format(x)) for x in acc_d]))
    for var, bar, avg_acc in zip(vars, error_bars, acc_d):
        print('Noise', var, [float('{:.2f}'.format(v)) for v in bar], '{:.2f}'.format(avg_acc))
    if args.distort_w_test and args.var_name is not None:
        return [float('{0:.2f}'.format(x)) for x in acc_d]
    elif args.noise > 0:
        return avg_te_acc_dist


def merge_batchnorm(model, args):
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
                param.data *= model.state_dict()[bn_weight].data.view(-1, 1, 1, 1) / torch.sqrt(model.state_dict()[bn_running_var].data.view(-1, 1, 1, 1) + args.eps)
                if name == 'module.features.15.conv2.conv.weight' and args.debug:
                    print('\n\nAfter:\n', param[0, :10])  #, model.module.features.15.conv2.conv.weight[0, :10])

    elif args.arch == 'resnet18':
        for name, param in model.state_dict().items():  #model.named_parameters():
            if name == 'module.conv1.weight':
                if args.debug:
                    print(name)
                    print('\n\nBefore:\n', model.module.conv1.weight[0, 0, 0])
                bn_weight = 'module.bn1.weight'
                bn_running_var = 'module.bn1.running_var'
            elif 'conv' in name:
                bn_prefix = name[:16]
                bn_num = name[20]
                bn_weight = bn_prefix + 'bn' + bn_num + '.weight'
                bn_running_var = bn_prefix + 'bn' + bn_num + '.running_var'
                if args.debug:
                    print(name)
                    print('bn_prefix', bn_prefix)
                    print('bn_num', bn_num)
                    print('bn_weight', bn_weight)
                    print('bn_running_var', bn_running_var)
            elif 'downsample.0' in name:
                bn_prefix = name[:16]
                bn_weight = bn_prefix + 'downsample.1.weight'
                bn_running_var = bn_prefix + 'bn' + bn_num + '.running_var'
            if 'conv' in name or 'downsample.0' in name:
                param.data *= model.state_dict()[bn_weight].data.view(-1, 1, 1, 1) / torch.sqrt(model.state_dict()[bn_running_var].data.view(-1, 1, 1, 1) + args.eps)
            if name == 'module.conv1.weight':
                if args.debug:
                    print('\n\nAfter:\n', model.module.conv1.weight[0, 0, 0])

    elif args.arch == 'noisynet':
        # scale1 = model.bn1.weight.data.view(-1, 1, 1, 1) / torch.sqrt(model.bn1.running_var.data.view(-1, 1, 1, 1) + 0.0000001)
        bn1_weights = model.bn1.weight
        bn1_biases = model.bn1.bias
        bn1_run_var = model.bn1.running_var
        bn1_run_mean = model.bn1.running_mean
        bn1_scale = bn1_weights.data.view(-1, 1, 1, 1) / torch.sqrt(bn1_run_var.data.view(-1, 1, 1, 1) + 0.0000001)
        if args.debug:
            print('\nconv1 bn1.weight\n', bn1_weights.detach().cpu().numpy())
            print('\nconv1 bn1.bias\n', bn1_biases.detach().cpu().numpy())
            print('\nconv1 bn1 run_vars\n', bn1_run_var.detach().cpu().numpy())
            print('\nbn1 run_means\n', bn1_run_mean.detach().cpu().numpy())
            print('\nconv1 bn1 scale\n', bn1_scale.view(-1).detach().cpu().numpy())
        model.conv1.weight.data *= bn1_scale  # (64,3,5,5) x (64)

        bn2_weights = model.bn2.weight
        bn2_biases = model.bn2.bias
        bn2_run_var = model.bn2.running_var
        bn2_run_mean = model.bn2.running_mean
        bn2_scale = bn2_weights.data.view(-1, 1, 1, 1) / torch.sqrt(bn2_run_var.data.view(-1, 1, 1, 1) + 0.0000001)
        if args.debug:
            print('\nconv2 bn2.weight\n', bn2_weights.detach().cpu().numpy())
            print('\nconv1 bn2.bias\n', bn2_biases.detach().cpu().numpy())
            print('\nconv2 bn2 run_vars\n', bn2_run_var.detach().cpu().numpy())
            print('\nbn2 run_means\n', bn2_run_mean.detach().cpu().numpy())
            print('\nconv2 bn2 scale\n', bn2_scale.view(-1).detach().cpu().numpy())
        model.conv2.weight.data *= bn2_scale

        bn3_weights = model.bn3.weight
        bn3_biases = model.bn3.bias
        bn3_run_var = model.bn3.running_var
        bn3_run_mean = model.bn3.running_mean
        bn3_scale = bn3_weights.data.view(-1, 1) / torch.sqrt(bn3_run_var.data.view(-1, 1) + 0.0000001)
        if args.debug:
            print('\nbn3.weight\n', bn3_weights.detach().cpu().numpy())
            print('\nbn3.bias\n', bn3_biases.detach().cpu().numpy())
            print('\nbn3 run_vars\n', bn3_run_var.detach().cpu().numpy())
            print('\nbn3 run_means\n', bn3_run_mean.detach().cpu().numpy())
            print('\nbn3 scale\n', bn3_scale.view(-1).detach().cpu().numpy())
        model.linear1.weight.data *= bn3_scale

        bn4_weights = model.bn4.weight
        bn4_biases = model.bn4.bias
        bn4_run_var = model.bn4.running_var
        bn4_run_mean = model.bn4.running_mean
        bn4_scale = bn4_weights.data.view(-1, 1) / torch.sqrt(bn4_run_var.data.view(-1, 1) + 0.0000001)
        if args.debug:
            print('\nbn4.weight\n', bn4_weights.detach().cpu().numpy())
            print('\nbn4.bias\n', bn4_biases.detach().cpu().numpy())
            print('\nbn4 run_vars\n', bn4_run_var.detach().cpu().numpy())
            print('\nbn4 run_means\n', bn4_run_mean.detach().cpu().numpy())
            print('\nbn4 scale\n', bn4_scale.view(-1).detach().cpu().numpy())
        model.linear2.weight.data *= bn4_scale


def validate(val_loader, model, args, epoch=0, plot_acc=0.0):
    model.eval()
    te_accs = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if args.dali:
                input = data[0]["data"]
                target = data[0]["label"].squeeze().cuda().long()
                images = Variable(input)
            else:
                images, target = data
            if args.fp16 and not args.amp:
                images = images.half()
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(images, epoch=epoch, i=i, acc=plot_acc)
            if i == 0:
                args.print_shapes = False
            acc = utils.accuracy(output, target)
            te_accs.append(acc)

            if False and args.q_a > 0 and args.calculate_running and i == 4:
                if args.debug:
                    print('\n')
                with torch.no_grad():
                    for m in model.modules():
                        if isinstance(m, QuantMeasure):
                            m.calculate_running = False
                            m.running_max = torch.tensor(m.running_list, device='cuda:0').mean()
                            if args.debug:
                                print('(val) running_list:', m.running_list, 'running_max:', m.running_max)

        mean_acc = np.mean(te_accs)
        print('\n{}\tEpoch {:d}  Validation Accuracy: {:.2f}\n'.format(str(datetime.now())[:-7], epoch, mean_acc))
        if args.dali:
            val_loader.reset()
    return mean_acc


def build_model(args):
    if args.var_name is None:
        if args.pretrained or args.resume:
            print("\n\n\tLoading pre-trained {}\n\n".format(args.arch))
        else:
            print("\n\n\tTraining {}\n\n".format(args.arch))

    if args.arch == 'mobilenet_v2':
        #model = models.mobilenet_v2(pretrained=args.pretrained)
        #model = MobileNetV2()
        model = mobilenet_v2(args)
    else:
        model = ResNet18(args)
        if args.pretrained:
            model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))

    model = model.cuda()
    if args.fp16 and not args.amp:
        model = model.half()
        #keep BN in FP32 because there's no CUDNN ops for it in FP32 (causes slowdown) TODO need to verify this!:
        if args.L3 == 0:   # does not work with L3
            for layer in model.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.float()

    if args.debug:
        utils.print_model(model, args, full=True)
        args.print_shapes = True
    elif args.var_name is not None:
        utils.print_model(model, args, full=False)

    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()

    if args.amp and torch.cuda.device_count() > 1:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.fp16 and not args.amp:  #loss scaling for SGD with weight decay:
        args.lr /= args.loss_scale
        args.weight_decay *= args.loss_scale
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.amp:
        from apex import amp
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, keep_batchnorm_fp32=args.keep_batchnorm_fp32, loss_scale=args.loss_scale)
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_level, keep_batchnorm_fp32=args.keep_batchnorm_fp32)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if torch.cuda.device_count() > 1:
        if args.amp:
            from apex.parallel import DistributedDataParallel as DDP
            # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
            # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
            # By default, apex.parallel.DistributedDataParallel overlaps communication with
            # computation in the backward pass.
            # delay_allreduce delays all communication to the end of the backward pass.
            model = DDP(model, delay_allreduce=True)
        else:
            model = torch.nn.DataParallel(model)

    return model, criterion, optimizer


def train(train_loader, val_loader, model, criterion, optimizer, start_epoch, best_acc, args):
    for epoch in range(start_epoch, args.epochs):
        utils.adjust_learning_rate(optimizer, epoch, args)
        print('LR {:.5f}'.format(float(optimizer.param_groups[0]['lr'])), 'wd', optimizer.param_groups[0]['weight_decay'], 'L1', args.L1, 'L3', args.L3, 'n_w', args.n_w)
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
                images = Variable(input)
                target = Variable(target)
            else:
                images, target = data
                train_loader_len = len(train_loader)
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            if args.fp16 and not args.amp:
                images = images.half()

            output = model(images, epoch=epoch, i=i)
            loss = criterion(output, target)

            if i == 0:
                args.print_shapes = False
            acc = utils.accuracy(output, target)
            tr_accs.append(acc)

            '''
            print('\n\n\nIteration', i)
            for n, p in model.named_parameters():
                if 'bn' in n:
                    if p.grad is not None:
                        print('\n\n{}\nvalue\n{}\ngradient\n{}\n'.format(n, p[:4], p.grad[:4]))
                    else:
                        print('\n\n{}\nvalue\n{}\ngradient\n{}\n'.format(n, p[:4], p.grad))
            '''
            if args.L3 > 0:  # L2 penalty for gradient size
                params = [p for n, p in model.named_parameters() if ('conv' in n or 'fc' in n) and 'weight' in n]
                param_grads = torch.autograd.grad(loss, params, create_graph=True, only_inputs=True)
                # torch.autograd.grad does not accumuate the gradients into the .grad attributes. It instead returns the gradients as Variable tuples.
                # now compute the 2-norm of the param_grads
                grad_norm = 0
                for grad in param_grads:
                    grad_norm += args.L3 * grad.pow(2).sum()
                # take the gradients wrt grad_norm. backward() will accumulate the gradients into the .grad attributes
                # grad_norm.backward(retain_graph=False)  # or like this:
                loss = loss + grad_norm

            if args.L1 > 0:
                if epoch == 0 and i == 0:
                    print('\n\nApplying L1 loss penalty {} to model weights\n\n'.format(args.L1))
                for n, p in model.named_parameters():
                    if ('conv' in n or 'fc' in n or 'linear' in n) and 'weight' in n:
                        loss = loss + args.L1 * p.norm(p=1)

            if args.w_max > 0:
                if epoch == 0 and i == 0:
                    print('\n\nClipping weights to ({}, {}) range\n\n'.format(-args.w_max, args.w_max))

            optimizer.zero_grad()
            if args.fp16:
                loss *= args.loss_scale
                #print('\nscaled_loss:', loss.item(), '\n\n')
                if False and i == 10:
                    raise (SystemExit)
                loss.backward()
            elif args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward(retain_graph=False)

            if args.grad_clip > 0:
                for n, p in model.named_parameters():
                    #if p.grad.data.max().item() > 1:
                        #print(i, n, p.grad.data.max().item())
                    p.grad.data.clamp_(-args.grad_clip, args.grad_clip)

            optimizer.step()

            if i % args.print_freq == 0:
                print('{}  Epoch {:>2d} Batch {:>4d}/{:d} LR {} | {:.2f}'.format(
                    str(datetime.now())[:-7], epoch, i, train_loader_len, optimizer.param_groups[0]["lr"], np.mean(tr_accs)))

            if args.q_a > 0 and args.calculate_running and epoch == start_epoch and i == 5:
                print('\n')
                with torch.no_grad():
                    for m in model.modules():
                        if isinstance(m, QuantMeasure):
                            m.calculate_running = False
                            m.running_max = torch.tensor(m.running_list, device='cuda:0').mean()
                            print('(train) running_list:', m.running_list, 'running_max:', m.running_max)

            if args.w_max > 0:
                for n, p in model.named_parameters():
                    if ('conv' in n or 'fc' in n) and 'weight' in n:
                        #print(n, p.shape)
                        p.data.clamp_(-args.w_max, args.w_max)

        acc = validate(val_loader, model, args, epoch=epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(), 'best_acc': best_acc,
                        'optimizer': optimizer.state_dict()}, 'checkpoints/' + args.tag + '.pth')

        if args.dali:
            train_loader.reset()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('****** You have chosen to seed training. This will turn on the CUDNN deterministic setting, and training will be SLOW! ******')
    else:
        cudnn.benchmark = True

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train_loader, val_loader = utils.setup_data(args)

    if args.act_max > 0:
        print('\n\nClipping activations to (0, {:.1f}) range\n\n'.format(args.act_max))
    if args.q_a > 0:
        print('\n\nQuantizing activations to {:d} bits, calculate_running is {}, pctl={:.3f}\n\n'.format(args.q_a, args.calculate_running, args.pctl))
    if args.q_w > 0:
        print('\n\nQuantizing weights to {:d} bits, calculate_running is {}, pctl={:.3f}\n\n'.format(args.q_w, args.calculate_running, args.pctl))
    if args.n_w > 0:
        print('\n\nAdding {:.1f}% noise to weights during training\n\n'.format(100.*args.n_w))
    if args.n_w_test > 0:
        print('\n\nAdding {:.1f}% noise to weights during test\n\n'.format(100.*args.n_w_test))

    if args.var_name is not None:
        total_list = []
        if args.var_name == 'pctl':
            var_list = [99.94, 99.95, 99.96, 99.97, 99.98, 99.99, 99.992, 99.994, 99.996, 99.998, 99.999]
            var_list = [99.99, 99.992, 99.994, 99.996, 99.998, 99.999, 99.9995]
        if args.var_name == 'q_scale':
            #var_list = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
            var_list = [0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]
        if args.var_name == 'selected_weights':
            var_list = [0, 1, 2, 3, 5, 10, 20, 30]
            acc_lists = []

        for var in var_list:
            setattr(args, args.var_name, var)
            acc_list = []
            print('\n*******************  {} {}  *********************\n'.format(args.var_name, var))
            for s in range(args.num_sims):
                #print('\nSimulation', s)
                if args.resume:
                    model, criterion, optimizer, best_acc, best_epoch, start_epoch = load_from_checkpoint(args)

                    if args.fp16 and not args.amp:
                        model = model.half()

                    if args.merge_bn:
                        merge_batchnorm(model, args)

                    if args.distort_w_test and args.var_name is not None:
                        noise_levels = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]
                        #noise_levels = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3]
                        accs = test_distortion(model, args, val_loader=val_loader, mode='weights', vars=noise_levels)
                        print('\n\n{:>2d}% selected weights: {}'.format(int(var), accs))
                        acc_lists.append(accs)
                        if var == var_list[-1]:
                            print('\n\nNoise levels (%):', noise_levels, '\n')
                            for v, accs in zip(var_list, acc_lists):
                                print('sel_{:02d}_scale_{:d} = {}'.format(int(v), int(args.selected_weights_noise_scale * 100), accs))
                            print('\n\n')
                            raise (SystemExit)
                        break

                if args.calculate_running:
                    for m in model.modules():
                        if isinstance(m, QuantMeasure):
                            m.calculate_running = True
                            m.running_list = []
                acc = validate(val_loader, model, args, epoch=best_epoch, plot_acc=best_acc)
                acc_list.append(acc)

            if args.distort_w_test and args.var_name is not None:  # ugly...
                continue

            total_list.append((np.mean(acc_list), np.min(acc_list), np.max(acc_list)))
            print('\n{:d} runs:  {} {} {:.2f} ({:.2f}/{:.2f})'.format(args.num_sims, args.var_name, var, *total_list[-1]))
        for var, (mean, min, max) in zip(var_list, total_list):
            print('{} {} acc {:.2f} ({:.2f}/{:.2f})'.format(args.var_name, var, mean, min, max))  #raise(SystemExit)
        #raise (SystemExit)
        return  #might fail with DataParallel

    if args.resume:
        model, criterion, optimizer, best_acc, best_epoch, start_epoch = load_from_checkpoint(args)

        for param_group in optimizer.param_groups:
            if args.weight_decay != param_group['weight_decay']:
                print("\n\nRestored L2: param_group['weight_decay'] {}, Specified L2: args.weight_decay {}\n\nAdjusting...\n\n\n".format(
                    param_group['weight_decay'], args.weight_decay))
                param_group['weight_decay'] = args.weight_decay

        if args.fp16 and not args.amp:
            model = model.half()

        if args.w_max > 0:
            for n, p in model.named_parameters():
                if ('conv' in n or 'fc' in n) and 'weight' in n:
                    # print(n, p.shape)
                    p.data.clamp_(-args.w_max, args.w_max)

        if args.merge_bn:
            merge_batchnorm(model, args)

        if args.w_max > 0:
            for n, p in model.named_parameters():
                if ('conv' in n or 'fc' in n) and 'weight' in n:
                    # print(n, p.shape)
                    p.data.clamp_(-0.25, 0.25)

        if args.distort_w_test or args.distort_act_test:
            noise_levels = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]
            #noise_levels = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3]
            if args.distort_w_test:
                mode = 'weights'
            if args.distort_act_test:
                mode = 'acts'
            test_distortion(model, args, val_loader=val_loader, mode=mode, vars=noise_levels)
            raise(SystemExit)

        print('\n\nTesting accuracy on validation set (should be {:.2f})...\n'.format(best_acc))
        validate(val_loader, model, args, epoch=best_epoch, plot_acc=best_acc)
        if args.evaluate:
            #raise (SystemExit)
            return  #might fail with DataParallel
    else:
        model, criterion, optimizer = build_model(args)
        best_acc, best_epoch, start_epoch = 0, 0, 0
    best_acc, best_epoch = 0, 0

    train(train_loader, val_loader, model, criterion, optimizer, start_epoch, best_acc, args)


if __name__ == '__main__':
    main()