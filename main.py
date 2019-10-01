import argparse
import os
import math
import time
import warnings
import numpy as np
from datetime import datetime
import copy

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
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--L3', type=float, default=0.000, metavar='', help='L2 for param grads')
    parser.add_argument('--L3_old', type=float, default=0.000, metavar='', help='L2 for param grads (original version)')
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
    parser.add_argument('--num_sims', default=1, type=int, help='number of simulations.')
    parser.add_argument('--var_name', default=None, type=str, help='var name for hyperparam search. ')
    parser.add_argument('--q_a', default=4, type=int, help='number of bits to quantize layer input')
    parser.add_argument('--local_rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--act_max', default=0, type=float, help='clipping threshold for activations')
    parser.add_argument('--eps', default=1e-7, type=float, help='epsilon to add to avoid dividing by zero')
    parser.add_argument('--grad_clip', default=0, type=float, help='max value of gradients')
    parser.add_argument('--q_scale', default=1, type=float, help='scale upper value of quantized tensor by this value')
    parser.add_argument('--pctl', default=99.98, type=float, help='percentile to show when plotting')
    parser.add_argument('--gpu', default=None, type=str, help='GPU to use, if None use all')
    parser.add_argument('--amp_level', default='O1', type=str, help='GPU to use, if None use all')
    parser.add_argument('--loss_scale', default=128.0, type=float, help='when using FP16 precision, scale loss by this value')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)

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
    feature_parser.add_argument('--plot_normalize', dest='plot_normalize', action='store_true')
    feature_parser.add_argument('--no-plot_normalize', dest='plot_normalize', action='store_false')
    parser.set_defaults(plot_normalize=False)

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

def distort_weights(model, args, s=0):
    with torch.no_grad():
        for n, p in model.named_parameters():
            #print('\n\n{}\n{}\n'.format(n, p.shape))
            if ('conv' in n or 'fc' in n or 'classifier' in n) and 'weight' in n:
                if args.debug and n == 'module.conv1.weight':
                    print('\n\n\nBefore: {} {}\n{}'.format(n, p.shape, p[0,0]))
                #p_noise = torch.cuda.FloatTensor(p.size()).uniform_(1. - args.noise, 1. + args.noise)
                p_noise = p * torch.cuda.FloatTensor(p.size()).uniform_(-args.noise, args.noise)
                if args.debug and n == 'module.conv1.weight':
                    print('\n\np_noise:\n{}\n'.format(p_noise.detach().cpu().numpy()[0, 0, 0]))
                #p.data.mul_(p_noise)
                p.data.add_(p_noise)
                print('\nAfter:  {} {}\n{}'.format(n, p.shape, p[0, 0]))
            elif 'bn' in n:
                pass


def test_weights_distortion(val_loader, model, args):
    orig_m = copy.deepcopy(model.state_dict())
    acc_d = []
    vars = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
    for args.noise in vars:
        print('\n\nDistorting weights by {}%\n\n'.format(args.noise * 100))
        te_acc_dists = []

        if args.debug:
            print('\n\nbefore:\n{}\n'.format(model.module.conv1.weight.data.detach().cpu().numpy()[0, 0, 0]))

        for s in range(args.num_sims):
            te_accuracies_dist = []
            distort_weights(model, args, s=s)
            te_acc_d = validate(val_loader, model, args)
            te_accuracies_dist.append(te_acc_d.item())
            te_acc_dist = np.mean(te_accuracies_dist)
            te_acc_dists.append(te_acc_dist)

            if args.debug:
                print('after:\n{}\n'.format(model.module.conv1.weight.data.detach().cpu().numpy()[0, 0, 0]))

            model.load_state_dict(orig_m)

            if args.debug:
                print('restored:\n{}\n'.format(model.module.conv1.weight.data.detach().cpu().numpy()[0, 0, 0]))

        avg_te_acc_dist = np.mean(te_acc_dists)
        acc_d.append(avg_te_acc_dist)
        print('\nNoise {:4.2f}: acc {:.2f}\n'.format(args.noise, avg_te_acc_dist))
    print('\n\n{}\n{}\n\n\n'.format(vars, acc_d))


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
    else:
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


def validate(val_loader, model, args, epoch=0, plot_acc=0.0):
    model.eval()
    te_accs = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if args.dali:
                input = data[0]["data"]
                target = data[0]["label"].squeeze().cuda().long()
                input_var = Variable(input)
                if args.fp16 and not args.amp:
                    input_var = input_var.half()
                output = model(input_var, epoch=epoch, i=i, acc=plot_acc)
            else:
                images, target = data
                if args.fp16 and not args.amp:
                    input = input.half()
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                output = model(images, epoch=epoch, i=i, acc=plot_acc)
            if i == 0:
                args.print_shapes = False
            acc = utils.accuracy(output, target)
            te_accs.append(acc)

            if args.q_a > 0 and args.calculate_running and i == 4:
                if args.debug:
                    print('\n')
                with torch.no_grad():
                    for m in model.modules():
                        if isinstance(m, QuantMeasure):
                            m.calculate_running = False
                            m.running_max = torch.tensor(m.running_list, device='cuda:0').mean()
                            if args.debug:
                                print('running_list:', m.running_list, 'running_max:', m.running_max)

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
    else:
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
        #model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, keep_batchnorm_fp32=args.keep_batchnorm_fp32, loss_scale=args.loss_scale)
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_level, keep_batchnorm_fp32=args.keep_batchnorm_fp32)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if torch.cuda.device_count() > 1:
        if args.amp:
            from apex.parallel import DistributedDataParallel as DDP
            # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
            # for the older version of APEX please use shared_param, for newer one it is delay_allreduce

            #By default, apex.parallel.DistributedDataParallel overlaps communication with
            # computation in the backward pass.
            # delay_allreduce delays all communication to the end of the backward pass.
            model = DDP(model, delay_allreduce=True)
        else:
            model = torch.nn.DataParallel(model)

    return model, criterion, optimizer


def train(train_loader, val_loader, model, criterion, optimizer, start_epoch, best_acc, args):
    for epoch in range(start_epoch, args.epochs):
        utils.adjust_learning_rate(optimizer, epoch, args)
        print('lr', args.lr, 'wd', args.weight_decay, 'L3', max(args.L3, args.L3_old), 'dropout', args.dropout)
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
                input_var = Variable(input)
                target_var = Variable(target)
                if args.fp16 and not args.amp:
                        input_var = input_var.half()
                output = model(input_var, epoch=epoch, i=i)
                #print('\n\n\noutput', output)
                loss = criterion(output, target_var)
            else:
                images, target = data
                if args.fp16 and not args.amp:
                    images = images.half()
                train_loader_len = len(train_loader)
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                output = model(images, epoch=epoch, i=i)
                loss = criterion(output, target)

            if i == 0:
                args.print_shapes = False
            acc = utils.accuracy(output, target)
            tr_accs.append(acc)
            #optimizer.zero_grad()
            #loss.backward()
            #loss.backward(retain_graph=True)
            '''
            print('\n\n\nIteration', i)
            for n, p in model.named_parameters():
                if 'bn' in n:
                    if p.grad is not None:
                        print('\n\n{}\nvalue\n{}\ngradient\n{}\n'.format(n, p[:4], p.grad[:4]))
                    else:
                        print('\n\n{}\nvalue\n{}\ngradient\n{}\n'.format(n, p[:4], p.grad))
            '''
            if args.L3 > 0:  #L2 penalty for gradient size
                #for n, p in model.named_parameters():
                    #if ('conv' in n or 'fc' in n) and 'weight' in n:
                        #print(n, list(p.shape))
                #raise(SystemExit)
                params = [p for n, p in model.named_parameters() if ('conv' in n or 'fc' in n) and 'weight' in n]
                #params = [model.conv1.weight, model.conv2.weight, model.linear1.weight, model.linear2.weight]
                param_grads = torch.autograd.grad(loss, params, create_graph=True, only_inputs=True)
                # torch.autograd.grad does not accumuate the gradients into the .grad attributes. It instead returns the gradients as Variable tuples.
                # now compute the 2-norm of the param_grads
                grad_norm = 0
                for grad in param_grads:
                    #print('param_grad {}:\n{}\ngrad.pow(2).mean(): {:.4f}'.format(grad.shape, grad[0,0], grad.pow(2).mean().item()))
                    grad_norm += args.L3 * grad.pow(2).mean()
                # take the gradients wrt grad_norm. backward() will accumulate the gradients into the .grad attributes
                #grad_norm.backward(retain_graph=False)  # or like this:
                #print('loss {:.4f} grad_norm {:.4f}'.format(loss.item(), grad_norm.item()))
                loss = loss + grad_norm
                #optimizer.zero_grad()
                #loss.backward(retain_graph=True)
            #else:

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
                if args.L3_old > 0:
                    retain_graph = True
                else:
                    retain_graph = False
                loss.backward(retain_graph=retain_graph)

            if args.L3_old > 0:  #L2 penalty for gradient size
                params = [p for n, p in model.named_parameters() if ('conv' in n or 'fc' in n) and 'weight' in n]
                #TODO only_inputs should be True here, but the optimal L2_old param should be adjusted:
                param_grads = torch.autograd.grad(loss, params, create_graph=True, only_inputs=False)
                # torch.autograd.grad does not accumuate the gradients into the .grad attributes. It instead returns the gradients as Variable tuples.
                # now compute the 2-norm of the param_grads
                grad_norm = 0
                for grad in param_grads:
                    grad_norm += args.L3 * grad.pow(2).sum()
                # take the gradients wrt grad_norm. backward() will accumulate the gradients into the .grad attributes
                grad_norm.backward(retain_graph=False)

            if args.grad_clip > 0:
                for n, p in model.named_parameters():
                    #if p.grad.data.max().item() > 1:
                        #print(i, n, p.grad.data.max().item())
                    p.grad.data.clamp_(-args.grad_clip, args.grad_clip)

            optimizer.step()

            if i % args.print_freq == 0:
                print('{}  Epoch {:>2d} Batch {:>4d}/{:d} LR {} | {:.2f}'.format(
                    str(datetime.now())[:-7], epoch, i, train_loader_len, optimizer.param_groups[0]["lr"], np.mean(tr_accs)))

            if args.calculate_running and epoch == start_epoch and i == 5:
                print('\n')
                with torch.no_grad():
                    for m in model.modules():
                        if isinstance(m, QuantMeasure):
                            m.calculate_running = False
                            m.running_max = torch.tensor(m.running_list, device='cuda:0').mean()
                            print('running_list:', m.running_list, 'running_max:', m.running_max)

            if args.distort_w_train:
                distort_weights(model, args)

                #torch.save({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(), 'best_acc': 0.0,
                # 'optimizer': optimizer.state_dict()}, args.tag + '.pth')
                #raise(SystemExit)

        acc = validate(val_loader, model, args, epoch=epoch)
        if acc > best_acc:
            best_acc = acc
            if args.distort_w_train:
                tag = args.tag + 'noise_{:.2f}_'.format(args.noise)
            else:
                tag = args.tag
            torch.save({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(), 'best_acc': best_acc,
                        'optimizer': optimizer.state_dict()}, tag + '.pth')

        if args.dali:
            train_loader.reset()

    print('\n\nBest Accuracy {:.2f}\n\n'.format(best_acc))


def main():
    cudnn.benchmark = True
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train_loader, val_loader = utils.setup_data(args)

    if args.var_name is not None:
        total_list = []
        if args.var_name == 'pctl':
            var_list = [99.94, 99.95, 99.96, 99.97, 99.98, 99.99, 99.992, 99.994, 99.996, 99.998, 99.999]
            var_list = [99.99, 99.992, 99.994, 99.996, 99.998, 99.999, 99.9995]
        if args.var_name == 'q_scale':
            #var_list = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
            var_list = [0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]

        for var in var_list:
            setattr(args, args.var_name, var)
            acc_list = []
            print('\n*******************  {} {}  *********************\n'.format(args.var_name, var))
            for s in range(args.num_sims):
                #print('\nSimulation', s)
                if args.resume:
                    model, criterion, optimizer, best_acc, best_epoch, start_epoch = load_from_checkpoint(args)
                    if args.merge_bn:
                        merge_batchnorm(model, args)
                for m in model.modules():
                    if isinstance(m, QuantMeasure):
                        m.calculate_running = True
                        m.running_list = []
                acc = validate(val_loader, model, args, epoch=best_epoch, plot_acc=best_acc)
                acc_list.append(acc)
            total_list.append((np.mean(acc_list), np.min(acc_list), np.max(acc_list)))
            print('\n{:d} runs:  {} {} {:.2f} ({:.2f}/{:.2f})'.format(args.num_sims, args.var_name, var, *total_list[-1]))
        for var, (mean, min, max) in zip(var_list, total_list):
            print('{} {} acc {:.2f} ({:.2f}/{:.2f})'.format(args.var_name, var, mean, min, max))  #raise(SystemExit)
        #raise (SystemExit)
        return  #might fail with DataParallel

    if args.resume:
        model, criterion, optimizer, best_acc, best_epoch, start_epoch = load_from_checkpoint(args)

        if args.merge_bn:
            merge_batchnorm(model, args)

        if args.distort_w_test:
            test_weights_distortion(val_loader, model, args)
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