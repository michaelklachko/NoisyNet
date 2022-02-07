import argparse
import warnings
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from models.resnet import ResNet18
import utils
from hardware_model import QuantMeasure

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='/data/imagenet/', metavar='DIR', help='path to dataset')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', '--batchsize', '--batch-size', '--bs', default=256, type=int, metavar='N')
    parser.add_argument('--lr', '--LR', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr_step', type=float, default=0.1, help='LR is multiplied by this value on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--L2', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay', dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N', help='number of workers for dataloader')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--tag', default='', type=str, metavar='PATH', help='tag')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--debug', dest='debug', action='store_true', help='debug')
    parser.add_argument('--distort_act', dest='distort_act', action='store_true', help='distort activations')
    parser.add_argument('--distort_pre_act', dest='distort_pre_act', action='store_true', help='distort pre-activations')
    parser.add_argument('--distort_act_test', dest='distort_act_test', action='store_true', help='distort activations during test')
    parser.add_argument('--stochastic', default=0.0, type=float, help='stochastic uniform noise to add before rounding during quantization')
    parser.add_argument('--step-after', default=30, type=int, help='reduce LR after this number of epochs')
    parser.add_argument('--q_a', default=0, type=int, help='number of bits to quantize layer input')
    parser.add_argument('--q_a_first', default=0, type=int, help='number of bits to quantize first layer input (RGB dataset)')
    parser.add_argument('--q_w', default=0, type=int, help='number of bits to quantize layer weights')
    parser.add_argument('--act_max', default=0, type=float, help='clipping threshold for activations')
    parser.add_argument('--eps', default=1e-7, type=float, help='epsilon to add to avoid dividing by zero')
    parser.add_argument('--grad_clip', default=0, type=float, help='max value of gradients')
    parser.add_argument('--q_scale', default=1, type=float, help='scale upper value of quantized tensor by this value')
    parser.add_argument('--pctl', default=99.98, type=float, help='percentile to use for input/activation clipping (usually for quantization)')
    parser.add_argument('--warmup', action='store_true', help='set lower initial learning rate to warm up the training')
    parser.add_argument('--lr-decay', type=str, default='step', help='mode for learning rate decay')

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--debug_quant', dest='debug_quant', action='store_true')
    feature_parser.add_argument('--no-debug_quant', dest='debug_quant', action='store_false')
    parser.set_defaults(debug_quant=False)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--normalize', dest='normalize', action='store_true')
    feature_parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=False)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--merge_bn', dest='merge_bn', action='store_true')
    feature_parser.add_argument('--no-merge_bn', dest='merge_bn', action='store_false')
    parser.set_defaults(merge_bn=False)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--bn_out', dest='bn_out', action='store_true')
    feature_parser.add_argument('--no-bn_out', dest='bn_out', action='store_false')
    parser.set_defaults(bn_out=False)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--track_running_stats', dest='track_running_stats', action='store_true')
    feature_parser.add_argument('--no-track_running_stats', dest='track_running_stats', action='store_false')
    parser.set_defaults(track_running_stats=True)

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

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--ignore_best_acc', dest='ignore_best_acc', action='store_true')
    feature_parser.add_argument('--no-ignore_best_acc', dest='ignore_best_acc', action='store_false')
    parser.set_defaults(ignore_best_acc=True)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--reset_start_epoch', dest='reset_start_epoch', action='store_true')
    feature_parser.add_argument('--no-reset_start_epoch', dest='reset_start_epoch', action='store_false')
    parser.set_defaults(reset_start_epoch=False)

    warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)

    args = parser.parse_args()
    return args


def validate(val_loader, model, args, epoch=0, plot_acc=0.0):
    model.eval()
    te_accs = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, target = data
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(images, epoch=epoch, i=i, acc=plot_acc)
            if i == 0:
                args.print_shapes = False
            acc = utils.accuracy(output, target)
            te_accs.append(acc)

            if args.q_a > 0 and args.calculate_running and epoch == 0 and i == 4:
                if args.debug:
                    print('\n')
                with torch.no_grad():
                    for m in model.modules():
                        if isinstance(m, QuantMeasure):
                            m.calculate_running = False
                            m.running_max = torch.tensor(m.running_list, device='cuda:0').mean()
                            print('(val) running_list:', ['{:.2f}'.format(v.item()) for v in m.running_list], 
                            'running_max: {:.3f}'.format(m.running_max.item()))

        mean_acc = np.mean(te_accs, dtype=np.float64)
        print('\n{}\tEpoch {:d}  Validation Accuracy: {:.2f}\n'.format(str(datetime.now())[:-7], epoch, mean_acc))

    return mean_acc


def build_model(args):
    if args.resume:
        print(f"\n\n\tLoading R18 checkpoint from {args.resume}\n\n")
    else:
        print(f"\n\n\tTraining R18 from scratch for {args.epochs} epochs\n\n")

    model = ResNet18(args)

    if args.debug:
        utils.print_model(model, args, full=True)
        args.print_shapes = True

    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model = torch.nn.DataParallel(model).cuda()
            
    return model, criterion, optimizer


def train(train_loader, val_loader, model, criterion, optimizer, start_epoch, best_acc, args):
    best_epoch = start_epoch
    train_loader_len = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        #utils.adjust_learning_rate(optimizer, epoch, args)
        print('LR {:.5f}'.format(float(optimizer.param_groups[0]['lr'])), 'wd', 
            optimizer.param_groups[0]['weight_decay'], 'q_a', args.q_a, 'act_max', args.act_max, 'bn_out', args.bn_out)

        model.train()
        tr_accs = []

        for i, data in enumerate(train_loader):
            images, target = data
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            utils.adjust_learning_rate(args, optimizer, epoch, i, train_loader_len)

            output = model(images, epoch=epoch, i=i)
            loss = criterion(output, target)

            if i == 0:
                args.print_shapes = False
            acc = utils.accuracy(output, target)
            tr_accs.append(acc)

            optimizer.zero_grad()

            loss.backward(retain_graph=False)

            optimizer.step()

            if i % args.print_freq == 0:
                print('{}  Epoch {:>2d} Batch {:>4d}/{:d} LR {:.5f} | {:.2f}'.format(
                    str(datetime.now())[:-7], epoch, i, train_loader_len, float(optimizer.param_groups[0]["lr"]), 
                    np.mean(tr_accs, dtype=np.float64)))

            if args.q_a > 0 and args.calculate_running and epoch == start_epoch and i == 5:
                print('\n')
                with torch.no_grad():
                    for m in model.modules():
                        if isinstance(m, QuantMeasure):
                            m.calculate_running = False
                            m.running_max = torch.tensor(m.running_list, device='cuda:0').mean()
                            print('(train) running_list:', ['{:.2f}'.format(v.item()) for v in m.running_list], 
                            'running_max: {:.3f}'.format(m.running_max.item()))

        acc = validate(val_loader, model, args, epoch=epoch)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()}, 'checkpoints/' + args.tag + '.pth')

    return best_acc, best_epoch


if __name__ == '__main__':
    args = parse_args()
    cudnn.benchmark = True

    train_loader, val_loader = utils.setup_data(args)

    if args.act_max > 0:
        print('\n\nClipping activations to (0, {:.1f}) range\n\n'.format(args.act_max))
    if args.q_a > 0:
        print('\n\nQuantizing activations to {:d} bits, calculate_running is {}, pctl={:.3f}\n\n'.format(args.q_a, args.calculate_running, args.pctl))
    if args.q_w > 0:
        print('\n\nQuantizing weights to {:d} bits, calculate_running is {}, pctl={:.3f}\n\n'.format(args.q_w, args.calculate_running, args.pctl))

    if args.resume:
        model, criterion, optimizer = build_model(args)
        model, criterion, optimizer, best_acc, start_epoch = utils.load_from_checkpoint(args, model, criterion, optimizer)

        for param_group in optimizer.param_groups:
            if args.weight_decay != param_group['weight_decay']:
                print("\n\nRestored L2: param_group['weight_decay'] {}, Specified L2: args.weight_decay {}\n\nAdjusting...\n\n\n".format(
                    param_group['weight_decay'], args.weight_decay))
                param_group['weight_decay'] = args.weight_decay

        if args.merge_bn:
            utils.merge_batchnorm(model, args)

        print('\n\nTesting accuracy on validation set (should be {:.2f})...\n'.format(best_acc))
        if (args.q_a > 0 and start_epoch != 0 and args.calculate_running) or args.reset_start_epoch:
            print('\n\nSetting start_epoch to zero to run validation\n\n')
            acc = validate(val_loader, model, args, epoch=0, plot_acc=best_acc)
        else:
            acc = validate(val_loader, model, args, epoch=start_epoch, plot_acc=best_acc)

        print(f'\n\nTest Accuracy {acc:.2f}\n\n')

        if args.ignore_best_acc:  # if doing training after restoring, save model even if new best accuracy is worse (default=True)
            best_acc = 0

        if args.evaluate:
            raise SystemExit 
    else:
        model, criterion, optimizer = build_model(args)
        best_acc, start_epoch = 0, 0

    if args.reset_start_epoch:   # do not scale LR based on start_epoch (default=False)
        start_epoch = 0

    if args.q_a > 0 and args.calculate_running:
        for m in model.modules():
            if isinstance(m, QuantMeasure):
                m.calculate_running = True
                m.running_list = []

    best_acc, best_epoch = train(train_loader, val_loader, model, criterion, optimizer, start_epoch, best_acc, args)
    print('\n\nBest Accuracy {:.2f} (epoch {:d})\n\n'.format(best_acc, best_epoch))
