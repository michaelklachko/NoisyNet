from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from hardware_model import add_noise_calculate_power, NoisyConv2d, NoisyLinear, QuantMeasure
from plot_histograms import plot_layers
import utils
import scipy.io
import os


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.debug = args.debug
        self.q_a = args.q_a
        self.fc1 = nn.Linear(784*3, 390, bias=args.use_bias)
        self.fc2 = nn.Linear(390, 10, bias=args.use_bias)
        self.quantize = QuantMeasure(args.q_a, stochastic=args.stochastic, max_value=1, debug=args.debug)
        self.quantize1 = QuantMeasure(4, stochastic=args.stochastic, max_value=1, debug=args.debug)
        self.quantize2 = QuantMeasure(2, stochastic=args.stochastic, max_value=1, debug=args.debug)
        self.quantize3 = QuantMeasure(1, stochastic=args.stochastic, max_value=1, debug=args.debug)
        #self.quantize2 = QuantMeasure(args.q_a, stochastic=args.stochastic, max_value=args.act_max, debug=args.debug)
        self.drop_p_input = args.dropout_input
        self.drop_p_act = args.dropout_act
        self.dropout_act = nn.Dropout(p=args.dropout_act)
        self.dropout_input = nn.Dropout(p=args.dropout_input)

    def forward(self, x):
        self.input = x
        if self.q_a > 0:
            #x = self.quantize(x)
            x1 = self.quantize1(x)
            x2 = self.quantize2(x)
            x3 = self.quantize3(x)
            x = torch.cat([x1, x2, x3], dim=1)
            #print(x.shape)
            self.quantized_input = x

        if self.drop_p_input > 0:
            x = self.dropout_input(x)

        self.preact = self.fc1(x)
        x = F.relu(self.preact)
        self.act = x
        if self.debug:
            print('\nbefore\n{}\n'.format(x[0, :100]))

        if self.drop_p_act > 0:
            x = self.dropout_act(x)

        self.output = self.fc2(x)
        if self.training:
            return F.log_softmax(self.output, dim=1)
        else:
            return self.output


def train(args, model, num_train_batches, images, labels, optimizer):
    model.train()
    correct = 0
    for i in range(num_train_batches):
        batch = images[i * args.batch_size : (i + 1) * args.batch_size]
        batch_labels = labels[i * args.batch_size : (i + 1) * args.batch_size]

        optimizer.zero_grad()
        output = model(batch)
        loss = F.nll_loss(output, batch_labels)

        if args.L3 > 0:
            param_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            grad_norm = 0
            for grad in param_grads:
                grad_norm += grad.pow(2).sum()
            loss = loss + args.L3 * grad_norm

        if args.L1_1 > 0:
            loss = loss + args.L1_1 * model.fc1.weight.norm(p=1)
        if args.L1_2 > 0:
            loss = loss + args.L1_2 * model.fc2.weight.norm(p=1)

        loss.backward()
        optimizer.step()

        if args.w_max > 0:
            for n, p in model.named_parameters():
                if 'weight' in n:
                    p.data.clamp_(-args.w_max, args.w_max)

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(batch_labels.view_as(pred)).sum().item()
    return 100. * correct / len(images)


def test(model, images, labels):
    model.eval()
    with torch.no_grad():
        output = model(images)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(labels.view_as(pred)).sum().item()
    return 100. * correct / len(images)


def prune_weights(args, model):
    sparsities = []
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'fc1' in n:
                prune_weights = args.prune_weights1
            elif 'fc2' in n:
                prune_weights = args.prune_weights2
            w = p.clone()
            w_pos = w.data[w.data >= 0]
            w_neg = w.data[w.data < 0]
            # set args.prune_weights per cent of smallest weights to zero
            pos_thr, _ = torch.kthvalue(torch.abs(w_pos.view(-1)), int(w_pos.numel() * prune_weights / 100.0))
            neg_thr, _ = torch.kthvalue(torch.abs(w_neg.view(-1)), int(w_neg.numel() * prune_weights / 100.0))
            sparsity = p.data[torch.abs(p.data) < 0.01 * p.data.max()].numel() / p.data.numel() * 100.0
            print('\n\nPruning {:.1f}% of {}, full range ({:.3f}, {:.3f}), thresholds ({:.3f}, {:.3f}) sparsity {:.1f}%\n{}'.format(
                prune_weights, n, p.data.min().item(), p.data.max().item(), -neg_thr, pos_thr, sparsity, p.data[0, :40].detach().cpu().numpy()))
            w_pos[w_pos < pos_thr] = 0
            w_neg[w_neg > -neg_thr] = 0
            p.data[w.data < 0] = w_neg
            p.data[w.data > 0] = w_pos
            sparsity = p.data[torch.abs(p.data) < 0.01 * p.data.max()].numel() / p.data.numel() * 100.0
            print('\n\nAfter pruning, full range ({:.3f}, {:.3f}) sparsity {:.1f}%\n{}\n\n'.format(
                p.data.min().item(), p.data.max().item(), sparsity, p.data[0, :40].detach().cpu().numpy()))
            sparsities.append(sparsity)
        return sparsities

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default='data/mnist.npy', help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=101, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--LR', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--L2', type=float, default=0.0001, metavar='L2', help='L2 weight decay strength')
    parser.add_argument('--L1_1', type=float, default=5e-4, metavar='L2', help='L1 weight decay strength')
    parser.add_argument('--L1_2', type=float, default=1e-5, metavar='L2', help='L1 weight decay strength')
    parser.add_argument('--L3', type=float, default=0.05, metavar='L3', help='gradient decay strength')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--use_bias', dest='use_bias', action='store_true', help='use biases')
    parser.add_argument('--q_a', type=int, default=4, metavar='S', help='quantize activations to this number of bits')
    parser.add_argument('--act_max', type=float, default=1.0, help='clipping threshold for activations')
    parser.add_argument('--w_max', type=float, default=0., help='clipping threshold for weights')
    parser.add_argument('--stochastic', type=float, default=0.5, help='stochastic quantization')
    parser.add_argument('--debug', dest='debug', action='store_true', help='debug')
    parser.add_argument('--calculate_running', dest='calculate_running', action='store_true', help='calculate_running')
    parser.add_argument('--plot', dest='plot', action='store_true', help='plot')
    parser.add_argument('--save', dest='save', action='store_true', help='save')
    parser.add_argument('--augment', dest='augment', action='store_true', help='augment')
    parser.add_argument('--dropout_input', type=float, default=0.2, help='dropout_input drop prob')
    parser.add_argument('--dropout_act', type=float, default=0.4, help='dropout_act drop prob')
    parser.add_argument('--prune_weights1', type=float, default=0.0, help='percentage of smallest weights to set to zero')
    parser.add_argument('--prune_weights2', type=float, default=0.0, help='percentage of smallest weights to set to zero')
    parser.add_argument('--prune_epoch', type=float, default=90, help='do pruning at the end of this epoch')
    parser.add_argument('--var_name', type=str, default='', help='var_name')
    parser.add_argument('--gpu', type=str, default=None, help='gpu')
    parser.add_argument('--num_sims', type=int, default=1, help='number of simulation runs')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    np.set_printoptions(precision=4, linewidth=200, suppress=True)

    data = np.load(args.dataset, allow_pickle=True)
    train_data, val_data = data
    train_inputs, train_labels = train_data
    test_inputs, test_labels = val_data
    train_inputs = torch.from_numpy(train_inputs).cuda()
    train_labels = torch.from_numpy(train_labels).cuda()
    test_inputs = torch.from_numpy(test_inputs).cuda()
    test_labels = torch.from_numpy(test_labels).cuda()

    results = {}

    if args.var_name == 'L1_1':
        var_list = [0, 1e-6, 2e-6, 3e-6, 5e-6, 7e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 7e-5, 1e-4, 2e-4]
    elif args.var_name == 'L1_2':
        var_list = [0, 1e-6, 2e-6, 3e-6, 5e-6, 7e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 7e-5, 1e-4, 2e-4]
    elif args.var_name == 'L3':
        var_list = [0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2]
    elif args.var_name == 'L2':
        var_list = [0, 5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 7e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 0.001]
    else:
        var_list = [' ']

    total_list = []

    for var in var_list:
        if args.var_name != '':
            print('\n\n********** Setting {} to {} **********\n\n'.format(args.var_name, var))
            setattr(args, args.var_name, var)

        results[var] = []
        best_accs = []

        for s in range(args.num_sims):
            model = Net(args).cuda()
            optimizer = optim.SGD(model.parameters(), lr=args.LR, momentum=args.momentum, weight_decay=args.L2)
            num_train_batches = int(len(train_inputs) / args.batch_size)
            best_acc = 0

            if s == 0:
                utils.print_model(model, args)

            for epoch in range(args.epochs):

                rnd_idx = np.random.permutation(len(train_inputs))
                train_inputs = train_inputs[rnd_idx]
                train_labels = train_labels[rnd_idx]

                if epoch == 80:
                    print('\nReducing learning rate ')
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.LR / 10.
                train_acc = train(args, model, num_train_batches, train_inputs, train_labels, optimizer)
                val_acc = test(model, test_inputs, test_labels)

                if (args.prune_weights1 > 0 or args.prune_weights2 > 0) and epoch % args.prune_epoch == 0 and epoch != 0:
                    print('\n\nAccuracy before pruning: {:.2f}\n\n'.format(val_acc))
                    sparsities = prune_weights(args, model)
                    val_acc = test(model, test_inputs, test_labels)
                    print('\n\nAccuracy after pruning: {:.2f}\n\n'.format(val_acc))
                else:
                    sparsities = [p.data[torch.abs(p.data) < 0.01 * p.data.max()].numel() / p.data.numel() * 100.0 for _, p in model.named_parameters()]
                print('Epoch {:>2d} train acc {:>.2f} test acc {:>.2f} sparsity {:>3.1f} {:>3.1f}'.format(epoch, train_acc, val_acc, sparsities[0], sparsities[1]))
                if val_acc > best_acc:
                    best_acc = val_acc

                    if epoch > 80 and (args.save or args.plot):

                        sparsities = prune_weights(args, model)
                        val_acc = test(model, test_inputs, test_labels)
                        print('\n\nAccuracy after pruning: {:.2f}\n\n'.format(val_acc))

                        w_pos = model.fc1.weight.clone()
                        w_pos[w_pos < 0] = 0
                        w_neg = model.fc1.weight.clone()
                        w_neg[w_neg >= 0] = 0
                        pos = F.linear(model.quantized_input, w_pos)
                        neg = F.linear(model.quantized_input, w_neg)
                        sep1 = torch.cat((neg, pos), 0)

                        w_pos = model.fc2.weight.clone()
                        w_pos[w_pos < 0] = 0
                        w_neg = model.fc2.weight.clone()
                        w_neg[w_neg >= 0] = 0
                        pos = F.linear(model.act, w_pos)
                        neg = F.linear(model.act, w_neg)
                        sep2 = torch.cat((neg, pos), 0)

                        dict_names = ['input', 'fc1_weights', 'preact', 'diff_preact', 'act', 'fc2_weights', 'output', 'diff_output']
                        tensors = [model.quantized_input, model.fc1.weight, model.preact, sep1, model.act, model.fc2.weight, model.output, sep2]
                        shapes = [list(t.shape) for t in tensors]
                        arrays = [t.detach().cpu().half().numpy() for t in tensors]
                        mlp_dict = {key: value for key, value in zip(dict_names, shapes)}
                        if args.save:
                            print('\n\nSaving MLP:\n{}\n'.format(mlp_dict))
                            # np.save('mlp.npy', arrays[1:])
                            # scipy.io.savemat('chip_plots/mnist_val.mat', mdict={key: value for key, value in zip(names[:], values[:])})
                            # scipy.io.savemat('chip_plots/mnist_labels.mat', mdict={'mnist_test_labels': test_labels.detach().cpu().numpy()})
                            # print('\nLabels:', test_labels.detach().cpu().numpy().shape, test_labels.detach().cpu().numpy()[:20], '\n\n')
                            scipy.io.savemat('chip_plots/mlp.mat', mdict={key: value for key, value in zip(dict_names[1:], arrays[1:])})
                            # scipy.io.savemat('chip_plots/mlp_first_layer_q4_act_1_acc_.mat', mdict={dict_names[2]: arrays[2], dict_names[3]: arrays[3]})

                        if args.plot:
                            names = ['input', 'weights', 'output', 'diff_output']
                            layers = []
                            layer = []
                            print('\n\nlen(arrays) // len(names):', len(arrays), len(names), len(arrays) // len(names), '\n\n')
                            num_layers = len(arrays) // len(names)
                            for k in range(num_layers):
                                print('layer', k, names)
                                for j in range(len(names)):
                                    layer.append([arrays[len(names) * k + j]])
                                layers.append(layer)
                                layer = []

                            info = []
                            neuron_inputs = []
                            for n, p in model.named_parameters():
                                if 'weight' in n:
                                    neuron_inputs.append(np.prod(p.shape[1:]))

                            for idx in range(len(neuron_inputs)):
                                temp = []
                                temp.append('{:d} neuron inputs '.format(neuron_inputs[idx]))
                                #if args.plot_power:
                                    #temp.append('{:.2f}mW '.format(self.power[idx][0]))
                                info.append(temp)

                            if args.plot:
                                print('\nPlotting {}\n'.format(names))
                                plot_layers(num_layers=len(layers), models=['chip_plots/'], epoch=epoch, i=0, layers=layers, names=names, var='', vars=[''], infos=info, pctl=99.9, acc=val_acc)

                            #plot_grid([[[v] for v in values]], ['input', 'quantized_input', 'weights', 'output'], path='chip_plots/epoch_' + str(epoch), filename='_mlp_histograms.png')
                            #layers = [[[a1, aa1], [a2, aa2]]]
                            #raise(SystemExit)
            if args.plot and os.path.exists('chip_plots/mlp.mat'):
                os.rename(r'chip_plots/mlp.mat', r'chip_plots/mlp_act_max_{:.1f}_w_max_{:.1f}_L2_{:.4f}_L3_{:.1f}_drop_{:.2f}_{:.2f}_LR_{:.3f}_acc_{:.2f}.mat'.format(
                    args.act_max, args.w_max, args.L2, args.L3, args.dropout_input, args.dropout_act, args.LR, best_acc))

            print('\nSimulation {:d}  Best Accuracy: {:.2f}\n\n'.format(s, best_acc))
            best_accs.append(best_acc)

        total_list.append((np.mean(best_accs), np.min(best_accs), np.max(best_accs)))
        print('\n{:d} runs:  {} {} {:.2f} ({:.2f}/{:.2f})\n'.format(args.num_sims, args.var_name, var, *total_list[-1]))

    print('\n\n')
    for var, (mean, min, max) in zip(var_list, total_list):
        print('{} {:>5} acc {:.2f} ({:.2f}/{:.2f})'.format(args.var_name, var, mean, min, max))
    print('\n\n')
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
