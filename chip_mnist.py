from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from hardware_model import add_noise_calculate_power, NoisyConv2d, NoisyLinear, QuantMeasure
from plot_histograms import plot_layers
import scipy.io
import os


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.debug = args.debug
        self.q_a = args.q_a
        self.fc1 = nn.Linear(784, 390, bias=args.use_bias)
        self.fc2 = nn.Linear(390, 10, bias=args.use_bias)
        self.quantize1 = QuantMeasure(args.q_a, stochastic=args.stochastic, max_value=1, debug=args.debug)
        self.quantize2 = QuantMeasure(args.q_a, stochastic=args.stochastic, max_value=args.act_max, debug=args.debug)
        self.drop_p_input = args.dropout_input
        self.drop_p_act = args.dropout_act
        self.dropout_act = nn.Dropout(p=args.dropout_act)
        self.dropout_input = nn.Dropout(p=args.dropout_input)

    def forward(self, x):
        self.input = x
        if self.q_a > 0:
            x = self.quantize1(x)
            self.quantized_input = x

        if self.drop_p_input > 0:
            x = self.dropout_input(x)

        self.preact = self.fc1(x)
        x = F.relu(self.preact)
        self.act = x
        if self.debug:
            print('\nbefore\n{}\n'.format(x[0, :100]))
        if self.q_a > 0:
            x = self.quantize1(x)
            self.quantized_act = x
            if self.debug:
                print('\nafter\n{}\n'.format(x[0, :100]))
                raise(SystemExit)

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


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default='data/mnist.npy', help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--LR', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--L2', type=float, default=0.000, metavar='L2', help='weight decay strength')
    parser.add_argument('--L3', type=float, default=0., metavar='L3', help='gradient decay strength')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--use_bias', dest='use_bias', action='store_true', help='use biases')
    parser.add_argument('--q_a', type=int, default=0, metavar='S', help='quantize activations to this number of bits')
    parser.add_argument('--act_max', type=float, default=1.0, help='clipping threshold for activations')
    parser.add_argument('--w_max', type=float, default=0., help='clipping threshold for weights')
    parser.add_argument('--stochastic', type=float, default=0.5, help='stochastic quantization')
    parser.add_argument('--debug', dest='debug', action='store_true', help='debug')
    parser.add_argument('--calculate_running', dest='calculate_running', action='store_true', help='calculate_running')
    parser.add_argument('--plot', dest='plot', action='store_true', help='plot')
    parser.add_argument('--save', dest='save', action='store_true', help='save')
    parser.add_argument('--dropout_input', type=float, default=0.0, help='dropout_input drop prob')
    parser.add_argument('--dropout_act', type=float, default=0.0, help='dropout_act drop prob')
    parser.add_argument('--gpu', type=str, default=None, help='gpu')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    data = np.load(args.dataset, allow_pickle=True)
    train_data, val_data = data
    train_inputs, train_labels = train_data
    test_inputs, test_labels = val_data
    train_inputs = torch.from_numpy(train_inputs).cuda()
    train_labels = torch.from_numpy(train_labels).cuda()
    test_inputs = torch.from_numpy(test_inputs).cuda()
    test_labels = torch.from_numpy(test_labels).cuda()

    model = Net(args).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.LR, momentum=args.momentum, weight_decay=args.L2)
    num_train_batches = int(len(train_inputs) / args.batch_size)
    best_acc = 0
    for epoch in range(args.epochs):
        train_acc = train(args, model, num_train_batches, train_inputs, train_labels, optimizer)
        val_acc = test(model, test_inputs, test_labels)
        print('Epoch {:>2d} train acc {:.2f} test acc {:.2f}'.format(epoch, train_acc, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            if epoch > 50:
                dict_names = ['input', 'quantized_input', 'fc1_weights', 'preact', 'act', 'quantized_act', 'fc2_weights', 'output']
                tensors = [model.input, model.quantized_input, model.fc1.weight, model.preact, model.act, model.quantized_act, model.fc2.weight, model.output]
                shapes = [list(t.shape) for t in tensors]
                arrays = [t.detach().cpu().half().numpy() for t in tensors]
                mlp_dict = {key: value for key, value in zip(dict_names, shapes)}
                if args.save:
                    print('\n\nSaving MLP:\n{}\n'.format(mlp_dict))
                    # np.save('mlp.npy', arrays[2:])
                    # scipy.io.savemat('chip_plots/mnist_val.mat', mdict={key: value for key, value in zip(names[:2], values[:2])})
                    scipy.io.savemat('chip_plots/mlp.mat', mdict={key: value for key, value in zip(dict_names[2:], arrays[2:])})

                if args.plot:
                    names = ['input', 'quantized_input', 'weights', 'output']
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
                        plot_layers(num_layers=len(layers), models=['chip_plots/'], epoch=epoch, i=0, layers=layers, names=names, var='', vars=[''], infos=info, pctl=99.9, acc=best_acc)

                    #plot_grid([[[v] for v in values]], ['input', 'quantized_input', 'weights', 'output'], path='chip_plots/epoch_' + str(epoch), filename='_mlp_histograms.png')
                    #layers = [[[a1, aa1], [a2, aa2]]]
                    #raise(SystemExit)
    if args.plot and os.path.exists('chip_plots/mlp.mat'):
        os.rename(r'chip_plots/mlp.mat', r'chip_plots/mlp_act_max_{:.1f}_w_max_{:.1f}_L2_{:.4f}_L3_{:.1f}_drop_{:.2f}_{:.2f}_LR_{:.3f}_acc_{:.2f}.mat'.format(
            args.act_max, args.w_max, args.L2, args.L3, args.dropout_input, args.dropout_act, args.LR, best_acc))

    print('\nBest Accuracy: {:.2f}\n\n'.format(best_acc))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
