from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from hardware_model import add_noise_calculate_power, NoisyConv2d, NoisyLinear, QuantMeasure
from plot_histograms import plot
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

    def forward(self, x):
        self.input = x
        if self.q_a > 0:
            x = self.quantize1(x)
            self.quantized_input = x

        self.preact1 = self.fc1(x)
        x = F.relu(self.preact1)
        self.act = x
        if self.debug:
            print('\nbefore\n{}\n'.format(x[0, :100]))
        if self.q_a > 0:
            x = self.quantize1(x)
            self.quantized_act = x
            if self.debug:
                print('\nafter\n{}\n'.format(x[0, :100]))
                raise(SystemExit)
        self.preact2 = self.fc2(x)
        if self.training:
            return F.log_softmax(self.preact2, dim=1)
        else:
            return self.preact2


def train(args, model, num_train_batches, images, labels, optimizer):
    model.train()
    correct = 0
    for i in range(num_train_batches):
        batch = images[i * args.batch_size : (i + 1) * args.batch_size]
        batch_labels = labels[i * args.batch_size : (i + 1) * args.batch_size]
        optimizer.zero_grad()
        output = model(batch)
        loss = F.nll_loss(output, batch_labels)
        if args.L3 > 0 and args.L3_new == 0:
            loss.backward(retain_graph=args.L3 > 0)

        if args.L3 > 0 or args.L3_new > 0:
            param_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            grad_norm = 0
            for grad in param_grads:
                grad_norm += grad.pow(2).sum()
            if args.L3 > 0:
                (args.L3 * grad_norm).backward(retain_graph=False)
            else:
                loss = loss + args.L3_new * grad_norm

        if args.L3_new > 0 and args.L3 == 0:
            loss.backward()

        optimizer.step()

        if args.w_max > 0:
            for n, p in model.named_parameters():
                if 'weight' in n:
                    p.data.clamp_(-args.w_max, args.w_max)

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(batch_labels.view_as(pred)).sum().item()
    return 100. * correct / 50000.


def test(model, images, labels):
    model.eval()
    with torch.no_grad():
        output = model(images)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(labels.view_as(pred)).sum().item()
    return 100. * correct / 10000.


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
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    train_data, val_data = data
    train_inputs, train_labels = train_data
    test_inputs, test_labels = val_data
    train_inputs = torch.from_numpy(train_inputs).cuda()
    train_labels = torch.from_numpy(train_labels).cuda()
    test_inputs = torch.from_numpy(test_inputs).cuda()
    test_labels = torch.from_numpy(test_labels).cuda()

    model = Net(args).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.L2)
    num_train_batches = int(50000 / args.batch_size)
    best_acc = 0
    for epoch in range(args.epochs):
        train_acc = train(args, model, num_train_batches, train_inputs, train_labels, optimizer)
        val_acc = test(model, test_inputs, test_labels)
        print('Epoch {:>2d} train acc {:.2f} test acc {:.2f}'.format(epoch, train_acc, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            if epoch > 0:
                params = []
                for n, p in model.named_parameters():
                    #print(n)
                    pp = p.detach().cpu().half().numpy()
                    if args.plot:
                        pass
                        #plot(pp, title=n, log=True, path='chip_plots/epoch_' + str(epoch) + '_' + n + '.png')
                    params.append(pp)

                #np.save('mlp_params.npy', params)
                #np.save('mlp_preact1.npy', model.preact1.detach().cpu().half().numpy())
                #np.save('mlp_preact2.npy', model.preact2.detach().cpu().half().numpy())
                scipy.io.savemat('chip_plots/weights.mat', mdict={
                    'fc1': params[0],
                    'fc2': params[1]
                })
                scipy.io.savemat('chip_plots/signals.mat', mdict={
                    'input': model.input.detach().cpu().half().numpy(),
                    'quantized_input': model.quantized_input.detach().cpu().half().numpy(),
                    'preact1': model.preact1.detach().cpu().half().numpy(),
                    'act': model.act.detach().cpu().half().numpy(),
                    'quantized_act': model.quantized_act.detach().cpu().half().numpy(),
                    'preact2': model.preact2.detach().cpu().half().numpy()
                })
                #raise(SystemExit)


                if args.plot:
                    plot(model.input.detach().cpu().half().numpy(), title='input', log=True, path='chip_plots/epoch_' + str(epoch) + '_input.png')
                    plot(model.quantized_input.detach().cpu().half().numpy(), title='quantized_input', log=True, path='chip_plots/epoch_' + str(epoch) + '_quantized_input.png')
                    plot(model.preact1.detach().cpu().half().numpy(), title='preact1', log=True, path='chip_plots/epoch_'+str(epoch)+'_preact1.png')
                    plot(model.act.detach().cpu().half().numpy(), title='act', log=True, path='chip_plots/epoch_' + str(epoch) + '_act.png')
                    plot(model.quantized_act.detach().cpu().half().numpy(), title='quantized_act', log=True, path='chip_plots/epoch_' + str(epoch) + '_quantized_act.png')
                    plot(model.preact2.detach().cpu().half().numpy(), title='preact2', log=True, path='chip_plots/epoch_'+str(epoch)+'_preact2.png')
                    raise(SystemExit)

    os.rename(r'chip_plots/weights.mat', r'chip_plots/weights_act_max_{:d}_w_max_{:.1f}_L2_{:.4f}_L3_{:.1f}_drop_{:.2f}_LR_{:.3f}_acc_{:.2f}.mat'.format(
        args.act_max, args.w_max, args.L2, args.L3, args.dropout, args.LR, best_acc))
    os.rename(r'chip_plots/signals.mat', r'chip_plots/signals_act_max_{:d}_w_max_{:.1f}_acc_{:.2f}.mat'.format(args.act_max, args.w_max, best_acc))

    print('\nBest Accuracy: {:.2f}\n\n'.format(best_acc))


if __name__ == '__main__':
    main()
