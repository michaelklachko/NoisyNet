import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn.functional as F


def get_layers(arrays, input, weight, output, stride=1, padding=1, layer='conv', basic=False, debug=False):
    print('\n\nInput:', list(input.shape), 'weights:', list(weight.shape), 'output:', list(output.shape), 'layer type:', layer)
    with torch.no_grad():
        print('\n\nLayer:', layer)
        arrays.append([input.half().detach().cpu().numpy()])
        print('adding input, len(arrays):', len(arrays))
        arrays.append([weight.half().detach().cpu().numpy()])
        print('adding weights, len(arrays):', len(arrays))
        arrays.append([output.half().detach().cpu().numpy()])
        print('adding vmms, len(arrays):', len(arrays))

        if basic:
            return

        blocks = []
        pos_blocks = []
        neg_blocks = []
        weight_sums_blocked = []
        weight_sums_sep_blocked = []
        block_size = 64
        dim = weight.shape[1]  # weights shape: (fm_out, fm_in, fs, fs)

        num_blocks = max(dim // block_size, 1)  # min 1 block, must be cleanly divisible!

        if layer == 'conv':
            '''Weight blocking: fm_in is the dimension to split into blocks.  Merge filter size into fm_out, and extract dimx1x1 blocks. 
			Split input (bs, fms, h, v) into blocks of fms dim, and convolve with weight blocks. This could probably be done with grouped convolutions, but meh'''
            f = weight.permute(2, 3, 0, 1).contiguous().view(-1, dim, 1, 1)

        for b in range(num_blocks):
            if layer == 'conv':
                input_block = input[:, b * block_size: (b + 1) * block_size, :, :]
                weight_block = f[:, b * block_size: (b + 1) * block_size, :, :]
            elif layer == 'linear':
                input_block = input[:, b * block_size: (b + 1) * block_size]
                weight_block = weight[:, b * block_size: (b + 1) * block_size]
            weight_block_pos = weight_block.clone()
            weight_block_neg = weight_block.clone()
            weight_block_pos[weight_block_pos <= 0] = 0
            weight_block_neg[weight_block_neg > 0] = 0
            if b == 0 and debug:
                print(
                '\n\nNumber of blocks:', num_blocks, 'weight block shape:', weight_block.shape, '\nweights for single output neuron:', weight_block[0].shape,
                '\nActual weights (one block):\n', weight_block[0].detach().cpu().numpy().ravel())
            if layer == 'conv':
                if b == 0 and debug:
                    print('\nWeight block sum(0) shape:', weight_block.sum((1, 2, 3)).shape, '\n\n')
                blocks.append(F.conv2d(input_block, weight_block, stride=stride, padding=padding))
                pos_blocks.append(F.conv2d(input_block, weight_block_pos, stride=stride, padding=padding))
                neg_blocks.append(F.conv2d(input_block, weight_block_neg, stride=stride, padding=padding))
                weight_sums_blocked.append(torch.abs(weight_block).sum((1, 2, 3)))
                weight_sums_sep_blocked.extend([weight_block_pos.sum((1, 2, 3)), weight_block_neg.sum((1, 2, 3))])
            elif layer == 'linear':
                if b == 0 and debug:
                    print('\nWeight block sum(0) shape:', weight_block.sum(1).shape, '\n\n')
                blocks.append(F.linear(weight_block, input_block))
                pos_blocks.append(F.linear(input_block, weight_block_pos))
                neg_blocks.append(F.linear(input_block, weight_block_neg))
                weight_sums_blocked.append(torch.abs(weight_block).sum(1))
                weight_sums_sep_blocked.extend([weight_block_pos.sum(1), weight_block_neg.sum(1)])

        blocked = torch.cat(blocks, 1)  # conv_out shape: (bs, fms, h, v)
        pos_blocks = torch.cat(pos_blocks, 1)
        neg_blocks = torch.cat(neg_blocks, 1)
        # print('\n\nconv2_pos_blocks:\n', pos_blocks.shape, '\n', pos_blocks[2,2])
        # print('\n\nconv2_neg_blocks:\n', neg_blocks.shape, '\n', neg_blocks[2, 2], '\n\n')
        # raise(SystemExit)
        sep_blocked = torch.cat((pos_blocks, neg_blocks), 0)
        # print('\nblocks shape', blocks.shape, '\n')
        # print(blocks.detach().cpu().numpy()[60, 234, :8, :8])
        weight_sums_blocked = torch.cat(weight_sums_blocked, 0)
        weight_sums_sep_blocked = torch.cat(weight_sums_sep_blocked, 0)

        w_pos = weight.clone()
        w_pos[w_pos < 0] = 0
        w_neg = weight.clone()
        w_neg[w_neg >= 0] = 0
        if layer == 'conv':
            weight_sums = torch.abs(weight).sum((1, 2, 3))
            pos = F.conv2d(input, w_pos, stride=stride, padding=padding)
            neg = F.conv2d(input, w_neg, stride=stride, padding=padding)
            weight_sums_sep = torch.cat((w_pos.sum((1, 2, 3)), w_neg.sum((1, 2, 3))), 0)
        elif layer == 'linear':
            weight_sums = torch.abs(weight).sum(1)
            pos = F.linear(input, w_pos)
            neg = F.linear(input, w_neg)
            weight_sums_sep = torch.cat((w_pos.sum(1), w_neg.sum(1)), 0)

        sep = torch.cat((neg, pos), 0)

        arrays.append([sep.half().detach().cpu().numpy()])
        arrays.append([blocked.half().detach().cpu().numpy()])
        arrays.append([sep_blocked.half().detach().cpu().numpy()])

        arrays.append([weight_sums.half().detach().cpu().numpy()])
        arrays.append([weight_sums_sep.half().detach().cpu().numpy()])
        arrays.append([weight_sums_blocked.half().detach().cpu().numpy()])
        arrays.append([weight_sums_sep_blocked.half().detach().cpu().numpy()])


def plot(values1, values2=None, bins=120, range_=None, labels=['1', '2'], title='', log=False, path=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    if values2:
        alpha = 0.5
    else:
        alpha = 1
    if range_ is not None:
        ax.hist(values1.ravel(), alpha=alpha, bins=bins, range=range_, color='b', label=labels[0])
        if values2:
            ax.hist(values2.ravel(), alpha=alpha, bins=bins, range=range_, color='r', label=labels[1])
    else:
        if values2:
            range_ = (min(np.min(values1), np.min(values2)), max(np.max(values1), np.max(values2)))
        else:
            range_ = (np.min(values1), np.max(values1))
        ax.hist(values1.ravel(), alpha=alpha, bins=bins, range=range_, color='b', label=labels[0])
        if values2:
            ax.hist(values2.ravel(), alpha=alpha, bins=bins, range=range_, color='r', label=labels[1])

    plt.title(title, fontsize=18)
    # plt.xlabel('Value', fontsize=16)
    # plt.ylabel('Frequency', fontsize=16)
    plt.legend(loc='upper right')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if log:
        plt.semilogy()
    ax.legend(loc='upper right', prop={'size': 14})
    print('\n\nSaving plot to {}\n'.format(path))
    plt.savefig(path, dpi=120, bbox_inches='tight')


def place_fig(arrays, rows=1, columns=1, r=0, c=0, bins=100, range_=None, title=None, name=None, max_input=0, max_weight=0, infos=None, pctl=99.9, labels=['1'],
              log=True, normalize=True):
    ax = plt.subplot2grid((rows, columns), (r, c))
    min_value = max_value = 0
    if range_ is None:  #TODO what is this for???
        for a in arrays:
            min_value = min(min_value, np.min(a))
            max_value = max(max_value, np.max(a))
        range_ = [min_value, max_value]
    range_ = None

    if len(arrays) == 1:
        histtype = 'bar'
        alpha = 1
        infos = [infos]
    else:
        histtype = 'step'
        alpha = 1  # 2.0 / len(arrays)

    thr_neg = 0
    thr_pos = 0
    show = True

    for array, label, info, color in zip(arrays, labels, infos, ['blue', 'red', 'green', 'black', 'magenta', 'cyan', 'orange', 'yellow', 'gray']):
        if normalize:
            # thr = max(abs(thr_neg), thr_pos)
            if name != 'input' and name != 'weights':
                if "weight" in name:
                    array = array / max_weight
                else:
                    array = array / (max_weight * max_input)
                thr_neg = np.percentile(array, 100 - pctl)
                thr_pos = np.percentile(array, pctl)

            if r == 0:  # only display accuracy on first row figures
                if "weight" in name:
                    label = label + ' {:.1f}%  {:.2f} {:.2f}'.format(pctl, thr_neg, thr_pos)

        if 'power' in name:
            label = info[1] + label
        if show and 'input' in name:
            label = info[0] + label
            show = False

        ax.hist(array.ravel(), alpha=alpha, bins=bins, density=False, color=color, range=range_, histtype=histtype, label=label, linewidth=1.5)

    ax.set_title(title + name, fontsize=18)
    # plt.xlabel('Value', fontsize=16)
    # plt.ylabel('Frequency', fontsize=16)
    # plt.legend(loc='upper right')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if log:
        plt.semilogy()
    ax.legend(loc='upper right', prop={'size': 15})


def plot_grid(layers, names, path=None, filename='', info=None, pctl=99.9, labels=['1'], normalize=False):
    figsize = (len(names) * 7, len(layers) * 6)
    # figsize = (len(names) * 7, 2 * 6)
    plt.figure(figsize=figsize)
    rows = len(layers)
    columns = len(layers[0])
    thr = 0
    max_input = 0
    if info is None:
        info = [['', '']] * len(layers)  #TODO get rid of this

    for r, layer, layer_info in zip(range(rows), layers, info):
        for c, name in zip(range(columns), names):
            array = layer[c]
            if normalize:
                if name == 'input':
                    max_input = np.max(array[0])
                    array[0] = array[0] / max_input
                if name == 'weights':
                    thr_neg = np.percentile(array[0], 100 - pctl)
                    thr_pos = np.percentile(array[0], pctl)
                    thr = max(abs(thr_neg), thr_pos)
                    # print('\nthr:', thr)
                    array[0][array[0] > thr] = thr
                    array[0][array[0] < -thr] = -thr
                    # print(name, 'np.max(array)', np.max(array[0]))
                    # print('before\n', array[0].ravel()[20:40])
                    array[0] = array[0] / thr
                # print('after\n', array[0].ravel()[20:40])
            place_fig(array, rows=rows, columns=columns, r=r, c=c, title='layer' + str(r) + ' ', name=name, infos=layer_info, pctl=pctl, max_input=max_input,
                      max_weight=thr, labels=labels, normalize=normalize)
    print('\n\nSaving plot to {}\n'.format(path + filename))
    plt.savefig(path + filename, dpi=120, bbox_inches='tight')
    print('\nDone!\n')
    plt.close()


def plot_layers(num_layers=4, models=None, epoch=0, i=0, layers=None, names=None, var='', vars=[0.0], infos=None, pctl=99.9, acc=0.0, tag='', normalize=False):
    accs = [acc]

    if len(models) > 1:
        names = np.load(models[0] + 'array_names.npy', allow_pickle=True)
        layers = []
        accs = []
        infos = []
        inputs = []
        power = []
        for l in range(num_layers):  # add placeholder for each layer
            layers.append([])
            for n in range(len(names)):
                layers[l].append([])

        for model in models:
            print('\n\nLoading model {}\n\n'.format(model))
            flist = os.listdir(model)  # extract best accuracy from model name
            for fname in flist:
                if 'model' in fname:
                    acc = float(fname.split('_')[-1][:-4])
            accs.append(acc)

            model_layers = np.load(model + 'layers.npy', allow_pickle=True)  # construct layers (placeholders in layers will contain multiple arrays, one per model)

            inputs.append(np.load(model + 'input_sizes.npy', allow_pickle=True))
            if 'power' in names:
                power.append(np.load(model + 'layer_power.npy', allow_pickle=True))

            for l in range(num_layers):
                for col in range(len(model_layers[l])):
                    layers[l][col].append(model_layers[l][col][0])
                '''
                if 'noise' in names:  # add noise/signal ratio array to each layer, if noise present
                    out = model_layers[l][2][0]  # TODO fix this fragility
                    noise = model_layers[l][-1][0]  # assume vmm out to be 3rd array in layer and noise last array:
                    full_range = np.max(out) - np.min(out)
                    clipped_range = np.percentile(out, 99) - np.percentile(out, 1)
                    if clipped_range == 0:
                        clipped_range = max(np.max(out) / 100., 1)
                    error = noise / clipped_range
                    print('Layer {:d}  pre-act range: clipped (99th-1st pctl)/full {:>5.1f}/{:>5.1f}  error range {:.2f}-{:.2f}'.format(
                        l, clipped_range, full_range, np.min(error), np.max(error)))
                    layers[l][-1].append(error)
                '''

        for lr in range(len(inputs[0])):
            info = []
            for mod in range(len(inputs)):
                temp = ['{:d} inputs\n'.format(inputs[mod][lr])]
                if 'power' in names:
                    temp.append('{:.2f}mW '.format(power[mod][lr]))
                info.append(temp)
            infos.append(info)


    labels = []
    print('\n')
    for k in range(len(accs)):
        labels.append(var + ' ' + str(vars[k]) + ' ({:.1f}%)'.format(accs[k]))
        print('epoch {:d} batch {:d} plotting var {}'.format(epoch, i, labels[-1]))

    if len(models) > 1:
        filename = 'comparison_of_{}{}'.format(var, tag)
    else:
        filename = 'epoch_{:d}_iter_{:d}_acc_{:.2f}_{}.png'.format(epoch, i, acc, tag)

    if infos is None:
        infos = [['', '']] * num_layers  #TODO get rid of this

    plot_grid(layers, names, path=models[0], filename=filename, labels=labels, info=infos, pctl=pctl, normalize=normalize)


if __name__ == "__main__":
    # for comparative figures, first save all values as numpy arrays using --plot arg in noisynet.py

    model2 = 'results/power_c1_10_L2_1_0.001_current-10.0-10.0-10.0-10.0_L3-0.0_L3_act-0.0_L2-0.001-0.0-0.0-0.0_actmax-0.0-0.0-0.0_w_max1-0.0-0.0-0.0-0.0_bn-True_LR-0.001_grad_clip-0.0_2019-10-05_14-15-35/'
    model1 = 'results/power_c1_10_L2_1_0.00_current-10.0-10.0-10.0-10.0_L3-0.0_L3_act-0.0_L2-0.0-0.0-0.0-0.0_actmax-0.0-0.0-0.0_w_max1-0.0-0.0-0.0-0.0_bn-True_LR-0.001_grad_clip-0.0_2019-10-05_14-31-00/'
    model3 = 'results/current-1.0-1.0-1.0-1.0_L3-0.0_L3_act-0.0_L2-0.0-0.0-0.0-0.0_actmax-100.0-0.0-0.0_w_max1-0.0-0.0-0.0-0.0_bn-True_LR-0.005_grad_clip-0.0_2019-01-01_13-18-31/'

    model3 = 'results/power_c1_10_L2_1_0.00_clipped_current-10.0-10.0-10.0-10.0_L3-0.0_L3_act-0.0_L2-0.0-0.0-0.0-0.0_actmax-2.0-2.0-2.0_w_max1-0.2-0.2-0.2-0.2_bn-True_LR-0.001_grad_clip-0.0_2019-10-05_15-09-26/'
    model4 = 'results/power_c1_10_L2_1_0.001_clipped_current-10.0-10.0-10.0-10.0_L3-0.0_L3_act-0.0_L2-0.001-0.0-0.0-0.0_actmax-2.0-2.0-2.0_w_max1-0.2-0.2-0.2-0.2_bn-True_LR-0.001_grad_clip-0.0_2019-10-05_15-11-09/'
    models = [model1, model2, model3, model4]

    print('\n\nPlotting histograms for {:d} models\n'.format(len(models)))
    var = ''
    vars = ['no L2 no clip', 'L2 no clip', 'no L2 clip', 'L2 clip']

    tag = '_all_four___'
    plot_layers(num_layers=4, models=models, epoch=0, i=0, var=var, vars=vars, tag=tag, pctl=99.9, normalize=False)

'''
#first layer:
filter1 = abs(conv1.weight)
abs_out1 = conv2d(RGB_input, filter1)
sample_sums1 = sum(abs_out1, dim=(1, 2, 3))
w_max1 = max(filter1)
x_max1 = 1  #max(RGB_input) is always 1
if merged_dac:  #merged DAC digital input (for the current chip - first and third layer input):
    p1 = 1.0e-6 * 1.2 * max_current1 * mean(sample_sums1) / (x_max1 * w_max1)
    p1_values = abs_out1 / (x_max1 * w_max1)
    noise1 = Normal(mean=0, std=sqrt(0.1 * abs_out1 * w_max1 / max_current1))
else:  #external DAC (for the next gen hardware) or analog input in the current chip (for layers 2 and 4)
    p1 = 1.0e-6 * 1.2 * max_current1 * mean(sample_sums) / x_max1
    p1_values = abs_out1 / x_max1
    #noise:
    f1 = filter1.pow(2) + filter1
    abs_out_noise1 = F.conv2d(RGB_input, f1)
    noise1 = Normal(mean=0, std=sqrt(0.1 * abs_out_noise1 * x_max1 / max_current1))

# second layer: either analog input or external DAC
filter2 = abs(conv2.weight)
f2 = filter2.pow(2) + filter2
abs_out2 = conv2d(relu1, f2)
x_max2 = max(relu1)
sample_sums2 = sum(abs_out2, dim=(1, 2, 3))
p2 = 1.0e-6 * 1.2 * max_current2 * mean(sample_sums2) / x_max2
p2_values = abs_out2 / x_max2

#abs_out2 = conv2d(relu1, filter2)  ???
noise2 = Normal(mean=0, std=torch.sqrt(0.1 * abs_out2 * x_max2 / max_current2))
'''