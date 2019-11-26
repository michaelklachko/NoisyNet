import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn.functional as F


def get_layers(arrays, input, weight, output, stride=1, padding=1, layer='conv', basic=False, debug=False, block_size=None):
    # print('\nLayer type:', layer, 'Input:', list(input.shape), 'weights:', list(weight.shape), 'output:', list(output.shape))#,
    # '\ndot product vector length:', np.prod(list(weight.shape)[1:]), 'fanout:', list(weight.shape)[0])

    print('input {} ({:.2f}, {:.2f}) weights {} ({:.2f}, {:.2f})  output {} ({:.2f}, {:.2f})'.format(
        list(input.shape), input.min().item(), input.max().item(), list(weight.shape), weight.min().item(), weight.max().item(),
        list(output.shape), output.min().item(), output.max().item()))

    with torch.no_grad():
        arrays.append([input.half().detach().cpu().numpy()])
        arrays.append([weight.half().detach().cpu().numpy()])
        arrays.append([output.half().detach().cpu().numpy()])
        if debug:
            print('\n\nLayer:', layer)
            print('adding input, len(arrays):', len(arrays))
            print('adding weights, len(arrays):', len(arrays))
            print('adding vmms, len(arrays):', len(arrays))

        if basic:
            return

        sources = []
        sources_sep = []

        w_pos = weight.clone()
        w_pos[w_pos < 0] = 0
        w_neg = weight.clone()
        w_neg[w_neg >= 0] = 0

        if layer == 'conv':
            pos = F.conv2d(input, w_pos, stride=stride, padding=padding)
            neg = F.conv2d(input, w_neg, stride=stride, padding=padding)
        elif layer == 'linear':
            pos = F.linear(input, w_pos)
            neg = F.linear(input, w_neg)

        sep = torch.cat((neg, pos), 0)
        arrays.append([sep.half().detach().cpu().numpy()])

        fan_out = weight.shape[0]  # weights shape: (fm_out, fm_in, fs, fs) or (out_neurons, in_neurons)

        if block_size is None:  # if no block size provided, compute for entire layer
            block_sizes = [fan_out, 128, 64, 32]
        elif block_size > fan_out or block_size == 0:
            block_sizes = [fan_out]
        else:
            block_sizes = [block_size]

        for block_size in block_sizes:

            weight_block_sums = []
            weight_block_sums_sep = []
            weight_sums_blocked = []
            weight_sums_sep_blocked = []

            num_blocks = max(fan_out // block_size, 1)  # min 1 block, must be cleanly divisible!
            #print('block size', block_size, 'num_blocks', num_blocks)

            if layer == 'conv':
                '''Weight blocking: fm_in is the dimension to split into blocks.  Merge filter size into fm_out, and extract dimx1x1 blocks. 
                Split input (bs, fms, h, v) into blocks of fms fan_out, and convolve with weight blocks. This could probably be done with grouped convolutions, but meh
                
                For each pixel in a single input feature map:
                     for each location in a fs x fs filters, accross fm_out filters
                          one block of weights is a single location in a fs x fs filter, accross block_size of fm_out filters
                          sum of these weights will produce a value that will be multiplied by each input in the input feature map
                          
                          there are fs x fs filter locations, and there are fm_out // block_size blocks accross fm_out dimension, for each filter location
                          
                     fm_in_x x fm_in_y input pixels, each multiplied by fs x fs x (fm_out // block_size) weight sums
                     
                each input feature map will have its own set of fs x fs x (fm_out // block_size) weight sums
                
                1. construct weight_sums:  
                
                fm_out = weight.shape[0]  # weights shape: (fm_out, fm_in, fs, fs)
                num_blocks = max(fm_out // block_size, 1)  # min 1 block, must be cleanly divisible!
                
                inputs: (bs, fm_in, x, y) --> (fm_in, bs, x, y) --> (fm_in, -1) --> (fm_in, 1, -1)
                
                weight_sums: (fm_out, fm_in, fs, fs) --> (fm_out // block_size, fm_in, fs, fs) --> (fm_in, fm_out // block_size, fs, fs) --> (fm_in, -1) --> (fm_in, -1, 1)
                
                result = inputs * weight_sums (hadamard) = (fm_in, num_weights, num_inputs)
                '''

            if layer == 'conv':

                fm_in = weight.shape[1]

                for b in range(num_blocks):
                    weight_block = weight[b * block_size: (b + 1) * block_size, :, :, :]
                    weight_block_sum = weight_block.sum(0)  # should be (fm_in, fs, fs)
                    weight_block_sums.append(weight_block_sum.contiguous().view(fm_in, -1, 1))

                    weight_block_pos = weight_block.clone()
                    weight_block_neg = weight_block.clone()
                    weight_block_pos[weight_block_pos < 0] = 0
                    weight_block_neg[weight_block_neg > 0] = 0

                    weight_block_sum_pos = weight_block_pos.sum(0)
                    weight_block_sum_neg = weight_block_neg.sum(0)

                    weight_block_sums_sep.append(weight_block_sum_pos.contiguous().view(fm_in, -1, 1))
                    weight_block_sums_sep.append(weight_block_sum_neg.contiguous().view(fm_in, -1, 1))

                weight_sums_blocked = torch.cat(weight_block_sums, 1)    # (fm_in, -1, 1) weight_block_sum
                weight_sums_sep_blocked = torch.cat(weight_block_sums_sep, 1)

                inputs = input.permute(1, 0, 2, 3).contiguous().view(fm_in, 1, -1)

            elif layer == 'linear':
                # weights shape (1000, 512), inputs shape (bs, 512), outputs shape (bs, 1000)
                in_neurons = input.shape[1]
                out_neurons = weight.shape[0]
                bs = input.shape[0]
                for b in range(num_blocks):
                    weight_block = weight[b * block_size: (b + 1) * block_size, :]  # [out_neurons, in_neurons] --> [block_size, in_neurons]
                    weight_block_sum = weight_block.sum(0, keepdim=True)  # [1, in_neurons]
                    weight_block_sums.append(weight_block_sum)  #  num_blocks x [1, in_neurons]

                    weight_block_pos = weight_block.clone()
                    weight_block_neg = weight_block.clone()
                    weight_block_pos[weight_block_pos < 0] = 0
                    weight_block_neg[weight_block_neg > 0] = 0

                    weight_block_sum_pos = weight_block_pos.sum(0, keepdim=True)
                    weight_block_sum_neg = weight_block_neg.sum(0, keepdim=True)

                    weight_block_sums_sep.append(weight_block_sum_pos)
                    weight_block_sums_sep.append(weight_block_sum_neg)

                #print('\nweight_block_sums[0]:', weight_block_sums[0].shape, 'num_blocks, in_neurons:', num_blocks, in_neurons)
                weight_sums_blocked = torch.cat(weight_block_sums, 0).contiguous().view(num_blocks, in_neurons, 1)
                weight_sums_sep_blocked = torch.cat(weight_block_sums_sep, 0).contiguous().view(num_blocks * 2, in_neurons, 1)

                inputs = input.permute(1, 0).view(1, in_neurons, bs)  # [in_neurons, bs]

            source_sums = inputs * weight_sums_blocked
            sources.append(source_sums.half().detach().cpu().numpy())
            source_sums_sep = inputs * weight_sums_sep_blocked
            sources_sep.append(source_sums_sep.half().detach().cpu().numpy())

        for source in sources:
            arrays.append([source])

        for source_sep in sources_sep:
            arrays.append([source_sep])

        input_sums_total = []
        for block_size in block_sizes:
            """
            The blocking done below is done along different dimension from the blocking above
            
            inputs: (bs, fm_in, x, y) --> (fm_in, bs, x, y) --> (fm_in, -1) --> (fm_in, 1, -1)
    
            weights: (fm_out, fm_in, fs, fs)
            
            1. Split input feature maps into groups of block_size
            2. Do 1x1 convolution on each group  [bs, 64, 1, 1] which is really just [bs, 64] goes through weight slices: [fm_out, 64, 1, 1], which is really just [fm_out, 64]. 
            3. The result is [bs, fm_out] - compare with 1x1 conv output: [bs, 64, x, y] --> [bs, fm_out, x, y], times filter_size^2. 
            """
            input_sums = []

            if layer == 'conv':
                bs, fm_in, x, y = list(input.shape)
                fm_out, fm_in, fs, fs = list(weight.shape)
                num_blocks = max(fm_in // block_size, 1)
                if debug:
                    print('\n\nnum blocks, bs, fm_in, x, y, fm_out, fm_in, fs, fs')
                    print(num_blocks, bs, fm_in, x, y, fm_out, fm_in, fs, fs)

                for i in range(num_blocks):
                    input_block = input[:, i:i+block_size, :, :]
                    if debug:
                        #print('\nweight_block shape', weight_block.shape)
                        print('weight[:, i:i + block_size, :, :].shape', weight[:, i:i+block_size, :, :].shape)
                    weight_block = weight[:, i:i+block_size, :, :].view(fm_out, min(block_size, fm_in), -1)  # (fm_out, fm_in, fs, fs) --> (fm_out, 64, fs, fs)

                    weight_block_pos = weight_block.clone()
                    weight_block_neg = weight_block.clone()
                    weight_block_pos[weight_block_pos > 0] = 1
                    weight_block_pos[weight_block_pos < 0] = 0
                    weight_block_neg[weight_block_neg > 0] = 0
                    weight_block_neg[weight_block_neg < 0] = -1

                    for j in range(fs * fs):
                        weight_block_pos_1x1 = weight_block_pos[:, :, j].view(fm_out, min(block_size, fm_in), 1, 1)     # (fm_out, 64, fs, fs) --> (fm_out, 64, 1, 1)
                        weight_block_neg_1x1 = weight_block_neg[:, :, j].view(fm_out, min(block_size, fm_in), 1, 1)

                        input_sum_block_pos = F.conv2d(input_block, weight_block_pos_1x1, stride=stride, padding=0)  # (bs, fm_out, x, y)
                        input_sum_block_neg = F.conv2d(input_block, weight_block_neg_1x1, stride=stride, padding=0)
                        input_sums.append(input_sum_block_pos)
                        input_sums.append(input_sum_block_neg)
                if debug:
                    print(len(input_sums))
                    print(input_sum_block_neg.shape)
                    print(bs ,num_blocks ,fs ,fs, fm_out, x, y)
                    print(2 * bs * num_blocks * fs * fs, fm_out, x, y)
                    print(stride, padding)
                input_sums = torch.cat(input_sums, 0).contiguous().view(2 * bs * num_blocks * fs * fs, fm_out, x//stride, y//stride)  # the view is just a test

            elif layer == 'linear':
                bs, fm_in = list(input.shape)
                num_blocks = max(fm_in // block_size, 1)

                for i in range(num_blocks):
                    # weights are [1000, 512], inputs are [bs, 512], outputs [bs, 1000]
                    input_block = input[:, i:i + block_size]
                    weight_block = weight[:, i:i + block_size]

                    weight_block_pos = weight_block.clone()
                    weight_block_neg = weight_block.clone()
                    weight_block_pos[weight_block_pos > 0] = 1
                    weight_block_pos[weight_block_pos < 0] = 0
                    weight_block_neg[weight_block_neg > 0] = 0
                    weight_block_neg[weight_block_neg < 0] = -1

                    input_sum_block_pos = F.linear(input_block, weight_block_pos)  # (bs, 512).(512, 1000)=(bs, 1000)
                    input_sum_block_neg = F.linear(input_block, weight_block_neg)
                    input_sums.append(input_sum_block_pos)
                    input_sums.append(input_sum_block_neg)

                input_sums = torch.cat(input_sums, 0).contiguous().view(2 * bs * num_blocks, 1000)

            input_sums_total.append(input_sums.half().detach().cpu().numpy())

        for input_sum in input_sums_total:
            arrays.append([input_sum])

        """
        blocks = []
        pos_blocks = []
        neg_blocks = []
        
        f = weight.permute(2, 3, 0, 1).contiguous().view(-1, fan_out, 1, 1)  # the whole receptive field becomes a single vector

        for b in range(num_blocks):
            if layer == 'conv':
                input_block = input[:, b * block_size: (b + 1) * block_size, :, :]
                weight_block = f[:, b * block_size: (b + 1) * block_size, :, :]
            elif layer == 'linear':
                input_block = input[:, b * block_size: (b + 1) * block_size]
                weight_block = weight[:, b * block_size: (b + 1) * block_size]
                # weights shape (1000, 512), inputs shape (bs, 512), outputs shape (bs, 1000)
                # weight_block = weight[:, 0: 64]
                # weight_block: (64, 512) for each input neuron, we extract block of 64 weights going to the chunk of 64 output neurons (1000/64 chunks)
                # as a result, we have 14 blocks of weights per input neuron.
                # for 512 input neurons, we have 512 x 14 chunks of weights, that is 512 x 14 sums of weights
                # plot histograms of 512x14 values (sums of weights) by their corresponding inputs (bs, 512)

            weight_block_pos = weight_block.clone()
            weight_block_neg = weight_block.clone()
            weight_block_pos[weight_block_pos <= 0] = 0
            weight_block_neg[weight_block_neg > 0] = 0
            if b == 0 and debug:
                print('\n\nNumber of blocks:', num_blocks, 'weight block shape:', weight_block.shape, '\nweights for single output neuron:',
                      weight_block[0].shape, '\nActual weights (one block):\n', weight_block[0].detach().cpu().numpy().ravel())
            if layer == 'conv':
                if b == 0 and debug:
                    print('\nWeight block sum(0) shape:', weight_block.sum((1, 2, 3)).shape, '\n\n')
                blocks.append(F.conv2d(input_block, weight_block, stride=stride, padding=padding))
                pos_blocks.append(F.conv2d(input_block, weight_block_pos, stride=stride, padding=padding))
                neg_blocks.append(F.conv2d(input_block, weight_block_neg, stride=stride, padding=padding))
                # weight_sums_blocked.append(torch.abs(weight_block).sum((1, 2, 3)))
                weight_sums_sep_blocked.extend([weight_block_pos.sum((1, 2, 3)), weight_block_neg.sum((1, 2, 3))])
            elif layer == 'linear':
                if b == 0 and debug:
                    print('\nWeight block sum(0) shape:', weight_block.sum(1).shape, '\n\n')
                blocks.append(F.linear(weight_block, input_block))
                pos_blocks.append(F.linear(input_block, weight_block_pos))
                neg_blocks.append(F.linear(input_block, weight_block_neg))
                # weight_sums_blocked.append(torch.abs(weight_block).sum(1))
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
        # weight_sums_blocked = torch.cat(weight_sums_blocked, 0)
        weight_sums_sep_blocked = torch.cat(weight_sums_sep_blocked, 0)

        w_pos = weight.clone()
        w_pos[w_pos < 0] = 0
        w_neg = weight.clone()
        w_neg[w_neg >= 0] = 0
        if layer == 'conv':
            # weight_sums = torch.abs(weight).sum((1, 2, 3))  # assuming weights shape: (out_fms, in_fms, x, y)
            # now multiply every pixel in every input feature map by the corersponding value in weight_sums vector:
            # e.g. 64 input feature maps, 20x20 pixels each, and 64 corresponding values in weight_sums vector
            # the result will be 64x20x20 scaled values (each input feature map has its own unique scaling factor)
            # implementation: first reshape (expand) weight_sums to (1, 64, 1, 1) , then multiply (bs, 64, x, y) by this vector
            # source_values = weight_sums.view(1, len(weight_sums), 1, 1) * input
            pos = F.conv2d(input, w_pos, stride=stride, padding=padding)
            neg = F.conv2d(input, w_neg, stride=stride, padding=padding)
            weight_sums_sep = torch.cat((w_pos.sum((1, 2, 3)), w_neg.sum((1, 2, 3))), 0)
        elif layer == 'linear':
            # weight_sums = torch.abs(weight).sum(1)
            pos = F.linear(input, w_pos)
            neg = F.linear(input, w_neg)
            weight_sums_sep = torch.cat((w_pos.sum(1), w_neg.sum(1)), 0)

        sep = torch.cat((neg, pos), 0)

        if layer == 'conv':
            '''
            calculating sums of currents along source lines:
            assume input shape (256, 3, 32, 32) and weights shape (64, 3, 5, 5)
            we need to calculate for every input pixel (input current) the sum of products of its values and all the weights it will encounter along the line
            each input pixel will encounter exactly 64 weights (one per output feature map), and:
            In any given single input feature map, there will be N sets of 64 weights for each pixel where N = 5x5
            Different input feature maps will have different sets of 5x5x64 weights
            Sums of products of a pixel with 64 weights is a sum of 64 weights multiply with the pixel
            Therefore, we will have 3 sets of weights and 3 sets of pixels (3, 25) and (3, 256*32*32)
            and the output will be 3 sets of 25*256*32*32 combined, which we will plot as a histogram

            1. transpose inputs to (3, 256*32*32) and weights to (3, 64, 5, 5)
            2. reshape weights to (3, 64, 25) and use abs values
            3. reduce weights to (3, 1, 25)
            4. expand inputs to (3, 256*32*32, 1)
            5. multiply them element wise (hadamard product)
            5. the result will be (3, 256*32*32, 25), which we flatten and plot
            '''
            in_fms = list(input.shape)[1]
            out_fms = list(weight.shape)[0]
            input_t = torch.transpose(input, 0, 1).reshape(in_fms, -1, 1)
            weight_t = torch.transpose(weight, 0, 1).reshape(in_fms, out_fms, -1)
            weight_sums = torch.abs(weight_t).sum(1, keepdim=True)
            source_sums = input_t * weight_sums
            # print('\n\ninput {} weight {} input_t {} weight_t {} weight_sums {} source_sums {}\n\n'.format(
            # list(input.shape), list(weight.shape), list(input_t.shape), list(weight_t.shape), list(weight_sums.shape), list(source_sums.shape)))

        elif layer == 'linear':
            # Input: [16, 512] weights: [1000, 512] output: [16, 1000]
            # make 512 weight sums (abs values) weight_sums: (1, 512)
            # make 16 * 512 products
            weight_sums_neg = torch.abs(w_neg).sum(0, keepdim=True)
            weight_sums_pos = torch.abs(w_pos).sum(0, keepdim=True)
            weight_sums = torch.abs(weight).sum(0, keepdim=True)
            source_sums = weight_sums * input
            source_sums_pos = weight_sums_pos * input
            source_sums_neg = weight_sums_neg * input

            # print('\n\ninput {} weight {} weight_sums {} source_sums {}\n\n'.format(
            # list(input.shape), list(weight.shape), list(weight_sums.shape), list(source_sums.shape)))

        arrays.append([sep.half().detach().cpu().numpy()])
        arrays.append([blocked.half().detach().cpu().numpy()])
        arrays.append([sep_blocked.half().detach().cpu().numpy()])

        # arrays.append([weight_sums.half().detach().cpu().numpy()])
        arrays.append([weight_sums_sep.half().detach().cpu().numpy()])
        # arrays.append([weight_sums_blocked.half().detach().cpu().numpy()])
        arrays.append([weight_sums_sep_blocked.half().detach().cpu().numpy()])

        for source in sources:
            arrays.append([source])

        for source_sep in sources_sep:
            arrays.append([source_sep])
        """


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


def place_fig(arrays, rows=1, columns=1, r=0, c=0, bins=100, range_=None, title=None, name=None, infos=None, labels=['1'],
              log=True):
    ax = plt.subplot2grid((rows, columns), (r, c))
    min_value = max_value = 0
    if range_ is None and len(arrays) > 1:  # if overlapping histograms, use largest range
        for a in arrays:
            min_value = min(min_value, np.min(a))
            max_value = max(max_value, np.max(a))
        range_ = [min_value, max_value]

    if len(arrays) == 1:
        histtype = 'bar'
        alpha = 1
        infos = [infos]
    else:
        histtype = 'step'
        alpha = 1  # 2.0 / len(arrays)

    show = True

    for array, label, info, color in zip(arrays, labels, infos, ['blue', 'red', 'green', 'black', 'magenta', 'cyan', 'orange', 'yellow', 'gray']):
        if 'power' in name:
            label = info[1] + label
        if show and 'input' in name:
            label = info[0] + label
            show = False
        # if 'input' in name or 'weight' in name:
        # label = None
        # else:
        label = '({:.1f}, {:.1f})'.format(np.min(array), np.max(array))

        ax.hist(array.ravel(), alpha=alpha, bins=bins, density=False, color=color, range=range_, histtype=histtype, label=label, linewidth=1.5)

    ax.set_title(title + name, fontsize=18)
    # plt.xlabel('Value', fontsize=16)
    # plt.ylabel('Frequency', fontsize=16)
    # plt.legend(loc='upper right')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if log:
        plt.semilogy()
    ax.legend(loc='best', prop={'size': 16})


def plot_grid(layers, names, path=None, filename='', info=None, pctl=99.9, labels=['1'], normalize=False):
    figsize = (len(names) * 7, len(layers) * 6)
    # figsize = (len(names) * 7, 2 * 6)
    plt.figure(figsize=figsize)
    rows = len(layers)
    columns = len(layers[0])
    thr = 0
    max_input = 0
    if info is None:
        info = [['', '']] * len(layers)  # TODO get rid of this

    for r, layer, layer_info in zip(range(rows), layers, info):
        for c, name in zip(range(columns), names):
            array = layer[c]
            if normalize:
                if name == 'input':   # input sums are multiplied with weights set to one, so w_max = 1
                    max_input = np.max(array[0])
                    if max_input == 0:
                        print('\n\nLayer {}, array {} (column {}) error when normalizing the array\nmax_input = {} = zero\n'
                              '\nexiting...\n\n'.format(r, name, c, max_input))
                        raise (SystemExit)
                    array[0] = array[0] / max_input
                elif name == 'weights':
                    thr = np.max(np.abs(array[0]))
                    '''
                    thr_neg = np.percentile(array[0], 100 - pctl)
                    thr_pos = np.percentile(array[0], pctl)
                    thr = max(abs(thr_neg), thr_pos)
                    # print('\nthr:', thr)
                    # TODO is the below assignment safe???
                    array[0][array[0] > thr] = thr
                    array[0][array[0] < -thr] = -thr
                    '''
                    # print(name, 'np.max(array)', np.max(array[0]))
                    # print('before\n', array[0].ravel()[20:40])
                    if False and thr == 0:
                        # print('\n\nLayer {}, array {} (column {}) error when normalizing the array\nmax_weight = {} = zero\n'
                        # 'weights are clipped at ({}, {}), pctl: {}\nexiting...\n\n'.format(thr_neg, thr_pos, pctl, r, name, c, thr))
                        raise (SystemExit)
                    array[0] = array[0] / thr
                elif name == 'weight sums diff' or name == 'weight sums diff blocked':
                    array[0] = array[0] / thr
                elif 'input sums' in name:    # input sums are multiplied with weights set to one, so w_max = 1
                    array[0] = array[0] / max_input
                else:
                    array[0] = array[0] / (max_input * thr)  # TODO fragile - inputs and weights must be the first two arrays in each layer for this to work
                # print('after\n', array[0].ravel()[20:40])

            place_fig(array, rows=rows, columns=columns, r=r, c=c, title='layer' + str(r) + ' ', name=name, infos=layer_info, labels=labels)

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

            model_layers = np.load(model + 'layers.npy',
                                   allow_pickle=True)  # construct layers (placeholders in layers will contain multiple arrays, one per model)

            inputs.append(np.load(model + 'input_sizes.npy', allow_pickle=True))
            if 'power' in names:
                power.append(np.load(model + 'layer_power.npy', allow_pickle=True))

            for l in range(num_layers):
                for col in range(len(model_layers[l])):
                    layers[l][col].append(model_layers[l][col][0])

                if 'noise' in names:  # add noise/signal ratio array to each layer, if noise present
                    print('\n\nNeed to fix noise plotting! Exiting...\n\n')
                    raise (SystemExit)
                    '''
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
        infos = [['', '']] * num_layers  # TODO get rid of this

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
