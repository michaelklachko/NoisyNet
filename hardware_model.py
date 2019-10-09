import torch
from torch import nn
import random
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

#random.seed(1)
#torch.manual_seed(1)
#torch.backends.cudnn.deterministic = True

def add_noise_calculate_power(self, args, arrays, input, weights, output, layer_type='conv', i=0, layer_num=0, merged_dac=True):
    merged_dac = True
    with torch.no_grad():
        if (args.uniform_ind > 0 and self.training) or (args.uniform_ind > 0 and args.noise_test):
            sigmas = torch.ones_like(output) * args.uniform_ind * torch.max(torch.abs(output))
            noise_distr = Uniform(-sigmas, sigmas)
            noise = noise_distr.sample()

        elif (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
            noise_distr = Uniform(torch.ones_like(output) * args.uniform_dep, torch.ones_like(output) / args.uniform_dep)
            noise = noise_distr.sample()

        elif (args.normal_ind > 0 and self.training) or (args.normal_ind > 0 and args.noise_test):
            sigmas = (torch.ones_like(output) * args.normal_ind * torch.max(torch.abs(output))).pow(2)
            noise_distr = Normal(loc=0, scale=torch.ones_like(output) * args.normal_ind * torch.max(torch.abs(output)))
            noise = noise_distr.sample()

        elif (args.normal_dep > 0 and self.training) or (args.normal_dep > 0 and args.noise_test):
            sigmas = (args.normal_dep * output).pow(2)
            noise_distr = Normal(loc=0, scale=args.normal_dep * output)
            noise = noise_distr.sample()

        else:
            abs_weights = torch.abs(weights)
            input_max = torch.max(input)  # always 1 for RGB input, unless < 5 bits Imagenet.
            if merged_dac:  # merged DAC digital input (for the current chip - first and third layer input):
                w_max = torch.max(abs_weights)
                if layer_type == 'conv':
                    sigmas = F.conv2d(input, abs_weights)
                    dim = (1, 2, 3)
                elif layer_type == 'linear':
                    sigmas = F.linear(input, abs_weights, bias=None)
                    dim = 1

                if i < 20:
                    sample_sums = torch.sum(sigmas, dim=dim)
                    p = 1.0e-6 * 1.2 * args.layer_currents[layer_num] * torch.mean(sample_sums) / (input_max * w_max)

                noise_distr = Normal(loc=0, scale=torch.sqrt(0.1 * (w_max / args.layer_currents[layer_num]) * sigmas))

            else:  # external DAC (for the next gen hardware) or analog input in the current chip (layers 2 and 4)
                abs_w_squared = abs_weights.pow(2) + abs_weights

                if layer_type == 'conv':
                    sigmas_w_squared = F.conv2d(input, abs_w_squared)
                    dim = (1, 2, 3)
                    if i < 20:
                        sigmas = F.conv2d(input, abs_weights)

                elif layer_type == 'linear':
                    sigmas_w_squared = F.linear(input, abs_w_squared, bias=None)
                    dim = 1

                    if i < 20:
                        sigmas = F.linear(input, abs_weights, bias=None)

                if i < 20:
                    sample_sums = torch.sum(sigmas, dim=dim)
                    p = 1.0e-6 * 1.2 * args.layer_currents[layer_num] * torch.mean(sample_sums) / input_max

                noise_distr = Normal(loc=0, scale=torch.sqrt(0.1 * (input_max / args.layer_currents[layer_num]) * sigmas_w_squared))

            noise = noise_distr.sample()

            if i < 20:
                self.power[layer_num].append(p.item())
                self.nsr[layer_num].append(torch.mean(torch.abs(noise) / torch.max(output)).item())
                self.input_sparsity[layer_num].append(input[input > 0].numel() / input.numel())

    if (args.plot or args.write):
        if merged_dac:
            if args.plot_noise:
                arrays += ([sigmas.half()], [noise.half()])

                clipped_range = np.percentile(output.detach().cpu().numpy(), 99) - np.percentile(output.detach().cpu().numpy(), 1)
                if clipped_range == 0:
                    print('\n\n***** np.percentile(output, 99) = np.percentile(output, 1) *****\n\n')
                    raise(SystemExit)
                    #clipped_range = max(np.max(output) / 100., 1)
                nsr = noise / clipped_range
                arrays.append([nsr.half()])

                print('adding sigmas and noise and snr, len(arrays):', len(arrays))
            if args.plot_power:
                arrays.append([(sigmas / (input_max * w_max)).half()])
                print('adding power, len(arrays):', len(arrays))
        else:
            if args.plot_noise:
                arrays += ([sigmas_w_squared.half()], [noise.half()])

                clipped_range = np.percentile(output.detach().cpu().numpy(), 99) - np.percentile(output.detach().cpu().numpy(), 1)
                if clipped_range == 0:
                    print('\n\n***** np.percentile(output, 99) = np.percentile(output, 1) *****\n\n')
                    raise (SystemExit)
                    # clipped_range = max(np.max(output) / 100., 1)
                nsr = noise / clipped_range
                arrays.append([nsr.half()])

            if args.plot_power:
                arrays.append([(sigmas / input_max).half()])

    if (args.uniform_dep > 0 and self.training) or (args.uniform_dep > 0 and args.noise_test):
        noisy_out = output * noise.cuda()
    else:
        noisy_out = output + noise.cuda()

    return noisy_out
