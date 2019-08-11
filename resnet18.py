import argparse
import os
import random
import math
import time
import warnings
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

import utils
from quantized_modules_clean import QConv2d, QLinear, QuantMeasure

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
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
parser.add_argument('--q_a', default=0, type=int, help='number of bits to quantize layer input')
parser.add_argument('--act_max', default=0, type=float, help='clipping threshold for activations')

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--pretrained', dest='pretrained', action='store_true')
feature_parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
parser.set_defaults(pretrained=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--debug_quant', dest='debug_quant', action='store_true')
feature_parser.add_argument('--no-debug_quant', dest='debug_quant', action='store_false')
parser.set_defaults(debug_quant=False)

warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)


class BasicBlock(nn.Module):
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		if args.act_max > 0:
			self.relu = nn.Hardtanh(0.0, args.act_max, inplace=True)
		else:
			self.relu = nn.ReLU(inplace=True)
		#self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.quantize = QuantMeasure(args.q_a, stochastic=args.stochastic, debug=args.debug_quant)

	def forward(self, x):
		if args.q_a > 0:
			x = self.quantize(x)
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		if args.q_a > 0:
			out = self.quantize(out)
		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, num_classes=1000):
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		if args.act_max > 0:
			self.relu = nn.Hardtanh(0.0, args.act_max, inplace=True)
		else:
			self.relu = nn.ReLU(inplace=True)
		#self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.quantize = QuantMeasure(args.q_a, stochastic=args.stochastic, debug=args.debug_quant)

		self.layer1 = self._make_layer(block, 64)
		self.layer2 = self._make_layer(block, 128, stride=2)
		self.layer3 = self._make_layer(block, 256, stride=2)
		self.layer4 = self._make_layer(block, 512, stride=2)


		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc = nn.Linear(512, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes),
			)

		layers = [block(self.inplanes, planes, stride, downsample), block(planes, planes)]
		self.inplanes = planes

		return nn.Sequential(*layers)

	def forward(self, x):
		if args.q_a > 0:
			x = self.quantize(x)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		if args.q_a > 0:
			x = self.quantize(x)
		x = self.fc(x)

		return x

"""
class ResNet(nn.Module):

	def __init__(self, num_classes=1000):
		super(ResNet, self).__init__()

		self.quantize = QuantMeasure(args.q_a, stochastic=args.stochastic, debug=args.debug_quant)

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		'''First block'''
		#block(64, 64)
		self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(64)

		#block(64, 64)
		self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(64)
		self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn5 = nn.BatchNorm2d(64)

		'''Second block'''
		self.downsample1 = nn.Sequential(
				nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
				nn.BatchNorm2d(128),)
		#block(64, 128, 2, downsample)
		self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn6 = nn.BatchNorm2d(128)
		self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn7 = nn.BatchNorm2d(128)

		#block(128, 128)
		self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn8 = nn.BatchNorm2d(128)
		self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn9 = nn.BatchNorm2d(128)

		'''Third block'''
		self.downsample2 = nn.Sequential(
				nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
		        nn.BatchNorm2d(256),)
		#block(128, 256, 2, downsample)
		self.conv10 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn10 = nn.BatchNorm2d(256)
		self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn11 = nn.BatchNorm2d(256)

		#block(256, 256)
		self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn12 = nn.BatchNorm2d(256)
		self.conv13 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn13 = nn.BatchNorm2d(256)

		'''Fourth block'''
		self.downsample3 = nn.Sequential(
				nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
				nn.BatchNorm2d(512),)
		#block(256, 512, 2, downsample)
		self.conv14 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn14 = nn.BatchNorm2d(512)
		self.conv15 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn15 = nn.BatchNorm2d(512)

		#block(512, 512)
		self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn16 = nn.BatchNorm2d(512)
		self.conv17 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn17 = nn.BatchNorm2d(512)

		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc = nn.Linear(512, num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		#x = self.layer1(x)
		residual = x
		out = self.conv2(x)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv3(out)
		out = self.bn3(out)
		out += residual
		x = self.relu(out)

		residual = x
		out = self.conv4(x)
		out = self.bn4(out)
		out = self.relu(out)
		out = self.conv5(out)
		out = self.bn5(out)
		out += residual
		x = self.relu(out)

		#x = self.layer2(x)
		out = self.conv6(x)
		out = self.bn6(out)
		out = self.relu(out)
		out = self.conv7(out)
		out = self.bn7(out)
		residual = self.downsample1(x)
		out += residual
		x = self.relu(out)

		residual = x
		out = self.conv8(x)
		out = self.bn8(out)
		out = self.relu(out)
		out = self.conv9(out)
		out = self.bn9(out)
		out += residual
		x = self.relu(out)

		#x = self.layer3(x)
		out = self.conv10(x)
		out = self.bn10(out)
		out = self.relu(out)
		out = self.conv11(out)
		out = self.bn11(out)
		residual = self.downsample2(x)
		out += residual
		x = self.relu(out)

		residual = x
		out = self.conv12(x)
		out = self.bn12(out)
		out = self.relu(out)
		out = self.conv13(out)
		out = self.bn13(out)
		out += residual
		x = self.relu(out)

		#x = self.layer4(x)
		out = self.conv14(x)
		out = self.bn14(out)
		out = self.relu(out)
		out = self.conv15(out)
		out = self.bn15(out)
		residual = self.downsample3(x)
		out += residual
		x = self.relu(out)

		residual = x
		out = self.conv16(x)
		out = self.bn16(out)
		out = self.relu(out)
		out = self.conv17(out)
		out = self.bn17(out)
		out += residual
		x = self.relu(out)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x
"""

def validate(val_loader, model, epoch=0):
	model.eval()
	te_accs = []
	data_times = []
	compute_times = []
	other_times = []
	with torch.no_grad():
		start = time.time()
		end = time.time()
		for i, (images, target) in enumerate(val_loader):
			target = target.cuda(non_blocking=True)
			data_time = time.time()
			if i != 0:
				data_times.append(data_time - end)
			output = model(images)
			compute_time = time.time()
			if i != 0:
				compute_times.append(compute_time - end)
			acc = accuracy(output, target)
			te_accs.append(acc)
			other_time = time.time()
			if i != 0:
				other_times.append(other_time - end)
			end = time.time()
			if i == 2000:
				break
		mean_acc = np.mean(te_accs)
		data_time = np.mean(data_times)
		compute_time = np.mean(compute_times)
		other_time = np.mean(other_times)
		total_time = np.sum(data_times) + np.sum(compute_times) + np.sum(other_times)

		print('\n{}\tEpoch {:d}  Validation Accuracy: {:.2f}  time: {:.1f} ({:.2f}/{:.2f}/{:.2f}/{:.1f}({:.2f}m)\n'.format(
			str(datetime.now())[:-7], epoch, mean_acc, (end-start)/60.0, data_time, compute_time, other_time, total_time, total_time/60.))
	return mean_acc


def adjust_learning_rate(optimizer, epoch, args):
	lr = args.lr * (0.1 ** (epoch // args.step_after))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target):
	with torch.no_grad():
		batch_size = target.size(0)
		pred = output.data.max(1)[1]
		acc = pred.eq(target.data).sum().item() * 100.0 / batch_size
		return acc

ctimes = []
vtimes = []
ttimes = []
#workers_list = [0, 1 ,2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48]
#for workers in workers_list:
for kk in [1]:
	args = parser.parse_args()

	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. this will slow down training')

	if args.pretrained:
		print("\n\n\tLoading pre-trained {}\n\n".format(args.arch))
	else:
		#pass
		print("\n\n\tTraining {}\n\n".format(args.arch))

	if args.arch == 'mobilenet_v2':
		model = models.mobilenet_v2(pretrained=args.pretrained)
	else:
		model = ResNet(BasicBlock)
		if args.pretrained:
			model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))

	model = torch.nn.DataParallel(model).cuda()
	utils.print_model(model, args)
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

	best_acc = 0

	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_acc = checkpoint['best_acc']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	traindir = os.path.join(args.data, 'train')
	valdir = os.path.join(args.data, 'val')
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]))

	val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		]))

	if args.evaluate:
		if args.distort_w_test:
			acc_d = []
			vars = [0]  #0.05, 0.1, 0.15, 0.2, 0.25]
			for args.noise in vars:
				print('\n\nDistorting weights by {}%\n\n'.format(args.noise * 100))
				te_acc_dists = []
				orig_params = []

				if args.debug:
					print('\n\nbefore:\n{}\n'.format(model.module.conv1.weight.data.detach().cpu().numpy()[0, 0, 0]))

				for n, p in model.named_parameters():
					orig_params.append(p.clone())

				for s in range(args.num_sims):
					te_accuracies_dist = []
					with torch.no_grad():
						for n, p in model.named_parameters():
							if ('conv' in n or 'fc' in n or 'classifier' in n) and 'weight' in n:
								if s == 0 and args.noise == vars[0]:
									print(n)
								p_noise = torch.cuda.FloatTensor(p.size()).uniform_(1. - args.noise, 1. + args.noise)
								if args.debug and n == 'module.conv1.weight':
									print('\n\np_noise:\n{}\n'.format(p_noise.detach().cpu().numpy()[0, 0, 0]))
								p.data.mul_(p_noise)
							elif 'bn' in n:
								print('\n\n{}\n{}\n'.format(n, p))

					te_acc_d = validate(val_loader, model)
					te_accuracies_dist.append(te_acc_d.item())

					te_acc_dist = np.mean(te_accuracies_dist)
					te_acc_dists.append(te_acc_dist)

					if args.debug:
						print('after:\n{}\n'.format(model.module.conv1.weight.data.detach().cpu().numpy()[0, 0, 0]))

					for (n, p), orig_p in zip(model.named_parameters(), orig_params):
						p.data = orig_p.clone().data
					if args.debug:
						print('restored:\n{}\n'.format(model.module.conv1.weight.data.detach().cpu().numpy()[0, 0, 0]))

				avg_te_acc_dist = np.mean(te_acc_dists)
				acc_d.append(avg_te_acc_dist)
				print('\nNoise {:4.2f}: acc {:.2f}\n'.format(args.noise, avg_te_acc_dist))
			print('\n\n{}\n{}\n\n\n'.format(vars, acc_d))

		else:
			validate(val_loader, model)
		raise (SystemExit)


	#print('\nWorkers: {:d}\n'.format(workers))
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers, pin_memory=False)
	val_loader =   torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
	start = time.time()
	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch, args)
		model.train()
		tr_accs = []
		data_times = []
		compute_times = []
		other_times = []
		backprop_times = []
		distort_times = []
		total_times = []
		losses = []

		#end = time.time()
		for i, (images, target) in enumerate(train_loader):
			target = target.cuda(non_blocking=True)
			#end_data = time.time()
			#data_time = end_data - end
			output = model(images)
			#end_compute = time.time()
			#compute_time = end_compute - end_data
			loss = criterion(output, target)
			losses.append(loss.item())
			acc = accuracy(output, target)
			tr_accs.append(acc)
			#end_other = time.time()
			#other_time = end_other - end_compute
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#end_backprop = time.time()
			#backprop_time = end_backprop - end_other
			#total_time = data_time + compute_time + other_time + backprop_time
			'''
			if args.distort_w_train:
				for n, p in model.named_parameters():
					if 'conv' in n or 'fc' in n and 'weight' in n:
						p.data.mul_(torch.cuda.FloatTensor(p.size()).uniform_(1. - args.noise, 1. + args.noise))
				end_distort = time.time()
				distort_time = end_distort - end_backprop
				distort_times.append(distort_time)
				total_time += distort_time

			data_times.append(data_time)
			compute_times.append(compute_time)
			other_times.append(other_time)
			backprop_times.append(backprop_time)
			total_times.append(total_time)
			'''
			if i % args.print_freq == 0:
				'''
				data_time = np.mean(data_times)
				compute_time = np.mean(compute_times)
				other_time = np.mean(other_times)
				backprop_time = np.mean(backprop_times)
				#distort_time = np.mean(distort_times)
				total_time = np.mean(total_times)
				'''
				loss = np.mean(losses)
				#print('{}  Epoch {:>2d} Batch {:>4d}/{:d} | Time {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f} | Loss {:.2f} | Acc {:.2f}/{:.2f}'.format(
					#str(datetime.now())[:-7], epoch, i, len(train_loader), data_time, compute_time, other_time, backprop_time, total_time, loss, acc, np.mean(tr_accs)))
				print('{}  Epoch {:>2d} Batch {:>4d}/{:d} | Loss {:.2f} | Acc {:.2f}/{:.2f}'.format(str(datetime.now())[:-7], epoch, i, len(train_loader), loss, acc, np.mean(tr_accs)))
				'''
				data_times = []
				compute_times = []
				other_times = []
				backprop_times = []
				distort_times = []
				total_times = []
				'''

			#end = time.time()
			#if i == 2000:
				#ctime = end - start
				#break

		acc = validate(val_loader, model, epoch=epoch)
		#vtime = time.time() - end
		if acc > best_acc:
			best_acc = acc
			if args.distort_w_train:
				tag = args.tag + 'noise_{:.2f}_'.format(args.noise)
			else:
				tag = args.tag
			torch.save({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(), 'best_acc': best_acc, 'optimizer': optimizer.state_dict()}, tag+'checkpoint.pth.tar')


	#ttime = time.time() - start
	print('\n\nBest Accuracy {:.2f}\n\n'.format(best_acc))
	'''
	print('Workers {:d}  Total time: {:.2f}\n'.format(workers, ttime/60.))
	ctimes.append(ctime)
	vtimes.append(vtime)
	ttimes.append(ttime)

for w, ct, vt, tt in zip(workers_list, ctimes, vtimes, ttimes):
	print('workers {:>2d} time {:5.1f} ({:5.1f} {:5.1f})'.format(w, tt/60., ct/60., vt/60.))
	'''
