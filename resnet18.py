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

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
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
parser.add_argument('--step-after', default=30, type=int, help='reduce LR after this number of epochs')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--num_sims', default=1, type=int, help='number of simulations.')

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--pretrained', dest='pretrained', action='store_true')
feature_parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
parser.set_defaults(blocked=False)

warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)

class BasicBlock(nn.Module):
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
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
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


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
		x = self.fc(x)

		return x


def resnet18(pretrained=False, **kwargs):
	model = ResNet(BasicBlock, **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
	return model


def validate(val_loader, model, epoch=0):
	model.eval()
	te_accs = []
	with torch.no_grad():
		start = time.time()
		for i, (images, target) in enumerate(val_loader):
			target = target.cuda(non_blocking=True)
			output = model(images)
			acc = accuracy(output, target)
			te_accs.append(acc)
		mean_acc = np.mean(te_accs)
		end = time.time()
		print('\n{}\tEpoch {:d}  Validation Accuracy: {:.2f}  time: {:.2f}s\n'.format(str(datetime.now())[:-7], epoch, mean_acc, end-start))
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


def main():
	args = parser.parse_args()

	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. this will slow down training')

	if args.pretrained:
		print("\n\n\tLoading pre-trained {}\n\n".format(args.arch))
	else:
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

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers, pin_memory=True)
	val_loader =   torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

	if args.evaluate:
		if args.distort_w_test:
			acc_d = []
			vars = [0]#0.05, 0.1, 0.15, 0.2, 0.25]
			for args.noise in vars:
				print('\n\nDistorting weights by {}%\n\n'.format(args.noise * 100))
				te_acc_dists = []
				orig_params = []

				if args.debug:
					print('\n\nbefore:\n{}\n'.format(model.module.conv1.weight.data.detach().cpu().numpy()[0,0,0]))

				for n, p in model.named_parameters():
					orig_params.append(p.clone())

				for s in range(args.num_sims):
					te_accuracies_dist = []
					with torch.no_grad():
						for n, p in model.named_parameters():
							if ('conv' in n or 'fc' in n or 'classifier' in n) and 'weight' in n:
								if s == 0 and args.noise == vars[0]:
									print(n)
								p_noise = torch.cuda.FloatTensor(p.size()).uniform_(1.-args.noise, 1.+args.noise)
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
		return

	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch, args)
		model.train()
		tr_accs = []
		end = time.time()
		for i, (images, target) in enumerate(train_loader):
			data_time = time.time() - end
			target = target.cuda(non_blocking=True)
			end_data = time.time()
			output = model(images)
			end_compute = time.time()
			compute_time = end_compute - end_data
			loss = criterion(output, target)
			acc = accuracy(output, target)
			tr_accs.append(acc)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if args.distort_w_train:
				for n, p in model.named_parameters():
					if 'conv' in n or 'fc' in n and 'weight' in n:
						p.data.mul_(torch.cuda.FloatTensor(p.size()).uniform_(1. - args.noise, 1. + args.noise))

			end = time.time()
			rest = end - end_compute
			total_time = data_time + compute_time + rest

			if i % args.print_freq == 0:
				print(
				'{}\tEpoch {:d} batch {:d}/{:d} | Time: data {:.3f} compute {:.3f} total {:.3f} | train loss {:.3f} | train acc: this batch {:.2f} mean {:.2f}'.format(
					str(datetime.now())[:-7], epoch, i, len(train_loader), data_time, compute_time, total_time, loss.item(), acc, np.mean(tr_accs)))

		acc = validate(val_loader, model, epoch=epoch)

		if acc > best_acc:
			best_acc = acc
			if args.distort_w_train:
				tag = args.tag + 'noise_{:.2f}_'.format(args.noise)
			else:
				tag = args.tag
			torch.save({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(), 'best_acc': best_acc, 'optimizer': optimizer.state_dict()}, tag+'checkpoint.pth.tar')
