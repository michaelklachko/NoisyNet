import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

def plot(values1, values2=None, bins=120, range_=None, labels=['1', '2'], title='', log=False):
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
	#plt.xlabel('Value', fontsize=16)
	#plt.ylabel('Frequency', fontsize=16)
	plt.legend(loc='upper right')
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	if log:
		plt.semilogy()
	ax.legend(loc='upper right', prop={'size': 14})
	plt.show()


def place_fig(arrays, rows=1, columns=1, r=0, c=0, bins=100, range_=None, title=None, labels=['1'], log=True):
	ax = plt.subplot2grid((rows, columns), (r, c))
	min_value = max_value = 0
	if range_ is None:
		for a in arrays:
			min_value = min(min_value, np.min(a))
			max_value = max(max_value, np.max(a))
		range_ = [min_value, max_value]

	if len(arrays) == 1:
		histtype = 'bar'
		alpha = 1
	else:
		histtype = 'step'
		alpha = 1    #2.0 / len(arrays)

	for array, label, color in zip(arrays, labels, ['blue', 'red', 'green', 'black', 'magenta', 'cyan', 'orange', 'yellow', 'gray']):
		ax.hist(array.ravel(), alpha=alpha, bins=bins, density=False, color=color, range=range_, histtype=histtype, label=label, linewidth=1.5)

	ax.set_title(title, fontsize=13)
	#plt.xlabel('Value', fontsize=16)
	#plt.ylabel('Frequency', fontsize=16)
	plt.legend(loc='upper right')
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	if log:
		plt.semilogy()
	ax.legend(loc='upper right', prop={'size': 12})
	#plt.show()


def plot_grid(arrays, names, path=None, filename='', figsize=(30, 16), labels=['1']):
	#print('plot_grid')
	#arrays has 6 arrays of diff type, with num_layers of tuples in each array, each tuple holding num_models arrays.
	rows = len(arrays[0])
	columns = len(arrays)
	plt.figure(figsize=figsize)
	for c, name in zip(range(columns), names):  #6 types, arrays[0] is inputs, ...
		for array, r in zip(arrays[c], range(rows)):  #layers for this type
			place_fig(array, rows=rows, columns=columns, r=r, c=c, title='layer' + str(r + 1) + ' ' + name, labels=labels)
	plt.savefig(path + filename, dpi=300, bbox_inches='tight')
	plt.close()


def plot_layers(layers=4, models=[], epoch=0, i=0, tensors=None, var='var', vars=[0.0], figsize=(42, 24), acc=0.0, tag=''):

	types = ['input', 'weights', 'out', 'sigmas', 'noise']
	#names = ['input', 'weights', 'out', 'sigmas', 'noise', 'noise/signal range']
	names = ['input', 'weights', 'out', 'pos/neg', 'after bn']

	inputs = []
	outs = []
	weights = []
	noises = []
	sigmas = []
	ems = []

	arrays = [inputs, weights, outs, sigmas, noises]

	if tensors is None:   #extract best accuracy for each model
		accs = []
		epochs = []
		for model in models:
			flist = os.listdir(model)
			for fname in flist:
				if 'model' in fname:
					epoch = int(fname.split('_')[2])  #model_epoch_xxx_acc_yy.yy.pth
					acc = float(fname.split('_')[-1][:-4])
			accs.append(acc)
			epochs.append(epoch)
	else:
		accs = [acc]

	labels = []
	for k in range(len(accs)):
		labels.append(var + ' ' + str(vars[k]) + ' ({:.1f}%)'.format(accs[k]))
		print('epoch {:d} batch {:d} plotting var {}'.format(epoch, i, labels[-1]))

	for l in range(layers):    #for each layer, we either go through tensors (if given, from single model), or through models
		if tensors is None:
			for atype, array in zip(types, arrays):
				temp = []
				for model, epoch in zip(models, epochs):
					name = model + 'layer' + str(l + 1) + '_' + atype + '_epoch_' + str(epoch) + '_iter_0.npy'
					temp.append(np.load(name))  #noises = [[model1_noise1, model2_noise1], [model1_noise2, model2_noise2], ...]
				array.append(temp)
			epoch = 0   #for filename

			temp = []
			for out, noise in zip(outs[l], noises[l]):
				full_range = np.max(out) - np.min(out)
				clipped_range = np.percentile(out, 99) - np.percentile(out, 1)
				if clipped_range == 0:
					clipped_range = max(np.max(out) / 100., 1)
				em = noise / clipped_range
				temp.append(em)
			ems.append(temp)

		else:
			for array, t in zip(arrays, tensors[l]):
				array.append([t])    #noises = [[noise1], [noise2], ...]

			full_range = np.max(outs[l][0]) - np.min(outs[l][0])
			clipped_range = np.percentile(outs[l][0], 99) - np.percentile(outs[l][0], 1)
			if clipped_range == 0:
				clipped_range = max(np.max(outs[l][0]) / 100., 1)
			em = noises[l][0] / clipped_range
			ems.append([em])

			print('Layer {:d}  pre-act range: clipped (99th-1st pctl)/full {:>5.1f}/{:>5.1f}  error range {:.2f}-{:.2f}'.format(
				l, clipped_range, full_range, np.min(em), np.max(em)))

	if len(models) > 1:
		filename = 'comparison_of_act_max1_final{}'.format(var)
	else:
		filename = 'epoch_{:d}_iter_{:d}_{}.png'.format(epoch, i, tag)


	#plot_grid([inputs, weights, outs, sigmas, noises, ems], names, path=models[0], filename=filename, figsize=figsize, labels=labels)
	plot_grid([inputs, weights, outs, sigmas, noises], names, path=models[0], filename=filename, figsize=figsize, labels=labels)
	print('plot is saved to {}\n'.format(filename))


if __name__ == "__main__":
	#for comparative figures, first save all values as numpy arrays using --plot arg in noisynet.py

	model1 = 'results/current-1.0-1.0-1.0-1.0_L3-0.0_L3_act-0.0_L2-0.0-0.0-0.0-0.0_actmax-0.5-0.0-0.0_w_max1-0.0-0.0-0.0-0.0_bn-True_LR-0.005_grad_clip-0.0_2019-01-01_13-16-33/'
	model2 = 'results/current-1.0-1.0-1.0-1.0_L3-0.0_L3_act-0.0_L2-0.0-0.0-0.0-0.0_actmax-5.0-0.0-0.0_w_max1-0.0-0.0-0.0-0.0_bn-True_LR-0.005_grad_clip-0.0_2019-01-01_13-17-24/'
	model3 = 'results/current-1.0-1.0-1.0-1.0_L3-0.0_L3_act-0.0_L2-0.0-0.0-0.0-0.0_actmax-100.0-0.0-0.0_w_max1-0.0-0.0-0.0-0.0_bn-True_LR-0.005_grad_clip-0.0_2019-01-01_13-18-31/'

	models = [model1, model2, model3]

	tag = ''
	plot_layers(layers=4, models=models, epoch=0, i=0, tensors=None, var='', vars=[0.5, 5.0, 100], figsize=(46, 18), acc=0.0, tag=tag)
