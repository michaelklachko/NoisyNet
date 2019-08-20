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


def place_fig(arrays, rows=1, columns=1, r=0, c=0, bins=100, range_=None, title=None, name=None, max_input=0, max_weight=0, pctl=99.9, labels=['1'], log=True):
	ax = plt.subplot2grid((rows, columns), (r, c))
	min_value = max_value = 0
	if range_ is None:
		for a in arrays:
			min_value = min(min_value, np.min(a))
			max_value = max(max_value, np.max(a))
		range_ = [min_value, max_value]
	range_ = None

	if len(arrays) == 1:
		histtype = 'bar'
		alpha = 1
	else:
		histtype = 'step'
		alpha = 1    #2.0 / len(arrays)

	thr_neg = 0
	thr_pos = 0

	for array, label, color in zip(arrays, labels, ['blue', 'red', 'green', 'black', 'magenta', 'cyan', 'orange', 'yellow', 'gray']):
		#thr = max(abs(thr_neg), thr_pos)
		if name != 'input' and name != 'weights':
			if "weight" in name:
				array = array / max_weight
			else:
				array = array / (max_weight * max_input)
			thr_neg = np.percentile(array, 100 - pctl)
			thr_pos = np.percentile(array, pctl)

		if r == 0:  #only display accuracy on first row figures
			label = label + ' {:.1f}%  {:.2f} {:.2f}'.format(pctl, thr_neg, thr_pos)
		else:
			label = '{:.1f}%  {:.2f} {:.2f}'.format(pctl, thr_neg, thr_pos)


		ax.hist(array.ravel(), alpha=alpha, bins=bins, density=False, color=color, range=range_, histtype=histtype, label=label, linewidth=1.5)

	ax.set_title(title+name, fontsize=18)
	#plt.xlabel('Value', fontsize=16)
	#plt.ylabel('Frequency', fontsize=16)
	#plt.legend(loc='upper right')
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	if log:
		plt.semilogy()
	ax.legend(loc='upper right', prop={'size': 15})


def plot_grid(layers, names, path=None, filename='', pctl=99.9, labels=['1']):
	figsize = (len(names) * 7, len(layers) * 6)
	#figsize = (len(names) * 7, 2 * 6)
	plt.figure(figsize=figsize)
	rows = len(layers)
	columns = len(layers[0])
	thr = 0
	for r, layer in zip(range(rows), layers):
		for c, name in zip(range(columns), names):
			array = layer[c]
			if name == 'input':
				max_input = np.max(array[0])
				array[0] = array[0] / max_input
			if name == 'weights':
				thr_neg = np.percentile(array[0], 100 - pctl)
				thr_pos = np.percentile(array[0], pctl)
				thr = max(abs(thr_neg), thr_pos)
				#print('\nthr:', thr)
				array[0][array[0] > thr] = thr
				array[0][array[0] < -thr] = -thr
				#print(name, 'np.max(array)', np.max(array[0]))
				#print('before\n', array[0].ravel()[20:40])
				array[0] = array[0] / thr
				#print('after\n', array[0].ravel()[20:40])
			place_fig(array, rows=rows, columns=columns, r=r, c=c, title='layer' + str(r + 1) + ' ', name=name, pctl=pctl, max_input=max_input, max_weight=thr, labels=labels)
	plt.savefig(path + filename, dpi=200, bbox_inches='tight')
	plt.close()


def plot_layers(num_layers=4, models=None, epoch=0, i=0, layers=None, names=None, var='', vars=[0.0], pctl=99.9, acc=0.0, tag=''):
	accs = [acc]

	if len(models) > 1:
		layers = []
		accs = []
		for l in range(num_layers):  #add placeholder for each layer
			layers.append([])
			for n in range(len(names)):
				layers[l].append([])

		for model in models:
			print('\n\nLoading model {}\n\n'.format(model))
			flist = os.listdir(model)  #extract best accuracy from model name
			for fname in flist:
				if 'model' in fname:
					acc = float(fname.split('_')[-1][:-4])
			accs.append(acc)

			model_layers = np.load(model + 'layers.npy', allow_pickle=True)  #construct layers (placeholders in layers will contain multiple arrays, one per model)
			for l in range(num_layers):
				for col in range(len(model_layers[l])):
					layers[l][col].append(model_layers[l][col][0])

				if 'noise' in names:  #add noise/signal ratio array to each layer, if noise present
					out = model_layers[l][2][0]  #TODO fix this fragility
					noise = model_layers[l][-1][0]  #assume vmm out to be 3rd array in layer and noise last array:
					full_range = np.max(out) - np.min(out)
					clipped_range = np.percentile(out, 99) - np.percentile(out, 1)
					if clipped_range == 0:
						clipped_range = max(np.max(out) / 100., 1)
					error = noise / clipped_range
					print('Layer {:d}  pre-act range: clipped (99th-1st pctl)/full {:>5.1f}/{:>5.1f}  error range {:.2f}-{:.2f}'.format(
						l, clipped_range, full_range, np.min(error), np.max(error)))
					layers[l][-1].append(error)

	labels = []
	print('\n')
	for k in range(len(accs)):
		labels.append(var + ' ' + str(vars[k]) + ' ({:.1f}%)'.format(accs[k]))
		print('epoch {:d} batch {:d} plotting var {}'.format(epoch, i, labels[-1]))

	if len(models) > 1:
		filename = 'comparison_of_{}'.format(var)
	else:
		filename = 'epoch_{:d}_iter_{:d}_{}.png'.format(epoch, i, tag)


	plot_grid(layers, names, path=models[0], filename=filename, labels=labels, pctl=pctl)
	print('\nplot is saved to {}\n'.format(filename))


if __name__ == "__main__":
	#for comparative figures, first save all values as numpy arrays using --plot arg in noisynet.py

	model1 = 'results/a_atest_current-5.0-5.0-5.0-5.0_L3-0.0_L3_act-0.0_L2-0.0005-0.0005-0.0005-0.0005_actmax-2.0-2.0-2.0_w_max1-0.0-0.0-0.0-0.0_bn-True_LR-0.0006_grad_clip-0.0_2019-07-30_15-01-21/'
	model2 = 'results/a_btest_current-5.0-5.0-5.0-5.0_L3-0.0_L3_act-0.0_L2-0.0-0.0-0.0-0.0_actmax-2.0-2.0-2.0_w_max1-0.0-0.0-0.0-0.0_bn-True_LR-0.0006_grad_clip-0.0_2019-07-30_15-08-13/'
	model3 = 'results/current-1.0-1.0-1.0-1.0_L3-0.0_L3_act-0.0_L2-0.0-0.0-0.0-0.0_actmax-100.0-0.0-0.0_w_max1-0.0-0.0-0.0-0.0_bn-True_LR-0.005_grad_clip-0.0_2019-01-01_13-18-31/'

	models = [model1, model2]

	names = ['input', 'weights', 'vmm', 'vmm diff']
	if 1:#args.blocked:
		names.append('vmm blocked')
	if 1:#args.merge_bn:
		names.append('biases')
	names.append('pre-activation')
	if 1:#args.current1 > 0:
		names.extend(['sigmas', 'noise', 'noise/range'])
	print('\n\nPlotting histograms for {}\n'.format(names))
	#names = ['input', 'weights', 'out', 'pos/neg', 'after bn']
	#figsize = (len(names) * 8, 4 * 4.5)
	figsize = (len(names) * 7, 4 * 6)
	var = 'L2'
	vars = [0.0005, 0.0]

	tag = 'test_tag'
	plot_layers(num_layers=4, models=models, epoch=0, i=0, names=names, var=var, vars=vars, figsize=figsize, tag=tag)
