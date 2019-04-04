# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 20:02:18 2015

@author: Michael Klachko
"""

import numpy as np
import pickle
import os
import time
from pylab import imshow, show, cm

def view_image(image, image_size, colors):
	if colors == 1:
		imshow(np.reshape(image, (image_size,image_size)), cmap=cm.gray)
	else:
		imshow(np.reshape(image, (image_size,image_size,colors), order='F'))
		#also can do this: im = c.reshape(3,32,32).transpose(1,2,0)
		# http://stackoverflow.com/questions/28005669/how-to-view-an-rgb-image-with-pylab
	show()

def load_dataset(path):
	print("loading dataset...")
	f = open(path, 'rb')
	dataset = pickle.load(f)
	#dataset = np.load(f)
	f.close()
	return dataset


def write_to_disk(dataset, filename):
	print("writing dataset to disk...")
	f = open(filename, 'wb')
	pickle.dump(dataset, f, -1)
	print("saved to ", filename)
	f.close()


def combine_batches(input_path):
	print("opening batches...")
	f1 = open(input_path + 'data_batch_1', 'rb')
	f2 = open(input_path + 'data_batch_2', 'rb')
	f3 = open(input_path + 'data_batch_3', 'rb')
	f4 = open(input_path + 'data_batch_4', 'rb')
	f5 = open(input_path + 'data_batch_5', 'rb')
	t1 = open(input_path + 'test_batch', 'rb')
	print("unpickling batches...")

	d1 = pickle.load(f1)
	d2 = pickle.load(f2)
	d3 = pickle.load(f3)
	d4 = pickle.load(f4)
	d5 = pickle.load(f5)
	t = pickle.load(t1)

	print("transposing each batch to MNIST (row major?) order...")

	d1["data"] = np.transpose(d1["data"])
	d2["data"] = np.transpose(d2["data"])
	d3["data"] = np.transpose(d3["data"])
	d4["data"] = np.transpose(d4["data"])
	d5["data"] = np.transpose(d5["data"])
	t["data"] = np.transpose(t["data"])

	d = []  #all training data (images)
	l = []  #all train labels

	#t["data"] contains all test data, t["labels"] contains all test labels

	print("combining batches into single file...")
	d.extend(d1["data"])
	d.extend(d2["data"])
	d.extend(d3["data"])
	d.extend(d4["data"])
	d.extend(d5["data"])

	l.extend(d1["labels"])
	l.extend(d2["labels"])
	l.extend(d3["labels"])
	l.extend(d4["labels"])
	l.extend(d5["labels"])

	print("normalizing and converting to float32...")

	train_data = np.asarray(d, dtype=np.float32) / 255.0
	test_data = np.asarray(t["data"], dtype=np.float32) / 255.0

	print("dataset contains {:d} training images, and {:d} test images".format(len(train_data), len(train_data)))
	print("\ntrain_data:", np.shape(train_data))
	print("train_labels:", np.shape(l))
	print("test_data:", np.shape(test_data))
	print("test_labels:", np.shape(t["labels"]), '\n\n')

	return ((train_data, l), ((), ()), (test_data, t["labels"]))  #tuple: (training data+labels, validation data/labels (empty), testing data/labels)


def load_batch(filename):
	""" load single batch of cifar """
	with open(filename, 'rb') as f:
		datadict = pickle.load(f)
		X = datadict['data']
		Y = datadict['labels']
		X = np.array(X / 256.0)
		Y = np.array(Y)
		return X, Y


def load_cifar(input_path, img_size=32, colors=3):
	""" load all of cifar """
	print('\nLoading CIFAR-10 dataset...\n')

	Xtr = []
	Ytr = []
	for b in range(1, 6):
		f = os.path.join(input_path, 'data_batch_%d' % b)
		X, Y = load_batch(f)
		Xtr.extend(X)
		Ytr.extend(Y)
		del X, Y

	Xte, Yte = load_batch(os.path.join(input_path, 'test_batch'))
	#convert training images into column vectors:
	training_inputs = [np.reshape(x, (colors, img_size, img_size)) for x in Xtr]
	training_results = [onehot(y) for y in Ytr]
	training_data = list(zip(training_inputs, training_results))

	test_inputs = [np.reshape(x, (colors, img_size, img_size)) for x in Xte]
	test_data = list(zip(test_inputs, Yte))

	return (training_data, test_data)


def onehot(label):
	"""converts scalar labels into one hot encoded 10-vectors
	(9 zeroes and a single 1 in the proper position"""
	vector = np.zeros((10, 1))
	vector[label] = 1.0
	return vector

def quantize_old(value, half_bin, levels, min_=0.0, max_=1.0):
	for level in levels:
		if abs(level - value) <= half_bin:
			return level
	if value < min_: return levels[0]
	if value > max_: return levels[-1]


#quantize_vec_old = np.vectorize(quantize, excluded=[1,2,3,4])

def quantize_images_old(input_path, output_path, low_prec=1, high_prec=8, min_=0.0, max_=1.0):
	#min_, max_ = 0.0, 1.0      #range of quantized weights, based on final weights distr.
	range_ = max_ - min_

	quantize_vec_old = np.vectorize(quantize_old, excluded=[1, 2, 3, 4])

	dataset = load_dataset(input_path)

	tr_set = dataset[0][0]
	test_set = dataset[2][0]

	for num_bits in range(low_prec, high_prec + 1):
		tr_set_q = []
		test_set_q = []
		levels = []

		print("\nPrecision =", num_bits, "bits\n")

		num_bins = 2 ** num_bits
		bin_size = range_ / num_bins
		half_bin = bin_size / 2
		for i in range(num_bins):
			levels.append(min_ + (i + 0.5) * bin_size)
		print("Quantization thresholds:\n", levels)


		print("\nStarting quantization of training images...")

		for i, image in enumerate(tr_set):
			if i % 100 == 0:
				print('.', end=' ')

			image_q = quantize_vec_old(image, half_bin, levels)
			tr_set_q.append(image_q)
		tr_set_q = np.array(tr_set_q, np.float32, copy=False)

		print("\nStarting quantization of test images...\n")

		for i, image in enumerate(test_set):
			if i % 100 == 0:
				print('.', end=' ')
			image_q = quantize_vec_old(image, half_bin, levels)
			test_set_q.append(image_q)

		test_set_q = np.array(test_set_q, np.float32, copy=False)

		print("\nbefore:", tr_set[0][:20])
		print("\nafter:", tr_set_q[0][:20])

		view_image(tr_set_q[0], 32, 3)

		dataset_q = ((tr_set_q, dataset[0][1]), ((), ()), (test_set_q, dataset[2][1]))

		write_to_disk(dataset_q, output_path + str(num_bits) + 'bit')


def quantize(value, bits, offset):
	#value = int(np.round(value*256))
	value = int(value * 256)
	return (((value >> bits) << bits) + offset) / 256.0


quantize_vec = np.vectorize(quantize, excluded=[1, 2])


def quantize_images(dataset, output_path, bits):
	offset = 2 ** (bits - 1)

	tr_set = dataset[0][0]
	test_set = dataset[2][0]

	print("\nbefore:", tr_set[0][:20])

	tr_set = quantize_vec(tr_set, bits, offset)
	tr_set = np.array(tr_set, np.float32, copy=False)

	test_set = quantize_vec(test_set, bits, offset)
	test_set = np.array(test_set, np.float32, copy=False)

	print("\nafter:", tr_set[0][:20])

	view_image(tr_set[0], 64, 3)

	dataset_q = ((tr_set, dataset[0][1]), ((), ()), (test_set, dataset[2][1]))

	write_to_disk(dataset_q, output_path + str(8 - bits) + 'bit')



cifar_path = ""  #batches folder
output_path = ""
filename = output_path+'cifar.pkl'

cifar_10 = combine_batches(cifar_path)
write_to_disk(cifar_10, filename)


for bits in range(8):
	print("\nPrecision =", 8 - bits, "bits")

	start_time = time.time()
	quantize_images(cifar_10, filename[:-4], bits)
	print("\n--- Program ran for {:.1f} minutes ---\n".format((time.time() - start_time) / 60.0))





