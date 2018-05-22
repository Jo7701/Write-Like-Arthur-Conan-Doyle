import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import random
import os
import time
import sys

def process():
	text = open('2books.txt').read()
	chars = sorted(list(set(text)))
	char2int = dict((c, i) for i,c in enumerate(chars))
	int2char = dict((i, c) for i,c in enumerate(chars))
	seq_len = 108

	data = []	
	for i in range(len(text) - seq_len):
		string = text[i : i + seq_len]
		output_string = text[i+seq_len]

		output = np.zeros(len(chars))
		output[char2int[output_string]] = 1
		data.append([[char2int[i] for i in string], output])
	#random.shuffle(data)

	x = np.asarray([np.array(i[0]) for i in data])
	y = np.asarray([np.array(i[1]) for i in data])

	return x,y, char2int, int2char

features, labels, c2i, i2c = process()

chunk_size = 1
n_chunks = 108
batch_size = 110
rnn_size = 512
n_classes = len(c2i)

x = tf.placeholder('float', [None, n_chunks, chunk_size], name = 'x')
y = tf.placeholder('float', name = 'y')

def recurrent_nerual_network(x):
	layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, (-1, chunk_size))
	x = tf.split(x, n_chunks)

	deep_lstm = tf.nn.rnn_cell.DropoutWrapper(rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(rnn_size) for _ in range(3)]), .9, .6)

	outputs, states = rnn.static_rnn(deep_lstm, x, dtype = tf.float32)

	output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'], name = 'output')
	return output

def train(x):
	prediction = recurrent_nerual_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	n_epochs = 100

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		for epoch in range(n_epochs):
			start_time = time.time()
			loss = 0
			i = 0
			while(i < len(features)):
				start = i
				end = i + batch_size
				epoch_x = features[start:end]
				epoch_y = labels[start:end]
				epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				loss += c
				i+=batch_size

			print 'Epoch ', epoch + 1, ' Loss: ', loss
			print 'Epoch ', epoch + 1, ' took ', time.time() - start_time, ' seconds.'
			if epoch != 0 and (epoch+1) % 10 == 0:
				os.makedirs('Epoch'+str(epoch+1))
				saver.save(sess, 'Epoch'+str(epoch+1)+'/model'+str(int(loss))) 

train(x)
