import tensorflow as tf
import numpy as np
import random
from tensorflow.python.ops import rnn, rnn_cell
import os

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

lol = tf.Variable('float')

with tf.Session() as sess:
	saver = tf.train.import_meta_graph('model5770.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./'))

	graph = tf.get_default_graph()
	x = graph.get_tensor_by_name('x:0')
	y = graph.get_tensor_by_name('y:0')
	output = graph.get_tensor_by_name('output:0')

	start_index = random.randrange(0, len(features))
	vec = features[start_index]

	for i in range(300):
		temp = vec[i:i+108].reshape((1, 108, 1))
		prediction = tf.argmax(sess.run(output, feed_dict = {x: temp}), 1)
		print prediction
		vec = np.append(vec, sess.run(prediction))

	output_string = ''
	for num in vec:
		output_string += i2c[num]

	print output_string
