# Testing Neural Programmer Interpreters (NPIs): {e_t, i_t, a_t} --> {i_t+1, a_t+1, r_t}
# Objective: Distinguish all non-zero elements in a matrix and mark(reverse) them according to their original value

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
import argparse


# Test with a specified matrix
def Single_Test(feature_map, showcommands):
	x = 0 # x coord of matrix
	y = 0 # y coord of matrix
	seqlen = 0 # Current Timestep

	# Initialize the program ID and arguments
	i = 0
	a = 0

	# Initialize the state ({e_t, i_t, a_t}) and result matrix
	state = np.zeros([1, 199, 3])
	result = np.zeros([10,10])

	# Reverse the matrix until the last element
	while(1):

		# Get the input state {e_t, i_t, a_t} at current time step
		e = feature_map[x, y]
		state[0, seqlen, :] = np.array([e, i, a])
		seqlen += 1

		# Calculte the output {i_t+1, a_t+1, r_t}
		a_result, i_result, r_result = sess.run([prediction_a, prediction_i, prediction_r], feed_dict = {X:state, seq_len:[seqlen]})
		a = np.argmax(a_result[0, :])
		i = np.argmax(i_result[0, :])
		r = np.argmax(r_result[0, :])
		if showcommands:
			print('Step {0} --> a: {1}, i: {2}, r : {3}'.format(seqlen, a, i, r))

		# Execute commands according to the output program ID
		if seqlen == 198: # Jump out of the Program
			if showcommands:
				print('Command: Terminate the Program')
			break

		if i == 0: # Move to the next element
			if showcommands:
				print('Command: Move to next')
			x += 1
			if x == 10:
				x = 0
				y += 1

		elif i == 1: # Judge the current number and reverse it
			if showcommands:
				print('Command: judge as {0}'.format(a))
			result[x, y] = -a

	# Show reversed results and accuracy
	print('Original Matrix: \n {0}'.format(feature_map))
	print('Reversed Results: \n {0}'.format(result))
	wrong_num = len(np.nonzero(feature_map)[0]) - len(np.nonzero(feature_map + result)[0])
	total_num = len(np.nonzero(feature_map)[0])
	accuracy = wrong_num/total_num
	print('Accuracy: {0}'.format(accuracy))
	print('Test Finished!')

	return wrong_num, total_num


# Training Parameters
learning_rate = 0.01
training_steps = 10000
batch_size = 1
display_step = 1
train_dir = './checkpoints2/'

# Network Parameters
num_input = 3 # {e_t, i_t, a_t} with size 3
timesteps = 199 # timesteps/length of execution traces
num_hidden = 1280 # hidden layer num of features
num_classes_i = 2 # 0 (move to next) or 1 (judge the number)
num_classes_r = 2 # 0 (continue) or 1 (terminate)
num_classes_a = 3 # 0, 1, 2 (numbers appear in the matrix)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input]) # [batch_size, 199, 3]
Y_a = tf.placeholder("float", [None, num_classes_a]) # [batch_size, 3]
Y_i = tf.placeholder("float", [None, num_classes_i]) # [batch_size, 2]
Y_r = tf.placeholder("float", [None, num_classes_r]) # [batch_size, 2]
seq_len = tf.placeholder(dtype=tf.int32, shape=[None]) # [batch_size]

# Define weights
weights_a = {
	'out': tf.Variable(tf.random_normal([num_hidden, num_classes_a])) # [1280, 3]
}
biases_a = {
	'out': tf.Variable(tf.random_normal([num_classes_a])) # [3]
}
weights_i = {
	'out': tf.Variable(tf.random_normal([num_hidden, num_classes_i])) # [1280, 2]
}
biases_i = {
	'out': tf.Variable(tf.random_normal([num_classes_i])) # [2]
}
weights_r = {
	'out': tf.Variable(tf.random_normal([num_hidden, num_classes_r])) # [1280, 2]
}
biases_r = {
	'out': tf.Variable(tf.random_normal([num_classes_r])) # [2]
}

# NPI Model: x: {e_t, i_t, a_t} --> {i_t+1, a_t+1, r_t}
def NPI(x, weights_a, biases_a, weights_i, biases_i, weights_r, biases_r, seq_len):

	# Define a lstm cell
	lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias = 1.0)

	# Get dynamic lstm output for each time step --> h_t
	outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype = tf.float32, sequence_length = seq_len)
	output_result = outputs[:, seq_len[0] - 1, :] # [batch_size, 1280]

	# Get {i_t+1, a_t+1, r_t}
	a = tf.matmul(output_result, weights_a['out']) + biases_a['out'] # [batch_size, 3]
	i = tf.matmul(output_result, weights_i['out']) + biases_i['out'] # [batch_size, 2]
	r = tf.matmul(output_result, weights_r['out']) + biases_r['out'] # [batch_size, 2]

	return a, i, r

# Define Graph
prediction_a, prediction_i, prediction_r = NPI(X, weights_a, biases_a, weights_i, biases_i, weights_r, biases_r, seq_len)

# Define Loss and Optimizer
with tf.name_scope("Loss_a") as scope:
	loss_a = tf.nn.softmax_cross_entropy_with_logits(logits = prediction_a, labels = Y_a)
	tf.summary.scalar("Loss_a", tf.squeeze(loss_a))

with tf.name_scope("Loss_i") as scope:
	loss_i = tf.nn.softmax_cross_entropy_with_logits(logits = prediction_i, labels = Y_i)
	tf.summary.scalar("Loss_i", tf.squeeze(loss_i))

with tf.name_scope("Loss_r") as scope:
	loss_r = tf.nn.softmax_cross_entropy_with_logits(logits = prediction_r, labels = Y_r)
	tf.summary.scalar("Loss_r", tf.squeeze(loss_r))

with tf.name_scope("Total_loss") as scope:
	loss_op = tf.reduce_mean(loss_a + loss_i + loss_r)
	tf.summary.scalar("Total_loss", loss_op)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

# Model Evaluation
correct_pred_a = tf.equal(tf.argmax(prediction_a, 1), tf.argmax(Y_a, 1))
accuracy_a = tf.reduce_mean(tf.cast(correct_pred_a, tf.float32))
correct_pred_i = tf.equal(tf.argmax(prediction_i, 1), tf.argmax(Y_i, 1))
accuracy_i = tf.reduce_mean(tf.cast(correct_pred_i, tf.float32))
correct_pred_r = tf.equal(tf.argmax(prediction_r, 1), tf.argmax(Y_r, 1))
accuracy_r = tf.reduce_mean(tf.cast(correct_pred_r, tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

# Session config --> Using GPU0
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.gpu_options.visible_device_list = '0'

with tf.Session(config = sess_config) as sess:

	# Run the initializer
	sess.run(init)

	# Restore the Latest Trained Model
	train_vars = tf.all_variables()
	saver = tf.train.Saver(train_vars, max_to_keep = 50000)
	ckpt = tf.train.get_checkpoint_state(train_dir)
	print('model_path', ckpt.model_checkpoint_path)
	saver.restore(sess, ckpt.model_checkpoint_path)

	# Specify Matrix for testing
	feature_map = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	                        [1, 1, 2, 2, 2, 0, 2, 2, 2, 2],
	                        [0, 0, 1, 0, 0, 0, 0, 2, 0, 2],
	                        [0, 0, 2, 0, 0, 0, 0, 2, 0, 2],
	                        [2, 2, 2, 1, 2, 0, 2, 2, 2, 2],
	                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 2],
	                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 2],
	                        [0, 0, 0, 0, 0, 0, 0, 2, 1, 2],
	                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
	
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--TestType", type = str, default = 'single', help = "Single Test (single) or Multiple Test (multiple)")
	parser.add_argument("--ShowCommands", action='store_true', default = False, help = "Show generated commands or not")
	args = parser.parse_args()
	showcommands = args.ShowCommands

	# Test with the specified matrix and show results and accruacy
	if args.TestType == "single":
		Single_Test(feature_map, showcommands)
	# Test with Multiple randomly generated matrix and show results and overall accruacy
	elif args.TestType == "multiple":
		wrong_num_all = 0
		total_num_all = 0
		for i in range(5):
			feature_map = np.random.randint(3, size = (10, 10))
			wrong_num, total_num = Single_Test(feature_map, showcommands)
			wrong_num_all += wrong_num
			total_num_all += total_num
			accuracy = wrong_num_all/total_num_all
		print('Overall Accuracy: {0}'.format(accuracy))



