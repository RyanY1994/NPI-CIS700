# Training Neural Programmer Interpreters (NPIs): {e_t, i_t, a_t} --> {i_t+1, a_t+1, r_t}
# Objective: Distinguish all non-zero elements in a matrix and mark(reverse) them according to their original value

from __future__ import print_function
from trace import generate_trace, generate_label, generate_sample
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os


# Training Parameters
learning_rate = 0.01
training_steps = 10000
batch_size = 1
display_step = 1
epoch_num = 30
train_dir = './checkpoints/'
tensorboard_dir = './Tensorboard/'

# Create folders
if not os.path.exists(train_dir):
	os.makedirs(train_dir)
if not os.path.exists(tensorboard_dir):
	os.makedirs(tensorboard_dir)

# Network Parameters
num_input = 3 # {e_t, i_t, a_t} with size 3
timesteps = 199 # timesteps/length of execution traces
num_hidden = 1280 # hidden layer size
num_classes_i = 2 # 0 (move to next) or 1 (judge the number)
num_classes_r = 2 # 0 (continue) or 1 (terminate)
num_classes_a = 3 # 0, 1, 2 (numbers appear in the matrix)

# Model Inputs
X = tf.placeholder("float", [None, timesteps, num_input]) # [batch_size, 199, 3]
Y_a = tf.placeholder("float", [None, num_classes_a]) # [batch_size, 3]
Y_i = tf.placeholder("float", [None, num_classes_i]) # [batch_size, 2]
Y_r = tf.placeholder("float", [None, num_classes_r]) # [batch_size, 2]
seq_len = tf.placeholder(dtype = tf.int32, shape = [None])

# Define Weights
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
# optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
# optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate)
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
saver = tf.train.Saver(tf.global_variables())

# Session config --> Using GPU0
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.gpu_options.visible_device_list = '0'

# Start training
with tf.Session(config = sess_config) as sess:

	# Run the initializer
	sess.run(init)

	# Define Tensorboard
	merged_summary_op = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(tensorboard_dir, graph_def = sess.graph_def)

	# Matrix for training
	feature_map = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
							[0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
							[0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
							[0, 0, 1, 1, 0, 0, 0, 2, 2, 0],
							[0, 0, 1, 1, 0, 0, 0, 2, 2, 2],
							[0, 0, 1, 0, 0, 0, 0, 2, 0, 0],
							[0, 0, 1, 0, 0, 0, 0, 2, 2, 2],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

	# Execution traces: {e_t, i_t, a_t, r_t}
	trace_list = generate_trace(feature_map) # list with length 199, each elem is a dict of {'e':e, 'i':i, 'a':a, 'r':r}
	# Input of execution traces: {e_t, i_t, a_t} 
	trace_sample = generate_sample(trace_list) # np.array with shape [199, 3]
	# Output of execution traces: {i_t+1, a_t+1, r_t}
	trace_label = generate_label(trace_list) # list with length 199, each elem is a dict of {'i':i_emb, 'a':a_emb, 'r':r_emb} embedding

	# Training Process
	for epoch in range(epoch_num):
		for step in range(0, timesteps):
			# Current timestep for dynamic lstm
			seqlen = [step + 1]

			# Input: {e_t, i_t, a_t}
			batch_x = trace_sample # [199, 3]
			batch_x = np.expand_dims(batch_x, axis = 0) # [1, 199, 3]

			# Label: {i_t+1, a_t+1, r_t}
			batch_y_i = trace_label[step]['i'] # [2]
			batch_y_i = np.expand_dims(batch_y_i, axis=0) # [1, 2]
			batch_y_a = trace_label[step]['a'] # [3]
			batch_y_a = np.expand_dims(batch_y_a, axis=0) # [1, 3]
			batch_y_r = trace_label[step]['r'] # [2]
			batch_y_r = np.expand_dims(batch_y_r, axis=0) # [1, 2]

			# Optimization
			sess.run(train_op, feed_dict={X:batch_x, Y_i:batch_y_i, Y_a:batch_y_a, Y_r:batch_y_r, seq_len:seqlen})

			# Display Training Loss and Accuracy
			if step % display_step == 0 or step == 1:

				# Calculate batch loss and accuracy
				loss, acca, acci, accr = sess.run([loss_op, accuracy_a, accuracy_i, accuracy_r], feed_dict = {X:batch_x, Y_i:batch_y_i, Y_a:batch_y_a, Y_r:batch_y_r, seq_len:seqlen})
				print('Step ' + str(step) + ', Minibatch Loss = ' + '{:.4f}'.format(loss) + ', Training Accuracy = ' + 'a:', acca, 'i', acci, 'r', accr)

				# Add Losses to tensorboard
				train_summary_str = sess.run(merged_summary_op, feed_dict = {X:batch_x, Y_i:batch_y_i, Y_a:batch_y_a, Y_r:batch_y_r, seq_len:seqlen})
				summary_writer.add_summary(train_summary_str, epoch * 200 + step)

		# Save Model for each epoch
		print('Epoch {0} Optimization Finished'.format(epoch))
		checkpoint_path = os.path.join(train_dir, 'model.ckpt')
		saver.save(sess, checkpoint_path, global_step = epoch)
		print('Model saved to {0}'.format(train_dir))
