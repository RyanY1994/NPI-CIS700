# Generate execution traces {e_t, i_t, a_t, r_t}: (Input {e_t, i_t, a_t}, Output {i_t+1, a_t+1, r_t}) for training
import numpy as np

# Generate execution trace according to given matrix --> list of {e_t, i_t, a_t, r_t}
def generate_trace(map): # map should be a 2D map
	assert len(map.shape) == 2, 'shape_error'
	xmax = map.shape[0] # 10
	ymax = map.shape[1] # 10

	# Generate execution
	trace_dict = []
	program_list = [0, 1] # 0 (move to next) or 1 (judge the number)
	step = 0
	x_index = 0

	# For each timestep assgin {e_t, i_t, a_t, r_t}
	for y in range(ymax):
		for x in range(xmax * 2):
			# Assign e_t
			e = map[x_index, y]
			# Assign i_t
			if step % 2 == 0:
				i = 1
			elif step % 2 == 1:
				i = 0
			# Assign a_t
			if i == 1: # judge the number --> a = e
				a = e
			elif i == 0: # move to next --> a = 0
				if x_index == xmax - 1:
					a = 0
					x_index = 0
				else:
					a = 0
					x_index += 1
			# Assign r_t
			if x_index == xmax - 1 and y == ymax - 1 and i == 1:
				r = 1
			else:
				r = 0
			# Store execution trace of each timestep
			trace = {'e':e, 'i':i, 'a':a, 'r':r}
			trace_dict.append(trace)
			step += 1
			if r == 1:
				break
	return trace_dict

# Generate output of execution trace --> list of {i_t+1, a_t+1, r_t}
def generate_label(trace_dict):
	label = []
	for i in range(len(trace_dict)):
		i_label = np.zeros(2)
		a_label = np.zeros(3)
		r_label = np.zeros(2)
		trace = trace_dict[i]
		i = trace['i']  # 0 or 1
		i_label[i] = 1
		a = trace['a']  # 0, 1, 2
		a_label[a] = 1
		r = trace['r']  # 0, 1
		r_label[r] = 1
		label.append({'a':a_label, 'i':i_label, 'r':r_label})
	return label

# Generate input of execution trace --> np.array of {e_t, i_t, a_t}
def generate_sample(trace_dict):
	sample = []
	for i in range(len(trace_dict)):
		result = []
		trace = trace_dict[i]
		trace_last = trace_dict[i - 1]
		if i - 1 < 0:
			trace_last = {'e': 0, 'r': 0, 'a': 0, 'i': 0}
		e = trace['e']
		result.extend([e])
		i = trace_last['i']
		result.extend([i])
		a = trace_last['a']
		result.extend([a])
		sample.append(result)
	return np.array(sample)


# Test
# feature_map = np.array([[0,0,0,0,0,0,0,0,0,0],
# 						[0,1,1,1,1,0,0,0,0,0],
# 						[0,1,0,0,1,0,0,0,0,0],
# 						[0,1,1,1,1,0,0,0,0,0],
# 						[0,0,1,1,0,0,0,2,2,0],
# 						[0,0,1,1,0,0,0,2,2,2],
# 						[0,0,1,0,0,0,0,2,0,0],
# 						[0,0,1,0,0,0,0,2,2,2],
# 						[0,0,0,0,0,0,0,0,0,0],
# 						[0,0,0,0,0,0,0,0,0,0]])

# trace_list = generate_trace(feature_map) # list with length 199, each elem is a dict of {'e':e, 'i':i, 'a':a, 'r':r}
# print(trace_list)
# print(len(trace_list))

# trace_sample = generate_sample(trace_list) # np.array with shape [199, 3]
# print(trace_sample)
# print(trace_sample.shape)

# trace_label = generate_label(trace_list) # list with length 199, each elem is a dict of {'i':i, 'a':a, 'r':r} embedding
# print(trace_label)
# print(len(trace_label))