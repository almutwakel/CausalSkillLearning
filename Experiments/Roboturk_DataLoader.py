# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from headers import *

def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]

class Roboturk_Dataset(Dataset):

	# LINK TO DATASET and INFO: http://roboturk.stanford.edu/dataset.html

	# Class implementing instance of Roboturk dataset. 
	def __init__(self, args):
		# self.dataset_directory = '/checkpoint/tanmayshankar/Roboturk/RoboTurkPilot'		
		
		self.args = args
		if self.args.datadir is None:
			# self.dataset_directory = '/checkpoint/tanmayshankar/MIME/'
			# self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/MIME/'
			self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/Roboturk/'
		else:
			self.dataset_directory = self.args.datadir			
		self.stat_dir_name='Roboturk'
		# Require a task list. 

		# The task name is needed for setting the environment, rendering. 
		# We shouldn't need the environment for .. training though, should we? 

		self.task_list = ["bins-Bread", "bins-Can", "bins-Cereal", "bins-full", "bins-Milk", "pegs-full", "pegs-RoundNut", "pegs-SquareNut"]
		self.num_demos = np.array([1069, 1069, 1069, 1069, 1069, 1145, 1144, 1145])
		self.cummulative_num_demos = self.num_demos.cumsum()
		self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)
		# Append -1 to the start of cummulative_num_demos. This has two purposes. 
		# The first is that when we are at index 0 of the dataset, if we appended 0, np.searchsorted returns 0, rather than 1. 
		# For index 1, it returns 1. This was becoming inconsistent behavior for demonstrations in the same task. 
		# Now with -1 added to cumm_num_demos, when we are at task index 0, it would add -1 to the demo index. This is necessary for ALL tasks, not just the first...  
		# So that foils our really clever idea. 
		# Well, if the searchsorted returns the index of the equalling element, it probably consistently does this irrespective of vlaue. 
		# This means we can use this...

		# No need for a clever solution, searchsorted has a "side" option that takes care of this. 

		self.total_length = self.num_demos.sum()		

		# Seems to follow joint angles order:
		# ('time','right_j0', 'head_pan', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6', 'r_gripper_l_finger_joint', 'r_gripper_r_finger_joint', 'Milk0', 'Bread0', 'Cereal0', 'Can0').
		# Extract these into... 
		self.joint_angle_indices = [1,3,4,5,6,7,8]
		self.gripper_indices = [9,10]	
		self.ds_freq = 20
		self.state_size = 8
		# self.r_gripper_r_finger_joint = np.array([-0.0116,   0.020833])
		# self.r_gripper_l_finger_joint = np.array([-0.020833, 0.0135])

		# [l,r]
		# gripper_open = [0.0115, -0.0115]
		# gripper_closed = [-0.020833, 0.020833]

		# Set files. 
		self.setup()

	def setup(self):
		# Load data from all tasks. 			
		self.files = []
		for i in range(len(self.task_list)):
			self.files.append(h5py.File("{0}/{1}/demo.hdf5".format(self.dataset_directory,self.task_list[i]),'r'))	

	def __len__(self):
		return self.total_length

	def __getitem__(self, index):

		if index>=self.total_length:
			print("Out of bounds of dataset.")
			return None

		# Get bucket that index falls into based on num_demos array. 
		task_index = np.searchsorted(self.cummulative_num_demos, index, side='right')-1
		
		if index==self.total_length-1:
			task_index-=1

		# Decide task ID, and new index modulo num_demos.
		# Subtract number of demonstrations in cumsum until then, and then 				
		new_index = index-self.cummulative_num_demos[max(task_index,0)]+1
		
		try:
			# Get raw state sequence. 
			state_sequence = self.files[task_index]['data/demo_{0}/states'.format(new_index)].value
		except:
			# If this failed, return invalid. 
			data_element = {}
			data_element['is_valid'] = False

			return data_element

		# Performing another check that makes sure data element actually has states.
		if state_sequence.shape[0]==0:
			data_element = {}
			data_element['is_valid'] = False
			return data_element

		# If we are here, the data element is presumably valid till now.
		# Get joint angles from this state sequence.
		joint_values = state_sequence[:,self.joint_angle_indices]
		# Get gripper values from state sequence. 
		gripper_finger_values = state_sequence[:,self.gripper_indices]

		# Normalize gripper values. 

		# 1 is right finger. 0 is left finger. 
		# 1-0 is right-left. 
		
		gripper_values = gripper_finger_values[:,1]-gripper_finger_values[:,0]
		gripper_values = (gripper_values-gripper_values.min()) / (gripper_values.max()-gripper_values.min())
		gripper_values = 2*gripper_values-1

		concatenated_demonstration = np.concatenate([joint_values,gripper_values.reshape((-1,1))],axis=1)
		downsampled_demonstration = resample(concatenated_demonstration, concatenated_demonstration.shape[0]//self.ds_freq)

		# Performing another check that makes sure data element actually has states.
		if downsampled_demonstration.shape[0]==0:
			data_element = {}
			data_element['is_valid'] = False
			return data_element

		data_element = {}

		if self.args.smoothen:
			data_element['demo'] = gaussian_filter1d(downsampled_demonstration,self.args.smoothing_kernel_bandwidth,axis=0,mode='nearest')
		else:
			data_element['demo'] = downsampled_demonstration
		# Trivially setting is valid to true until we come up wiuth a better strategy. 
		data_element['is_valid'] = True

		return data_element
	
	def close(self):
		for file in self.files:
			file.close()

	def preprocess_dataset(self):

		# for task_index in range(len(self.task_list)):
		# for task_index in [3,5]:
		for task_index in [0,1,2,4,6,7]:

			print("#######################################")
			print("Preprocessing task index: ", task_index)
			print("#######################################")

			# Get the name of environment.
			environment_name = self.files[task_index]['data'].attrs['env']
			# Create an actual robo-suite environment. 
			import robosuite
			self.env = robosuite.make(environment_name)

			# Get sizes. 
			obs = self.env._get_observation()
			robot_state_size = obs['robot-state'].shape[0]
			object_state_size = obs['object-state'].shape[0]	


			# Create list of files for this task. 
			task_demo_list = []

			# For every element in the filelist of the element,
			for i in range(1,self.num_demos[task_index]+1):

				print("Preprocessing task index: ", task_index, " Demo Index: ", i, " of: ", self.num_demos[task_index])
			
				# Create list of datapoints for this demonstrations. 
				datapoint = {}

				# Get SEQUENCE of flattened states.
				try:
					flattened_state_sequence = self.files[task_index]['data/demo_{0}/states'.format(i)].value
					joint_action_sequence = self.files[task_index]['data/demo_{0}/joint_velocities'.format(i)].value
					gripper_action_sequence = self.files[task_index]['data/demo_{0}/gripper_actuations'.format(i)].value

					flattened_state_sequence = resample(flattened_state_sequence, flattened_state_sequence.shape[0]//self.ds_freq)

					number_timesteps = flattened_state_sequence.shape[0]
					robot_state_array = np.zeros((number_timesteps, robot_state_size))
					object_state_array = np.zeros((number_timesteps, object_state_size))

					# Get joint angle values from 
					joint_values = flattened_state_sequence[:,self.joint_angle_indices]
					# Get gripper values from state sequence. 
					gripper_finger_values = flattened_state_sequence[:,self.gripper_indices]

					# Normalize gripper values. 

					# 1 is right finger. 0 is left finger. 
					# 1-0 is right-left. 
		
					gripper_values = gripper_finger_values[:,1]-gripper_finger_values[:,0]
					gripper_values = (gripper_values-gripper_values.min()) / (gripper_values.max()-gripper_values.min())
					gripper_values = 2*gripper_values-1

					concatenated_demonstration = np.concatenate([joint_values,gripper_values.reshape((-1,1))],axis=1)
					concatenated_actions = np.concatenate([joint_action_sequence,gripper_action_sequence.reshape((-1,1))],axis=1)

					# For every element in sequence, set environment state. 
					for t in range(flattened_state_sequence.shape[0]):

						self.env.sim.set_state_from_flattened(flattened_state_sequence[t])
						# Seems like we necessarily need to call this, because otherwise states are wrong
						# This was the difference?
						# Also seems to be 
						self.env.sim.forward()

						# Now get observation.
						observation = self.env._get_observation()

						# Robot and Object state appended to datapoint dictionary. 
						robot_state_array[t] = observation['robot-state']
						object_state_array[t] = observation['object-state']

				except: 

					datapoint['robot_state_array'] = np.zeros((1, robot_state_size))
					datapoint['object_state_array'] = np.zeros((1, object_state_size))				

				# Put both lists in a dictionary.
				datapoint['flat-state'] = flattened_state_sequence
				datapoint['robot-state'] = robot_state_array
				datapoint['object-state'] = object_state_array
				datapoint['demo'] = concatenated_demonstration				
				datapoint['demonstrated_actions'] = concatenated_actions

				# Add this dictionary to the file_demo_list. 
				task_demo_list.append(datapoint)

			# Create array.
			task_demo_array = np.array(task_demo_list)

			# Now save this file_demo_list. 
			# np.save(os.path.join(self.dataset_directory,self.task_list[task_index],"New_Task_Demo_Array.npy"),task_demo_array)
			np.save(os.path.join(self.dataset_directory,self.task_list[task_index],"New_Task_Demo_Array_with_Objects.npy"),task_demo_array)

	def compute_statistics(self, suffix=""):
		
		self.total_length = self.__len__()
		mean = np.zeros((self.state_size))
		variance = np.zeros((self.state_size))
		mins = np.zeros((self.total_length, self.state_size))
		maxs = np.zeros((self.total_length, self.state_size))
		lens = np.zeros((self.total_length))

		# And velocity statistics. 
		vel_mean = np.zeros((self.state_size))
		vel_variance = np.zeros((self.state_size))
		vel_mins = np.zeros((self.total_length, self.state_size))
		vel_maxs = np.zeros((self.total_length, self.state_size))

		
		for i in range(self.total_length):

			print("Phase 1: DP: ",i)
			data_element = self.__getitem__(i)

			if data_element['is_valid']:
				demo = data_element['demo']
				vel = np.diff(demo,axis=0)
				mins[i] = demo.min(axis=0)
				maxs[i] = demo.max(axis=0)
				mean += demo.sum(axis=0)
				lens[i] = demo.shape[0]

				vel_mins[i] = abs(vel).min(axis=0)
				vel_maxs[i] = abs(vel).max(axis=0)
				vel_mean += vel.sum(axis=0)			

		mean /= lens.sum()
		vel_mean /= lens.sum()

		for i in range(self.total_length):

			print("Phase 2: DP: ",i)
			data_element = self.__getitem__(i)
			
			# Just need to normalize the demonstration. Not the rest. 
			if data_element['is_valid']:
				demo = data_element['demo']
				vel = np.diff(demo,axis=0)
				variance += ((demo-mean)**2).sum(axis=0)
				vel_variance += ((vel-vel_mean)**2).sum(axis=0)

		variance /= lens.sum()
		variance = np.sqrt(variance)

		vel_variance /= lens.sum()
		vel_variance = np.sqrt(vel_variance)

		max_value = maxs.max(axis=0)
		min_value = mins.min(axis=0)

		vel_max_value = vel_maxs.max(axis=0)
		vel_min_value = vel_mins.min(axis=0)

		np.save("Roboturk{}_Mean.npy".format(suffix), mean)
		np.save("Roboturk{}_Var.npy".format(suffix), variance)
		np.save("Roboturk{}_Min.npy".format(suffix), min_value)
		np.save("Roboturk{}_Max.npy".format(suffix), max_value)
		np.save("Roboturk{}_Vel_Mean.npy".format(suffix), vel_mean)
		np.save("Roboturk{}_Vel_Var.npy".format(suffix), vel_variance)
		np.save("Roboturk{}_Vel_Min.npy".format(suffix), vel_min_value)
		np.save("Roboturk{}_Vel_Max.npy".format(suffix), vel_max_value)

class Roboturk_FullDataset(Roboturk_Dataset):
	
	def __init__(self, args):
		
		super(Roboturk_FullDataset, self).__init__(args)
		
		self.environment_names = ["SawyerPickPlaceBread","SawyerPickPlaceCan","SawyerPickPlaceCereal","SawyerPickPlace","SawyerPickPlaceMilk","SawyerNutAssembly", "SawyerNutAssemblyRound","SawyerNutAssemblySquare"]

	def setup(self):
		self.files = []
		for i in range(len(self.task_list)):
			if i==3 or i==5:
				self.files.append(np.load("{0}/{1}/FullDataset_Task_Demo_Array.npy".format(self.dataset_directory, self.task_list[i]), allow_pickle=True))
			else:
				# self.files.append(np.load("{0}/{1}/New_Task_Demo_Array.npy".format(self.dataset_directory, self.task_list[i]), allow_pickle=True))
				self.files.append(np.load("{0}/{1}/New_Task_Demo_Array_with_Objects.npy".format(self.dataset_directory, self.task_list[i]), allow_pickle=True))

	def __getitem__(self, index):

		if index>=self.total_length:
			print("Out of bounds of dataset.")
			return None

		# Get bucket that index falls into based on num_demos array. 
		task_index = np.searchsorted(self.cummulative_num_demos, index, side='right')-1
		
		# Decide task ID, and new index modulo num_demos.
		# Subtract number of demonstrations in cumsum until then, and then 				
		new_index = index-self.cummulative_num_demos[max(task_index,0)]		
		data_element = self.files[task_index][new_index]

		resample_length = len(data_element['demo'])//self.args.ds_freq
		# print("Orig:", len(data_element['demo']),"New length:",resample_length)

		self.kernel_bandwidth = self.args.smoothing_kernel_bandwidth
		
		# Trivially adding task ID to data element.
		data_element['task_id'] = task_index
		data_element['task-id'] = task_index

		if resample_length<=1 or data_element['robot-state'].shape[0]<=1:
			data_element['is_valid'] = False			
		else:
			data_element['is_valid'] = True

			if self.args.smoothen: 
				data_element['demo'] = gaussian_filter1d(data_element['demo'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['robot-state'] = gaussian_filter1d(data_element['robot-state'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['object-state'] = gaussian_filter1d(data_element['object-state'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['flat-state'] = gaussian_filter1d(data_element['flat-state'],self.kernel_bandwidth,axis=0,mode='nearest')

			data_element['environment-name'] = self.environment_names[task_index]

			if self.args.ds_freq>1:
				data_element['demo'] = resample(data_element['demo'], resample_length)
				data_element['robot-state'] = resample(data_element['robot-state'], resample_length)
				data_element['object-state'] = resample(data_element['object-state'], resample_length)
				data_element['flat-state'] = resample(data_element['flat-state'], resample_length)

		return data_element

class Roboturk_SegmentedDataset(Roboturk_Dataset):

	def __init__(self):

		super(Roboturk_SegmentedDataset, self).__init__()		

		self.args = args
		# self.dataset_directory = '/checkpoint/tanmayshankar/Roboturk/RoboTurkPilot'
		if self.args.datadir is None:
			# self.dataset_directory = '/checkpoint/tanmayshankar/MIME/'
			# self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/MIME/'
			self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/Roboturk/'
		else:
			self.dataset_directory = self.args.datadir	
		
		# Require a task list. 
		# The task name is needed for setting the environment, rendering. 
		# We shouldn't need the environment for .. training though, should we? 

		self.task_list = ["bins-Bread", "bins-Can", "bins-Cereal", "bins-Milk", "pegs-RoundNut", "pegs-SquareNut"]
		self.num_demos = np.array([1069, 1069, 1069, 1069, 1144, 1145])
		self.cummulative_num_demos = self.num_demos.cumsum()
		self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)
		# Append -1 to the start of cummulative_num_demos. This has two purposes. 
		# The first is that when we are at index 0 of the dataset, if we appended 0, np.searchsorted returns 0, rather than 1. 
		# For index 1, it returns 1. This was becoming inconsistent behavior for demonstrations in the same task. 
		# Now with -1 added to cumm_num_demos, when we are at task index 0, it would add -1 to the demo index. This is necessary for ALL tasks, not just the first...  
		# So that foils our really clever idea. 
		# Well, if the searchsorted returns the index of the equalling element, it probably consistently does this irrespective of vlaue. 
		# This means we can use this...

		# No need for a clever solution, searchsorted has a "side" option that takes care of this. 

		self.total_length = self.num_demos.sum()		

		# Load data from all tasks. 			
		self.files = []
		for i in range(len(self.task_list)):
			self.files.append(h5py.File("{0}/{1}/demo.hdf5".format(self.dataset_directory,self.task_list[i]),'r'))

		# Seems to follow joint angles order:
		# ('time','right_j0', 'head_pan', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6', 'r_gripper_l_finger_joint', 'r_gripper_r_finger_joint', 'Milk0', 'Bread0', 'Cereal0', 'Can0').
		# Extract these into... 
		self.joint_angle_indices = [1,3,4,5,6,7,8]
		self.gripper_indices = [9,10]	
		self.ds_freq = 20


class Roboturk_NewSegmentedDataset(Dataset):

	def __init__(self, args):

		super(Roboturk_NewSegmentedDataset, self).__init__()
		self.stat_dir_name='Roboturk'
		# self.dataset_directory = '/checkpoint/tanmayshankar/Roboturk/RoboTurkPilot'
		# self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/Roboturk/'
		self.args = args
		if self.args.datadir is None:
			# self.dataset_directory = '/checkpoint/tanmayshankar/MIME/'
			# self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/MIME/'
			self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/Roboturk/'
		else:
			self.dataset_directory = self.args.datadir	
		
		self.args = args
		# Require a task list. 
		# The task name is needed for setting the environment, rendering. 
		# We shouldn't need the environment for .. training though, should we? 

		self.task_list = ["bins-Bread", "bins-Can", "bins-Cereal", "bins-Milk", "pegs-RoundNut", "pegs-SquareNut"]
		self.task_names = {	"bins-Bread": "SawyerPickPlaceBread",
						 	"bins-Can": "SawyerPickPlaceCan",
							"bins-Cereal": "SawyerPickPlaceCereal", 
							"bins-Milk": "SawyerPickPlaceMilk", 
							"pegs-RoundNut": "SawyerNutAssemblyRound", 
							"pegs-SquareNut": "SawyerNutAssemblySquare"
						}
		self.environment_names = ["SawyerPickPlaceBread","SawyerPickPlaceCan","SawyerPickPlaceCereal","SawyerPickPlaceMilk","SawyerNutAssemblyRound","SawyerNutAssemblySquare"]
		self.num_demos = np.array([1069, 1069, 1069, 1069, 1144, 1145])
		self.cummulative_num_demos = self.num_demos.cumsum()
		self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)

		self.bad_original_index_list = [4900,537]
		self.state_size = 8
		# self.bad_original_index_list = []
		
		# Append -1 to the start of cummulative_num_demos. This has two purposes. 
		# The first is that when we are at index 0 of the dataset, if we appended 0, np.searchsorted returns 0, rather than 1. 
		# For index 1, it returns 1. This was becoming inconsistent behavior for demonstrations in the same task. 
		# Now with -1 added to cumm_num_demos, when we are at task index 0, it would add -1 to the demo index. This is necessary for ALL tasks, not just the first...  
		# So that foils our really clever idea. 
		# Well, if the searchsorted returns the index of the equalling element, it probably consistently does this irrespective of vlaue. 
		# This means we can use this...

		# No need for a clever solution, searchsorted has a "side" option that takes care of this. 

		self.total_length = self.num_demos.sum()		

		# Load data from all tasks. 			
		self.files = []
		# for i in range(len(self.task_list)):
		for i in range(len(self.task_list)):
			# self.files.append(np.load("{0}/{1}/New_Task_Demo_Array.npy".format(self.dataset_directory, self.task_list[i]), allow_pickle=True))
			self.files.append(np.load("{0}/{1}/New_Task_Demo_Array_with_Objects.npy".format(self.dataset_directory, self.task_list[i]), allow_pickle=True))

		# # Seems to follow joint angles order:
		# # ('time','right_j0', 'head_pan', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6', 'r_gripper_l_finger_joint', 'r_gripper_r_finger_joint', 'Milk0', 'Bread0', 'Cereal0', 'Can0').
		# # Extract these into... 
		# self.joint_angle_indices = [1,3,4,5,6,7,8]
		# self.gripper_indices = [9,10]	
		# self.ds_freq = 20
		# # self.r_gripper_r_finger_joint = np.array([-0.0116,   0.020833])
		# # self.r_gripper_l_finger_joint = np.array([-0.020833, 0.0135])

		# # [l,r]
		# # gripper_open = [0.0115, -0.0115]
		# # gripper_closed = [-0.020833, 0.020833]

		# Get dataset trajectory lengths for smart batching
		self.dataset_trajectory_lengths = np.zeros(self.total_length)
		for index in range(self.total_length):
			# Get bucket that index falls into based on num_demos array. 
			task_index = np.searchsorted(self.cummulative_num_demos, index, side='right')-1
			
			# Decide task ID, and new index modulo num_demos.
			# Subtract number of demonstrations in cumsum until then, and then 				
			new_index = index-self.cummulative_num_demos[max(task_index,0)]		
			data_element = self.files[task_index][new_index]

			self.dataset_trajectory_lengths[index] = len(data_element['demo'])

		######################################################
		# Now implementing dataset_trajectory_length_limits. 
		######################################################
		
		if self.args.dataset_traj_length_limit>0:
			# Essentially will need new self.cummulative_num_demos and new .. file index map list things. 
			# Also will need to set total_length. 

			self.full_max_length = self.dataset_trajectory_lengths.max()
			self.full_length = copy.deepcopy(self.total_length)
			self.full_cummulative_num_demos = copy.deepcopy(self.cummulative_num_demos)
			self.full_num_demos = copy.deepcopy(self.num_demos)
			self.full_files = copy.deepcopy(self.files)
			self.files = [[] for i in range(len(self.task_list))]
			self.full_dataset_trajectory_lengths = copy.deepcopy(self.dataset_trajectory_lengths)
			self.dataset_trajectory_lengths = []
			self.num_demos = np.zeros(len(self.task_list),dtype=int)

			for index in range(self.full_length):
				# Get bucket that index falls into based on num_demos array. 
				task_index = np.searchsorted(self.full_cummulative_num_demos, index, side='right')-1
				# Get the demo index in this task list. 
				new_index = index-self.full_cummulative_num_demos[max(task_index,0)]

				# Check the length of this particular trajectory and its validity. 
				if (self.full_dataset_trajectory_lengths[index] < self.args.dataset_traj_length_limit) and (index not in self.bad_original_index_list):
					# Add from old list to new. 
					self.files[task_index].append(self.full_files[task_index][new_index])
					self.dataset_trajectory_lengths.append(self.full_dataset_trajectory_lengths[index])
					self.num_demos[task_index] += 1
				else:
					pass

					# Reduce count. 
					# self.num_demos[task_index] -= 1
					
					# # Pop item from files. It's still saved in full_files. 					
					# # self.files[task_index].pop(new_index)
					# self.files[task_index] = np.delete(self.files[task_index],new_index)
					# Approach with opposite pattern.. instead of deleting invalid files, add valid ones.
					
					# # Pop item from dataset_trajectory_lengths. 
					# self.dataset_trajectory_lengths = np.delete(self.dataset_trajectory_lengths, index)

			# Set new cummulative num demos. 
			self.cummulative_num_demos = self.num_demos.cumsum()
			self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)
			# Set new total length.
			self.total_length = self.cummulative_num_demos[-1]
			# Make array.
			self.dataset_trajectory_lengths = np.array(self.dataset_trajectory_lengths)

			for t in range(len(self.task_list)):
				self.files[t] = np.array(self.files[t])

			# By popping element from files / dataset_traj_lengths, we now don't need to change indexing.
	

	def __len__(self):
		return self.total_length

	def __getitem__(self, index):

		if index>=self.total_length:
			print("Out of bounds of dataset.")
			return None
		
		if index in self.bad_original_index_list:
			index += 1

		# Get bucket that index falls into based on num_demos array. 
		task_index = np.searchsorted(self.cummulative_num_demos, index, side='right')-1
		
				# Decide task ID, and new index modulo num_demos.
		# Subtract number of demonstrations in cumsum until then, and then 				
		new_index = index-self.cummulative_num_demos[max(task_index,0)]
		data_element = self.files[task_index][new_index]

		resample_length = len(data_element['demo'])//self.args.ds_freq
		# print("Orig:", len(data_element['demo']),"New length:",resample_length)

		self.kernel_bandwidth = self.args.smoothing_kernel_bandwidth

		if resample_length<=1 or ((index in self.bad_original_index_list) and ((self.args.dataset_traj_length_limit==-1) or (self.args.dataset_traj_length_limit>self.full_max_length))):
			# Only skip elements here if we didn't artificially shorten trajs
			data_element['is_valid'] = False			
		else:
			data_element['is_valid'] = True
			
			if self.args.smoothen:
				data_element['demo'] = gaussian_filter1d(data_element['demo'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['robot-state'] = gaussian_filter1d(data_element['robot-state'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['object-state'] = gaussian_filter1d(data_element['object-state'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['flat-state'] = gaussian_filter1d(data_element['flat-state'],self.kernel_bandwidth,axis=0,mode='nearest')			

			data_element['environment-name'] = self.environment_names[task_index]
			data_element['task-id'] = task_index
			# Trivially adding task ID to data element.
			data_element['task_id'] = task_index

			if self.args.ds_freq>1:
				data_element['demo'] = resample(data_element['demo'], resample_length)
				data_element['robot-state'] = resample(data_element['robot-state'], resample_length)
				data_element['object-state'] = resample(data_element['object-state'], resample_length)
				data_element['flat-state'] = resample(data_element['flat-state'], resample_length)


		return data_element

	def get_number_task_demos(self, task_index):
		return self.num_demos[task_index]

	def get_task_demo(self, task_index, index):

		if index>=self.num_demos[task_index]:
			print("Out of bounds of dataset.")
			return None		

		data_element = self.files[task_index][index]

		resample_length = len(data_element['demo'])//self.args.ds_freq
		# print("Orig:", len(data_element['demo']),"New length:",resample_length)

		self.kernel_bandwidth = self.args.smoothing_kernel_bandwidth

		if resample_length<=1 or data_element['robot-state'].shape[0]==0:			
			data_element['is_valid'] = False			
		else:
			data_element['is_valid'] = True

			if self.args.smoothen:
				data_element['demo'] = gaussian_filter1d(data_element['demo'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['robot-state'] = gaussian_filter1d(data_element['robot-state'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['object-state'] = gaussian_filter1d(data_element['object-state'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['flat-state'] = gaussian_filter1d(data_element['flat-state'],self.kernel_bandwidth,axis=0,mode='nearest')			

			data_element['environment-name'] = self.environment_names[task_index]

			if self.args.ds_freq>1:
				data_element['demo'] = resample(data_element['demo'], resample_length)
				data_element['robot-state'] = resample(data_element['robot-state'], resample_length)
				data_element['object-state'] = resample(data_element['object-state'], resample_length)
				data_element['flat-state'] = resample(data_element['flat-state'], resample_length)

		return data_element
	def compute_statistics(self, suffix=""):
		
		self.total_length = self.__len__()
		mean = np.zeros((self.state_size))
		variance = np.zeros((self.state_size))
		mins = np.zeros((self.total_length, self.state_size))
		maxs = np.zeros((self.total_length, self.state_size))
		lens = np.zeros((self.total_length))

		# And velocity statistics. 
		vel_mean = np.zeros((self.state_size))
		vel_variance = np.zeros((self.state_size))
		vel_mins = np.zeros((self.total_length, self.state_size))
		vel_maxs = np.zeros((self.total_length, self.state_size))

		
		for i in range(self.total_length):

			print("Phase 1: DP: ",i)
			data_element = self.__getitem__(i)

			if data_element['is_valid']:
				demo = data_element['demo']
				vel = np.diff(demo,axis=0)
				mins[i] = demo.min(axis=0)
				maxs[i] = demo.max(axis=0)
				mean += demo.sum(axis=0)
				lens[i] = demo.shape[0]

				vel_mins[i] = abs(vel).min(axis=0)
				vel_maxs[i] = abs(vel).max(axis=0)
				vel_mean += vel.sum(axis=0)			

		mean /= lens.sum()
		vel_mean /= lens.sum()

		for i in range(self.total_length):

			print("Phase 2: DP: ",i)
			data_element = self.__getitem__(i)
			
			# Just need to normalize the demonstration. Not the rest. 
			if data_element['is_valid']:
				demo = data_element['demo']
				vel = np.diff(demo,axis=0)
				variance += ((demo-mean)**2).sum(axis=0)
				vel_variance += ((vel-vel_mean)**2).sum(axis=0)

		variance /= lens.sum()
		variance = np.sqrt(variance)

		vel_variance /= lens.sum()
		vel_variance = np.sqrt(vel_variance)

		max_value = maxs.max(axis=0)
		min_value = mins.min(axis=0)

		vel_max_value = vel_maxs.max(axis=0)
		vel_min_value = vel_mins.min(axis=0)

		np.save("Roboturk{}_Mean.npy".format(suffix), mean)
		np.save("Roboturk{}_Var.npy".format(suffix), variance)
		np.save("Roboturk{}_Min.npy".format(suffix), min_value)
		np.save("Roboturk{}_Max.npy".format(suffix), max_value)
		np.save("Roboturk{}_Vel_Mean.npy".format(suffix), vel_mean)
		np.save("Roboturk{}_Vel_Var.npy".format(suffix), vel_variance)
		np.save("Roboturk{}_Vel_Min.npy".format(suffix), vel_min_value)
		np.save("Roboturk{}_Vel_Max.npy".format(suffix), vel_max_value)


class Roboturk_ObjectDataset(Roboturk_NewSegmentedDataset):

	def __init__(self, args):

		super(Roboturk_ObjectDataset, self).__init__(args)
		self.state_size = 7

	def __getitem__(self, index):
		
		data_element = copy.deepcopy(super().__getitem__(index))

		# Copy over the demo to the robot-demo key.
		data_element['robot-demo'] = copy.deepcopy(data_element['demo'])
		# Set demo to object-state trajectory. 

		# Also try ignoring the relative positions for now.
		# print("Embedding in get el")
		# embed()

		if self.args.object_pure_relative_state:
			start_index = 7
		else:
			start_index = 0

		data_element['demo'] = data_element['object-state'][:,start_index:start_index+7]

		return data_element

class Roboturk_RobotObjectDataset(Roboturk_NewSegmentedDataset):

	def __init__(self, args):

		super(Roboturk_RobotObjectDataset, self).__init__(args)
		self.state_size = 15
		self.stat_dir_name = 'RoboturkRobotObject'

	def super_getitem(self, index):

		return super().__getitem__(index)

	def __getitem__(self, index):

		data_element = copy.deepcopy(super().__getitem__(index))

		# Now concatenate the robot and object states. 
		# if data_element['demo'].shape[-1]==15:
		# print("######################")
		# print(data_element['task-id'])
		# print("SHAPE OF 1st DEMO",data_element['demo'].shape)
		data_element['robot-demo'] = copy.deepcopy(data_element['demo'])

		if self.args.object_pure_relative_state:
			start_index = 7
		else:
			start_index = 0
		
		demo = np.concatenate([data_element['demo'],data_element['object-state'][:,start_index:start_index+7]],axis=-1)
		data_element['demo'] = copy.deepcopy(demo)

		# print("SHAPE OF 2nd DEMO",data_element['demo'].shape)

		return data_element

class Roboturk_Dataloader_Tester(unittest.TestCase):
	
	def test_Roboturkdataloader(self):

		self.dataset = Roboturk_Dataset()

		# Check the first index of the dataset.
		data_element = self.dataset[0]

		validity = data_element['is_valid']
		check_demo_data = (data_element['demo']==np.load("Test_Data/Roboturk_Dataloader_DE.npy")).all()

		self.assertTrue(validity and check_demo_data)

class RealRoboturk_Dataset(Dataset):
	
	def __init__(self, args):

		self.dataset_directory = '/data1/tanmayshankar/RealRoboturk'
		self.args = args
		
		embed()
		pass 

if __name__ == '__main__':
	# Run all tests defined for the dataloader.
	unittest.main()