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

class OrigFrankaKitchen_Dataset(Dataset): 

	# Class implementing instance of Robomimic dataset. 
	def __init__(self, args):		
		
		self.args = args

		if self.args.datadir is None:
			self.dataset_directory = '/data/tanmayshankar/Datasets/FrankaKitchen'
		else:
			self.dataset_directory = self.args.datadir

		# Require a task list. 
		# The task name is needed for setting the environment, rendering. 		

		# self.task_list = ['table_cleanup_to_dishwasher', "table_cleanup_to_sink", "table_setup_from_dishwasher", "table_setup_from_dresser"]
		self.task_list = ['kitchen-partial-v0']
		# self.task_list = ["can","lift","square","tool_hang"]
		self.environment_names = ['kitchen-partial-v0']
						
		# self.num_demos = np.array([612])
		# self.cummulative_num_demos = self.num_demos.cumsum()
		# self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)	
		# self.total_length = self.num_demos.sum()		

		self.ds_freq = 1.5

		# Set files. 
		self.setup()

		self.stat_dir_name='FrankaKitchen'

	def setup(self):
		
		import d4rl
		self.create_environment() 		
		self.franka_kitchen_generator = d4rl.d4rl.sequence_dataset(self.environment)
	

		# # Load data from all tasks. 			
		# self.files = []
		# self.file_list = sorted(glob.glob(os.path.join(self.dataset_directory, "*.hdf5")))

		# for k, file in enumerate(self.file_list):
		# 	# Changing file name.
		# 	self.files.append(h5py.File(file,'r'))
		# 	# self.files.append(h5py.File("{0}/{1}/ph/low_dim.hdf5".format(self.dataset_directory,	self.task_list[i]),'r'))

		self.length_threshold = 100
		self.total_length = 604
		self.set_relevant_indices()

	def __len__(self):			
		return self.total_length
	
	def __getitem__(self, index):

		return {}
	
	def create_environment(self):
		import d4rl
		import gym
		self.environment = gym.make('kitchen-partial-v0')		

	def set_relevant_indices(self):

		self.robot_pose_indices = np.arange(0,9)
		self.object_pose_indices = np.arange(9,30)
		self.goal_pose_indices = np.arange(30,60)

	def preprocess_dataset(self):

		# Create list of demos. 
		task_demo_list = []	

		# For however many trajectories we have: 
		for i, demo in enumerate(self.franka_kitchen_generator):
			
			# Create list of datapoints for this demonstrations. 
			datapoint = {}
			print("Preprocessing Demo Index: ", i, " of: 613" )
		
			# print("Embed in preproc")
			# embed()

			# Get SEQUENCE of flattened states.
			flattened_state_sequence = demo['observations']
			robot_state_sequence = demo['observations'][...,self.robot_pose_indices]
			object_state_sequence = demo['observations'][...,self.object_pose_indices]
			goal_state_sequence = demo['observations'][...,self.goal_pose_indices]
			robot_action_sequence = demo['actions']

			traj_length = flattened_state_sequence.shape[0]
			if traj_length>self.length_threshold:

				# Number of timesteps to downsample to. 
				number_timesteps = int(traj_length//self.ds_freq)

				flattened_state_sequence = resample(flattened_state_sequence, number_timesteps)
				robot_state_sequence = resample(robot_state_sequence, number_timesteps)
				object_state_sequence = resample(object_state_sequence, number_timesteps)
				goal_state_sequence = resample(goal_state_sequence, number_timesteps)
				robot_action_sequence = resample(robot_action_sequence, number_timesteps)			
				concatenated_demonstration = np.concatenate([robot_state_sequence, object_state_sequence], axis=-1)

				# Put both lists in a dictionary.
				datapoint['flat-state'] = flattened_state_sequence
				datapoint['robot-state'] = robot_state_sequence
				datapoint['object-state'] = object_state_sequence
				datapoint['goal-state'] = goal_state_sequence
				datapoint['actions'] = robot_action_sequence
				datapoint['demo'] = concatenated_demonstration

				# Add this dictionary to the file_demo_list. 
				task_demo_list.append(copy.deepcopy(datapoint))

		# Create array.
		task_demo_array = np.array(task_demo_list)

		# Now save this file_demo_list. 
		np.save(os.path.join(self.dataset_directory,"New_Task_Demo_Array.npy"),task_demo_array)


class FrankaKitchen_Dataset(OrigFrankaKitchen_Dataset):
	
	def __init__(self, args):
		
		super(FrankaKitchen_Dataset, self).__init__(args)	

		
		# Now that we've run setup, compute dataset_trajectory_lengths for smart batching.
		self.dataset_trajectory_lengths = np.zeros(self.total_length)

		for index in range(self.total_length):

			data_element = self.files[index]
			self.dataset_trajectory_lengths[index] = len(data_element['demo'])

		# Now implementing the dataset trajectory length limits. 
		######################################################
		# Now implementing dataset_trajectory_length_limits. 
		######################################################
		
		if self.args.dataset_traj_length_limit>0:
			# Essentially will need new self.cummulative_num_demos and new .. file index map list things. 
			# Also will need to set total_length. 

			self.full_max_length = self.dataset_trajectory_lengths.max()
			self.full_length = copy.deepcopy(self.total_length)
			self.full_files = copy.deepcopy(self.files)
			self.files = []
			self.full_dataset_trajectory_lengths = copy.deepcopy(self.dataset_trajectory_lengths)
			self.dataset_trajectory_lengths = []
			self.num_demos = 0

			for index in range(self.full_length):

				# Check the length of this particular trajectory and its validity. 
				if (self.full_dataset_trajectory_lengths[index] < self.args.dataset_traj_length_limit):
					# and (index not in self.bad_original_index_list):
					# Add from old list to new. 
					self.files.append(self.full_files[index])
					self.dataset_trajectory_lengths.append(self.full_dataset_trajectory_lengths[index])
					self.num_demos += 1

			# Set new total length.
			self.total_length = self.num_demos
			# Make array.
			self.dataset_trajectory_lengths = np.array(self.dataset_trajectory_lengths)

			self.files = np.array(self.files)			


	def setup(self):
		self.files = np.load("{0}/New_Task_Demo_Array.npy".format(self.dataset_directory), allow_pickle=True)
		self.total_length = 604
		self.set_relevant_indices()

	def __getitem__(self, index):

		if index>=self.total_length:
			print("Out of bounds of dataset.")
			return None

		data_element = self.files[index]		
		self.kernel_bandwidth = self.args.smoothing_kernel_bandwidth

		# Dummy task ID. 
		data_element['task-id'] = 0
		data_element['task_id'] = 0
		
		if self.args.smoothen: 
			data_element['demo'] = gaussian_filter1d(data_element['demo'],self.kernel_bandwidth,axis=0,mode='nearest')
			data_element['robot-state'] = gaussian_filter1d(data_element['robot-state'],self.kernel_bandwidth,axis=0,mode='nearest')
			data_element['object-state'] = gaussian_filter1d(data_element['object-state'],self.kernel_bandwidth,axis=0,mode='nearest')
			data_element['flat-state'] = gaussian_filter1d(data_element['flat-state'],self.kernel_bandwidth,axis=0,mode='nearest')
			data_element['goal-state'] = gaussian_filter1d(data_element['goal-state'],self.kernel_bandwidth,axis=0,mode='nearest')
			data_element['actions'] = gaussian_filter1d(data_element['actions'],self.kernel_bandwidth,axis=0,mode='nearest')

		
			# data_element['environment-name'] = self.environment_names[task_index]

		return data_element
	
	def compute_statistics(self):

		self.state_size = 30
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

		np.save("FrankaKitchenRO_Mean.npy", mean)
		np.save("FrankaKitchenRO_Var.npy", variance)
		np.save("FrankaKitchenRO_Min.npy", min_value)
		np.save("FrankaKitchenRO_Max.npy", max_value)
		np.save("FrankaKitchenRO_Vel_Mean.npy", vel_mean)
		np.save("FrankaKitchenRO_Vel_Var.npy", vel_variance)
		np.save("FrankaKitchenRO_Vel_Min.npy", vel_min_value)
		np.save("FrankaKitchenRO_Vel_Max.npy", vel_max_value)

class FrankaKitchen_ObjectDataset(FrankaKitchen_Dataset):

	def __init__(self, args):

		super(FrankaKitchen_ObjectDataset, self).__init__(args)

	def super_getitem(self, index):

		return super().__getitem__(index)

	def __getitem__(self, index):
		
		data_element = copy.deepcopy(super().__getitem__(index))

		# Copy over the demo to the robot-demo key.
		data_element['robot-demo'] = copy.deepcopy(data_element['demo'])
		# Set demo to object-state trajectory. 
		data_element['demo'] = data_element['object-state']
		# [:,start_index:start_index+7]

		return data_element

class FrankaKitchen_RobotObjectDataset(FrankaKitchen_Dataset):

	def __init__(self, args):

		super(FrankaKitchen_RobotObjectDataset, self).__init__(args)
		self.stat_dir_name = 'FrankaKitchenRO'
		
	def super_getitem(self, index):

		return super().__getitem__(index)

	# def __getitem__(self, index):

	# 	data_element = copy.deepcopy(super().__getitem__(index))
		
	# 	# data_element['demo'] = data_element['demo'][...,7:]
	# 	data_element['demo'] = data_element['demo']

	# 	return data_element