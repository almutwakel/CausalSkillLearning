from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from headers import *

def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]

# def invert(homogenous_matrix):
	
# 	from scipy.spatial.transform import Rotation as R
# 	inverse = np.zeros((4,4))
# 	rotation = R.from_matrix(homogenous_matrix[:3,:3])
# 	inverse[:3, :3] = rotation.inv().as_matrix()
# 	inverse[:3, -1] = -rotation.inv().apply(homogenous_matrix[:3,-1])
# 	inverse[-1, -1] = 1.

# 	return inverse

# def invert_batch_matrix(homogenous_matrix):

# 	from scipy.spatial.transform import Rotation as R
# 	inverse = np.zeros((homogenous_matrix.shape[0], 4,4))
# 	rotation = R.from_matrix(homogenous_matrix[:,:3,:3])
# 	inverse[:, :3, :3] = rotation.inv().as_matrix()
# 	inverse[:, :3, -1] = -rotation.inv().apply(homogenous_matrix[:, :3,-1])
# 	inverse[:, -1, -1] = 1.

# 	return inverse

# def create_matrix(position, orientation):

# 	homogenous_matrix = np.zeros((4,4))
# 	homogenous_matrix[:3,-1] = position
# 	homogenous_matrix[:3,:3] = R.from_quat(orientation).as_matrix()
# 	homogenous_matrix[-1,-1] = 1.

# 	return homogenous_matrix

# def create_batch_matrix(position, orientation):

# 	traj_length = position.shape[0]
# 	homogenous_matrix = np.zeros((traj_length, 4,4))
# 	homogenous_matrix[:, :3,-1] = position
# 	homogenous_matrix[:, :3,:3] = R.from_quat(orientation).as_matrix()
# 	homogenous_matrix[:,-1,-1] = 1.

# 	return homogenous_matrix

# def normalize_quaternion(q):
# 	# Define quaternion normalization function.
# 	return q/np.linalg.norm(q)

def check_diff_invalidity(current_data_value, previous_average):    
	epsilon = 0.75
	return not(np.linalg.norm(current_data_value - previous_average) <= epsilon)

def check_stat_invalidity(current_data_value, previous_average):
	epsilon = 1.0
	statistical_mean = np.array([3.45, -2.4, 0.75])
	return np.linalg.norm(current_data_value - statistical_mean) <= epsilon

def check_stat_and_diff_invalidity(current_data_value, previous_average, t, latest_valid_index):

	if latest_valid_index==-1:
		previous_average = current_data_value
			
	invalid = check_diff_invalidity(current_data_value, previous_average) or check_stat_invalidity(current_data_value, previous_average)
	
	return invalid

class NDAXInterface_PreDataset(Dataset): 

	# Class implementing instance of RealWorld Rigid Body Dataset. 
	def __init__(self, args):
		
		self.args = args
		if self.args.datadir is None:
			# self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/NDAX/'
			self.dataset_directory = '/data/tanmayshankar/Datasets/NDAX/'
		else:
			self.dataset_directory = self.args.datadir			
		
		# Require a task list. 
		# The task name is needed for setting the environment, rendering. 
		self.task_list = ['open_door', 'close_door', 'eggs', 'stock_cupboard', 'stock_fridge']
		self.environment_names = ['open_door', 'close_door', 'eggs', 'stock_cupboard', 'stock_fridge']
		
		self.num_demos = np.array([3, 2, 2, 4, 4])

		# Each task has 200 demos according to RoboMimic.
		self.number_tasks = len(self.task_list)
		self.cummulative_num_demos = self.num_demos.cumsum()
		self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)
		self.total_length = self.num_demos.sum()		

		# self.ds_freq = 1*np.ones(self.number_tasks).astype(int)
		self.ds_freq = np.array([6.5])

		# Set files. 
		self.setup()	

		self.stat_dir_name ='NDAX'
	
	def interpolate_position(self, valid=None, position_sequence=None):
		
		from scipy import interpolate

		# Interp1d from Scipy expects the last dimension to be the dimension we are ninterpolating over. 
		valid_positions = np.swapaxes(position_sequence, 1, 0)
		valid_times = valid
		step = (valid[-1]-valid[0])/(len(valid)//self.ds_freq)

		print("Embed in Interpolate Position")
		# Resample uniform timesteps with the same length as original data. 
		query_times = np.arange(valid[0], valid[-1], step)

		# Create interpolating function. 
		interpolating_function = interpolate.interp1d(valid_times, valid_positions)

		# Query interpolating function. 
		interpolated_positions = interpolating_function(query_times)

		# Swap axes back and return. 
		return np.swapaxes(interpolated_positions, 1, 0)
	
	def interpolate_orientation(self, valid=None, orientation_sequence=None):

		from scipy.spatial.transform import Rotation as R
		from scipy.spatial.transform import Slerp

		# Resample uniform timesteps with the same length as original data. 
		valid_times = valid
		step = (valid[-1]-valid[0])/(len(valid)//self.ds_freq)
		query_times = np.arange(valid[0], valid[-1], step)

		valid_orientations = orientation_sequence
		rotation_sequence = R.concatenate(R.from_quat(valid_orientations))
		
		# Create slerp object. 
		slerp_object = Slerp(valid_times, rotation_sequence)

		# Query the slerp object. 
		interpolated_rotations = slerp_object(query_times)

		# Convert to quaternions.
		interpolated_quaternion_sequence = interpolated_rotations.as_quat()
		
		return interpolated_quaternion_sequence

	def interpolate_pose(self, pose_sequence):

		# Pose here is an array with Motor angles, Position and orientation. 
		# The dimensions and ordering of these are - 6D motor angles, 4D orientation, then 3D position. 
		times = pose_sequence[:,-1]
		orientations = pose_sequence[:,6:10]
		position_indices = np.concatenate([np.arange(0,6), np.arange(10,13)])
		positions = pose_sequence[:,position_indices]

		# Interpolate positions and orientations. 
		interpolated_positions = self.interpolate_position(valid=times, position_sequence=positions)
		interpolated_orientations = self.interpolate_orientation(valid=times, orientation_sequence=orientations)
		interpolated_poses = np.concatenate([interpolated_positions, interpolated_orientations], axis=1)

		return interpolated_poses

	def downsample_data(self, demonstration, task_index, ds_freq=None):

		if ds_freq is None:
			ds_freq = self.ds_freq[task_index]
		# Downsample each stream.
		number_timepoints = int(demonstration['demo'].shape[0] // ds_freq)

		# for k in demonstration.keys():
		key_list = ['robot-state', 'object-state', 'eef-state', 'demo']
		if self.args.images_in_real_world_dataset:
			key_list.append('images')
		for k in key_list:
			demonstration[k] = resample(demonstration[k], number_timepoints)

	def dictify_data(self, data):

		# Assumes data is from interpolate_pose, which is: Motor Positions, Hand Position, Hand Orientation.
		demonstration = {}
		demonstration['demo'] = data
		demonstration['motor_positions'] = data[:,:6]
		demonstration['hand_pose'] = data[6:10]
		demonstration['hand_orientation'] = data[10:]

		return demonstration

	def filter_demo_outliers(self, demo):

		# Function to remove statistical outliers that arise from the markers being occluded. 
		# Works by: 
		# 	1) Maintaining a moving window average of the data. \\ 
		# 		Rejecting points that deviate from the previous average by a amount larger thna threshold. 
		# 		Window size is = filter_size. 
		# 	2) Maintaing a statistical average of all of the invalid data points \\ 
		# 		(that, for some reason, belong to one mode). Reject points that \\
		#		belong to this mode within a certain threshold. 

		# Copying over data into another variable, so we can rewrite it. 
		alternate_demo = copy.deepcopy(demo)	
		averages = np.zeros_like(demo)

		# Initialize validity, averages, etc. 
		validity = np.ones(demo.shape[0])		
		latest_valid_index = -1 
		averages[0] = alternate_demo.shape[0]

		# Iterating for every timestep in the demonstration. 
		for k in range(1, demo.shape[0]):

			# Check (in)validity. 
			if check_stat_and_diff_invalidity(alternate_demo[k], averages[k-1], k, latest_valid_index):
				# If Invalid, then rewrite current datapoint to previous datapoint. 
				# Assumes this is constant. 
				alternate_demo[k] = copy.deepcopy(alternate_demo[k-1])
				
				# Set validity to invalid. 
				validity[k] = 0
				# The average is now defined as the mean over the previous filter_size - 1 elements, and the current element.. 
				averages[k] = alternate_demo[max(latest_valid_index, k+1-filter_size):k+1].mean(axis=0)
			else:            
				# Special case ofr us encountering the first valid data point afer a stream of invalid data points. 
				if latest_valid_index == -1:
					# Set average to current value.. 
					averages[k] = alt_data[k]
				else:
					# The average is now defined as the mean over the previous filter_size - 1 elements, and the current element.. 
					averages[k] = alternate_demo[max(latest_valid_index, k+1-filter_size):k+1].mean(axis=0)

				# Set the last valid index to current timepoint. 
				latest_valid_index = k
		
		return averages, validity
	
	def interpolate_valid_pose_data(valid_positions, valid_orientations, validity):

		###########################
		# 3) Interpolate valid pose data to invalid timepoints. 
		###########################

		# 3a) The first few elements may not be valid. If so, backfill first few elements with the first valid element. 
		
		first_valid_index = np.where(validity)[0][0]
		last_valid_index = np.where(validity)[0][-1]

		if first_valid_index>0:
			valid_positions[:first_valid_index] = valid_positions[first_valid_index]
			valid_orientations[:valid_orientations] = valid_orientations[first_valid_index]
			validity[:first_valid_index] = 1

		# 3b) For positions, the last elements will always be valid because of the copying mechanism. 
		# This is not true for orientations, so fill last invalid section of orientations as well. 

		if last_valid_index<valid_orientations.shape[0]-1:
			valid_orientations[last_valid_index+1:] = valid_orientations[last_valid_index]
			validity[last_valid_index+1:] = 1

		# 3c) Now that we have valid starts and ends, interpolate the data. 		
		interpolated_positions = self.interpolate_position(valid=validity, position_sequence=valid_positions)
		interpolated_orientations = self.interpolate_orientation(valid=validity, position_sequence=valid_orientations)

		# 3d) Concatenate into Oirentation, Position style pose, so that we maintain consistent ordering for timestep based interpolation. 
		# interpolated_data = np.concatenate([interpolated_positions, interpolated_orientations], axis=-1)
		interpolated_data = np.concatenate([interpolated_orientations, interpolated_positions], axis=-1)
				
		return interpolated_data
	
	def split_demonstration_stream(self, data):
		# Split into Motor Angles, Position, Orientation. 
		return data[:,:6], data[:,10:13], data[:,6:10]

	def process_demonstration(self, raw_data, raw_file_name):

		###########################
		# 1) Throw away irrelevant information. 
		###########################

		# 1a) Throw away first timestep of CSV, because it's just header. 					
		data = raw_data[1:]
		# 1b) Parse into 3 components. 
		motor_data, pos_data, orientation_data = self.split_demonstration_stream(data)

		###########################
		# 2) Filter out outliers based on Pose. 
		###########################
		
		filtered_data, validity = self.filter_demo_outliers(pos_data)

		###########################
		# 3) Interpolate valid pose data to invalid timepoints. 
		###########################
				
		interpolated_data = interpolate_valid_pose_data(valid_positions=filtered_data, \
											 valid_orientations=orientation_data, validity)

		###########################
		# 4) Interpolate valid pose data to invalid timepoints. 
		###########################

		# 4a) First reintergrate into single stream. 
		# Also add last column of data, because this is the list of timesteps.. 
		data = np.concatenate([motor_data, interpolated_data, data[:,-1]], axis=-1)
		
		# # 4b) Now interpolate the data at uniform downsampled frequency.
		# # This will reorder data to Motor Positions, Hand Position, Hand Orientation.
		interpolated_pose = self.interpolate_pose(data)
		
		# Dictify the data. 
		demonstration = self.dictify_data(interpolated_pose)
		demonstration['task_id'] = raw_file_name[:-6]

		return demonstration
		
	def setup(self):
		
		###########################
		# Load data from all tasks.
		###########################
		  			
		data_path = os.path.join(self.dataset_directory, "*.csv")
		self.file_list = sorted(glob.glob(data_path))
		self.files = []

		# For all demos.
		for k, v in enumerate(self.file_list):
			
			print("#######################")
			print("Currently loading file: ", k, " of ", len(self.file_list), " with ID: ", v)
			# Actually load numpy files. 
			raw_data = np.genfromtxt(v, delimiter=',')
			
			# Proces data. 
			demonstration = self.process_demonstration(raw_data=raw_data, raw_file_name=v)

			# Process data. 
			self.files.append(demonstration)

		# For each task, save task_file_list to One numpy. 
		suffix = ""
		task_numpy_path = os.path.join(self.dataset_directory, "New_Task_Demo_Array{0}.npy".format(suffix))
		np.save(task_numpy_path, self.files)

	def __len__(self):
		return self.total_length
	
	def __getitem__(self, index):

		return {}	

class NDAXInterface_Dataset(NDAXInterface_PreDataset):
	
	def __init__(self, args):
		
		super(NDAXInterface_Dataset, self).__init__(args)	

		# Now that we've run setup, compute dataset_trajectory_lengths for smart batching.
		self.dataset_trajectory_lengths = np.zeros(self.total_length)
		for index in range(self.total_length):

			data_element = self.files[index]
			self.dataset_trajectory_lengths[index] = len(data_element['demo'])

		
		# # ######################################################
		# # # Now implementing dataset_trajectory_length_limits. 
		# # ######################################################
		
		# # if self.args.dataset_traj_length_limit>0:
		# # 	# Essentially will need new self.cummulative_num_demos and new .. file index map list things. 
		# # 	# Also will need to set total_length. 

		# # 	self.full_max_length = self.dataset_trajectory_lengths.max()
		# # 	self.full_length = copy.deepcopy(self.total_length)
		# # 	self.full_cummulative_num_demos = copy.deepcopy(self.cummulative_num_demos)
		# # 	self.full_num_demos = copy.deepcopy(self.num_demos)
		# # 	self.full_files = copy.deepcopy(self.files)
		# # 	self.files = [[] for i in range(len(self.task_list))]
		# # 	self.full_dataset_trajectory_lengths = copy.deepcopy(self.dataset_trajectory_lengths)
		# # 	self.dataset_trajectory_lengths = []
		# # 	self.num_demos = np.zeros(len(self.task_list),dtype=int)

		# # 	for index in range(self.full_length):
		# # 		# Get bucket that index falls into based on num_demos array. 
		# # 		task_index = np.searchsorted(self.full_cummulative_num_demos, index, side='right')-1
		# # 		# Get the demo index in this task list. 
		# # 		new_index = index-self.full_cummulative_num_demos[max(task_index,0)]

		# # 		# Check the length of this particular trajectory and its validity. 
		# # 		if (self.full_dataset_trajectory_lengths[index] < self.args.dataset_traj_length_limit):
		# # 			# Add from old list to new. 
		# # 			self.files[task_index].append(self.full_files[task_index][new_index])
		# # 			self.dataset_trajectory_lengths.append(self.full_dataset_trajectory_lengths[index])
		# # 			self.num_demos[task_index] += 1
		# # 		else:
		# # 			pass

		# # 			# Reduce count. 
		# # 			# self.num_demos[task_index] -= 1
					
		# # 			# # Pop item from files. It's still saved in full_files. 					
		# # 			# # self.files[task_index].pop(new_index)
		# # 			# self.files[task_index] = np.delete(self.files[task_index],new_index)
		# # 			# Approach with opposite pattern.. instead of deleting invalid files, add valid ones.
					
		# # 			# # Pop item from dataset_trajectory_lengths. 
		# # 			# self.dataset_trajectory_lengths = np.delete(self.dataset_trajectory_lengths, index)

		# 	# Set new cummulative num demos. 
		# 	self.cummulative_num_demos = self.num_demos.cumsum()
		# 	self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)
		# 	# Set new total length.
		# 	self.total_length = self.cummulative_num_demos[-1]
		# 	# Make array.
		# 	self.dataset_trajectory_lengths = np.array(self.dataset_trajectory_lengths)

		# 	for t in range(len(self.task_list)):
		# 		self.files[t] = np.array(self.files[t])

		# 	# By popping element from files / dataset_traj_lengths, we now don't need to change indexing.
		
	def setup(self):
				
		data_path = os.path.join(self.dataset_directory, "New_Task_Demo_Array.npy")
		self.files = np.load(data_path, allow_pickle=True)		

	def __getitem__(self, index):

		data_element = self.files[index]

		# Embed here
		if 'task-id' not in data_element.keys():
			data_element['task-id'] = copy.deepcopy(data_element['task_id'])

			# print("###############################")
			# print('T_ID:', data_element['task_id'])
			# print('T-ID:', data_element['task-id'])
			task = os.path.split(data_element['task-id'])[-1]
			data_element['task_id'] = self.task_list.index(task)
			data_element['environment-name'] = task		

		if self.args.data in ['NDAXMotorAngles']:
			# Relabel to motor angles.
			data_element['old_demo'] = copy.deepcopy(data_element['demo'])
			data_element['demo'] = data_element['motor_positions']

		# print("Embedding in NDAX DL")
		# embed()

		return data_element
	
	def compute_statistics(self):

		self.state_size = 13
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

		np.save("NDAX_Mean.npy", mean)
		np.save("NDAX_Var.npy", variance)
		np.save("NDAX_Min.npy", min_value)
		np.save("NDAX_Max.npy", max_value)
		np.save("NDAX_Vel_Mean.npy", vel_mean)
		np.save("NDAX_Vel_Var.npy", vel_variance)
		np.save("NDAX_Vel_Min.npy", vel_min_value)
		np.save("NDAX_Vel_Max.npy", vel_max_value)
