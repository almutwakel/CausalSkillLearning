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

def pelvis_norm(relevant_joints_datapoint):
	relevant_joints_datapoint[:, 1:] -= relevant_joints_datapoint[:, 0].reshape(relevant_joints_datapoint.shape[0], 1, 3)
	return relevant_joints_datapoint

def shoulder_norm(relevant_joints_datapoint):
	relevant_joints_datapoint[:, 2:25] -= relevant_joints_datapoint[:, 1].reshape(relevant_joints_datapoint.shape[0], 1, 3)
	if len(relevant_joints_datapoint[0]) > 26:
		relevant_joints_datapoint[:, 27:] -= relevant_joints_datapoint[:, 26].reshape(relevant_joints_datapoint.shape[0], 1, 3)
	return relevant_joints_datapoint

def wrist_norm(relevant_joints_datapoint):
	
	# Normalize with one hand for now.
	relevant_joints_datapoint[:, 1:21] -= relevant_joints_datapoint[:, 0].reshape(relevant_joints_datapoint.shape[0], 1, 3)

	# If we're normalizing for both hands. 
	if len(relevant_joints_datapoint[0]) > 22:
		# Normalize with other hand's wrist. 
		relevant_joints_datapoint[:, 22:] -= relevant_joints_datapoint[:, 21].reshape(relevant_joints_datapoint.shape[0], 1, 3)

		wristless_joints = np.delete(relevant_joints_datapoint, [0, 22], axis=1)
		return wristless_joints
	return relevant_joints_datapoint

def alternate_wrist_norm(relevant_joints_datapoint):
	
	# In this, we are going to normalize the wrist position as well.... 
	# Normalize with one hand for now.
	relevant_joints_datapoint[:, :21] -= relevant_joints_datapoint[:, 0].reshape(relevant_joints_datapoint.shape[0], 1, 3)

	# If we're normalizing for both hands. 
	if len(relevant_joints_datapoint[0]) > 22:
		# Normalize with other hand's wrist. 
		relevant_joints_datapoint[:, 21:] -= relevant_joints_datapoint[:, 21].reshape(relevant_joints_datapoint.shape[0], 1, 3)
		
	return relevant_joints_datapoint



class GRAB_PreDataset(Dataset):

	def __init__(self, args, split='train', short_traj=False, traj_length_threshold=500):

		# Some book-keeping first. 
		self.args = args
		if self.args.datadir is None:
			# self.dataset_directory = '/checkpoint/tanmayshankar/MIME/'
			# self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/MIME/'
			self.dataset_directory = '/data/tanmayshankar/Datasets/GRAB_Joints/'
		else:
			self.dataset_directory = self.args.datadir
		   
		self.stat_dir_name='GRAB'
		# 1) Keep track of joints: 
		#   a) Full joint name list from https://github.com/vchoutas/smplx/blob/master/smplx/joint_names.py. 
		#   b) Relevant joint names. 
		# 2) This lets us subsample relevant joints from full joint name list by indexing...
	
		# Logging all the files we need. 
		self.file_path = os.path.join(self.dataset_directory, '*/*_body_joints.npz')
		self.filelist = sorted(glob.glob(self.file_path))

		# Get number of files. 
		self.total_length = len(self.filelist)

		# Set downsampling frequency.
		self.ds_freq = 16

		# Setup. 
		self.setup()

		self.compute_statistics()

	def set_relevant_joints(self):

		self.joint_names = np.array(['pelvis',
							'left_hip',
							'right_hip',
							'spine1',
							'left_knee',
							'right_knee',
							'spine2',
							'left_ankle',
							'right_ankle',
							'spine3',
							'left_foot',
							'right_foot',
							'neck',
							'left_collar',
							'right_collar',
							'head',
							'left_shoulder',
							'right_shoulder',
							'left_elbow',
							'right_elbow',
							'left_wrist',
							'right_wrist',
							'jaw',
							'left_eye_smplhf',
							'right_eye_smplhf',
							'left_index1',
							'left_index2',
							'left_index3',
							'left_middle1',
							'left_middle2',
							'left_middle3',
							'left_pinky1',
							'left_pinky2',
							'left_pinky3',
							'left_ring1',
							'left_ring2',
							'left_ring3',
							'left_thumb1',
							'left_thumb2',
							'left_thumb3',
							'right_index1',
							'right_index2',
							'right_index3',
							'right_middle1',
							'right_middle2',
							'right_middle3',
							'right_pinky1',
							'right_pinky2',
							'right_pinky3',
							'right_ring1',
							'right_ring2',
							'right_ring3',
							'right_thumb1',
							'right_thumb2',
							'right_thumb3',
							'nose',
							'right_eye',
							'left_eye',
							'right_ear',
							'left_ear',
							'left_big_toe',
							'left_small_toe',
							'left_heel',
							'right_big_toe',
							'right_small_toe',
							'right_heel',
							'left_thumb',
							'left_index',
							'left_middle',
							'left_ring',
							'left_pinky',
							'right_thumb',
							'right_index',
							'right_middle',
							'right_ring',
							'right_pinky',
							'right_eye_brow1',
							'right_eye_brow2',
							'right_eye_brow3',
							'right_eye_brow4',
							'right_eye_brow5',
							'left_eye_brow5',
							'left_eye_brow4',
							'left_eye_brow3',
							'left_eye_brow2',
							'left_eye_brow1',
							'nose1',
							'nose2',
							'nose3',
							'nose4',
							'right_nose_2',
							'right_nose_1',
							'nose_middle',
							'left_nose_1',
							'left_nose_2',
							'right_eye1',
							'right_eye2',
							'right_eye3',
							'right_eye4',
							'right_eye5',
							'right_eye6',
							'left_eye4',
							'left_eye3',
							'left_eye2',
							'left_eye1',
							'left_eye6',
							'left_eye5',
							'right_mouth_1',
							'right_mouth_2',
							'right_mouth_3',
							'mouth_top',
							'left_mouth_3',
							'left_mouth_2',
							'left_mouth_1',
							'left_mouth_5',  # 59 in OpenPose output
							'left_mouth_4',  # 58 in OpenPose output
							'mouth_bottom',
							'right_mouth_4',
							'right_mouth_5',
							'right_lip_1',
							'right_lip_2',
							'lip_top',
							'left_lip_2',
							'left_lip_1',
							'left_lip_3',
							'lip_bottom',
							'right_lip_3'])

		self.arm_joint_names = np.array(['pelvis',
							'left_collar',
							'right_collar',
							'left_shoulder',
							'right_shoulder',
							'left_elbow',
							'right_elbow',
							'left_wrist',
							'right_wrist'])

		self.arm_and_hand_joint_names = np.array(['pelvis',
							'left_collar',
							'right_collar',
							'left_shoulder',
							'right_shoulder',
							'left_elbow',
							'right_elbow',
							'left_wrist',
							'right_wrist',
							'left_index1',
							'left_index2',
							'left_index3',
							'left_middle1',
							'left_middle2',
							'left_middle3',
							'left_pinky1',
							'left_pinky2',
							'left_pinky3',
							'left_ring1',
							'left_ring2',
							'left_ring3',
							'left_thumb1',
							'left_thumb2',
							'left_thumb3',
							'right_index1',
							'right_index2',
							'right_index3',
							'right_middle1',
							'right_middle2',
							'right_middle3',
							'right_pinky1',
							'right_pinky2',
							'right_pinky3',
							'right_ring1',
							'right_ring2',
							'right_ring3',
							'right_thumb1',
							'right_thumb2',
							'right_thumb3',
							'left_thumb',
							'left_index',
							'left_middle',
							'left_ring',
							'left_pinky',
							'right_thumb',
							'right_index',
							'right_middle',
							'right_ring',
							'right_pinky'])

		# Create index arrays
		self.arm_joint_indices = np.zeros(len(self.arm_joint_names))
		self.arm_and_hand_joint_indices = np.zeros(len(self.arm_and_hand_joint_names))
		self.object_indices = np.zeros(6)


		for k, v in enumerate(self.arm_joint_names):			
			self.arm_joint_indices[k] = np.where(self.joint_names==v)[0][0]
		
		for k, v in enumerate(self.arm_and_hand_joint_indices):
			self.arm_and_hand_joint_indices[k] = np.where(self.joint_names==v)[0][0]

		# for i in range(self.object_indices.shape[0]):
			# self.object_indices[i] = len(self.arm_and_hand_joint_indices) + i

		# self.arm_hand_object_indices = self.arm_and_hand_joint_indices

		# Append zeros for object indices and fill when they're loaded
		self.arm_hand_object_indices = np.zeros(len(self.arm_and_hand_joint_names) + 6)
		self.arm_hand_object_indices[0:len(self.arm_and_hand_joint_names)] = self.arm_and_hand_joint_indices

		# self.arm_hand_object_indices[-6:] = self.object_indices
		
		
	def subsample_relevant_joints(self, datapoint):

		# Remember, the datapoint is going to be of the form.. 
		# Timesteps x Joints x 3 (dimensions). 
		# Index into it as: 

		# Figure out whether to use full hands, or just use the arm positions. 
		# Consider unsupervised translation to robots without articulated grippers. 
		# For now use arm joint indices. 
		# We can later consider adding other robots / hands.
		self.relevant_joint_indices = self.arm_joint_indices.astype(int)

		return datapoint[:, self.relevant_joint_indices]
		
	def setup(self):

		# Load all files.. 
		self.files = []
		self.dataset_trajectory_lengths = np.zeros(self.total_length)
		
		# set joints
		self.set_relevant_joints()

		# For all files. 
		for k, v in enumerate(self.filelist):
						
			if k%100==0:
				print("Loading file: ",k)

			# Now actually load file. 
			datapoint = np.load(v, allow_pickle=True)['body_joints']	

			if 'Object' in self.args.data:
				# Get object filepath
				object_path = v.replace("GRAB_Joints", "grab").replace("_body_joints.npz", ".npz")

				# Load object data
				object_dict_raw = np.load(object_path, allow_pickle=True)
				object_dict = object_dict_raw['object'].flatten()[0]
				object_transl = object_dict['params']['transl']
				object_orient = object_dict['params']['global_orient']

				object_datapoint = np.concatenate((object_transl, object_orient), axis=1)

		# Without normalizing object:
			# Subsample relevant joints. 
			relevant_joints_datapoint = self.subsample_relevant_joints(datapoint)

			# Normalize using the pelvis joint (i.e. the first joint).
			normalized_relevant_joint_datapoint = self.normalize(relevant_joints_datapoint)

			# Reshape. 
			reshaped_normalized_datapoint = normalized_relevant_joint_datapoint.reshape(normalized_relevant_joint_datapoint.shape[0],-1)

			# Combine object + body joints
			if 'Object' in self.args.data:
				# reshaped_normalized_datapoint = np.concatenate((reshaped_normalized_datapoint, object_datapoint), axis=1)
				reshaped_normalized_datapoint[:, -6:] = object_datapoint

			self.state_size = reshaped_normalized_datapoint.shape[1]
			# Subsample in time. 
			number_of_timesteps = datapoint.shape[0]//self.ds_freq
			# subsampled_data = resample(relevant_joints_datapoint, number_of_timesteps)
			subsampled_data = resample(reshaped_normalized_datapoint, number_of_timesteps)
			
			# Add subsampled datapoint to file. 
			self.files.append(subsampled_data)            

		# Create array. 
		self.file_array = np.array(self.files)

		# Now save this file.
		# np.save(os.path.join(self.dataset_directory,"GRAB_DataFile.npy"), self.file_array)
		np.save(os.path.join(self.dataset_directory, self.getname() + "_DataFile_BaseNormalize.npy"), self.file_array)
		np.save(os.path.join(self.dataset_directory, self.getname() + "_OrderedFileList.npy"), self.filelist)

	def get_state_size(self):
		return self.state_size

	def normalize(self, relevant_joints_datapoint):
		if len(relevant_joints_datapoint) == 0:
			return relevant_joints_datapoint
		if self.args.position_normalization == 'pelvis':
			return pelvis_norm(relevant_joints_datapoint)
		elif self.args.position_normalization == 'shoulder':
			return shoulder_norm(relevant_joints_datapoint)
		elif self.args.position_normalization == 'wrist':
			return wrist_norm(relevant_joints_datapoint)
		else:
			return relevant_joints_datapoint

	def getname(self):
		return "GRAB"

	def __len__(self):
		return self.total_length

	def __getitem__(self, index):
		
		if isinstance(index, np.ndarray):
			return list(self.file_array[index])
		else:
			return self.file_array[index]

	def compute_statistics(self):

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
			data_element = {}
			data_element['is_valid'] = True
			data_element['demo'] = self.file_array[i]
			data_element['file'] = self.filelist[i]

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
			data_element = {}
			data_element['is_valid'] = True
			data_element['demo'] = self.file_array[i]
			data_element['file'] = self.filelist[i]
			
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

		statdir = "Statistics/" + self.getname()

		np.save(os.path.join(statdir, self.getname() + "_Mean.npy"), mean)
		np.save(os.path.join(statdir, self.getname() + "_Var.npy"), variance)
		np.save(os.path.join(statdir, self.getname() + "_Min.npy"), min_value)
		np.save(os.path.join(statdir, self.getname() + "_Max.npy"), max_value)
		np.save(os.path.join(statdir, self.getname() + "_Vel_Mean.npy"), vel_mean)
		np.save(os.path.join(statdir, self.getname() + "_Vel_Var.npy"), vel_variance)
		np.save(os.path.join(statdir, self.getname() + "_Vel_Min.npy"), vel_min_value)
		np.save(os.path.join(statdir, self.getname() + "_Vel_Max.npy"), vel_max_value)

class GRAB_Dataset(Dataset):

	def __init__(self, args):

		# Some book-keeping first. 
		self.args = args
		self.stat_dir_name='GRAB'
		if self.args.datadir is None:
			# self.dataset_directory = '/checkpoint/tanmayshankar/MIME/'
			# self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/MIME/'
			self.dataset_directory = '/data/tanmayshankar/Datasets/GRAB_Joints/'
		else:
			self.dataset_directory = self.args.datadir
		   
		# Load file.
		self.data_list = np.load(os.path.join(self.dataset_directory, self.getname() + "_DataFile_BaseNormalize.npy"), allow_pickle=True)
		self.filelist = np.load(os.path.join(self.dataset_directory, self.getname() + "_OrderedFileList.npy"), allow_pickle=True)

		self.dataset_length = len(self.data_list)

		if self.args.dataset_traj_length_limit>0:			
			self.short_data_list = []
			self.short_file_list = []
			self.dataset_trajectory_lengths = []
			for i in range(self.dataset_length):
				if self.data_list[i].shape[0]<self.args.dataset_traj_length_limit:
					self.short_data_list.append(self.data_list[i])
					self.short_file_list.append(self.filelist[i])
					self.dataset_trajectory_lengths.append(self.data_list[i].shape[0])

			self.data_list = self.short_data_list
			self.filelist = self.short_file_list
			self.dataset_length = len(self.data_list)
			self.dataset_trajectory_lengths = np.array(self.dataset_trajectory_lengths)
				
		self.data_list_array = np.array(self.data_list)

	def get_state_size(self):
		if self.data_list is None or len(self.data_list)==0:
			raise ValueError("Data list is empty. Cannot get state size.")
		return self.data_list_array[0].shape[1]		

	def getname(self):
		return "GRAB"

	def __len__(self):
		# Return length of file list. 
		return self.dataset_length

	def __getitem__(self, index):
		# Return n'th item of dataset.
		# This has already processed everything.

		# if isinstance(index,np.ndarray):			
		# 	return list(self.data_list_array[index])
		# else:
		# 	return self.data_list[index]

		data_element = {}
		data_element['is_valid'] = True
		data_element['demo'] = self.data_list[index]
		data_element['file'] = self.filelist[index]
		data_element['task-id'] = index	

		return data_element

class GRABArmHand_Dataset(GRAB_Dataset):

	def __init__(self, args):
		super(GRABArmHand_Dataset, self).__init__(args=args)
		self.stat_dir_name="GRABArmHand"


	def subsample_relevant_joints(self, datapoint):

		self.relevant_joint_indices = self.arm_and_hand_joint_indices.astype(int)

		return datapoint[:, self.relevant_joint_indices]

	def getname(self):
		return "GRABArmHand"

class GRABArmHand_PreDataset(GRAB_PreDataset):

	def __init__(self, args, split='train', short_traj=False, traj_length_threshold=500):
		super(GRABArmHand_PreDataset, self).__init__(args, split=split, short_traj=short_traj, traj_length_threshold=traj_length_threshold)
		self.stat_dir_name="GRABArmHand"

	def set_relevant_joints(self):
		self.joint_names = np.array(['pelvis',
									 'left_hip',
									 'right_hip',
									 'spine1',
									 'left_knee',
									 'right_knee',
									 'spine2',
									 'left_ankle',
									 'right_ankle',
									 'spine3',
									 'left_foot',
									 'right_foot',
									 'neck',
									 'left_collar',
									 'right_collar',
									 'head',
									 'left_shoulder',
									 'right_shoulder',
									 'left_elbow',
									 'right_elbow',
									 'left_wrist',
									 'right_wrist',
									 'jaw',
									 'left_eye_smplhf',
									 'right_eye_smplhf',
									 'left_index1',
									 'left_index2',
									 'left_index3',
									 'left_middle1',
									 'left_middle2',
									 'left_middle3',
									 'left_pinky1',
									 'left_pinky2',
									 'left_pinky3',
									 'left_ring1',
									 'left_ring2',
									 'left_ring3',
									 'left_thumb1',
									 'left_thumb2',
									 'left_thumb3',
									 'right_index1',
									 'right_index2',
									 'right_index3',
									 'right_middle1',
									 'right_middle2',
									 'right_middle3',
									 'right_pinky1',
									 'right_pinky2',
									 'right_pinky3',
									 'right_ring1',
									 'right_ring2',
									 'right_ring3',
									 'right_thumb1',
									 'right_thumb2',
									 'right_thumb3',
									 'nose',
									 'right_eye',
									 'left_eye',
									 'right_ear',
									 'left_ear',
									 'left_big_toe',
									 'left_small_toe',
									 'left_heel',
									 'right_big_toe',
									 'right_small_toe',
									 'right_heel',
									 'left_thumb',
									 'left_index',
									 'left_middle',
									 'left_ring',
									 'left_pinky',
									 'right_thumb',
									 'right_index',
									 'right_middle',
									 'right_ring',
									 'right_pinky',
									 'right_eye_brow1',
									 'right_eye_brow2',
									 'right_eye_brow3',
									 'right_eye_brow4',
									 'right_eye_brow5',
									 'left_eye_brow5',
									 'left_eye_brow4',
									 'left_eye_brow3',
									 'left_eye_brow2',
									 'left_eye_brow1',
									 'nose1',
									 'nose2',
									 'nose3',
									 'nose4',
									 'right_nose_2',
									 'right_nose_1',
									 'nose_middle',
									 'left_nose_1',
									 'left_nose_2',
									 'right_eye1',
									 'right_eye2',
									 'right_eye3',
									 'right_eye4',
									 'right_eye5',
									 'right_eye6',
									 'left_eye4',
									 'left_eye3',
									 'left_eye2',
									 'left_eye1',
									 'left_eye6',
									 'left_eye5',
									 'right_mouth_1',
									 'right_mouth_2',
									 'right_mouth_3',
									 'mouth_top',
									 'left_mouth_3',
									 'left_mouth_2',
									 'left_mouth_1',
									 'left_mouth_5',  # 59 in OpenPose output
									 'left_mouth_4',  # 58 in OpenPose output
									 'mouth_bottom',
									 'right_mouth_4',
									 'right_mouth_5',
									 'right_lip_1',
									 'right_lip_2',
									 'lip_top',
									 'left_lip_2',
									 'left_lip_1',
									 'left_lip_3',
									 'lip_bottom',
									 'right_lip_3'])

		self.arm_and_hand_joint_names = np.array([ 'pelvis',
												'left_shoulder', # index 0
												'left_elbow',
												'left_collar',
												'left_wrist', 
												'left_index1',
												'left_index2',
												'left_index3',
												'left_middle1',
												'left_middle2',
												'left_middle3',
												'left_pinky1',
												'left_pinky2',
												'left_pinky3',
												'left_ring1',
												'left_ring2',
												'left_ring3',
												'left_thumb1',
												'left_thumb2',
												'left_thumb3',
												'left_thumb',
												'left_index',
												'left_middle',
												'left_ring',
												'left_pinky',
												'right_shoulder', # index 24
												'right_elbow',
												'right_collar',
												'right_wrist',
												'right_index1',
												'right_index2',
												'right_index3',
												'right_middle1',
												'right_middle2',
												'right_middle3',
												'right_pinky1',
												'right_pinky2',
												'right_pinky3',
												'right_ring1',
												'right_ring2',
												'right_ring3',
												'right_thumb1',
												'right_thumb2',
												'right_thumb3',
												'right_thumb',
												'right_index',
												'right_middle',
												'right_ring',
												'right_pinky'])

		self.left_arm_and_hand_joint_names = np.array([ 'pelvis',
												'left_shoulder',
												'left_elbow',
												'left_collar',
												'left_wrist', 
												'left_index1',
												'left_index2',
												'left_index3',
												'left_middle1',
												'left_middle2',
												'left_middle3',
												'left_pinky1',
												'left_pinky2',
												'left_pinky3',
												'left_ring1',
												'left_ring2',
												'left_ring3',
												'left_thumb1',
												'left_thumb2',
												'left_thumb3',
												'left_thumb',
												'left_index',
												'left_middle',
												'left_ring',
												'left_pinky'])

		self.right_arm_and_hand_joint_names = np.array([ 'pelvis',
												'right_shoulder',
												'right_elbow',
												'right_collar',
												'right_wrist',
												'right_index1',
												'right_index2',
												'right_index3',
												'right_middle1',
												'right_middle2',
												'right_middle3',
												'right_pinky1',
												'right_pinky2',
												'right_pinky3',
												'right_ring1',
												'right_ring2',
												'right_ring3',
												'right_thumb1',
												'right_thumb2',
												'right_thumb3',
												'right_thumb',
												'right_index',
												'right_middle',
												'right_ring',
												'right_pinky'])

		# Create index arrays
		if self.args.single_hand == "left":
			selection = self.left_arm_and_hand_joint_names
		elif self.args.single_hand == "right":
			selection = self.right_arm_and_hand_joint_names
		else:
			selection = self.arm_and_hand_joint_names

		self.arm_and_hand_joint_indices = np.zeros(len(selection))

		for k, v in enumerate(selection):
			self.arm_and_hand_joint_indices[k] = np.where(self.joint_names==v)[0][0]

	def subsample_relevant_joints(self, datapoint):

		self.relevant_joint_indices = self.arm_and_hand_joint_indices.astype(int)

		return datapoint[:, self.relevant_joint_indices]
	
	def normalize(self, relevant_joints_datapoint):
		joints = super().normalize(relevant_joints_datapoint)
		# Remove pelvis from joints, regardless of normalization type
		return joints[1:]

	def getname(self):
		return "GRABArmHand"

class GRABHand_Dataset(GRAB_Dataset):

	def __init__(self, args):
		super(GRABHand_Dataset, self).__init__(args=args)
		self.stat_dir_name='GRABHand'

	def subsample_relevant_joints(self, datapoint):

		self.relevant_joint_indices = self.hand_joint_indices.astype(int)

		return datapoint[:, self.relevant_joint_indices]

	def getname(self):
		return "GRABHand"

	def __getitem__(self, index):

		data_element = super().__getitem__(index)

		# print("Embedding in get item")
		# embed()
		# Zero out the wrist position for all timesteps..
		if self.args.skip_wrist:
			data_element['demo'][:, :3] = 0.

		return data_element

class GRABHand_PreDataset(GRAB_PreDataset):

	def __init__(self, args):
		super(GRABHand_PreDataset, self).__init__(args)
		self.stat_dir_name='GRABHand'

	def set_relevant_joints(self):
		self.joint_names = np.array(['pelvis',
									 'left_hip',
									 'right_hip',
									 'spine1',
									 'left_knee',
									 'right_knee',
									 'spine2',
									 'left_ankle',
									 'right_ankle',
									 'spine3',
									 'left_foot',
									 'right_foot',
									 'neck',
									 'left_collar',
									 'right_collar',
									 'head',
									 'left_shoulder',
									 'right_shoulder',
									 'left_elbow',
									 'right_elbow',
									 'left_wrist',
									 'right_wrist',
									 'jaw',
									 'left_eye_smplhf',
									 'right_eye_smplhf',
									 'left_index1',
									 'left_index2',
									 'left_index3',
									 'left_middle1',
									 'left_middle2',
									 'left_middle3',
									 'left_pinky1',
									 'left_pinky2',
									 'left_pinky3',
									 'left_ring1',
									 'left_ring2',
									 'left_ring3',
									 'left_thumb1',
									 'left_thumb2',
									 'left_thumb3',
									 'right_index1',
									 'right_index2',
									 'right_index3',
									 'right_middle1',
									 'right_middle2',
									 'right_middle3',
									 'right_pinky1',
									 'right_pinky2',
									 'right_pinky3',
									 'right_ring1',
									 'right_ring2',
									 'right_ring3',
									 'right_thumb1',
									 'right_thumb2',
									 'right_thumb3',
									 'nose',
									 'right_eye',
									 'left_eye',
									 'right_ear',
									 'left_ear',
									 'left_big_toe',
									 'left_small_toe',
									 'left_heel',
									 'right_big_toe',
									 'right_small_toe',
									 'right_heel',
									 'left_thumb',
									 'left_index',
									 'left_middle',
									 'left_ring',
									 'left_pinky',
									 'right_thumb',
									 'right_index',
									 'right_middle',
									 'right_ring',
									 'right_pinky',
									 'right_eye_brow1',
									 'right_eye_brow2',
									 'right_eye_brow3',
									 'right_eye_brow4',
									 'right_eye_brow5',
									 'left_eye_brow5',
									 'left_eye_brow4',
									 'left_eye_brow3',
									 'left_eye_brow2',
									 'left_eye_brow1',
									 'nose1',
									 'nose2',
									 'nose3',
									 'nose4',
									 'right_nose_2',
									 'right_nose_1',
									 'nose_middle',
									 'left_nose_1',
									 'left_nose_2',
									 'right_eye1',
									 'right_eye2',
									 'right_eye3',
									 'right_eye4',
									 'right_eye5',
									 'right_eye6',
									 'left_eye4',
									 'left_eye3',
									 'left_eye2',
									 'left_eye1',
									 'left_eye6',
									 'left_eye5',
									 'right_mouth_1',
									 'right_mouth_2',
									 'right_mouth_3',
									 'mouth_top',
									 'left_mouth_3',
									 'left_mouth_2',
									 'left_mouth_1',
									 'left_mouth_5',  # 59 in OpenPose output
									 'left_mouth_4',  # 58 in OpenPose output
									 'mouth_bottom',
									 'right_mouth_4',
									 'right_mouth_5',
									 'right_lip_1',
									 'right_lip_2',
									 'lip_top',
									 'left_lip_2',
									 'left_lip_1',
									 'left_lip_3',
									 'lip_bottom',
									 'right_lip_3'])

		self.hand_joint_names = np.array(['left_wrist', 
										  'left_index1',
										  'left_index2',
										  'left_index3',
										  'left_middle1',
										  'left_middle2',
										  'left_middle3',
										  'left_pinky1',
										  'left_pinky2',
										  'left_pinky3',
										  'left_ring1',
										  'left_ring2',
										  'left_ring3',
										  'left_thumb1',
										  'left_thumb2',
										  'left_thumb3',
										  'left_thumb',
										  'left_index',
										  'left_middle',
										  'left_ring',
										  'left_pinky',
										  'right_wrist',  # index 21
										  'right_index1',
										  'right_index2',
										  'right_index3',
										  'right_middle1',
										  'right_middle2',
										  'right_middle3',
										  'right_pinky1',
										  'right_pinky2',
										  'right_pinky3',
										  'right_ring1',
										  'right_ring2',
										  'right_ring3',
										  'right_thumb1',
										  'right_thumb2',
										  'right_thumb3',
										  'right_thumb',
										  'right_index',
										  'right_middle',
										  'right_ring',
										  'right_pinky'])

		self.left_hand_joint_names = np.array(['left_wrist',
										  'left_index1',
										  'left_index2',
										  'left_index3',
										  'left_middle1',
										  'left_middle2',
										  'left_middle3',
										  'left_pinky1',
										  'left_pinky2',
										  'left_pinky3',
										  'left_ring1',
										  'left_ring2',
										  'left_ring3',
										  'left_thumb1',
										  'left_thumb2',
										  'left_thumb3',
										  'left_thumb',
										  'left_index',
										  'left_middle',
										  'left_ring',
										  'left_pinky'])

		self.right_hand_joint_names = np.array(['right_wrist',
										  'right_index1',
										  'right_index2',
										  'right_index3',
										  'right_middle1',
										  'right_middle2',
										  'right_middle3',
										  'right_pinky1',
										  'right_pinky2',
										  'right_pinky3',
										  'right_ring1',
										  'right_ring2',
										  'right_ring3',
										  'right_thumb1',
										  'right_thumb2',
										  'right_thumb3',
										  'right_thumb',
										  'right_index',
										  'right_middle',
										  'right_ring',
										  'right_pinky'])

		# Create index arrays
		if self.args.single_hand == "left":
			selection = self.left_hand_joint_names
		elif self.args.single_hand == "right":
			selection = self.right_hand_joint_names
		else:
			selection = self.hand_joint_names

		self.hand_joint_indices = np.zeros(len(selection))

		for k, v in enumerate(selection):
			self.hand_joint_indices[k] = np.where(self.joint_names==v)[0][0]

	def subsample_relevant_joints(self, datapoint):

		self.relevant_joint_indices = self.hand_joint_indices.astype(int)

		return datapoint[:, self.relevant_joint_indices]
	
	def getname(self):
		return "GRABHand"



class GRABArmHandObject_Dataset(GRAB_Dataset):

	def __init__(self, args):
		super(GRABArmHandObject_Dataset, self).__init__(args=args)
		self.stat_dir_name="GRABArmHandObject"


	def subsample_relevant_joints(self, datapoint):

		self.relevant_joint_indices = self.arm_hand_object_indices.astype(int)

		return datapoint[:, self.relevant_joint_indices]

	def getname(self):
		return "GRABArmHandObject"

class GRABArmHandObject_PreDataset(GRAB_PreDataset):

	def __init__(self, args, split='train', short_traj=False, traj_length_threshold=500):
		super(GRABArmHandObject_PreDataset, self).__init__(args, split=split, short_traj=short_traj, traj_length_threshold=traj_length_threshold)
		self.stat_dir_name="GRABArmHandObject"

	def set_relevant_joints(self):
		self.joint_names = np.array(['pelvis',
									 'left_hip',
									 'right_hip',
									 'spine1',
									 'left_knee',
									 'right_knee',
									 'spine2',
									 'left_ankle',
									 'right_ankle',
									 'spine3',
									 'left_foot',
									 'right_foot',
									 'neck',
									 'left_collar',
									 'right_collar',
									 'head',
									 'left_shoulder',
									 'right_shoulder',
									 'left_elbow',
									 'right_elbow',
									 'left_wrist',
									 'right_wrist',
									 'jaw',
									 'left_eye_smplhf',
									 'right_eye_smplhf',
									 'left_index1',
									 'left_index2',
									 'left_index3',
									 'left_middle1',
									 'left_middle2',
									 'left_middle3',
									 'left_pinky1',
									 'left_pinky2',
									 'left_pinky3',
									 'left_ring1',
									 'left_ring2',
									 'left_ring3',
									 'left_thumb1',
									 'left_thumb2',
									 'left_thumb3',
									 'right_index1',
									 'right_index2',
									 'right_index3',
									 'right_middle1',
									 'right_middle2',
									 'right_middle3',
									 'right_pinky1',
									 'right_pinky2',
									 'right_pinky3',
									 'right_ring1',
									 'right_ring2',
									 'right_ring3',
									 'right_thumb1',
									 'right_thumb2',
									 'right_thumb3',
									 'nose',
									 'right_eye',
									 'left_eye',
									 'right_ear',
									 'left_ear',
									 'left_big_toe',
									 'left_small_toe',
									 'left_heel',
									 'right_big_toe',
									 'right_small_toe',
									 'right_heel',
									 'left_thumb',
									 'left_index',
									 'left_middle',
									 'left_ring',
									 'left_pinky',
									 'right_thumb',
									 'right_index',
									 'right_middle',
									 'right_ring',
									 'right_pinky',
									 'right_eye_brow1',
									 'right_eye_brow2',
									 'right_eye_brow3',
									 'right_eye_brow4',
									 'right_eye_brow5',
									 'left_eye_brow5',
									 'left_eye_brow4',
									 'left_eye_brow3',
									 'left_eye_brow2',
									 'left_eye_brow1',
									 'nose1',
									 'nose2',
									 'nose3',
									 'nose4',
									 'right_nose_2',
									 'right_nose_1',
									 'nose_middle',
									 'left_nose_1',
									 'left_nose_2',
									 'right_eye1',
									 'right_eye2',
									 'right_eye3',
									 'right_eye4',
									 'right_eye5',
									 'right_eye6',
									 'left_eye4',
									 'left_eye3',
									 'left_eye2',
									 'left_eye1',
									 'left_eye6',
									 'left_eye5',
									 'right_mouth_1',
									 'right_mouth_2',
									 'right_mouth_3',
									 'mouth_top',
									 'left_mouth_3',
									 'left_mouth_2',
									 'left_mouth_1',
									 'left_mouth_5',  # 59 in OpenPose output
									 'left_mouth_4',  # 58 in OpenPose output
									 'mouth_bottom',
									 'right_mouth_4',
									 'right_mouth_5',
									 'right_lip_1',
									 'right_lip_2',
									 'lip_top',
									 'left_lip_2',
									 'left_lip_1',
									 'left_lip_3',
									 'lip_bottom',
									 'right_lip_3'])

		self.arm_and_hand_joint_names = np.array([ 'pelvis',
												'left_shoulder', # index 0
												'left_elbow',
												'left_collar',
												'left_wrist', 
												'left_index1',
												'left_index2',
												'left_index3',
												'left_middle1',
												'left_middle2',
												'left_middle3',
												'left_pinky1',
												'left_pinky2',
												'left_pinky3',
												'left_ring1',
												'left_ring2',
												'left_ring3',
												'left_thumb1',
												'left_thumb2',
												'left_thumb3',
												'left_thumb',
												'left_index',
												'left_middle',
												'left_ring',
												'left_pinky',
												'right_shoulder', # index 24
												'right_elbow',
												'right_collar',
												'right_wrist',
												'right_index1',
												'right_index2',
												'right_index3',
												'right_middle1',
												'right_middle2',
												'right_middle3',
												'right_pinky1',
												'right_pinky2',
												'right_pinky3',
												'right_ring1',
												'right_ring2',
												'right_ring3',
												'right_thumb1',
												'right_thumb2',
												'right_thumb3',
												'right_thumb',
												'right_index',
												'right_middle',
												'right_ring',
												'right_pinky'])

		self.left_arm_and_hand_joint_names = np.array([ 'pelvis',
												'left_shoulder',
												'left_elbow',
												'left_collar',
												'left_wrist', 
												'left_index1',
												'left_index2',
												'left_index3',
												'left_middle1',
												'left_middle2',
												'left_middle3',
												'left_pinky1',
												'left_pinky2',
												'left_pinky3',
												'left_ring1',
												'left_ring2',
												'left_ring3',
												'left_thumb1',
												'left_thumb2',
												'left_thumb3',
												'left_thumb',
												'left_index',
												'left_middle',
												'left_ring',
												'left_pinky'])

		self.right_arm_and_hand_joint_names = np.array([ 'pelvis',
												'right_shoulder',
												'right_elbow',
												'right_collar',
												'right_wrist',
												'right_index1',
												'right_index2',
												'right_index3',
												'right_middle1',
												'right_middle2',
												'right_middle3',
												'right_pinky1',
												'right_pinky2',
												'right_pinky3',
												'right_ring1',
												'right_ring2',
												'right_ring3',
												'right_thumb1',
												'right_thumb2',
												'right_thumb3',
												'right_thumb',
												'right_index',
												'right_middle',
												'right_ring',
												'right_pinky'])

		# Create index arrays
		if self.args.single_hand == "left":
			selection = self.left_arm_and_hand_joint_names
		elif self.args.single_hand == "right":
			selection = self.right_arm_and_hand_joint_names
		else:
			selection = self.arm_and_hand_joint_names

		self.arm_and_hand_joint_indices = np.zeros(len(selection))
		self.object_indices = np.zeros(6)

		for k, v in enumerate(selection):
			self.arm_and_hand_joint_indices[k] = np.where(self.joint_names==v)[0][0]

		for i in range(self.object_indices.shape[0]):
			self.object_indices[i] = len(self.arm_and_hand_joint_names) + i


		self.arm_hand_object_indices = np.zeros(len(self.arm_and_hand_joint_indices) + 6)
		self.arm_hand_object_indices[0:len(self.arm_and_hand_joint_indices)] = self.arm_and_hand_joint_indices
		self.arm_hand_object_indices[-6:] = self.object_indices
		
		

	def subsample_relevant_joints(self, datapoint):

		self.relevant_joint_indices = self.arm_hand_object_indices.astype(int)

		return datapoint[:, self.relevant_joint_indices]

	def getname(self):
		return "GRABArmHandObject"
	


class GRABObject_Dataset(GRAB_Dataset):

	def __init__(self, args):
		super(GRABObject_Dataset, self).__init__(args=args)
		self.stat_dir_name="GRABObject"


	def subsample_relevant_joints(self, datapoint):

		self.relevant_joint_indices = self.object_indices.astype(int)

		return datapoint[:, self.relevant_joint_indices]

	def getname(self):
		return "GRABObject"

class GRABObject_PreDataset(GRAB_PreDataset):

	def __init__(self, args, split='train', short_traj=False, traj_length_threshold=500):
		super(GRABObject_PreDataset, self).__init__(args, split=split, short_traj=short_traj, traj_length_threshold=traj_length_threshold)
		self.stat_dir_name="GRABObject"

	def subsample_relevant_joints(self, datapoint):

		self.relevant_joint_indices = self.object_indices.astype(int)

		return datapoint[:, self.relevant_joint_indices]

	def getname(self):
		return "GRABObject"