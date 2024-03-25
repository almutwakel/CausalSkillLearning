from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from headers import *

def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]

class AICaringExerciseDatasetV1(Dataset):
	
    def __init__(self, args):
          
        # Book-keeping.
        self.args = args

        self.stat_dir_name = "AICaring_v1" 

        # self.task_list = ['BicepCurl_Good', 'BicepCurl_LowRange', 'BicepCurl_HighRange', 'LateralRaise_Good', 'LateralRaise_LowRange', 'LateralRaise_HighRange']
        self.task_list = ['BicepCurl_Good', 'BicepCurl_LowRange', 'LateralRaise_Good', 'LateralRaise_LowRange', 'LateralRaise_HighRange']
        self.exercise_list = ['BicepCurl', 'BicepCurl', 'LateralRaise', 'LateralRaise', 'LateralRaise']
        self.performance_level = ['Good', 'LowRange', 'Good', 'LowRange', 'HighRange'] 
        # self.number_demos = np.zeros(6)
        # self.number_demos = np.zeros(5)

        self.setup()

    def setup(self):
         
        file_path = os.path.join(self.args.datadir, "new_experts_smaller.pickle")        
        self.original_data = np.load(file_path, allow_pickle=True)

        reformatted_data = []
        
        exercise_list = list(self.original_data.keys())
        levels = list(self.original_data[exercise_list[0]].keys())

        self.dataset_trajectory_lengths = []
        self.number_demos = []

        c = 0 
        for k, ex in enumerate(exercise_list):
            for j, level in enumerate(levels):
                                
                # print("Exercise: ", ex, "Level: ", level, " Number Demos:", nd)            
                for i, demo in enumerate(self.original_data[ex][level]):
                     self.dataset_trajectory_lengths.append(len(demo))
                     c+=1   

                if not(ex=='bicep_curls' and level=='high range'):
                     reformatted_data.append(self.original_data[ex][level])                     
                     self.number_demos.append(len(self.original_data[ex][level]))

        self.dataset_trajectory_lengths = np.array(self.dataset_trajectory_lengths)
        self.number_demos = np.array(self.number_demos)
        self.files = reformatted_data
        self.cummulative_num_demos = self.number_demos.cumsum()
        self.cummulative_num_demos = np.insert(self.cummulative_num_demos, 0, 0)
        self.total_length = self.number_demos.sum()

    def __length__(self):
        
        return self.total_length
    
    def __getitem__(self, index):
         
        # Bin. 
        task_index = np.searchsorted(self.cummulative_num_demos, index, side='right')-1

		# Decide task ID, and new index modulo num_demos.
		# Subtract number of demonstrations in cumsum until then, and then 
        new_index = index-self.cummulative_num_demos[max(task_index,0)]
        
        # Create a data element. 
        data_element = {}
        
        data_element['demo'] = self.files[task_index][new_index][:,1:]
        data_element['exercise'] = self.exercise_list[task_index]
        data_element['performance_level'] = self.performance_level[task_index]
        data_element['task_id'] = task_index
        data_element['task-id'] = task_index

        return data_element
    






        