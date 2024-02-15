# Dummy run
# // CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=H2R_TransferTrial_debug --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=50. --task_based_aux_loss_weight=50. --auxillary_z_env_effect_z_loss_weight=200. --negative_component_weight=1. --positive_z_distance_margin=1. --negative_z_distance_margin=5. --initial_z_distance_threshold=5. --final_z_distance_threshold=1. --z_distance_threshold_decay_over=80000 --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=25 --env_state_size=14 --metric_distance_space='z_J' --kl_schedule='Monotonic' --initial_kl_weight=0.001 --final_kl_weight=0.001 --kl_begin_increment_epochs=10000 --datadir=/data/tanmayshankar/Datasets/RigidBodyHumanData_NDTAImageFiles/ --logdir=/data/tanmayshankar/TrainingLogs/


// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=H2R_TransferTrial_debug --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/home/tshankar/Research/Data/Datasets/RigidBodyHumanData_NDTAImageFiles/ --logdir=ExpWandbLogs/ --model=ExpWandbLogs/RWRP_ZEnvTaskAux_265/saved_models/Model_epoch94000 --images_in_real_world_dataset=1

# 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=H2R_TransferTrial_debug_RobotStream --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/home/tshankar/Research/Data/Datasets/RigidBodyHumanData_NDTAImageFiles/ --logdir=ExpWandbLogs/ --model=ExpWandbLogs/RWRP_ZEnvTaskAux_265/saved_models/Model_epoch94000 --images_in_real_world_dataset=1 --embedding_visualization_stream='robot'

# 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=H2R_TransferTrial_debug_EnvStream --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/home/tshankar/Research/Data/Datasets/RigidBodyHumanData_NDTAImageFiles/ --logdir=ExpWandbLogs/ --model=ExpWandbLogs/RWRP_ZEnvTaskAux_265/saved_models/Model_epoch94000 --images_in_real_world_dataset=1 --embedding_visualization_stream='env'

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=H2R_TransferTrial_debugplot --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/home/tshankar/Research/Data/Datasets/RigidBodyHumanData_NDTAImageFiles/ --logdir=ExpWandbLogs/ --model=ExpWandbLogs/RWRP_ZEnvTaskAux_265/saved_models/Model_epoch94000 --images_in_real_world_dataset=1

#####################################################
#####################################################
 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=H2R_TransferTrial_003Env --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/home/tshankar/Research/Data/Datasets/RigidBodyHumanData_NDTAImageFiles/ --logdir=ExpWandbLogs/ --model=ExpWandbLogs/RWRP_ZEnvTaskAux_265/saved_models/Model_epoch94000 --images_in_real_world_dataset=1 --embedding_visualization_stream='env'

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=H2R_TransferTrial_003EnvWandb --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/home/tshankar/Research/Data/Datasets/RigidBodyHumanData_NDTAImageFiles/ --logdir=ExpWandbLogs/ --model=ExpWandbLogs/RWRP_ZEnvTaskAux_265/saved_models/Model_epoch94000 

#####################################################
#####################################################
 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=H2R_TransferTrial_004Env --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/scratch/cchawla/RigidBodyHumanData_NDTAImageFiles_NewFreq/ --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_ZEnvTaskAux_265/saved_models/Model_epoch94000 --images_in_real_world_dataset=1 --embedding_visualization_stream='env'

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=H2R_TransferTrial_004EnvWandb --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/home/tshankar/Research/Data/Datasets/RigidBodyHumanData_NDTAImageFiles/ --logdir=ExpWandbLogs/ --datadir=/scratch/cchawla/RigidBodyHumanData_NDTAImageFiles_NewFreq/ --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_ZEnvTaskAux_265/saved_models/Model_epoch94000

#####################################################
#####################################################
 
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=H2R_TransferTrial_005Env --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/scratch/cchawla/RigidBodyHumanData_NDTAImageFiles_NewFreq/ --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_ZEnvTaskAux_265/saved_models/Model_epoch94000 --images_in_real_world_dataset=1 --embedding_visualization_stream='env'

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=H2R_TransferTrial_005EnvWandb --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/home/tshankar/Research/Data/Datasets/RigidBodyHumanData_NDTAImageFiles/ --logdir=ExpWandbLogs/ --datadir=/scratch/cchawla/RigidBodyHumanData_NDTAImageFiles_NewFreq/ --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_ZEnvTaskAux_265/saved_models/Model_epoch94000

# 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=H2R_TransferTrial_006EnvWandb --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/home/tshankar/Research/Data/Datasets/RigidBodyHumanData_NDTAImageFiles/ --logdir=ExpWandbLogs/ --model=ExpWandbLogs/RWRP_ZEnvTaskAux_265/saved_models/Model_epoch94000 

#####################################################
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=H2R_TransferTrial_010_EnvWandb --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/scratch/cchawla/RigidBodyHumanData_NDTAImageFiles_NewFreq/ --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_ZEnvTaskAux_300/saved_models/Model_epoch94000

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=H2R_TransferTrial_011_EnvWandb --data=RealWorldRigidHumanNNTransfer --var_skill_length=0 --traj_length=15 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/scratch/cchawla/RigidBodyHumanData_NDTAImageFiles_NewFreq/ --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_ZEnvTaskAux_300/saved_models/Model_epoch94000

#####################################################
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=H2R_TransferTrial_012_EnvWandb --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/scratch/cchawla/RigidBodyHumanData_NDTAImageFiles_NewFreq/ --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_ZEnvTaskAux_302/saved_models/Model_epoch94000

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=H2R_TransferTrial_013_EnvWandb --data=RealWorldRigidHumanNNTransfer --var_skill_length=0 --traj_length=15 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/scratch/cchawla/RigidBodyHumanData_NDTAImageFiles_NewFreq/ --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_ZEnvTaskAux_302/saved_models/Model_epoch94000


# debug
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=H2R_TransferTrial_debug --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/scratch/cchawla/RigidBodyHumanData_NDTAImageFiles_NewFreq/ --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_ZEnvTaskAux_302/saved_models/Model_epoch94000

# ORIGINAL
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=H2R_TransferTrial_021 --data=RealWorldRigidHumanNNTransfer --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.001 --epsilon_to=0.001 --epsilon_over=100 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --positional_encoding=1 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --datadir=/scratch/cchawla/RigidBodyHumanData_NDTAImageFiles/ --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_ZEnvTaskAux_302/saved_models/Model_epoch94000