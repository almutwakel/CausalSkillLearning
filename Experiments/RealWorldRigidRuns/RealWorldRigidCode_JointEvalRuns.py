# Template Eval
# // CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RWRP_300_TestNumz6 --data=RealWorldRigid --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.1 --epsilon_over=50000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.00001 --policy_variance_decay_over=50000 --state_scale_factor=10. --datadir=/scratch/cchawla/NewRealWorldRigidRelPose --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_300/saved_models/Model_epoch100000 --N_trajectories_to_visualize=500 --images_in_real_world_data=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=learntsub --name=RWRP_300_Joint --data=RealWorldRigid --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.1 --epsilon_over=50000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.00001 --policy_variance_decay_over=50000 --state_scale_factor=10. --load_latent=0 --datadir=/scratch/cchawla/NewRealWorldRigidRelPose --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_300/saved_models/Model_epoch100000 --N_trajectories_to_visualize=500 --images_in_real_world_data=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=queryjoint --name=RWRP_300_Joint --data=RealWorldRigid --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.1 --epsilon_over=50000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.00001 --policy_variance_decay_over=50000 --state_scale_factor=10. --load_latent=0 --traj_length=-1 --datadir=/scratch/cchawla/NewRealWorldRigidRelPose --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_300/saved_models/Model_epoch100000 --N_trajectories_to_visualize=500 --images_in_real_world_data=1

# 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=queryjoint --name=RWRP_300_Joint2 --data=RealWorldRigid --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.00001 --epsilon_to=0.00001 --epsilon_over=1 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.00001 --final_policy_variance=0.00001 --policy_variance_decay_over=1 --state_scale_factor=10. --load_latent=0 --traj_length=-1 --datadir=/scratch/cchawla/NewRealWorldRigidRelPose --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_300/saved_models/Model_epoch100000 --N_trajectories_to_visualize=100 --images_in_real_world_data=0


//CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=queryjoint --name=RWRP_300_Joint2_GIFS --data=RealWorldRigid --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.00001 --epsilon_to=0.00001 --epsilon_over=1 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.00001 --final_policy_variance=0.00001 --policy_variance_decay_over=1 --state_scale_factor=10. --load_latent=0 --traj_length=-1 --datadir=/scratch/cchawla/NewRealWorldRigidRelPose --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_300/saved_models/Model_epoch100000 --N_trajectories_to_visualize=100 --images_in_real_world_data=1

# 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=queryjoint --name=RWRP_322_Joint --data=RealWorldRigid --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.00001 --epsilon_to=0.00001 --epsilon_over=1 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.00001 --final_policy_variance=0.00001 --policy_variance_decay_over=1 --state_scale_factor=10. --load_latent=0 --traj_length=-1 --N_trajectories_to_visualize=100 --images_in_real_world_data=1 --datadir=/data/tanmayshankar/Datasets/NewRealWorldRelPose/ --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RWRP_322/saved_models/Model_epoch100000 

###########################################################
## TEMPLATE 350 pretrain run
# CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RWRP_350_Eval --data=RealWorldRigid --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.05 --epsilon_to=0.05 --epsilon_over=1 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000005 --final_policy_variance=0.000005 --policy_variance_decay_over=10 --state_scale_factor=10. --input_corruption_noise=0. --subpolicy_input_dropout=0. --cummulative_computed_state_reconstruction_loss_weight=100. --datadir=/scratch/cchawla/NewRealWorldRigidRelPose --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_350/saved_models/Model_epoch48000

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=queryjoint --name=RWRP_350_Joint --data=RealWorldRigid --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.00001 --epsilon_to=0.00001 --epsilon_over=1 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.00001 --final_policy_variance=0.00001 --policy_variance_decay_over=1 --state_scale_factor=10. --load_latent=0 --traj_length=-1 --N_trajectories_to_visualize=100 --images_in_real_world_data=0 --datadir=/scratch/cchawla/NewRealWorldRigidRelPose --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_350/saved_models/Model_epoch48000

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=queryjoint --name=RWRP_350_Joint --data=RealWorldRigid --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.00001 --epsilon_to=0.00001 --epsilon_over=1 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.00001 --final_policy_variance=0.00001 --policy_variance_decay_over=1 --state_scale_factor=10. --load_latent=0 --traj_length=-1 --N_trajectories_to_visualize=100 --images_in_real_world_data=0 --datadir=/scratch/cchawla/NewRealWorldRigidRelPose --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_350/saved_models/Model_epoch80000

# TEmplate 

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=queryjoint --name=RWRP_352_Joint --data=RealWorldRigid --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.00001 --epsilon_to=0.00001 --epsilon_over=1 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.00001 --final_policy_variance=0.00001 --policy_variance_decay_over=1 --state_scale_factor=10. --load_latent=0 --traj_length=-1 --N_trajectories_to_visualize=100 --images_in_real_world_data=0 --datadir=/scratch/cchawla/NewRealWorldRigidRelPose --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_352/saved_models/Model_epoch82000

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=queryjoint --name=RWRP_354_Joint --data=RealWorldRigid --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.00001 --epsilon_to=0.00001 --epsilon_over=1 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.00001 --final_policy_variance=0.00001 --policy_variance_decay_over=1 --state_scale_factor=10. --load_latent=0 --traj_length=-1 --N_trajectories_to_visualize=100 --images_in_real_world_data=0 --datadir=/scratch/cchawla/NewRealWorldRigidRelPose --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_354/saved_models/Model_epoch82000

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=queryjoint --name=RWRP_356_Joint --data=RealWorldRigid --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.00001 --epsilon_to=0.00001 --epsilon_over=1 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.00001 --final_policy_variance=0.00001 --policy_variance_decay_over=1 --state_scale_factor=10. --load_latent=0 --traj_length=-1 --N_trajectories_to_visualize=100 --images_in_real_world_data=0 --datadir=/scratch/cchawla/NewRealWorldRigidRelPose --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_356/saved_models/Model_epoch82000

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=queryjoint --name=RWRP_358_Joint --data=RealWorldRigid --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.00001 --epsilon_to=0.00001 --epsilon_over=1 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.00001 --final_policy_variance=0.00001 --policy_variance_decay_over=1 --state_scale_factor=10. --load_latent=0 --traj_length=-1 --N_trajectories_to_visualize=100 --images_in_real_world_data=0 --datadir=/scratch/cchawla/NewRealWorldRigidRelPose --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_358/saved_models/Model_epoch82000

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=queryjoint --name=RWRP_356_Joint_ObjectGeneralizationExpts --data=RealWorldRigid --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.00001 --epsilon_to=0.00001 --epsilon_over=1 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.00001 --final_policy_variance=0.00001 --policy_variance_decay_over=1 --state_scale_factor=10. --load_latent=0 --traj_length=-1 --N_trajectories_to_visualize=100 --images_in_real_world_data=0 --datadir=/scratch/cchawla/NewRealWorldRigidRelPose --logdir=/scratch/cchawla/TrainingLogs/ --model=/scratch/cchawla/TrainingLogs/RWRP_356/saved_models/Model_epoch82000
