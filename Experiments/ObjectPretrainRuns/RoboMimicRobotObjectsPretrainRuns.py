# 
# Sample
// python Master.py --train=1 --setting=pretrain_sub --name=RMP_105 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=minmax --epochs=2000


# Actually run the RMRO 
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_000 --data=RoboMimicRobotObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_001 --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/

#############################
# Eval
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_000_Env --data=RoboMimicRobotObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_000/saved_models/Model_epoch1995 --embedding_visualization_stream='env' --task_based_shuffling=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_000_Robot --data=RoboMimicRobotObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_000/saved_models/Model_epoch1995 --embedding_visualization_stream='robot' --task_based_shuffling=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_000 --data=RoboMimicRobotObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_000/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0


# Eval 001
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_001_Env --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 --embedding_visualization_stream='env' --task_based_shuffling=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_001_Robot --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 --embedding_visualization_stream='robot' --task_based_shuffling=1

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_001 --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0
    
# with video
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_001_Vid --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0
##########################################
# Run split stream encoder 
##########################################


# Actually run the RMRO 
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RMROP_000 --data=RoboMimicRobotObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=32 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --split_stream_encoder=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RMROP_001 --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --split_stream_encoder=1

# Eval
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMROP_000 --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RMROP_000/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0


// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMROP_001 --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RMROP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0

# 
scp /data/tanmayshankar/TrainingLogs/RMROP_001/saved_models/Model_epoch1995 tshankar@bach:~/../../data/tanmayshankar/TrainingLogs/RMROP_001/saved_models/

scp /data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 tshankar@bach:~/../../data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_001_Vid --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0