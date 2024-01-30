# 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_preproc_debug --data=RealWorldRigidHumanPreproc --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/

# 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_dset_gen --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/

# 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_trainingtrial --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/

# 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_001 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_002 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.2 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/

# Eval
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RWHP_002_EvalHand --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.0001 --epsilon_to=0.0001 --epsilon_over=10 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=10 --state_scale_factor=10. --input_corruption_noise=0.0 --subpolicy_input_dropout=0.0 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RWHP_002/saved_models/Model_epoch68000 --plot_index_min=0 --plot_index_max=22

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RWHP_002_EvalHand_10 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.0001 --epsilon_to=0.0001 --epsilon_over=10 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=10 --state_scale_factor=10. --input_corruption_noise=0.0 --subpolicy_input_dropout=0.0 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RWHP_002/saved_models/Model_epoch68000 --plot_index_min=0 --plot_index_max=10

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RWHP_002_EvalHand_Images --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.0001 --epsilon_to=0.0001 --epsilon_over=10 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=10 --state_scale_factor=10. --input_corruption_noise=0.0 --subpolicy_input_dropout=0.0 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RWHP_002/saved_models/Model_epoch68000 --plot_index_min=0 --plot_index_max=22 --images_in_real_world_dataset=1



# Different normalization
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_003 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_004 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.2 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/

# Now with positional encoding.
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_testposenc --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1


// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_005 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1

// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_006 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.2 --sub  olicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1

# With Pos Enc again, but with KLD
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_007 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1 --kl_weight=0.1 

# Debug pos enc
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_posenc --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1 --kl_weight=0.1

# Try Pos Enc again, but without Pos Enc in the decoder
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_010 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_011 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1 --kl_weight=0.1 

# 
 // CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_012 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1

# Actual dropout in the LSTM(s)
 // CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_020 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --dropout=0.2 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_021 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1

# Try dropout in the LSTMs, input dropout, and playing with Z size..
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_022 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=4 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_023 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=8 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_024 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=32 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1

# Eval
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RWHP_022_EvalHand --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.0001 --epsilon_to=0.0001 --epsilon_over=100 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=4 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=10 --state_scale_factor=10. --input_corruption_noise=0.0 --subpolicy_input_dropout=0.0 --dropout=0.0 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1 --model=/data/tanmayshankar/TrainingLogs/RWHP_022/saved_models/Model_epoch88000 --plot_index_min=0 --plot_index_max=21

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RWHP_022_EvalObj --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.0001 --epsilon_to=0.0001 --epsilon_over=100 --display_freq=2000 --epochs=10 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=4 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=0.000001 --final_policy_variance=0.000001 --policy_variance_decay_over=10 --state_scale_factor=10. --input_corruption_noise=0.0 --subpolicy_input_dropout=0.0 --dropout=0.0 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1 --model=/data/tanmayshankar/TrainingLogs/RWHP_022/saved_models/Model_epoch88000 --plot_index_min=25 --plot_index_max=28

##############################################
# Playing with KL
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_030 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=8 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100. --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --kl_schedule='Monotonic' --initial_kl_weight=0. --final_kl_weight=0.1 --kl_begin_increment_epochs=10000 --kl_increment_epochs=50000 --positional_encoding=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_031 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100. --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --kl_schedule='Monotonic' --initial_kl_weight=0. --final_kl_weight=0.1 --kl_begin_increment_epochs=10000 --kl_increment_epochs=50000 --positional_encoding=1

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_032 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=8 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100. --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --kl_schedule='Cyclic' --kl_increment_epochs=1000 --kl_begin_increment_epochs=1000 --kl_cyclic_phase_epochs=1000 --final_kl_weight=0.1 --positional_encoding=1

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_033 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=16 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100. --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --kl_schedule='Cyclic' --kl_increment_epochs=1000 --kl_begin_increment_epochs=1000 --kl_cyclic_phase_epochs=1000 --final_kl_weight=0.1 --positional_encoding=1

############################################
# Running with .. all the aux losses, Z_Env aux loss, 
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RWHP_024 --data=RealWorldRigidHuman --var_skill_length=1 --number_layers=3 --hidden_size=24 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=1.0 --epsilon_to=0.05 --epsilon_over=100000 --display_freq=2000 --epochs=100000 --save_freq=2000 --smoothen=0 --task_based_shuffling=0 --z_dimensions=32 --normalization=minmax --variance_mode='QuadraticAnnealed' --initial_policy_variance=1. --final_policy_variance=0.000005 --policy_variance_decay_over=100000 --state_scale_factor=10. --input_corruption_noise=0.5 --subpolicy_input_dropout=0.4 --dropout=0.4 --cummulative_computed_state_reconstruction_loss_weight=100.  --datadir=/data/tanmayshankar/Datasets/RealWorldHumanData --logdir=/data/tanmayshankar/TrainingLogs/ --positional_encoding=1
