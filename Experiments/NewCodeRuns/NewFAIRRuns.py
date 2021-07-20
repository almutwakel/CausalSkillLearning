# Logging bunch of runs we need to systematically rerun with the new corrected Z transform computation. 

################################################################
################################################################
# Rerun JFE_MIME_psup_runs that were with dataset_trajectory_length = 50, but now with the corected???? z tuple loss
# Debug
WANDB_MODE=offline python Master.py --name=JFE_MIME_psup_debug --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --dataset_traj_length_limit=50 --number_of_supervised_datapoints=30 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/

python cluster_run.py --name='JFE_100' --cmd='python Master.py --name=JFE_MIME_psup_500 --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --dataset_traj_length_limit=50 --number_of_supervised_datapoints=30 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='JFE_101' --cmd='python Master.py --name=JFE_MIME_psup_501 --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --dataset_traj_length_limit=50 --number_of_supervised_datapoints=62 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='JFE_102' --cmd='python Master.py --name=JFE_MIME_psup_502 --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --dataset_traj_length_limit=50 --number_of_supervised_datapoints=94 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='JFE_103' --cmd='python Master.py --name=JFE_MIME_psup_503 --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --dataset_traj_length_limit=50 --number_of_supervised_datapoints=190 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='JFE_104' --cmd='python Master.py --name=JFE_MIME_psup_504 --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --dataset_traj_length_limit=50 --number_of_supervised_datapoints=350 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

# Rerun
python cluster_run.py --name='JFE_100' --cmd='python Master.py --name=JFE_MIME_psup_500_cont --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --dataset_traj_length_limit=50 --number_of_supervised_datapoints=30 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --model=ExpWandbLogs/JFE_MIME_psup_500/saved_models/Model_epoch4340'

python cluster_run.py --name='JFE_101' --cmd='python Master.py --name=JFE_MIME_psup_501_cont --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --dataset_traj_length_limit=50 --number_of_supervised_datapoints=62 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --model=ExpWandbLogs/JFE_MIME_psup_501/saved_models/Model_epoch3900'


########################################
# Running JFE_MIME_psup_300-304 with corrected z transformation computation, with full datset trajectory length.
# Debug
python cluster_run.py --name='JFE_105' --cmd='python Master.py --name=JFE_MIME_psup_510 --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=30 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='JFE_106' --cmd='python Master.py --name=JFE_MIME_psup_511 --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=62 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='JFE_107' --cmd='python Master.py --name=JFE_MIME_psup_512 --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=94 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='JFE_108' --cmd='python Master.py --name=JFE_MIME_psup_513 --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=190 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='JFE_109' --cmd='python Master.py --name=JFE_MIME_psup_514 --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=350 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

# Debug
python Master.py --name=JFE_MIME_psup_debugwandb --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=30 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/

# DJFE_MIME_gmmtup runs with GMM variance = 0.5, various loss weights

# Debugging gmm tuple z tuple computaiton 
python Master.py --name=DJFE_MIME_gmmtup_debugz --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=94 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0.1 --gmm_variance_value=0.5 --backward_loss_weight=1. --forward_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=0.1 --dataset_traj_length_limit=50 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/

# 
# Running GMM tuple with GMM variance = 0.5, different weights for the cross domain z tuple loss..
python cluster_run.py --name='JFE_110' --cmd='python Master.py --name=DJFE_MIME_gmmtup_020 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=94 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0.1 --gmm_variance_value=0.5 --backward_loss_weight=1. --forward_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=0.1 --dataset_traj_length_limit=50 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='JFE_111' --cmd='python Master.py --name=DJFE_MIME_gmmtup_021 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=94 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0.1 --gmm_variance_value=0.5 --backward_loss_weight=1. --forward_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=0.2 --dataset_traj_length_limit=50 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='JFE_112' --cmd='python Master.py --name=DJFE_MIME_gmmtup_022 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=94 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0.1 --gmm_variance_value=0.5 --backward_loss_weight=1. --forward_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=0.4 --dataset_traj_length_limit=50 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

# 
# Run leftmime-rightmime 1 and 3 with corrected z trajectory loss 
python cluster_run.py --name='JFE_113' --cmd='python Master.py --name=JFE_leftmime_rightmime_010 --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=5000 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=10000 --save_freq=100 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --dataset_traj_length_limit=50 --number_of_supervised_datapoints=0 --metric_eval_freq=5000 --source_single_hand=left --target_single_hand=right --supervised_set_based_density_loss=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='JFE_114' --cmd='python Master.py --name=JFE_leftmime_rightmime_011 --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=5000 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=10000 --save_freq=100 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --dataset_traj_length_limit=100 --number_of_supervised_datapoints=0 --metric_eval_freq=5000 --source_single_hand=left --target_single_hand=right --supervised_set_based_density_loss=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='JFE_115' --cmd='python Master.py --name=JFE_leftmime_rightmime_012 --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=5000 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=10000 --save_freq=100 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=5000 --source_single_hand=left --target_single_hand=right --supervised_set_based_density_loss=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

# Debugging 
python Master.py --name=JFE_MIME_psup_fulllen --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=30 --metric_eval_freq=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --model=ExpWandbLogs/JFE_MIME_psup_511/saved_models/Model_epoch1540