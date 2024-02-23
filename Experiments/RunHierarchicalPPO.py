from headers import *
# sys.path.append("/private/home/tanmayshankar/Research/Code/spinningup")
from spinup.exercises.pytorch.problem_set_1 import exercise1_1
from spinup.exercises.pytorch.problem_set_1 import exercise1_2_auxiliary
from spinup.utils.run_utils import ExperimentGrid
from PolicyNetworks import mlp, MLPGaussianActor
from spinup import ppo_pytorch as ppo
# from spinup.algos.pytorch.ppo.hierarchical_ppo import hierarchical_ppo
from spinup.algos.pytorch.ppo.new_hierarchical_ppo import hierarchical_ppo
from spinup.exercises.common import print_result
from functools import partial
import gym, os, pandas as pd, psutil, time
import robosuite
from robosuite.wrappers import GymWrapper
from spinup.utils.test_policy import load_policy_and_env, run_policy, render_episode, hierarchical_run_policy, hierarchical_render_episode
import imageio 

"""
This file runs the OpenAI Spinning Up PPO baseline on all of the Robosuite environments. 
"""

if __name__ == '__main__':
	"""
	Run this file to verify your solution.
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument('--env', dest='env_name', type=str, default='Lift',                   		help='Specify the environment name of the Robosuite environment to run PPO On.')
	parser.add_argument('--robots', dest='robots', type=str, default="Swayer", 						help='Robot to use in environment.')
	parser.add_argument('--run_name', dest='run_name', type=str,                                    help='Set the name of the run.')
	# parser.add_argument('--train', dest='train', default=True, action='store_true',                 help='Whether to train or evaluate.')
	feature_parser = parser.add_mutually_exclusive_group(required=False)
	feature_parser.add_argument('--train', dest='train', action='store_true')
	feature_parser.add_argument('--no-train', dest='train', action='store_false')
	parser.set_defaults(train=True)
	parser.add_argument('--model', dest='model', type=str,                                          help='Model to load.')
	parser.add_argument('--debug', dest='debug', type=int, default=0,                             	help='Whether to run in debug mode.')
	parser.add_argument('--render', dest='render', type=int, default=1,                             help='Whether to render an episode.')
	parser.add_argument('--evaluate', dest='evaluate', type=int, default=1,                         help='Whether to evaluate.')
	parser.add_argument('--hierarchical', dest='hierarchical', type=int, default=0,                 help='Whether to run Hierarchical PPO or Flat PPO.')
	parser.add_argument('--epochs', dest='epochs', type=int, default=100,                           help='How many epochs to run.')

	####################################################
	# Arguments to instantiate low-level policy. 
	####################################################

	parser.add_argument('--lowlevel_policy_model', dest='lowlevel_policy_model', type=str, default=None,        help='Policy model file.')
	parser.add_argument('--data', dest='data', type=str, default=None,                                          help='What data domain to instantiate policy for.')
	parser.add_argument('--basedir', dest='basedir', type=str, default="/home/tshankar/Research/Code/CausalSkillLearning/Experiments/Statistics", help="Base directory to look for statistics files in.")
	parser.add_argument('--action_scaling', dest='action_scaling', type=float, default=1.,                      help='How much to scale actions by.')

	####################################################
	# Arguments to evaluate translated zs. 
	parser.add_argument('--evaluate_translated_zs', dest='evaluate_translated_zs', type=int, default=0, help='Whether to use translated zs to evaluate.')
	parser.add_argument('--translated_z_file', dest='translated_z_file', type=str, default=None, help='File to load zs from.')

	# parser.add_argument('--translated_z_file', dest='translated_z_file', type=str, default=None, help='File to load zs from.')

	####################################################
	# Policy arguments
	parser.add_argument('--positional_encoding',dest='positional_encoding',type=float, default=0., help='Whether or not to use positional encoding layers in the models.')
	parser.add_argument('--variance_factor',dest='variance_factor',type=float,default=0.01,help='Factor by which to multiple variance value predicted by network.')

	####################################################
	# PPO arguments
	####################################################

	parser.add_argument('--target_kl', dest='target_kl', type=float, default=0.01, help=' Roughly what KL divergence we think is appropriate \
		 between new and old policies after an update. This will get used \
		 for early stopping. (Usually small, 0.01 or 0.05.)')
	# parser.add_argument('--')
	
	args = parser.parse_args()

	# Remember, the environment names need to be from here. 
	# environment_names = ["SawyerLift", "SawyerStack", "BaxterLift", "BaxterPegInHole"]
	# environment_names = ["SawyerPickPlaceBread","SawyerPickPlaceCan","SawyerPickPlaceCereal","SawyerPickPlace","SawyerPickPlaceMilk","SawyerNutAssembly", "SawyerNutAssemblyRound","SawyerNutAssemblySquare"]

	####################################################
	# First make the robosuite environment. 
	####################################################    
     
	# Specify that we're going to use the Sawyer by default
	base_env = robosuite.make(args.env_name, robots=args.robots, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, reward_shaping=True)        

	
	# print("Embed after constructing env")
	# embed()
	# Now make a GymWrapped version of that environment.
	gym_env = GymWrapper(base_env)

	####################################################
	# Create a log directory.
	# logdir = "/tmp/experiments/%i"%int(time.time())
	# logdir = "/tmp/experiments/{0}".format(args.env_name+"_"+args.run_name)
	logdir = "Logs/{0}".format(args.env_name+"_"+args.run_name)

	# Create a policy / critic. 
	ActorCritic = partial(exercise1_2_auxiliary.ExerciseActorCritic, actor=MLPGaussianActor)
	print("We reached train")
	if args.train:
		print("Beginning Training.")
		# Actually call PPO.

		if args.hierarchical:
			hierarchical_ppo(lambda : gym_env, 
			ac_kwargs=dict(hidden_sizes=(64,)), 
			steps_per_epoch=1000, epochs=args.epochs,
			logger_kwargs=dict(output_dir=logdir), args=args, target_kl=args.target_kl)
		else:
			ppo(env_fn = lambda : gym_env,
				actor_critic=ActorCritic,
				ac_kwargs=dict(hidden_sizes=(64,)),
				steps_per_epoch=1000, epochs=args.epochs, logger_kwargs=dict(output_dir=logdir))

		# Get scores from last five epochs to evaluate success.
		data = pd.read_table(os.path.join(logdir,'progress.txt'))
		last_scores = data['AverageEpRet'][-5:]

		# Now evaluate last model over 100 episodes. 
		# Load model while evaluating. 
		_ , policy = load_policy_and_env(logdir)
		# print("Embedding before eval")
		# embed()
		# Now run the policy.
		if args.hierarchical:
			# Now run the policy.
			hierarchical_run_policy(gym_env, policy, render=False, args=args)
		else:
			run_policy(gym_env, policy, render=False)

	else: 
		print("Evaluating, not training.")

		# Load model while evaluating. 
		if args.model is None: 
			_ , policy = load_policy_and_env(logdir)
		else:
			_ , policy = load_policy_and_env(args.model)
	

		if args.render: 
			# Render an episode of the policy. 
			if args.hierarchical:
				episode_gif = hierarchical_render_episode(gym_env, policy, args=args)
			else:
				episode_gif = render_episode(gym_env, policy)
			path = os.path.join(logdir, "Images")
			if not(os.path.isdir(path)):
				os.mkdir(path)

			imageio.mimsave(os.path.join(path,"Trained_Rollout.gif"), episode_gif)

		if args.evaluate:
			if args.hierarchical:
				# Now run the policy.
				hierarchical_run_policy(gym_env, policy, render=False, args=args)
			else:
				run_policy(gym_env, policy, render=False)