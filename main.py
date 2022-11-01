import numpy as np
import torch
import gym
import argparse
import os
import d4rl
from torch.utils.tensorboard import SummaryWriter
import datetime
import utils
import TD3_BC
import time
import h5py

# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, iteration, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			# if done == True:
			# 	avg_reward += cumulative_reward
			# else:
			# 	avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
	writer.add_scalar('avg reward', avg_reward, iteration + 1)
	writer.add_scalar('D4RL Score', d4rl_score, iteration + 1)
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--env", default="hopper-medium-v0")        # OpenAI gym environment name
	parser.add_argument("--eval_env", default="maze2d-open-dense-v0") # Evaluation done on dense environment even when trained on sparse to compare results
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", default=True, action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=True)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Evaluation Env: {args.eval_env}, Seed: {args.seed}, {args.save_model}")
	print("---------------------------------------")

	log_path = 'Results7/{}/{}'.format(args.env,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
	writer = SummaryWriter(log_path)

	# if not os.path.exists("./results7"):
	# 	os.makedirs("./results7")

	if args.save_model and not os.path.exists("./models7"):
		os.makedirs("./models7")

	env = gym.make(args.env)
	start_time = time.time()

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	prev_actor_loss = 0

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha
	}

	# Initialize policy
	policy = TD3_BC.TD3_BC(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	# dataset = h5py.File('./single_reward_dataset/maze2d-open-v0.hdf5','r')
	# replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env, dataset = dataset))
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env), sparse = True)
	#print(args.env + "  " + str(replay_buffer.size))
	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1
	#print(replay_buffer.reward[:500])
	evaluations = []
	start_time = time.time()
	for t in range(int(args.max_timesteps)):
		actor_loss, critic_loss, iteration = policy.train(replay_buffer, prev_actor_loss, args.batch_size)
		prev_actor_loss = actor_loss
		writer.add_scalar('Actor Loss', actor_loss, iteration + 1)
		writer.add_scalar('Critic Loss', critic_loss, iteration + 1)
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			evaluations.append(eval_policy(policy, args.eval_env, args.seed, mean, std, t))
			#np.save(f"./results7/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models7/{file_name}_{t+1}")
		
		if t + 1 == int(args.max_timesteps):
			writer.add_text('Env_name', str(args.env))
			train_time = time.time() - start_time
			writer.add_text('Training_time', str(train_time))
		
	writer.close()
