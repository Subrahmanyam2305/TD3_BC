import numpy as np
import torch
import random


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def convert_D4RL(self, dataset, sparse = False, mask_percent = 1):
		if sparse:
			self.state = dataset['observations']
			self.action = dataset['actions']
			self.next_state = dataset['next_observations']
			self.reward = dataset['rewards'].reshape(-1,1)
			new_reward = np.zeros_like(self.reward)
			mask = np.zeros_like(self.reward)
			self.not_done = 1. - dataset['terminals'].reshape(-1,1)

			for k in range(self.reward.shape[0]):
				if dataset['timeouts'][k] or dataset['terminals'][k]:
					mask[k] = 1
			# i = 1
			# while i <= mask_percent * self.reward.shape[0]:
			# 	j = np.random.randint(self.reward.shape[0])
			# 	if mask[j] == 1:
			# 		continue
			# 	mask[j] = 1
			# 	i += 1
			sampled = random.sample(range(int(mask_percent*self.reward.shape[0])), k = int(mask_percent*self.reward.shape[0]))
			for i in sampled:
				mask[i] = 1
			add = 0
			print("Mask created!!")
			for i in range(dataset['rewards'].shape[0]-1, -1, -1):
				add += self.reward[i]
				new_reward[i] = add
				if i > 0 and dataset['terminals'][i] or dataset['timeouts'][i]:
					add = 0
			self.reward = np.multiply(new_reward, mask)
			print("New reward calculated!!")
			self.size = self.state.shape[0]
		else:
			#print(dataset.keys())
			self.state = dataset['observations']
			self.action = dataset['actions']
			self.next_state = dataset['next_observations']
			self.reward = dataset['rewards'].reshape(-1,1)
			self.not_done = 1. - dataset['terminals'].reshape(-1,1)
			#self.info = dataset['infos/qpos']
			#print(self.info)
			self.size = self.state.shape[0]


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std