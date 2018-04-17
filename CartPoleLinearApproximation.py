import numpy as np
import random
import gym
import math

env = gym.make("CartPole-v0")
observation = env.reset()

w = np.zeros((4, env.action_space.n))
discount_factor = 0.99

final_turns = []

for episode in range(1000):
	observation = env.reset()

	# Update rate
	learning_rate = max(0.1, min(0.5, 1.0 - math.log10((episode + 1) / 30)))
	explore_rate = max(0.01, min(1, 1.0 - math.log10((episode + 1) / 30)))
	for t in range(200):
		# if episode > 990:
		env.render()

		state0 = observation

		if random.random() > explore_rate:
			action = env.action_space.sample()
		else:
			action = np.argmax(np.dot(observation, w))

		observation, reward, done, info = env.step(action)

		# episode 100 final turn 9 average 11.370000 explore rate 0.472800
		# episode 999 final turn 11 average 20.790000 explore rate 0.010000
		# => Getting a little bit better, but not really
		best_q = np.amax(np.dot(observation, w))
		old_q = np.dot(state0, w)[action]
		target_q = reward + best_q * discount_factor
		print("target_q", target_q)
		print("old_q", old_q)
		r_gradient = (np.random.random_sample(w.shape) * 2.0 - 1.0) * (target_q - old_q) * learning_rate
		gradient_mask = np.zeros(w.shape)
		gradient_mask[:, action] = np.ones(4)
		r_gradient *= gradient_mask
		new_w = w + r_gradient
		new_q = np.dot(state0, new_w)[action]
		print("new_q", new_q)
		if pow(target_q - new_q, 2) < pow(target_q - old_q, 2):
			w += r_gradient

		if done:
			final_turns.append(t)
			if len(final_turns) > 100:
				final_turns.pop(0)
			print("episode %d" % episode, 
				"final turn %d" % t, 
				"average %f" % np.average(final_turns), 
				"explore rate %f" % explore_rate)
			break
print(w)
# [[-17397514.3585613  -22192311.867659  ]
#  [ -5042667.43622229  -3483446.373171  ]
#  [ 22749157.74101703  -7048400.73872941]
#  [ 54541725.58527956  41605385.67593958]] => diverges