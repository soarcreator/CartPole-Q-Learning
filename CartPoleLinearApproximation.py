import numpy as np
import random
import gym
import math
from gym.spaces.prng import seed

# init seed and environment
seed(10)
np.random.seed(10)
random.seed(10)
env = gym.make("CartPole-v0")
env.seed(10)

# The weights below was found in the episode 50. Total Turn 398000
# w = [[ 3.12413034,  3.24618892],
#  [ 8.99752649, 12.2026371 ],
#  [-0.56146245,  1.36947562],
#  [-6.95913086, -2.9225207 ]]

# w = [[-0.70014011, -0.74692615],
#  [-3.53716535,  2.12238875],
#  [-1.49991566,  3.11146644],
#  [-2.86390059,  4.90937748]]

# The weights below was found in the episode 250. Total Turn 395090
# w = [[-3.46177988, -3.15280476],
#  [-1.75402309, -1.66459222],
#  [-0.69378298, -0.20644719],
#  [-4.43401779, -2.91479131]]

# The weights below was found in the episode 2000. Total Turn 187280
# w = [[ -0.41704199,   0.08057702],
#  [  6.4459129,   -5.91601382],
#  [ -1.15037097,  -0.10012901],
#  [-16.26928897,   6.83005852]]

w = np.zeros((4, env.action_space.n))
discount_factor = 0.99

final_turns = []
total_turn = 0

for episode in range(2000):
	observation = env.reset()

	# Update rate
	# learning_rate = max(0.1, min(0.5, 1.0 - math.log10((episode + 1) / 30)))
	# explore_rate = max(0.01, min(1, 1.0 - math.log10((episode + 1) / 30)))
	learning_rate = max(0.001, min(0.05, 1.0 - math.log10((episode + 1) / 30)))
	explore_rate = max(0.01, min(1, 1.0 - math.log10((episode + 1) / 100)))
	# explore_rate = 0
	# learning_rate = 0
	if episode % 50 == 0:
		print(w)
	for t in range(200):
		if episode % 50 == 0:
			explore_rate = 0
			learning_rate = 0
			env.render()

		if random.random() < explore_rate:
			action = env.action_space.sample()
		else:
			action = np.argmax(np.dot(observation, w))

		state0 = observation
		observation, reward, done, info = env.step(action)

		q_table0 = np.dot(state0, w)
		# best_q = np.amax(q_table0) # Pattern 1
		best_q = np.amax(np.dot(observation, w)) # Pattern 2
		old_q = q_table0[action]
		target_q = reward + best_q * discount_factor
		delta = target_q - old_q
		new_w = w
		new_w[:, action] += state0 * delta * learning_rate
		new_q = np.dot(state0, new_w)[action]
		if pow(target_q - new_q, 2) < pow(target_q - old_q, 2):
			w = new_w

		if done:
			final_turns.append(t)
			total_turn += t
			if len(final_turns) > 100:
				final_turns.pop(0)
			print("episode %d" % episode, 
				"final turn %d" % t, 
				"average %f" % np.average(final_turns), 
				"explore rate %f" % explore_rate)
			break
print(w)
print("Total Turn", total_turn)

# Pattern 1
# [[ -0.41704199   0.08057702]
#  [  6.4459129   -5.91601382]
#  [ -1.15037097  -0.10012901]
#  [-16.26928897   6.83005852]]
# Total Turn 164341

# Pattern 2, which should be better than Pattern 1, however.
# [[  336.40010547    -6.12292096]
#  [ 2086.85692452  1824.79447566]
#  [ -273.7291308   -261.51979897]
#  [-2956.32903306 -3072.82419767]]
# Total Turn 37154