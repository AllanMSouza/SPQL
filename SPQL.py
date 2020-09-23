import numpy as np
import pylab as plt
import networkx as nx
import random
import time
import pickle

import pandas as pd

MATRIX_SIZE     = 5
ORIGIN          = 0
GOAL            = (MATRIX_SIZE * MATRIX_SIZE) - 1
MAX_EPISODES    = 10_000
LR              = 0.7
DISCOUNT_RATE   = 0
INITIAL_Q       = float('inf')


def gen_edges(edges, i):

	poits = []

	if edges['up'] == 1:
		poits.append((i, i - MATRIX_SIZE))

	if edges['down'] == 1:
		poits.append((i, i + MATRIX_SIZE))

	if edges['right'] == 1:
		poits.append((i, i + 1))

	if edges['left'] == 1:
		poits.append((i, i - 1))

	return poits

def get_edges_list():

	points_list = []

	for i in range(MATRIX_SIZE * MATRIX_SIZE):

		edges = {'up' : 1, 'down' : 1, 'left' : 1, 'right' : 1}

		if i < MATRIX_SIZE:
			edges['up'] = 0

		if (i + 1)%MATRIX_SIZE == 0:
			edges['right'] = 0

		if i%MATRIX_SIZE == 0:
			edges['left'] = 0

		if i >= MATRIX_SIZE * (MATRIX_SIZE - 1):
			edges['down'] = 0

		points_list += gen_edges(edges, i)

	return points_list

def build_graph(points_list, R):

	G = nx.DiGraph()

	for point in points_list:
		G.add_edge(point[0], point[1], weight= R[point])

	return G


def build_network(G):

	nx.draw(G)
	plt.show()

def build_reward_matrix(points_list):
	R  = np.matrix(np.zeros(shape=(MATRIX_SIZE * MATRIX_SIZE, MATRIX_SIZE * MATRIX_SIZE)))
	R += INITIAL_Q

	for point in points_list:
		R[point] = random.randint(2, 20) 

	return R

def update_reward_goal(G, R):

	for node in G[GOAL]:
		R[node, GOAL] += 1000

	return R

def build_Q_values_matrix(G):
	Q  = np.matrix(np.zeros(shape=(MATRIX_SIZE * MATRIX_SIZE, MATRIX_SIZE * MATRIX_SIZE)))
	Q += INITIAL_Q

	for node in G.nodes:
		for x in G[node]:
			Q[node, x] = 0
			Q[x, node] = 0

	return Q



def available_actions(state, R):
	current_state_row = R[state,]
	available_actions = np.where(current_state_row != INITIAL_Q)[1]

	return available_actions 

def get_best_action(available_actions, state, R):
	max_value = 99999
	action    = -1

	for act in available_actions:
		if R[state, act] < max_value:
			max_value = R[state, act]
			action    = act

	return act

def sample_next_action(available_actions, state, R):
	random_value = random.uniform(1, 0)

	if random_value < 0.1:
		return int(np.random.choice(available_actions, 1))

	else:
		return get_best_action(available_actions, state, R)

def update(current_state, action, lr, discont_rate, R, Q, penalty):

	max_index = np.where(Q[action,] == np.min(Q[action,]))[1]

	if max_index.shape[0] > 1:
		max_index = int(np.random.choice(max_index, size=1))

	else:
		max_index = int(max_index)

	max_value = Q[action, max_index]

	q_target = R[current_state, action] * penalty + discont_rate * max_value 
	q_update = q_target - Q[current_state, action]

	Q[current_state, action] += lr * (q_update)
	# Q[current_state, action] = R[current_state, action] + discont_rate * max_value

	return Q

def trainning(R, Q):
	# print('Training ...')

	score       = []
	policy_star = []

	for epsiode in range(MAX_EPISODES):
		# print(f'Training epsiode: {epsiode}')
		current_state    = ORIGIN #np.random.randint(0, int(Q.shape[0]))
		previous_actions = [ORIGIN,]

		epsiode_score = 0
		min_score     = 99999

		done = False

		while not done:

			available_act = available_actions(current_state, R)
			penalty       = 1
			action        = sample_next_action(available_act, current_state, R)
			
			if action in previous_actions:
				#penalty for cycles
				done    = True
				penalty = 100
				Q       = update(current_state, action, LR, DISCOUNT_RATE, R, Q, penalty)

			else:

				Q = update(current_state, action, LR, DISCOUNT_RATE, R, Q, penalty)
			
			epsiode_score += Q[current_state, action]

			current_state = action
			previous_actions.append(action)

			if current_state == GOAL:
				done = True


		score.append(epsiode_score)

	return Q, score, policy

def check_path(possible_next, path):

	for p_next in possible_next:
		if p_next in path:
			return True

	return False

def remove_path_from_possible_next(path, possible_next):

	possible_next = list(possible_next)

	for edge in path:
		if edge in possible_next:
			possible_next.remove(edge)

	return possible_next

def testing(Q):
	print('Testing ...')
	path      = [ORIGIN,]
	next_node = np.argmin(Q[ORIGIN, ])
	path.append(next_node)

	while next_node != GOAL:

		# possible_next = Q[next_node,][0]
		possible_next   = np.where(Q[next_node,] != float('inf'))[1]

		if check_path(possible_next, path):
			possible_next = remove_path_from_possible_next(path, possible_next)

		# next_node = np.argmax(possible_next)
		next_node  = get_best_action(possible_next, next_node, Q)
		path.append(next_node)

	print(f'Most efficient path  :{path}')

def plot_score(score):

	df = pd.DataFrame(score)
	print(df.head())
	df.plot()
	plt.show()

def main():

	df = pd.DataFrame()
	edges_list = get_edges_list()

	for seed in range(10):
		print(f'Training seed {seed} ...')
		R          = build_reward_matrix(edges_list)
		# R          = pickle.load(open(f'Rewards Matrix Ex1/{seed}.pickle', 'rb'))
		G          = build_graph(edges_list, R)
		Q          = build_Q_values_matrix(G)
		# R          = update_reward_goal(G, R)

		Q, score,policy = trainning(R, Q)
		df.insert(seed, f'seed {seed}', score, True)
		# plot_score(score)

		testing(Q)
		shortest_path = nx.shortest_path(G,ORIGIN,GOAL,'weight')
		print(f'Most efficient policy:{policy}')
		print(f'Shortest Path        :{shortest_path}')

	df['mean'] = df.mean(axis=1)
	plot_score(df['mean'].values)
	# df['mean'].to_csv('output/greedy.csv')


if __name__ == '__main__':
	main()