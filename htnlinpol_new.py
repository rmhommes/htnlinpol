from dijkstra_algorithm.dijkstra import *
import random
from collections import defaultdict
import copy
import yoyo
import subprocess
import sys
# from pyeda.inter import *
import time
from discreteMarkovChain import markovChain
import numpy as np
import uuid

###################################
# HTNLINPOL PROBLEM DOMAIN ELEMENTS
###################################

# Generate state variables V for world with size z
def V(z):
	if z > 20:
		return None
	return sum([['startingPos{},{}'.format(i,j),'currentPos{},{}'.format(i,j),'targetPos{},{}'.format(i,j), 'area_cleared{},{}'.format(i,j)] for i in range(z) for j in range(z)],[]) + \
		['areaValue{},{},{}'.format(i,j,k) for i in range(z) for j in range(z) for k in list(range(11))+[-1]] + \
		['gripper_full_expected', 'gripper_empty_unexpected', 'object_classified_as_garbage', 'garbage_localized', 'object_detected']

# Generate primitive task names for world with size z
def N_P(z):
	if z > 20:
		return None
	return ['n_move_from_to{},{},{}'.format(i,j,d) for i in range(z) for j in range(z) for d in directions if isMoveLegal(i,j,d,z)] + \
		['n_set_target{},{}'.format(i,j) for i in range(z) for j in range(z)] + \
		['n_report_success',
		'n_report_fail',
		'n_scan_area',
		'n_classify_object',
		'n_pick_up_garbage',
		'n_localize_garbage',
		'n_collect_garbage']

# Generate deterministic actions for world with size z
def A(z):
	if z > 20:
		return None
	return ['a_move_from_to{},{},{}'.format(i,j,d) for i in range(z) for j in range(z) for d in directions if isMoveLegal(i,j,d,z)] + \
		['a_set_target{},{}'.format(i,j) for i in range(z) for j in range(z)] + \
		['a_report_success',
		'a_report_fail']

# Generate non-deterministic actions for world with size z
def A_p():
	return ['a_scan_area', 
		'a_classify_object',
		'a_pick_up_garbage',
		'a_localize_garbage',
		'a_collect_garbage']

def state_transition_function(s, a, z):
	prefix = 'a_move_from_to'
	if a.startswith(prefix):
		args = a[len(prefix):]
		argsParts = args.split(',')
		currentPosition = 'currentPos{},{}'.format(int(argsParts[0]), int(argsParts[1]))
		precs = [currentPosition]
		if checkPrecs(precs, [], s):
			newPos = calcMove(int(argsParts[0]), int(argsParts[1]), int(argsParts[2]))
			newPosition = 'currentPos{},{}'.format(newPos[0], newPos[1])
			s = addToState([newPosition],delFromState([currentPosition],s))
		else:
			return None
	prefix = 'a_set_target'
	if a.startswith(prefix):
		args = a[len(prefix):]
		argsParts = args.split(',')
		precs = []
		if checkPrecs(precs, [], s):
			s = addToState(['targetPos{},{}'.format(int(argsParts[0]), int(argsParts[1]))],delFromState(['targetPos{},{}'.format(i,j) for i in range(z) for j in range(z)],s))
		else:
			return None
	if a == 'a_report_success':
		targetPos = getTargetPosition(s)
		if targetPos is None:
			return None
		if checkPrecs(['area_cleared{},{}'.format(targetPos[0], targetPos[1])], ['area_failed{},{}'.format(targetPos[0], targetPos[1])], s):
			pass
		else:
			return None
	if a == 'a_report_fail':
		targetPos = getTargetPosition(s)
		if targetPos is None:
			return None
		if checkPrecs(['area_failed{},{}'.format(targetPos[0], targetPos[1])], ['area_cleared{},{}'.format(targetPos[0], targetPos[1])], s):
			pass
		else:
			return None
	return s

############
# AUXILLIARY
############

# Calcute new position after taking one step in direction d from position i,j
def calcMove(i,j,d):
	if d == 0:
		return i, j-1
	if d == 45:
		return i+1, j-1
	if d == 90:
		return i+1, j
	if d == 135:
		return i+1, j+1
	if d == 180:
		return i, j+1
	if d == 225:
		return i-1, j+1
	if d == 270:
		return i-1, j
	if d == 315:
		return i-1, j-1

# Check if position calculated by calcMove does not contain a value below zero or higher than the world size
def isMoveLegal(i,j,d,z):
	i, j = calcMove(i,j,d)
	return i >= 0 and j >= 0 and i < z and j < z

def getAreaVal(s, i, j):
	for v in s:
		prefix = 'areaValue{},{},'.format(i, j)
		if v.startswith(prefix):
			return int(v[len(prefix):])

def getStartingPosition(s):
	for v in s:
		prefix = 'currentPos'
		if v.startswith(prefix):
			posStr = v[len(prefix):]
			posStrParts = posStr.split(',')
			return int(posStrParts[0]), int(posStrParts[1])

def getTargetPosition(s):
	for v in s:
		prefix = 'targetPos'
		if v.startswith(prefix):
			posStr = v[len(prefix):]
			posStrParts = posStr.split(',')
			return int(posStrParts[0]), int(posStrParts[1])

def worldToDijkstraGrid(s, dim):
	ll = []
	for i in range(dim):
		l = []
		for j in range(dim):
			u = getAreaVal(s, i, j)
			l.append(u)
		ll.append(l)
	G = {}
	for xi in range(len(ll)):
		row = ll[xi]
		for yi in range(len(row)):
			utility = row[yi]
			p = '{},{}'.format(xi, yi)
			v = {}
			if utility >= 0 and 'area_failed{},{}'.format(xi, yi) not in s:
				for xni in range(xi-1, xi+2):
					for yni in range(yi-1, yi+2):
						if xni >= 0 and xni < len(ll) and yni >= 0 and yni < len(ll[xni]) and ll[xni][yni] is not None and 'area_failed{},{}'.format(xni, yni) not in s:
							pn = '{},{}'.format(xni, yni)
							v[pn] = 1
			G[p] = v
	return G

def generateWorld(dim, p_obst):
	s = []
	for i in range(dim):
		for j in range(dim):
			u = None
			p_non_obst = 1 - p_obst
			randNum = random.random()
			if randNum <= p_non_obst or (i == 0 and j == 0):
				u = int(round(random.random()*10))
				s.append('areaValue{},{},{}'.format(i,j,u))
			else:
				s.append('areaValue{},{},{}'.format(i,j,-1))
	s.append('currentPos{},{}'.format(0,0))
	s.append('startingPos{},{}'.format(0,0))
	return s

def isStateSubSet(S1, S2):
	for s1 in S1:
		s1.sort()
	for s2 in S2:
		s2.sort()
	S1 = set(map(lambda e: str(e), S1))
	S2 = set(map(lambda e: str(e), S2))
	return S1.issubset(S2)

def isStateDoubleSubSet(S1, S2):
	for s1 in S1:
		b = False
		for s2 in S2:
			if set(s2).issubset(set(s1)):
				b = True
				break
		if not b:
			return False
	return True

def toPolicyFunction(s, Pi, gamma_p_dict):
	a = None
	gamma_p = gammaDictToFunc(gamma_p_dict)
	for S, a_prime in Pi:
		r = gamma_p(s, a_prime)
		if s in S and not (r is None or len(r) == 0):
			a = a_prime
			break
	if a is None:
		return None
	a_tail = a[1:]
	return ('t'+a_tail, [('t'+a_tail,'n'+a_tail)])

def updateS(S, pair):
	s, p = pair
	S_new = []
	pairAdded = False
	for (s_prime, p_prime) in S:
		if stateEquals(s, s_prime):
			S_new.append((s, (p + p_prime)))
			pairAdded = True
		else:
			S_new.append((s_prime, p_prime))
	if not pairAdded:
		S_new.append(pair)
	return S_new

def calcPolicyOutput(Pi, S, gamma_p, delta):
	output = []
	visited = set()
	while len(S) > 0:
		S_prime = []
		for s in S:
			s_converted = state_convert(s,state_transition_function_policy_dict())
			if s_converted in visited:
				continue
			visited.add(s_converted)
			action = Pi(s_converted)
			if action is None:
				output.append(s)
				continue
			a = delta(action[1][0][1])
			S_prime_2 = gamma_p(s, a)
			for s_prime, _ in S_prime_2:
				S_prime.append(s_prime)
		S = S_prime
	return output

def checkPrecs(precs, precsNeg, s):
	return set(precs).issubset(s) and len(set(s).intersection(set(precsNeg))) == 0

def delFromState(dels, s):
	return list(set(s) - set(dels))

def addToState(adds, s):
	return list(set(s).union(set(adds)))

def state_convert(s, gamma_p_dict):
	if type(s) == str:
		return s
	currentPosition = getStartingPosition(s)
	for state in gamma_p_dict['states']:
		if state in s or state+'{},{}'.format(currentPosition[0], currentPosition[1]) in s:
			return state
	return 'default'

def policy_convert(Pi, gamma_p_dict):
	Pi_new = []
	for states, action in Pi:
		states_new = list(map(lambda state: state_convert(state, gamma_p_dict), states))
		Pi_new.append((states_new, action))
	return Pi_new

def state_transition_function_policy_dict():
	d = {
		'states': ['default', 'object_detected', 'area_cleared', 'area_failed', 'object_classified_as_garbage', 'garbage_localized', 'gripper_full_expected', 'gripper_empty_unexpected'],
		'actions': ['a_scan_area', 'a_classify_object', 'a_pick_up_garbage', 'a_localize_garbage', 'a_collect_garbage'],
		'transitions': {'a_scan_area': [('default', 'object_detected', 0.8), ('default', 'area_cleared', 0.2 * 0.9), ('default', 'area_failed', 0.2 * 0.1)],
						'a_classify_object': [('object_detected', 'object_classified_as_garbage', 0.5), ('object_detected', 'default', 0.5)],
						'a_pick_up_garbage': [('garbage_localized', 'gripper_full_expected', 0.75), ('garbage_localized', 'gripper_empty_unexpected', 0.25)],
						'a_localize_garbage': [('object_classified_as_garbage', 'garbage_localized', 1), ('gripper_empty_unexpected', 'garbage_localized', 1)],
						'a_collect_garbage': [('gripper_full_expected', 'default', 1)]
						}
	}
	return d

def create_isomorphic_task_network(tn):
	T, prec, alpha = tn
	T_prime = []
	prec_prime = []
	alpha_prime = []
	mapping = {}
	for t in T:
		t_prime = str(uuid.uuid4())
		T_prime.append(t_prime)
		mapping[t] = t_prime
	for pair in prec:
		t_1, t_2 = pair
		pair_prime = mapping[t_1], mapping[t_2]
		prec_prime.append(pair_prime)
	for pair in alpha:
		t, n = pair
		pair_prime = mapping[t], n
		alpha_prime.append(pair_prime)
	return T_prime, prec_prime, alpha_prime

def getTaskIdForName(alpha, name):
	for (t, n) in alpha:
		if n == name:
			return t

def restrictPrecToTasks(prec, T):
	prec_prime = []
	for pair in prec:
		t_1, t_2 = pair
		if t_1 in T and t_2 in T:
			prec_prime.append(pair)
	return prec_prime

def restrictAlphaToTasks(alpha, T):
	alpha_prime = []
	for pair in alpha:
		t, _ = pair
		if t in T:
			alpha_prime.append(pair)
	return alpha_prime

def applyDecompositionMethod(m, tn):
	n_replace, tn_m = m
	T, prec, alpha = tn
	t_replace = getTaskIdForName(alpha, n_replace)
	tn_prime = create_isomorphic_task_network(tn_m)
	T_prime, prec_prime, alpha_prime = tn_prime
	T_2 = list((set(T) - set([t_replace])).union(T_prime))
	prec_X = list(set([(t_1, t_2) for t_1 in list(map(lambda e: e[0], prec)) for t_2 in T_prime if (t_1, t_replace) in prec]).union(set([(t_1, t_2) for t_2 in list(map(lambda e: e[1], prec)) for t_1 in T_prime if (t_replace, t_2) in prec])))
	
	prec_2 = restrictPrecToTasks(list(set(prec).union(set(prec_prime)).union(set(prec_X))), T_2)
	alpha_2 = restrictAlphaToTasks(list(set(alpha).union(set(alpha_prime))), T_2)
	return T_2, prec_2, alpha_2

def areTaskNetworksIsomorphic(tn_1, tn_2):
	T_1, prec_1, alpha_1 = tn_1
	T_2, prec_2, alpha_2 = tn_2
	b_1 = set(T_1) == set(T_2)
	b_2 = set(map(lambda e: e[1], alpha_1)) == set(map(lambda e: e[1], alpha_2))
	b_3 = set(map(lambda e: (dict(alpha_1)[e[0]], dict(alpha_1)[e[1]]), prec_1)) == set(map(lambda e: (dict(alpha_2)[e[0]], dict(alpha_2)[e[1]]), prec_2))
	return b_1 and b_2 and b_3

def getTaskNetworkSignature(tn):
	T, prec, alpha = tn
	prec_sig = list(map(lambda e: (dict(alpha)[e[0]], dict(alpha)[e[1]]), prec))
	alpha_sig = list(map(lambda e: e[1], alpha))
	return prec_sig, alpha_sig

def filterStatesForSubroutine(tnsAndS, placeholders, S, gamma, gamma_p, delta, dim):
	filterMap = defaultdict(lambda: [0, [], None])
	allStates = list(map(lambda e: e[1], tnsAndS))
	for tn, (s, pOrig) in tnsAndS:
		T, prec, alpha = tn
		sigStr = str(getTaskNetworkSignature(tn))
		d = state_transition_function_policy_dict()
		placeholders_converted = dict(map(lambda e: (e[0], (lambda state: toPolicyFunction(state, policy_convert(e[1][0], d), d), e[1][1])), placeholders.items()))
		p, _ = calcPlanRobustness((T, alpha), placeholders_converted, allStates, gamma, d, gamma_p, delta, dim)
		if p >= filterMap[sigStr][0]:
			filterMap[sigStr][0] = p
			filterMap[sigStr][1] = (s, pOrig)
			filterMap[sigStr][2] = tn
	sortedFilterList = list(filterMap.items())
	sortedFilterList.sort(key=lambda keyValuePair: keyValuePair[1][0])
	if len(sortedFilterList) == 0:
		return False
	result = list(map(lambda e: (e[1][2], e[1][1]), sortedFilterList))
	return result

def getOrderRoots(T, prec):
	T_remove = T.copy()
	roots = set()
	for pair in prec:
		t_1, t_2 = pair
		if t_1 in T_remove:
			T_remove.remove(t_1)
		if t_2 in T_remove:
			T_remove.remove(t_2)
		isRoot = True
		for pair_prime in prec:
			t_1_prime, t_2_prime = pair_prime
			if t_1 == t_2_prime:
				isRoot = False
				break
		if isRoot:
			roots.add(t_1)
	return list(roots) + T_remove

def isgoal(tn):
	return tn == ([], [], [])

def getMethods(n, M):
	methods = []
	for m in M:
		c, _ = m
		if c == n:
			methods.append(m)
	return methods

def stateEquals(s_1, s_2):
	return len(set(s_1) - set(s_2)) == 0 and len(set(s_2) - set(s_1)) == 0

def gammaDictToFunc(gamma):
	return lambda s, a: list(map(lambda tup: tup[1:], list(filter(lambda e: e[0] == s, gamma['transitions'][a]))))

def gammaDictToMatrix(gamma, Pi, delta):
	M = []
	transitions = gamma['transitions']
	states = gamma['states']
	for sFrom in states:
		row = []
		for sTo in states:
			p = getTransitionProb(transitions, sFrom, sTo, Pi, delta)
			row.append(p)
		M.append(row)
	return np.array(M)

def getTransitionProb(transitions, sFrom, sTo, Pi, delta):
	for a, tups in transitions.items():
		for sFrom_prime, sTo_prime, p in tups:
			if delta(Pi(sFrom_prime)[1][0][1]) == a:
				if sFrom == sFrom_prime and sTo == sTo_prime:
					return p
	return 0

###############################
# PLANNING SUBROUTINE FUNCTIONS
###############################

def reportToHuman(s):
	targetPos = getTargetPosition(s)
	if 'area_cleared{},{}'.format(targetPos[0], targetPos[1]) in s:
		return (['t_report_success'], [], [('t_report_success', 'n_report_success')]), s
	return False

def calcShortestPath(s, gamma, dim):
	sOrig = s.copy()
	dijkstraGrid = worldToDijkstraGrid(s, dim)
	startingPos = getStartingPosition(s)
	targetPos = getTargetPosition(s)
	dijkstraPath = shortestPath(dijkstraGrid, '{},{}'.format(startingPos[0], startingPos[1]), '{},{}'.format(targetPos[0], targetPos[1]))
	if dijkstraPath is not None:
		currentPosition = startingPos
		operators = []
		for i in range(len(dijkstraPath)):
			dijkstraPos = dijkstraPath[i]
			pParts = dijkstraPos.split(',')
			dijkstraPos = int(pParts[0]), int(pParts[1])
			x_new = dijkstraPos[0]
			y_new = dijkstraPos[1]
			for d in [0, 45, 90, 135, 180, 225, 270, 315]:
				if calcMove(currentPosition[0],currentPosition[1],d) == (x_new, y_new):
					operators.append('n_move_from_to{},{},{}'.format(currentPosition[0],currentPosition[1],d))
					s = gamma(s, operators[-1], dim)
					break
			currentPosition = x_new, y_new
		T = []
		prec = []
		alpha = []
		indexMap = defaultdict(lambda: 0)
		if len(operators) > 0:
			o1 = operators[0]
			t1 = 't'+o1[1:]
			T.append(t1)
			indexMap[t1] += 1
			lastTask = t1
			namingPair = t1, o1
			alpha.append(namingPair)
			for i in range(1, len(operators)):
				o = operators[i]
				t = 't'+o[1:]
				if indexMap[t] > 0:
					t += '{}'.format(indexMap[t])
				indexMap[t] += 1
				T.append(t)
				orderingPair = lastTask, t
				prec.append(orderingPair)
				namingPair = t, o
				alpha.append(namingPair) 
				lastTask = t
		return (T, prec, alpha), sOrig

def chooseWaypoint(s, gamma, dim):
	vals = []
	for i in range(dim):
		for j in range(dim):
			val = getAreaVal(s, i, j), i, j
			vals.append(val)
	vals.sort(key=lambda e: -e[0])
	for index in range(len(vals)):
		chosen = vals[index]
		if 'area_visited{},{}'.format(chosen[1], chosen[2]) in s:
			continue
		n = 'n_set_target{},{}'.format(chosen[1], chosen[2])
		T = ['t'+n[1:]]
		prec = []
		alpha = [(T[0], n)]
		a = 'a'+ n[1:]
		s_prime = gamma(s, a, dim)
		yield (T, prec, alpha), s

def cleanUpGarbage(S, z, gamma_p, NP, A, delta, dim):
	A = ['a_scan_area',
		'a_classify_object',
		'a_pick_up_garbage',
		'a_localize_garbage',
		'a_collect_garbage']
	X_0 = ['t_policy']
	G = []
	Pi = []
	M = [('t_policy', ['a_scan_area', 't_after_scan']),
		('t_after_scan', []),
		('t_after_scan', ['a_classify_object', 't_after_classify']),
		('t_after_classify', ['a_localize_garbage', 'a_pick_up_garbage', 't_after_pick_up']),
		('t_after_pick_up', ['t_after_classify']),
		('t_after_pick_up', ['a_collect_garbage', 't_policy'])]
	gamma = yoyo.state_transition_function_policy
	states = list(map(lambda e: e[0], S))
	F = [(states, X_0)]
	Pi = yoyo.yoyo(F, G, M, Pi, gamma, A, states, X_0, z)
	Pi_func = lambda state: toPolicyFunction(state, Pi, gamma_p)
	d = state_transition_function_policy_dict()
	_, S_result = calcPlanRobustness((['t_temp'], ('t_temp', 'n_temp')), {'t_temp': (lambda state: toPolicyFunction(state, policy_convert(Pi, d), d), None)}, S, None, d, gamma_p, delta, dim)	
	return Pi, S_result

#############################
# PROBABILISTIC EXECUTABILITY
#############################

def calcProbabilisticExecutabilityOfPolicy(pi, s_p, gamma_p, Pr):
	s, p = s_p
	S_O = exitStates(pi, [s], gamma, gamma_p, Pr)
	#output = calcPolicyOutput(Pi, list(map(lambda e: e[0], S)), gamma_p, delta)
	P = stateActionTransitionFunctionToMatrix(pi, gamma_p, Pr)
	mc = markovChain(P)
	mc.computePi('power')
	return list(filter(lambda x: x[0] in S_O, zip(getStates(gamma_p), (lambda y: p * y, mc.pi.values()))))

def calcProbabilisticExecutabilityOfPlan(Pi, s_p, gamma, gamma_p, Pr):
	s, p = s_p
	if len(T) == 0:
		return 1, S
	pi_1 = Pi[0]
	o = None, 0
	if type(pi_1) is not tuple:
		s_prime, p_prime = calcProbabilisticExecutabilityOfPolicy(pi_1, (s, p), gamma_p, Pr)[0]
		if p_prime == 0:
			return o
		o = s_prime, p_prime
	else:
		for i in range(pi_1):
			a = pi_1[i]
			s = gamma(s, a)
			if s is None:
				return o
		o = s, p
	if len(T) == 1:
		return o
	Pi = tuple(list(Pi)[1:])
	return calcProbabilisticExecutabilityOfPlan(Pi, o, gamma, gamma_p, Pi)

############
# SIMULATION
############

def simulate(Pi, s, n_steps, gamma, gamma_p):
	executionTrace = []
	step = 0
	while step < n_steps:
		if len(Pi) == 0:
			return executionTrace
		if type(Pi[0]) is tuple:
			for a in Pi[0]:
				s = gamma(s, a)
				if s is None:
					return False
				executionTrace.append(a)
				step += 1
		else:
			while step < n_steps:
				a = Pi[0](s)
				if a is None:
					break
				s = simulatePolicy(s, a, gamma_p)
				executionTrace.append(a)
				step += 1
			if step == n_steps and Pi[0](s) is not None:
				return False
		Pi = tuple(list(Pi)[1:])
	return executionTrace, s

def simulatePolicyTaskNetwork(s, t_and_alpha, delta, gamma_p):
	t, alpha = t_and_alpha
	path = []
	a = delta(dict(alpha)[t])
	path.append(a)
	state_prob_pairs = gamma_p(s, a)
	intervals_prob = []
	prob_cum = 0
	for state_prob_pair in state_prob_pairs:
		s_prime, prob = state_prob_pair
		interval_prob = prob_cum, prob_cum+prob
		intervals_prob.append(interval_prob)
		prob_cum += prob
	r = random.random()
	index = 0
	for interval_prob in intervals_prob:
		p_1, p_2 = interval_prob
		if r >= p_1 and r < p_2:
			break
		index += 1
	s = state_prob_pairs[index][0]
	return s, path

################
# HTNLINPOL-PLAN
################

def htnlinpolplan(V, N_c, N_p, N_sub, M, M_sub, M_sub_p, gamma, gamma_p, Pr, delta, s_I, tn, n_steps = 1000000):
	generators = {}
	Pi = ()
	step = 0
	fringe = [(s_I, tn, Pi)]
	while step < n_steps:
		step += 1
		if len(fringe) > 0:
			s, tn, Pi = fringe.pop()
			T, prec, alpha = tn
			T_s, alpha_s = sol
			if isgoal(tn):
				yield Pi
				continue
			roots = getOrderRoots(T, prec)
					
			for t in roots:
				n = dict(alpha)[t]
				if n in N_c:
					for m in getMethods(n, M):
						tn_prime = applyDecompositionMethod(m, tn)
						fringe.append((copy.deepcopy(s), tn_prime, Pi))
					break
				elif n in N_p:
					new_fringe_element = apply((s, tn, Pi), t, delta, gamma)
					if new_fringe_element is not None:
						fringe.append(new_fringe_element)
				else:
					for m in getMethods(n, M_sub + M_sub_p):
						r = applySubroutineExecutionMethod(m, tn, copy.deepcopy(s), G, t, gamma, gamma_p, Pr)
						if r is False:
							continue
						pi, tn_prime, s_prime, G = r
						Pi_prime = copy.deepcopy(Pi)
						Pi_prime = tuple(list(Pi_prime).append(pi))
						fringe.append((s_prime, tn_prime, Pi_prime))
	yield false

def applySubroutineExecutionMethod(m, tn, s, G, t, Sigma, Sigma_p, Pr):
	n, f = m
	T, prec, alpha = m
	if t in G:
		result = G[t]
	else:
		result = f(s)
		G[t] = result
	try:
		pi = next(r)
		if pi is None:
			return False
		s_prime = exitStates(pi, [s], gamma, gamma_p, Pr)[0]
	except StopIteration:
		return false
	T_prime = T.copy()
	T_prime.remove(t)
	prec_prime = list(filter(lambda pair: t not in pair, prec))
	alpha_prime = list(filter(lambda pair: t != pair[0], alpha))
	return pi, (T_prime, prec_prime, alpha_prime), s_prime, G

def apply(e, t, delta, gamma):
	s, tn, Pi = e
	T, prec, alpha = tn
	n = dict(alpha)[t]
	s_prime = gamma(s, delta(n))
	if len(s) is None:
		return False
	T.remove(t)
	prec_prime = list(filter(lambda pair: t not in pair, prec))
	alpha_prime = list(filter(lambda pair: t != pair[0], alpha))
	T_s.append(t)
	Pi_prime = copy.deepcopy(Pi)
	Pi_prime = tuple(list(Pi_prime).append((delta(n),)))
	return s_prime, (T, prec_prime, alpha_prime), Pi_prime

if __name__ == '__main__':

	pass
