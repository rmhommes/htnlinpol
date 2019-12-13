from dijkstra_algorithm.dijkstra import *
import random
from collections import defaultdict
import copy
import yoyo
import subprocess
import sys
import time

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

def getAreaVal(s, i, j):
	for v in s:
		prefix = 'areaValue{},{},'.format(i, j)
		if v.startswith(prefix):
			return int(v[len(prefix):])

def getCurrentPosition(s):
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

def toPolicyFunction(s, Pi, gamma_p):
	a = None
	for S, a_prime in Pi:
		r = gamma_p(s, a_prime)
		if isStateSubSet([s], S) and not (r is None or len(r) == 0): # isStateDoubleSubSet([s], S)
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

def checkPrecs(precs, precsNeg, s):
	return set(precs).issubset(s) and len(set(s).intersection(set(precsNeg))) == 0

def delFromState(dels, s):
	return list(set(s) - set(dels))

def addToState(adds, s):
	return list(set(s).union(set(adds)))

def create_isomorphic_task_network(tn, index):
	T, prec, alpha = tn
	T_prime = []
	prec_prime = []
	alpha_prime = []
	mapping = {}
	for t in T:
		t_prime = t+'{}'.format(index)
		index += 1
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
	return (T_prime, prec_prime, alpha_prime), index

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

def applyDecompositionMethod(m, tn, index):
	n_replace, tn_m = m
	T, prec, alpha = tn
	t_replace = getTaskIdForName(alpha, n_replace)
	tn_prime, index = create_isomorphic_task_network(tn_m, index)
	T_prime, prec_prime, alpha_prime = tn_prime
	T_2 = list((set(T) - set([t_replace])).union(T_prime))
	prec_X = list(set([(t_1, t_2) for t_1 in list(map(lambda e: e[0], prec)) for t_2 in T_prime if (t_1, t_replace) in prec]).union(set([(t_1, t_2) for t_2 in list(map(lambda e: e[1], prec)) for t_1 in T_prime if (t_replace, t_2) in prec])))
	
	prec_2 = restrictPrecToTasks(list(set(prec).union(set(prec_prime)).union(set(prec_X))), T_2)
	alpha_2 = restrictAlphaToTasks(list(set(alpha).union(set(alpha_prime))), T_2)
	return (T_2, prec_2, alpha_2), index

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
		p, _ = calcPlanRobustness((T, alpha), placeholders, allStates, gamma, gamma_p, delta, dim)
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

###############################
# PLANNING SUBROUTINE FUNCTIONS
###############################

def reportToHuman(s):
	targetPos = getTargetPosition(s)
	if 'area_cleared{},{}'.format(targetPos[0], targetPos[1]) in s:
		return (['t_report_success'], [], [('t_report_success', 'n_report_success')]), s
	if 'area_failed{},{}'.format(targetPos[0], targetPos[1]) in s:
		return (['t_report_fail'], [], [('t_report_fail', 'n_report_fail')]), s
	return False

def calcShortestPath(s, gamma, dim):
	sOrig = s.copy()
	dijkstraGrid = worldToDijkstraGrid(s, dim)
	startingPos = getCurrentPosition(s)
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
	_, S_result = calcPlanRobustness((['t_temp'], ('t_temp', 'n_temp')), {'t_temp': (Pi_func, None)}, S, None, gamma_p, delta, dim)	
	return Pi_func, S_result

#####################
# INTERNAL ROBUSTNESS
#####################

def calcPlanRobustness(sol, placeholders, S, gamma, gamma_p, delta, dim):
	T, alpha = sol
	if len(T) == 0:
		return 1, S
	pi_1 = T[0]
	visited = []
	S_output = []
	dep = []
	pDrop = 0
	pSum = sum(map(lambda e: e[1], S))
	while True:
		S_new = []
		for (s, p) in S:
			if isStateSubSet([list(s)], list(map(lambda e: list(e), visited))):
				if isStateSubSet([list(s)], list(map(lambda e: list(e[0]), S_output))):
					S_output = updateS(S_output, (s, p))
				else:
					for s_dep, S_dep_prime, _ in dep:
						if stateEquals(s_dep, s):
							for s_dep_prime, p_dep in S_dep_prime:
								if stateEquals(s_dep, s_dep_prime) or isStateSubSet([list(s_dep_prime)], list(map(lambda e: list(e[0]), S_output))):
									S_output = updateS(S_output, (s_dep_prime, p * p_dep))
				continue
			visited.append(s)
			if pi_1 in placeholders:
				policy = placeholders[pi_1][0]
				action = policy(s)
				if action is None:
					S_output = updateS(S_output, (s, p))
					continue
				a = delta(action[1][0][1])
				S_prime = gamma_p(s, a)
				dep.append((s, S_prime, a))
				for s_prime, p_prime in S_prime:
					S_new.append((s_prime, p * p_prime))
			else:
				n = dict(alpha)[pi_1]
				a = delta(n)
				s_prime = gamma(s, a, dim)
				if s_prime is not None:
					S_output.append((s_prime, p))
				else:
					pDrop += p
		if len(S_new) == 0:
			break
		else:
			for i in range(len(S_new)):
				for j in range(i+1, len(S_new)):
					if S_new[i] is None or S_new[j] is None:
						continue
					if stateEquals(S_new[i][0], S_new[j][0]):
						S_new[i] = (S_new[i][0], S_new[i][1]+S_new[j][1])
						S_new[j] = None
			S = list(filter(lambda e: e is not None, S_new))
	probs = list(map(lambda e: e[1], S_output))
	R = sum(probs)
	p_output = 0
	p_output = sum(map(lambda e: e[1], S_output)) + pDrop
	if pSum > 0:
		S_output = list(map(lambda e: (e[0], e[1] / float(p_output)), S_output))
		S_output = list(map(lambda e: (e[0], e[1] * float(pSum)), S_output))
	if len(T) == 1:
		return sum(map(lambda e: e[1], S_output)), S_output
	return calcPlanRobustness((T[1:], alpha), placeholders, S_output, gamma, gamma_p, delta, dim)

############
# SIMULATION
############

def simulate(sol, placeholders, s, n_steps, delta, gamma, gamma_p, dim):
	T, alpha = sol
	path = []
	step = 0
	while step < n_steps:
		if len(T) == 0:
			break
		t = T[0]
		T = T[1:]
		if t not in placeholders:
			n = dict(alpha)[t]
			a = delta(n)
			s = gamma(s, a, dim)
			if s is None:
				return s, path
			path.append(a)
			step += 1
		else:
			e = placeholders[t][0]
			t_and_alpha = e(s)
			stepped = False
			while t_and_alpha is not None and step < n_steps:
				s, path_add = simulatePolicyTaskNetwork(s, t_and_alpha, delta, gamma_p)
				path += path_add
				t_and_alpha = e(s)
				step += 1
				stepped = True
			if stepped is False:
				return s, path
	return s, path

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

def htnlinpolplan(P, dim, n_steps = 1000000):
	index = 0
	placeholders = {}
	generators = {}
	D, s_I, tn = P
	N_C, _N_P, N_S, M, M_S, M_S_p, M_M, M_M_p, Sigma, Sigma_p, delta = D[1:]
	A, gamma = Sigma
	A_p, gamma_p = Sigma_p
	step = 0
	S = [(s_I, 1)]
	sol = [], []
	fringe = [(S, tn, sol)]
	while step < n_steps:
		step += 1
		if len(fringe) > 0:
			S, tn, sol = fringe.pop()
			T, prec, alpha = tn
			T_s, alpha_s = sol
			if isgoal(tn):
				yield sol, placeholders, S
				continue
			roots = getOrderRoots(T, prec)
					
			for t in roots:
				n = dict(alpha)[t]
				if n in N_C:
					for m in getMethods(n, M):
						tn_prime, index = applyDecompositionMethod(m, tn, index)
						fringe.append((copy.deepcopy(S), tn_prime, sol))
					break
				elif n in _N_P and len(S) == 1:
					new_fringe_element = apply((S, tn, sol), t, delta, gamma, dim)
					if new_fringe_element is not None:
						fringe.append(new_fringe_element)
				else:
					execResults = []
					if n in _N_P and len(S) > 1:
						execResults = [preApply((S, tn, sol), t, delta, gamma, dim)]
						if execResults[0] is False:
							continue
					else:
						nthMethod = 0
						M_S_for_n = getMethods(n, M_S)
						M_S_p_for_n = getMethods(n, M_S_p)
						M_M_for_n = getMethods(n, M_M)
						M_M_p_for_n = getMethods(n, M_M_p)
						methods = M_S_for_n + M_S_p_for_n + M_M_for_n + M_M_p_for_n
						for m in methods:
							if m in M_S_for_n or m in M_S_p_for_n:
								hasPolicyResult = m in M_S_p_for_n
								execMethodResult = applySingleResultPlanningSubroutineExecutionMethod(m, tn, index, copy.deepcopy(S), hasPolicyResult, placeholders, Sigma, Sigma_p, delta, dim)
								if execMethodResult is False:
									continue
								execResult, placeholders = execMethodResult
							else:
								hasPolicyResult = m in M_M_p_for_n
								key = t+'{}'.format(nthMethod)
								nthMethod += 1
								execMethodResult = applyMultipleResultPlanningSubroutineExecutionMethod(m, tn, index, copy.deepcopy(S), hasPolicyResult, placeholders, generators, key, Sigma, Sigma_p, delta, dim)
								if execMethodResult is False:
									continue
								execResult, placeholders, generators = execMethodResult
							execResults.append(execResult)
							if m in M_S_p_for_n or m in M_M_p_for_n:
								T_s += [t]
								alpha_s += [(t, n)]
					for execResult in execResults:
						for (tn_prime, _), S_prime in execResult:
							fringe.append((S_prime, tn_prime, (T_s, alpha_s)))
	yield sol, placeholders, S

def applySingleResultPlanningSubroutineExecutionMethod(m_s, tn, index, S, hasPolicyResult, placeholders, Sigma, Sigma_p, delta, dim):
	n_replace, f = m_s
	if not hasPolicyResult:
		tnsAndS = []
		for s, p in S:
			result = f(list(s), Sigma[1])
			if result is False:
				continue
			tn_m, s_prime = result
			tnsAndS.append((tn_m, (s_prime, p)))
		filterStatesResult = filterStatesForSubroutine(tnsAndS, placeholders, S, Sigma[1], Sigma_p[1], delta, dim)
		if filterStatesResult is False:
			return False
		finalResult = list(map(lambda e: (applyDecompositionMethod((n_replace, e[0]), tn, index), e[1]), filterStatesResult))
		finalFinalResult = []
		for task_network, (s_new, p_new) in finalResult:
			finalFinalResult.append((task_network, [(s_new, p_new)]))
		return finalFinalResult, placeholders
	T, prec, alpha = tn
	nextResult = f(S, Sigma_p[1])
	if nextResult is None:
		return False
	resultPolicy, S_prime = nextResult
	t = getTaskIdForName(alpha, n_replace)
	placeholders[t] = resultPolicy, list(map(lambda e: e[0], S_prime))
	T_prime = T.copy()
	T_prime.remove(t)
	prec_prime = list(filter(lambda pair: t not in pair, prec))
	alpha_prime = list(filter(lambda pair: t != pair[0], alpha))
	return [(((T_prime, prec_prime, alpha_prime), index), S_prime)], placeholders

def applyMultipleResultPlanningSubroutineExecutionMethod(m_m, tn, index, S, hasPolicyResult, placeholders, generators, key, Sigma, Sigma_p, delta, dim):
	n_replace, f = m_m
	if not hasPolicyResult:
		tnsAndS = []
		for s, p in S:
			if key in generators:
				results = generators[key]
				for r, p_prime in results:
					try:
						tn_m, s_prime = next(r)
						tnsAndS.append((tn_m, (s_prime, p_prime)))
					except StopIteration:
						continue
				if len(tnsAndS) == 0:
					return False
			else:
				r = f(list(s), Sigma[1])
				try:
					tn_m, s_prime = next(r)
				except StopIteration:
					continue
			if tn_m is not None and s_prime is not None:
				tnsAndS.append((tn_m, (s_prime, p)))
		filterStatesResult = filterStatesForSubroutine(tnsAndS, placeholders, S, Sigma[1], Sigma_p[1], delta, dim)
		if filterStatesResult is False:
			return False
		finalResult = list(map(lambda e: (applyDecompositionMethod((n_replace, e[0]), tn, index), e[1]), filterStatesResult))
		finalFinalResult = []
		for task_network, (s_new, p_new) in finalResult:
			finalFinalResult.append((task_network, [(s_new, p_new)]))
		return finalFinalResult, placeholders, generators
	results = f(list(map(lambda e: S[0], S)), Sigma_p[1])
	T, prec, alpha = tn
	t = getTaskIdForName(alpha, n_replace)
	try:
		nextResult = next(results)
		if nextResult is None:
			return False
		resultPolicy, S_prime = nextResult
	except StopIteration:
		return False
	placeholders[t] = resultPolicy, list(map(lambda e: e[0], S_prime))
	T_prime = T.copy()
	T_prime.remove(t)
	prec_prime = list(filter(lambda pair: t not in pair, prec))
	alpha_prime = list(filter(lambda pair: t != pair[0], alpha))
	return [(((T_prime, prec_prime, alpha_prime), index), S_prime)], placeholders, generators

def preApply(e, t, delta, gamma, dim):
	S, tn, sol = e
	T, prec, alpha = tn
	T_s, alpha_s = sol
	n_t = dict(alpha)[t]
	for s, p in S:
		s_prime = gamma(s, delta(n_t), dim)
		if s_prime is None:
			continue
		tnsAndS.append((([t], [], alpha), (s_prime, p)))
	filterStatesResult = filterStatesForSubroutine(tnsAndS, placeholders, S, Sigma[1], Sigma_p[1], delta, dim)
	if filterStatesResult is False:
		return False
	finalResult = list(map(lambda e: (tn, e[1]), filterStatesResult))
	finalFinalResult = []
	for task_network, (s_new, p_new) in finalResult:
		finalFinalResult.append((task_network, [(s_new, p_new)]))
	return finalFinalResult

def apply(e, t, delta, gamma, dim):
	S, tn, sol = e
	T, prec, alpha = tn
	T_s, alpha_s = sol
	n_t = dict(alpha)[t]
	S_prime = []
	for s, p in S:
		s_prime = gamma(s, delta(n_t), dim)
		if s_prime is not None:
			S_prime.append((s_prime, p))
	if len(S_prime) != 1:
		return False
	T.remove(t)
	prec_prime = list(filter(lambda pair: t not in pair, prec))
	alpha_prime = list(filter(lambda pair: t != pair[0], alpha))
	T_s.append(t)
	alpha_s.append((t, n_t))
	return S_prime, (T, prec_prime, alpha_prime), (T_s, alpha_s)

if __name__ == '__main__':
	bddEnabled = True
	exp1 = False # Experiment in which a BDD is created to be used by Yoyo to implement the planning subroutine function of choosing the next best waypoint
	exp2 = True # Experiment in which HTNLINPOL-PLAN is empirically analyzed

	if exp1:
		for n in range(1, 11):
			v = n
			bdd = bddvar('o')
			s = 'o'
			ts = time.time()

			for i in range(n):
				for j in range(n):
					for k in range(v):
						if bddEnabled:
							bddAdd = bddvar('areaValue{}_{}_{}'.format(i, j, k))
							bdd = bdd & bddAdd
						s += ' & areaValue{}_{}_{}'.format(i, j, k)
						if bddEnabled:
							bddNest = bddvar('o')
						for i2 in range(n):
							for j2 in range(n):
								if i == i2 and j == j2:
									continue
								s += ' & ('
								pos = []
								for l in range(k):
									pos.append('areaValue{}_{}_{}'.format(i2, j2, l))
								sAdd = ' | '.join(pos)
								if len(sAdd) > 0:
									s += '('+sAdd+')'
								else:
									s += 'o'
								if bddEnabled:
									bddNestAdd1 = None
									if len(sAdd) > 0:
										bddNestAdd1 = expr2bdd(expr(sAdd))
								neg = []
								for l in range(k, v):
									neg.append('~areaValue{}_{}_{}'.format(i2, j2, l))
								sAdd = ' & '.join(neg)
								if len(sAdd) > 0:
									s += ' & ('+sAdd + ')'
								else:
									s += 'o'
								if bddEnabled:
									bddNestAdd2 = None
									if len(sAdd) > 0:
										bddNestAdd2 = expr2bdd(expr(sAdd))
								s += ' & ' + ' & '.join(neg)
								if bddEnabled:
									if bddNestAdd1:
										bddNest = bddNest & bddNestAdd1
									if bddNestAdd2:
										bddNest = bddNest & bddNestAdd2
									bddNest = bddNest & bddvar('targetPos_prime{}_{}'.format(i, j))
								s += ') & targetPos_prime{}_{}'.format(i, j)
						if bddEnabled:
							bdd = bdd | bddNest
			s += ' & '
			disj = []
			for i3 in range(n):
				for j3 in range(n):
					disj.append('targetPos{}_{}'.format(i3, j3) + ' & ' + 'targetPos_prime{}_{}'.format(i3, j3))
			s += '(' + ' | '.join(disj) + ')'
			if bddEnabled:
				bdd = bdd & expr2bdd(expr(' | '.join(disj)))
			t1 = time.time() - ts
			print('N: ', n)
			print('TIME 1: ', t1)
			print(len(s))

			dim = n
			p_obst = 0.0
			directions = [0, 45, 90, 135, 180, 225, 270, 315]
			N_C = ['n_mission', 'n_waypoint']
			_V = V(dim)
			_N_P = N_P(dim)
			_A = A(dim)
			_A_p = A_p()
			N_S = ['n_goto_waypoint', 'n_policy', 'n_choose_waypoint']
			M = [('n_mission', ([], [], [])),
				('n_mission', (['t_waypoint', 't_policy', 't_report_to_human', 't_mission'],[('t_waypoint', 't_policy'), ('t_policy', 't_report_to_human'), ('t_report_to_human', 't_mission')],[('t_waypoint', 'n_waypoint'), ('t_policy', 'n_policy'), ('t_report_to_human', 'n_report_to_human'), ('t_mission', 'n_mission')])), 
				('n_waypoint', (['t_choose_waypoint', 't_goto_waypoint'],[('t_choose_waypoint', 't_goto_waypoint')],[('t_choose_waypoint', 'n_choose_waypoint'), ('t_goto_waypoint', 'n_goto_waypoint')]))]
			gamma = state_transition_function
			gamma_p = lambda s, a: yoyo.state_transition_function_policy(s, a, dim, htnps=True)
			f_goto_waypoint = lambda s, g: calcShortestPath(s, g, dim)
			delta = lambda n_p: (dict(zip(_N_P, _A+_A_p)))[n_p]
			f_clean_garbage = lambda S, _: cleanUpGarbage(S, dim, gamma_p, _N_P, _A, delta, dim)
			f_choose_waypoint = lambda s, g: chooseWaypoint(s, g, dim)
			f_report_to_human = lambda s, _: reportToHuman(s)
			M_S = [('n_goto_waypoint', f_goto_waypoint), ('n_report_to_human', f_report_to_human)]
			M_S_p = [('n_policy', f_clean_garbage)]
			M_M = [('n_choose_waypoint', f_choose_waypoint)]
			M_M_p = []
			Sigma = _A, gamma
			Sigma_p = _A_p, gamma_p
			D = _V, N_C, _N_P, N_S, M, M_S, M_S_p, M_M, M_M_p, Sigma, Sigma_p, delta

			tn_I = ['t_mission'], [], [('t_mission', 'n_mission')]
			s_I = generateWorld(dim, p_obst)
			P = D, s_I, tn_I

			ts = time.time()
			resultGenerator = htnlinpolplan(P, dim)
			for i in range(1):
				try:
					result = next(resultGenerator)
				except StopIteration:
					pass

			t2 = time.time() - ts
			print('TIME 2: ', t2)
			s, path = simulate(result[0], result[1], s_I, 15000, delta, gamma, gamma_p, dim)
			R, _ = calcPlanRobustness(result[0], result[1], [(s_I, 1)], gamma, gamma_p, delta, dim)
			print('RAT: ', t2 / float(t1))
			print()

	if exp2:
		prevTime = None
		for n in range(2, 11):
			for obst in [0.5]:
				numbExp = 100
				totalTime = 0
				totalPlanLength = 0
				totalRobustness = 0
				totalNStates = 0
				for ind in range(numbExp):
					dim = n
					p_obst = obst
					directions = [0, 45, 90, 135, 180, 225, 270, 315]
					N_C = ['n_mission', 'n_waypoint']
					_V = V(dim)
					_N_P = N_P(dim)
					_A = A(dim)
					_A_p = A_p()
					N_S = ['n_goto_waypoint', 'n_policy', 'n_choose_waypoint']
					M = [('n_mission', ([], [], [])),
						('n_mission', (['t_waypoint', 't_policy', 't_report_to_human', 't_mission'],[('t_waypoint', 't_policy'), ('t_policy', 't_report_to_human'), ('t_report_to_human', 't_mission')],[('t_waypoint', 'n_waypoint'), ('t_policy', 'n_policy'), ('t_report_to_human', 'n_report_to_human'), ('t_mission', 'n_mission')])), 
						('n_waypoint', (['t_choose_waypoint', 't_goto_waypoint'],[('t_choose_waypoint', 't_goto_waypoint')],[('t_choose_waypoint', 'n_choose_waypoint'), ('t_goto_waypoint', 'n_goto_waypoint')]))]
					gamma = state_transition_function
					gamma_p = lambda s, a: yoyo.state_transition_function_policy(s, a, dim, htnps=True)
					f_goto_waypoint = lambda s, g: calcShortestPath(s, g, dim)
					delta = lambda n_p: (dict(zip(_N_P, _A+_A_p)))[n_p]
					f_clean_garbage = lambda S, _: cleanUpGarbage(S, dim, gamma_p, _N_P, _A, delta, dim)
					f_choose_waypoint = lambda s, g: chooseWaypoint(s, g, dim)
					f_report_to_human = lambda s, _: reportToHuman(s)
					M_S = [('n_goto_waypoint', f_goto_waypoint), ('n_report_to_human', f_report_to_human)]
					M_S_p = [('n_policy', f_clean_garbage)]
					M_M = [('n_choose_waypoint', f_choose_waypoint)]
					M_M_p = []
					Sigma = _A, gamma
					Sigma_p = _A_p, gamma_p
					D = _V, N_C, _N_P, N_S, M, M_S, M_S_p, M_M, M_M_p, Sigma, Sigma_p, delta

					tn_I = ['t_mission'], [], [('t_mission', 'n_mission')]
					s_I = generateWorld(dim, p_obst)
					P = D, s_I, tn_I

					ts = time.time()
					resultGenerator = htnlinpolplan(P, dim)
					for i in range(1):
						try:
							result = next(resultGenerator)
						except StopIteration:
							pass

					t2 = time.time() - ts
					totalTime += t2
					
					planLength = len(result[0][0])
					totalPlanLength += planLength
					totalNStates += len(result[2])

					R, _ = calcPlanRobustness(result[0], result[1], [(s_I, 1)], gamma, gamma_p, delta, dim)
					totalRobustness += R
			avgTime = totalTime / float(numbExp)
			avgPlanLength = totalPlanLength / float(numbExp)
			avgRobustness = totalRobustness / float(numbExp)
			avgNStates = totalNStates / float(numbExp)
			print('TIME: ', avgTime)
			print('PLANLENGTH: ', avgPlanLength)
			print('ROBUSTNESS: ', avgRobustness)
			if prevTime:
				print('INC TIME: ', avgTime / float(prevTime))
			prevTime = avgTime
