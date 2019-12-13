import random

##################################################################################################################################################################################################
# Based on the work of Kuter, U., Nau, D., Pistore, M., & Traverso, P. (2009). Task decomposition on abstract states, for planning under nondeterminism. Artiﬁcial Intelligence, 173(5-6), 669–695
##################################################################################################################################################################################################

##############################
# YOYO PROBLEM DOMAIN ELEMENTS
##############################

# Generate non-deterministic actions for world with size z
def genA(z):
	if z > 20:
		return None
	return ['a_move_from_to{},{},{}'.format(i,j,d) for i in range(z) for j in range(z) for d in directions if isMoveLegal(i,j,d,z)] + \
		['a_set_target{},{}'.format(i,j) for i in range(z) for j in range(z)] + \
		['a_scan_area',
		'a_classify_object',
		'a_pick_up_garbage',
		'a_localize_garbage',
		'a_collect_garbage']

def state_transition_function_policy(s, a, z, htnps=False, reverse=False):
	if a == 'a_collect_garbage':
		if not reverse:
			if checkPrecs(['gripper_full_expected'], [], s):
				if htnps:
					return [(addToState([], delFromState(['gripper_full_expected'],s)), 1.0)]
				return [addToState([], delFromState(['gripper_full_expected'],s))]
		else:
			if checkPrecs([], ['gripper_full_expected'], s):
				return [addToState(['gripper_full_expected'], delFromState([],s))]
	if a == 'a_localize_garbage':
		if not reverse:
			if checkPrecs(['object_classified_as_garbage'], ['garbage_localized'], s) or checkPrecs(['gripper_empty_unexpected'], ['garbage_localized'], s):
				if htnps:
					return [(addToState(['garbage_localized'], delFromState(['object_classified_as_garbage', 'gripper_empty_unexpected'],s)), 1.0)]
				return [addToState(['garbage_localized'], delFromState(['object_classified_as_garbage', 'gripper_empty_unexpected'],s))]
		else:
			if checkPrecs(['garbage_localized'], [], s):
				return [addToState(['gripper_empty_unexpected'], delFromState(['garbage_localized'],s)), addToState(['object_classified_as_garbage'], delFromState(['garbage_localized'],s))]
	if a == 'a_pick_up_garbage':
		if not reverse:
			if checkPrecs(['garbage_localized'], ['gripper_full_expected', 'gripper_empty_unexpected'], s):
				if htnps:
					return [(addToState(['gripper_full_expected'], delFromState(['garbage_localized'],s)), 0.75), (addToState(['gripper_empty_unexpected'], delFromState(['garbage_localized'],s)), 0.25)]
				return [addToState(['gripper_full_expected'], delFromState(['garbage_localized'],s)), addToState(['gripper_empty_unexpected'], delFromState(['garbage_localized'],s))]
		else:
			if checkPrecs(['gripper_full_expected'], ['garbage_localized'], s) or checkPrecs(['gripper_empty_unexpected'], ['garbage_localized'], s):
				return [addToState(['garbage_localized'], delFromState(['gripper_full_expected', 'gripper_empty_unexpected'],s))]
	if a == 'a_classify_object':
		if not reverse:
			if checkPrecs(['object_detected'], ['object_classified_as_garbage'], s):
				if htnps:
					return [(addToState(['object_classified_as_garbage'], delFromState(['object_detected'],s)), 0.5), (addToState([], delFromState(['object_detected'],s)), 0.5)]
				return [addToState(['object_classified_as_garbage'], delFromState(['object_detected'],s)), addToState([], delFromState(['object_detected'],s))]
		else:
			if checkPrecs(['object_classified_as_garbage'], ['object_detected'], s):
				return [addToState(['object_detected'], delFromState(['object_classified_as_garbage'],s)), addToState(['object_detected'], delFromState([],s))]
	if a == 'a_scan_area':
		pos = getTargetPosition(s)
		if pos is None:
			return []
		x, y = pos
		if not reverse:
			if checkPrecs(['targetPos{},{}'.format(x, y), 'currentPos{},{}'.format(x, y)], ['area_cleared{},{}'.format(x, y), 'area_failed{},{}'.format(x, y), 'area_visited{},{}'.format(x, y)], s):
				if htnps:
					return [(addToState(['object_detected'], delFromState([],s)), 0.8), 
							(addToState(['area_cleared{},{}'.format(x, y), 'area_visited{},{}'.format(x, y)], delFromState([],s)), 0.2 * 0.9), 
							(addToState(['area_failed{},{}'.format(x, y), 'area_visited{},{}'.format(x, y)], delFromState([],s)), 0.2 * 0.1)]
				return [addToState(['object_detected'], delFromState([],s)), addToState(['area_cleared{},{}'.format(x, y), 'area_visited{},{}'.format(x, y)], delFromState([],s)), addToState(['area_failed{},{}'.format(x, y), 'area_visited{},{}'.format(x, y)], delFromState([],s))]
		else:
			if checkPrecs(['object_detected'], [], s) or checkPrecs(['area_visited{},{}'.format(x, y), 'area_cleared{},{}'.format(x, y)], [], s) or checkPrecs(['area_visited{},{}'.format(x, y), 'area_failed{},{}'.format(x, y)], [], s):
				return [addToState([], delFromState(['object_detected', 'area_visited{},{}'.format(x, y), 'area_cleared{},{}'.format(x, y), 'area_failed{},{}'.format(x, y)],s))]
	prefix = 'a_move_from_to'
	if a.startswith(prefix):
		args = a[len(prefix):]
		argsParts = args.split(',')
		currentPosition = 'currentPos{},{}'.format(int(argsParts[0]), int(argsParts[1]))
		visitedPosition = 'visitedPos{},{}'.format(int(argsParts[0]), int(argsParts[1]))
		precs = [currentPosition]
		d = int(argsParts[2])
		if reverse:
			d = calcOppositeDirection(d)
		newPos = calcMove(int(argsParts[0]), int(argsParts[1]), d)
		newPosition = 'currentPos{},{}'.format(newPos[0], newPos[1])
		precsNeg = ['visitedPos{},{}'.format(newPos[0], newPos[1]), 'targetPos{},{}'.format(int(argsParts[0]), int(argsParts[1]))]
		if checkPrecs(precs, precsNeg+['object_detected', 'object_classified_as_garbage', 'garbage_localized', 'gripper_full_expected', 'gripper_empty_unexpected'], s):
			if htnps:
				return [(addToState([newPosition, visitedPosition],delFromState([currentPosition],s)), 1.0)]
			return [addToState([newPosition, visitedPosition],delFromState([currentPosition],s))]

	prefix = 'a_set_target'
	if a.startswith(prefix):
		args = a[len(prefix):]
		argsParts = args.split(',')
		if not reverse:
			precs = []
			precsNeg = ['area_visited{},{}'.format(int(argsParts[0]), int(argsParts[1]))]
			if checkPrecs(precs, precsNeg+['object_detected', 'object_classified_as_garbage', 'garbage_localized', 'gripper_full_expected', 'gripper_empty_unexpected'], s):
				if htnps:
					return [(addToState(['targetPos{},{}'.format(int(argsParts[0]), int(argsParts[1]))],delFromState(['targetPos{},{}'.format(i,j) for i in range(z) for j in range(z)]+['visitedPos{},{}'.format(i,j) for i in range(z) for j in range(z)],s)), 1.0)]
				return [addToState(['targetPos{},{}'.format(int(argsParts[0]), int(argsParts[1]))],delFromState(['targetPos{},{}'.format(i,j) for i in range(z) for j in range(z)]+['visitedPos{},{}'.format(i,j) for i in range(z) for j in range(z)],s))]
		else:
			pos = getStartingPosition(s)
			if checkPrecs(['targetPos{},{}'.format(int(argsParts[0]), int(argsParts[1]))], [], s):
				return [addToState(['targetPos{},{}'.format(pos[0], pos[1])],delFromState(['targetPos{},{}'.format(int(argsParts[0]), int(argsParts[1]))],s)), addToState([],delFromState(['targetPos{},{}'.format(int(argsParts[0]), int(argsParts[1]))],s))]
	return []

############
# AUXILLIARY
############

def policyStates(Pi):
	return list(sum(list(map(lambda e: e[0], Pi)),[]))

def statesInFringe(F):
	return list(sum(list(map(lambda e: e[0], F)),[]))

def isEmptyHTN(tn):
	return tn == []

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

def weakPreimage(S, gamma, A, dim):
	img = []
	for s in S:
		for a in A:
			S_add = gamma(s, a, dim, reverse=True)
			for s_add in S_add:
				img += [(s_add, a)]
	return img

def strongPreimage(S, gamma, A, dim):
	img = []
	for s in S:
		for a in A:
			S_add = gamma(s, a, dim, reverse=True)
			if isStateSubSet(gamma(s, a, dim), S) and len(S_add) > 0:
				for s_add in S_add:
					img += [(s_add, a)]
	return img

def isCandidateSolution(Pi, S_F, G, S_0, gamma, A, dim, strong=False):
	S_prime = []
	S = G + S_F
	while len(stateSetIntersection(S_prime, S)) != len(S_prime) or len(S_prime) != len(S):
		S_prime = S
		preimage = None
		if strong:
			preimage = strongPreimage(S, gamma, A, dim)
		else:
			preimage = weakPreimage(S, gamma, A, dim)
		Pi_prime = policySetIntersection(Pi, preimage)
		S = S + policyStates(Pi_prime)
		Pi = list(filter(lambda e: not isStateSubSet(e[0], S), Pi))
	return goodPolicy(Pi, S_0, S, strong=strong)

def goodPolicy(Pi, S_0, S, strong=False):
	if strong:
		if not isStateSubSet(S_0, S) or not len(Pi) == 0:
			return False
		return True
	if not isStateSubSet(S_0, S):
		return False
	return True

def getMethodsForTask(t, M):
	M_result = []
	for m in M:
		if m[0] == t:
			M_result.append(m)
	return M_result

def stateEquals(s_1, s_2):
	return len(set(s_1) - set(s_2)) == 0 and len(set(s_2) - set(s_1)) == 0

def isStateSubSet(S1, S2):
	for s1 in S1:
		s1.sort()
	for s2 in S2:
		s2.sort()
	S1 = set(map(lambda e: str(e), S1))
	S2 = set(map(lambda e: str(e), S2))
	return S1.issubset(S2)

def stateSetMinus(S1, S2):
	Sr = []
	for s1 in S1:
		b = False
		for s2 in S2:
			if stateEquals(s1, s2):
				b = True
				break
		if not b:
			Sr.append(s1)
	return Sr

def stateSetIntersection(S1, S2):
	Sr = []
	for s1 in S1:
		b = False
		for s2 in S2:
			if stateEquals(s1, s2):
				b = True
				break
		if b:
			Sr.append(s1)
	return Sr

def policySetIntersection(Pi1, Pi2):
	Pir = []
	for S1, t1 in Pi1:
		bPi = False
		for s2, t2 in Pi2:
			if isStateSubSet([list(s2)], S1) and t1 == t2:
				bPi = True
				break
		if bPi:
			Pir.append((S1, t1))
	return Pir

def isTaskPrimitive(t, A):
	return t in A

def setBasedTransition(S, a, gamma, dim):
	S_prime = []
	for s in S:
		S_prime = S_prime + gamma(s, a, dim)
	return S_prime

def checkPrecs(precs, precsNeg, s):
	return set(precs).issubset(s) and len(set(s).intersection(set(precsNeg))) == 0

def delFromState(dels, s):
	return list(set(s) - set(dels))

def addToState(adds, s):
	return list(set(s).union(set(adds)))

def calcOppositeDirection(d):
	if d == 0:
		return 180
	if d == 45:
		return 225
	if d == 90:
		return 270
	if d == 135:
		return 315
	if d == 180:
		return 0
	if d == 225:
		return 45
	if d == 270:
		return 90
	if d == 315:
		return 135

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

def isMoveLegal(i,j,d,z):
	i, j = calcMove(i,j,d)
	return i >= 0 and j >= 0 and i < z and j < z

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

def getValueOfTarget(pair):
	target, s = pair
	x, y = target[1][0].split('a_set_target')[1].split(',')
	return getAreaVal(s, int(x), int(y))

def getValueOfMove(pair):
	move, s = pair
	if len(move[1]) == 0 or not move[1][0].startswith('a_move_from_to'):
		return 0
	x, y, d = move[1][0].split('a_move_from_to')[1].split(',')
	xFrom, yFrom = calcMove(int(x), int(y), int(d))
	xTo, yTo = getTargetPosition(s)
	return - ((int(xFrom) - int(xTo))**2+(int(yFrom) - int(yTo))**2)**0.5

def resortMoveMethods(s, M, dim):
	if getTargetPosition(s) is None:
		return M
	indexNeg = dim*dim*8
	moveMethods = M[-indexNeg:]
	M = M[:-indexNeg]
	moveMethods = list(map(lambda e: (e, s), moveMethods))
	moveMethods.sort(key=getValueOfMove)
	moveMethods = list(map(lambda e: e[0], moveMethods))
	return M + moveMethods

############
# SIMULATION
############

def simulate(Pi, s, n_steps, gamma, dim):
	path = []
	step = 0
	while step < n_steps:
		stop = True
		for (S, a) in Pi:
			if isStateSubSet([list(s)], S):
				stop = False
				S_prime = gamma(s, a, dim)
				s = random.choice(S_prime)
				break
		if stop:
			return s, path
		path.append(a)
		step += 1
	return s, path	

######
# YOYO
######

# No BDDs are used in this version of Yoyo
def yoyo(F, G, M, Pi, gamma, A, S_0, X_0, dim, strong=False):
	r = list(filter(lambda e: isStateSubSet(S_0, G) and not isEmptyHTN(e[1]), F))
	if len(r) > 0:
		return False

	F = list(filter(lambda x: len(x[0]) > 0, list(map(lambda e: (list(stateSetMinus(e[0], G  + policyStates(Pi))), e[1]), F))))
	
	if len(F) == 0:
		return Pi

	while len(F) > 0:
		r = list(filter(lambda e: isStateSubSet(S_0, G) and not isEmptyHTN(e[1]), F))
		if len(r) > 0:
			return False

		F = list(filter(lambda x: len(x[0]) > 0, list(map(lambda e: (list(stateSetMinus(e[0], G  + policyStates(Pi))), e[1]), F))))
		
		if len(F) == 0:
			return Pi

		S, X = F.pop()
		if isEmptyHTN(X) or len(S) == 0:
			continue
		t = X[0]
		X = X[1:]
		if isTaskPrimitive(t, A):
			S_prime = []
			for s in S:
				if len(gamma(s, t, dim)) > 0:
					S_prime.append(s)
			if len(S_prime) > 0:
				Pi.append((S_prime, t))
				F.append((setBasedTransition(S_prime, t, gamma, dim), X))
			else:
				continue
		else:
			M = resortMoveMethods(S[0], M, dim)
			M_prime = getMethodsForTask(t, M)
			if len(M_prime) == 0:
				continue
			S_prime = S
			for m in M_prime:
				tn = m[1]
				F.append((stateSetMinus(S, S_prime), [t]+X))
				F.append((S_prime, tn+X))
		F_temp = []
		for i in range(len(F)):
			for j in range(i+1, len(F)):
				if F[i] is None or F[j] is None:
					continue
				if F[i][1] == F[j][1]:
					F_temp.append((F[i][0] + F[j][0], F[i][1]))
					F[i] = None
					F[j] = None
		F = list(filter(lambda e: e is not None, F+F_temp))
	F = [(S_0, X)] + F
	F = list(filter(lambda x: len(x[0]) > 0, list(map(lambda e: (stateSetMinus(e[0], policyStates(Pi)), e[1]), F))))
	if len(F) == 0 or len(sum(list(map(lambda e: e[1], F)), [])) == 0:
		return Pi
	Pi = yoyo(F, G, M, Pi, gamma, A, S_0, X_0, dim, strong=False)
	return Pi

if __name__ == '__main__':
	directions = [0, 45, 90, 135, 180, 225, 270, 315]
	dim = 2
	A = genA(dim)
	p_obst = 0.5
	s_I = generateWorld(dim, p_obst)
	X_0 = ['t_mission']
	F = [([s_I], X_0)]
	s_G = s_I[:-2]
	G = []
	chooseWaypointMethods = [('t_choose_waypoint', ['a_set_target{},{}'.format(i,j)]) for i in range(dim) for j in range(dim)]
	chooseWaypointMethods = list(map(lambda e: (e, s_I), chooseWaypointMethods))
	chooseWaypointMethods.sort(key=getValueOfTarget)
	chooseWaypointMethods = list(map(lambda e: e[0], chooseWaypointMethods))
	moveMethods = [('t_move', ['a_move_from_to{},{},{}'.format(i,j,d)]) for i in range(dim) for j in range(dim) for d in directions if isMoveLegal(i,j,d,dim)]

	M = [('t_mission', ['t_waypoint', 't_engagement', 't_mission']), 
		('t_mission', []),
		('t_waypoint', ['t_choose_waypoint', 't_goto_waypoint']),
		('t_engagement', ['a_scan_area', 't_after_scan']),
		('t_after_scan', []),
		('t_after_scan', ['a_classify_object', 't_after_classify']),
		('t_after_classify', ['a_localize_garbage', 'a_pick_up_garbage', 't_after_pick_up']),
		('t_after_pick_up', ['t_after_classify']),
		('t_after_pick_up', ['a_collect_garbage', 't_engagement']),
		('t_goto_waypoint', ['t_move', 't_goto_waypoint']),
		('t_goto_waypoint', [])] + chooseWaypointMethods + moveMethods
	gamma = state_transition_function_policy
	Pi = []
	Pi = yoyo(F, G, M, Pi, gamma, A, [s_I], X_0, dim, strong=False)
	print(Pi)
	s, path = simulate(Pi, s_I, 100, gamma, dim)
	print('PATH', path)
