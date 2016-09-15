__author__ = 'Liu Zebang <zeframlou@gmail.com>'
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__version__ = 1.0

import math
import random
import functools
import configparser
import time
import parallel


c = configparser.ConfigParser()
c.read('config v1.0.ini')
config = c['LOVEREM']

MU0 = 1.256637061 * 10 ** (-6)
BLOCK_LEN = int(config['BLOCK_LEN'])
BLOCK_WID = int(config['BLOCK_WID'])
V = int(config['V'])
A = float(config['A'])
TE = 0.5 * BLOCK_LEN * A / V
OMEGA = 2 * math.pi * int(config['FREQUENCY'])
NS = int(config['NS'])
NP = int(config['NP'])
BEGIN_POINT = config['BEGIN_POINT']
END_POINT = config['END_POINT']
TIRE_LEN = int(config['TIRE_LEN'])
TP0 = int(0.5 * (BLOCK_LEN - TIRE_LEN))
T1 = (BLOCK_LEN - TP0 - TIRE_LEN) * A / V
DEV = int(config['DEV'])
U0 = int(config['U0'])
DIS = float(config['DIS'])
COIL_LEN = int(0.5 * BLOCK_LEN)
COIL_WID = int(0.5 * BLOCK_WID)
P0 = int(0.5 * (BLOCK_LEN - COIL_LEN))
X0 = int(0.5 * (BLOCK_WID - COIL_WID))
CHR_LEN = int(config['CHR_LEN'])
STEPS = int(0.9 * OMEGA / (2 * math.pi))
SEG_COUNT = int(config['SEG_COUNT'])
R0 = 6.4 * 0.01 / math.pi * NP
R1 = 6.4 * 0.01 / math.pi * NS
SAMPLE_STEP = float(config['SAMPLE_STEP'])

POPULATION_SIZE = int(config['POPULATION_SIZE'])
GENERATION_COUNT = int(config['GENERATION_COUNT'])
ELITE_SIZE = int(config['ELITE_SIZE'])
SELECTION = int(config['SELECTION'])
C1 = float(config['C1'])
C2 = float(config['C2'])
C3 = float(config['C3'])
C4 = float(config['C4'])
MUTATION_RATE = float(config['MUTATION_RATE'])
CROSSOVER_RATE = float(config['CROSSOVER_RATE'])
RAD_THRESHOLD = float(config['RAD_THRESHOLD']) * POPULATION_SIZE
RAD_RATE = float(config['RAD_RATE'])
DELTA_X = int(config['DELTA_X'])
RAD_GENERATION = int(config['RAD_GENERATION'])
NEWB_THRESHOLD = float(config['NEWB_THRESHOLD'])
TOURNAMENT_SIZE = int(config['TOURNAMENT_SIZE'])


GENERATION = 0
CONVERGE_COUNTER = 0
CONVERGE_THRESHOLD = 15
NEWB_FLAG = False


class Coil(object):
	def __init__(self, chr):
		self.chr, self.f, self.eta, self.deta, self.w, self.cost = chr, 0.0, 0.0, 0.0, 0.0, 0.0


class Edge(object):
	def __init__(self, x1, y1, x2, y2):
		self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2


class Point(object):
	def __init__(self, x, y, z):
		self.x, self.y, self.z = x, y, z


def B(e, p):
	K = MU0 / (4 * math.pi)
	a = ((e.y2 - e.y1) ** 2 + (e.x2 - e.x1) ** 2) ** 0.5
	if a == 0:
		return 0
	d = ((e.y2 - e.y1) * p.x - (e.x2 - e.x1) * p.y + (e.x2 - e.x1) * e.y1 - (e.y2 - e.y1) * e.x1) / a
	if d == 0:
		return 0
	r = (d ** 2 + p.z ** 2) ** 0.5 * A
	cos2 = ((e.x2 - p.x) * (e.x2 - e.x1) + (e.y2 - p.y) * (e.y2 - e.y1)) / (
		(((e.x2 - p.x) ** 2 + (e.y2 - p.y) ** 2 + p.z ** 2) * ((e.x2 - e.x1) ** 2 + (e.y2 - e.y1) ** 2)) ** 0.5)
	cos1 = ((e.x1 - p.x) * (e.x2 - e.x1) + (e.y1 - p.y) * (e.y2 - e.y1)) / (
		(((e.x1 - p.x) ** 2 + (e.y1 - p.y) ** 2 + p.z ** 2) * ((e.x2 - e.x1) ** 2 + (e.y2 - e.y1) ** 2)) ** 0.5)
	b = K * (cos2 - cos1) / r
	x = b * ((e.x1 - p.x) * (e.y2 - e.y1) - (e.x2 - e.x1) * (e.y1 - p.y)) / ((((e.y2 - e.y1) * p.z) ** 2 + (
		(e.x2 - e.x1) * p.z) ** 2 + ((e.x1 - p.x) * (e.y2 - e.y1) - (e.x2 - e.x1) * (e.y1 - p.y)) ** 2) ** 0.5)
	return x


def spgen(pos, dx, z):
	pointlist = []
	i = X0 + dx
	while i < (X0 + dx + COIL_WID + 1):
		j = pos
		while j < (pos + COIL_LEN + 1):
			p = Point(i, j, z)
			pointlist.append(p)
			j += SAMPLE_STEP
		i += SAMPLE_STEP
	return pointlist


phis, points, edges = 0, spgen(0, 0, 0), []
edges.append(Edge(X0, 0, X0, COIL_LEN))
edges.append(Edge(X0, COIL_LEN, X0 + COIL_WID, COIL_LEN))
edges.append(Edge(X0 + COIL_WID, COIL_LEN, X0 + COIL_WID, 0))
edges.append(Edge(X0 + COIL_WID, 0, X0, 0))
for e in edges:
	for p in points:
		phis += B(e, p) * ((SAMPLE_STEP * A) ** 2)
Ls = NS * abs(phis)
Rs = R1 * 2 * (COIL_WID + COIL_LEN) * A


def begin(coils):
	#Initialize
	logfile = open(time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()), 'w')
	configfile = open('config.ini')
	print(configfile.read(), file=logfile)
	configfile.close()
	if len(coils) == 0:
		coils = coilgen(POPULATION_SIZE)
	elites, new_coils = [], []
	global GENERATION
	global CONVERGE_COUNTER
	for GENERATION in range(1,GENERATION_COUNT + 1):
		print('GEN', GENERATION, '/', GENERATION_COUNT)
		print('GEN', GENERATION, '/', GENERATION_COUNT, file=logfile)

		coils.extend(elites)

		coils = parallel.mapp(eva, coils)

		coils.sort(key=lambda x: x.f, reverse=True)

		coils = coils[:POPULATION_SIZE]
		elites = coils[:ELITE_SIZE]

		for coil in coils:
			print(coil.chr, coil.f, coil.eta, coil.deta, coil.w, coil.cost)

		coil = coils[0]
		print(coil.chr, coil.f, coil.eta, coil.deta, coil.w, coil.cost, file=logfile)
		print(sum([c.f for c in coils]) / len(coils), file=logfile)

		#Convergence detector
		if abs((coils[0].eta - coils[1].eta) / coils[0].eta) < 1e-04:
			CONVERGE_COUNTER += 1
		else:
			CONVERGE_COUNTER = 0
		if CONVERGE_COUNTER == CONVERGE_THRESHOLD:
			break  #End process

		if GENERATION != GENERATION_COUNT:
			if GENERATION >= RAD_GENERATION:
				#Radical mutation enabled
				for (i, c) in enumerate(coils):
					mutate(c, i)
			for j in range(int(POPULATION_SIZE / 2)):
				a, b = tournament(coils, 2, TOURNAMENT_SIZE)
				crossover(a, b, new_coils)
			if GENERATION < RAD_GENERATION:
				#Radical mutation disabled
				for c in new_coils:
					mutate(c, -1)
			coils, new_coils = new_coils, []

			#Remove redundant coils and fill the vacancies with randomly generated coils
			l1, l2 = [], []
			for c in coils:
				if c.chr not in l2:
					l1.append(c)
					l2.append(c.chr)
			l1.extend(coilgen(POPULATION_SIZE - len(l1)))
			coils = l1

	for i in range(SELECTION):
		coil = coils[i]
		print(coil.chr, coil.f, coil.eta, coil.deta, coil.w, coil.cost, file=logfile)
	print('FINISHED')
	print('MISSION COMPLETE. EL PSY CONGROO.', file=logfile)
	logfile.close()
	return None


def tournament(pop, num, t_size):
	new_pop = []
	for i in range(num):
		tour = []
		for j in range(t_size):
			r = random.randint(0, POPULATION_SIZE - 1)
			counter = 0
			while pop[r].chr in map(lambda x:x.chr, tour):
				r = random.randint(0, POPULATION_SIZE - 1)
				counter += 1
				if counter == POPULATION_SIZE:
					print('Prematurely Converged')
					break
			tour.append(pop[r])
		tour.sort(key=lambda x:x.f, reverse=True)
		new_pop.append(tour[0])
	return new_pop



def coilgen(n):
	coils = []
	for i in range(n):
		coil = Coil(BEGIN_POINT)
		for j in range(CHR_LEN):
			x, y = random.randint(0, BLOCK_WID + 1), random.randint(0, BLOCK_LEN + 1)
			if x < 10:
				coil.chr += '0'
			coil.chr += str(x)
			if y < 10:
				coil.chr += '0'
			coil.chr += str(y)
		coil.chr += END_POINT
		coils.append(coil)
	return coils


def seg(N, C):
	step = int(C // N)
	segs = [step for i in range(N)]
	return segs


def crossover(a, b, coils):
	aseg, bseg = seg(SEG_COUNT, len(a.chr) / 4), seg(SEG_COUNT, len(b.chr) / 4)
	aseg1 = [a.chr[int(sum(aseg[:i]) * 4):int(sum(aseg[:i + 1]) * 4)] for i in range(SEG_COUNT)]
	bseg1 = [b.chr[int(sum(bseg[:i]) * 4):int(sum(bseg[:i + 1]) * 4)] for i in range(SEG_COUNT)]
	for i in range(SEG_COUNT):
		r = random.random()
		if r <= CROSSOVER_RATE:
			aseg1[i], bseg1[i] = bseg1[i], aseg1[i]
	coils.append(Coil(functools.reduce(lambda x, y: x + y, aseg1)))
	coils.append(Coil(functools.reduce(lambda x, y: x + y, bseg1)))
	return None


def mutate(c, rank):
	for i in range(int(len(c.chr) / 4 - 2)):
		r = random.random()
		if POPULATION_SIZE - rank < RAD_THRESHOLD:
			det = RAD_RATE
		else:
			det = 1
		if r <= MUTATION_RATE * det:
			x, y, z, dev = int(c.chr[4 * i + 4:4 * i + 6]), int(c.chr[4 * i + 6:4 * i + 8]), '', DEV
			if POPULATION_SIZE - rank < RAD_THRESHOLD:
				dev = DEV * RAD_RATE
			dx, dy = random.randint(-dev, dev + 1), random.randint(-dev, dev + 1)
			while x + dx < 0 or x + dx > BLOCK_WID:
				dx = random.randint(-dev, dev + 1)
			while y + dy < 0 or y + dy > BLOCK_LEN:
				dy = random.randint(-dev, dev + 1)
			if x + dx < 10:
				z += '0'
			z += str(x + dx)
			if y + dy < 10:
				z += '0'
			z += str(y + dy)
			c.chr = c.chr[:4 * i + 4] + z + c.chr[4 * i + 8:]
	return None


def is_left(p0, p1, p2):
	return ((p1.x - p0.x) * (p2.y - p0.y)
			- (p2.x - p0.x) * (p1.y - p0.y))


def wn_poly(point, edges):
	wn = 0 #Winding number. See http://geomalgorithms.com/a03-_inclusion.html
	for e in edges:
		if e.y1 <= point.y and e.y2 > point.y and is_left(Point(e.x1, e.y1, 0), Point(e.x2, e.y2, 0), point) > 0:
			wn += 1
		elif e.y2 <= point.y and is_left(Point(e.x1, e.y1, 0), Point(e.x2, e.y2, 0), point) < 0:
			wn -= 1
	return wn


def ppgen_alt(edges):
	points, xs, ys = [], [e.x1 for e in edges], [e.y1 for e in edges]
	xs.extend([e.x2 for e in edges])
	ys.extend([e.y2 for e in edges])
	edges.append(Edge(edges[-1].x2, edges[-1].y2, edges[0].x1, edges[0].y1))
	for x in range(int(min(xs) / SAMPLE_STEP), int((max(xs) + 1) / SAMPLE_STEP)):
		for y in range(int(min(ys) / SAMPLE_STEP), int((max(ys) + 1) / SAMPLE_STEP)):
			x0, y0 = x * SAMPLE_STEP, y * SAMPLE_STEP
			flag = True
			for (i, e) in enumerate(edges):
				if e.x1 == x0 and e.y1 == y0:
					points.append(Point(x0, y0, 0))
					flag = False
					break
			if flag:
				if wn_poly(Point(x0, y0, 0), edges) != 0:
					points.append(Point(x0, y0, 0))
	edges.pop()
	return points


def eva(coil):
	if coil.f == 0:
		edges1, edges2, cost = [], [], 0
		for i in range(int(len(coil.chr) / 4 - 1)):
			x1, y1, x2, y2 = int(coil.chr[4 * i:4 * i + 2]), int(coil.chr[4 * i + 2:4 * i + 4]), int(
				coil.chr[4 * i + 4:4 * i + 6]), int(coil.chr[4 * i + 6:4 * i + 8])
			edges1.append(Edge(x1, y1, x2, y2))
			edges2.append(Edge(x1, y1 + BLOCK_LEN, x2, y2 + BLOCK_LEN))

		for e in edges1:
			cost += ((e.x1 - e.x2) ** 2 + (e.y1 - e.y2) ** 2) ** 0.5

		phip, pointsp = 0, ppgen_alt(edges1)
		for e in edges1:
			for p in pointsp:
				phip += B(e, p) * ((SAMPLE_STEP * A) ** 2)
		Lp = NP * abs(phip)
		Rp = R0 * cost * A

		coil.cost = cost
		tmp = etacalc_res(edges1, edges2, Lp, Rp, 0)
		coil.eta, coil.w = tmp[0], tmp[1] / TE
		coil.deta = abs((coil.eta - etacalc_res(edges1, edges2, Lp, Rp, DELTA_X)[0]) / coil.eta)

	global GENERATION
	global NEWB_FLAG
	k = 1
	if coil.eta < NEWB_THRESHOLD and not NEWB_FLAG:
		k = 10000000
	else:
		NEWB_FLAG = True
	fitness = k * C1 * coil.eta - C2 * coil.cost - C3 * coil.deta + C4 * coil.w
	coil.f = fitness
	return coil


def etacalc_res(edges1, edges2, Lp, Rp, dx):
	M1, M2 = [], []
	for i in range(STEPS + 1):
		t = i / STEPS * TE
		pos = P0 + V * t
		pointss = spgen(pos, dx, DIS)
		phis1, phis2 = 0, 0

		for e in edges1:
			for p in pointss:
				phis1 += B(e, p) * ((SAMPLE_STEP * A) ** 2)
		M1.append(NS * abs(phis1))
		if t < T1:
			M2.append(0)
		else:
			for e in edges2:
				for p in pointss:
					phis2 += B(e, p) * ((SAMPLE_STEP * A) ** 2)
			M2.append(NS * abs(phis2))
	dM1, dM2 = [], []
	for i in range(STEPS):
		d1 = M1[i + 1] - M1[i]
		d2 = M2[i + 1] - M2[i]
		dt = TE / STEPS
		dM1.append(d1 / dt)
		dM2.append(d2 / dt)

	Zl1, Zl2, Zc1, Zc2 = complex(Rp, OMEGA * Lp), complex(Rs, OMEGA * Ls), complex(0, -(OMEGA ** 2 + (Rp / Lp) ** 2) * Lp / OMEGA), complex(0, -(OMEGA ** 2 + (Rs / Ls) ** 2) * Ls / OMEGA)
	Z = 1
	Wz, W0 = 0, 0
	for i in range(int(T1 * STEPS / TE)):
		Zm = complex(dM1[i], OMEGA * M1[i])
		t = i / STEPS * TE
		U = U0 * (complex(math.cos(OMEGA * t), math.sin(OMEGA * t)))
		Iz = -U * Zm * Zc2 / ((Zl1 * Zl2 - Zm ** 2) * (Z + Zc2) + Z * Zc2 * Zl1)
		I1 = U * (Zc1 + Zl1) / (Zc1 * Zl1) * (Zm ** 2 * (Z + Zc2) / ((Zl1 * Zl2 - Zm ** 2) * (Z + Zc2) + Z * Zc2 * Zl1) + 1)
		Wz += (Iz * Z).real * Iz.real * TE / STEPS
		W0 += U.real * I1.real * TE / STEPS
	for i in range(int(T1 * STEPS / TE), STEPS):
		Zma = complex(dM1[i], OMEGA * M1[i])
		Zmb = complex(dM2[i], OMEGA * M2[i])
		Iz = -U * Zc2 * (Zma + Zmb) / ((Zl1 * Zl2 - Zma ** 2 - Zmb ** 2) * (Z + Zc2) + Z * Zc2 * Zl1)
		I1a = U * (Zc1 + Zl1) / (Zc1 * Zl1) * (Zma * (Zma + Zmb) * (Z + Zc2) / ((Zl1 * Zl2 - Zma ** 2 - Zmb ** 2) * (Z + Zc2) + Z * Zc2 * Zl1) + 1)
		I1b = U * (Zc1 + Zl1) / (Zc1 * Zl1) * (Zmb * (Zma + Zmb) * (Z + Zc2) / ((Zl1 * Zl2 - Zma ** 2 - Zmb ** 2) * (Z + Zc2) + Z * Zc2 * Zl1) + 1)
		Wz += (Iz * Z).real * Iz.real * TE / STEPS
		W0 += U.real * (I1a.real + I1b.real) * TE / STEPS
	eta = Wz / W0
	return eta, Wz

begin([])