# Thomas Glynne-Jones
# Python model based on 'A Landscape Theory of Aggregation' (Robert Axelrod and D. Scott Bennett, BJPS 1993)
# PLEASE NOTE I have no affiliation to either of the authors or related institutions and my work has not been endorsed by them. 
# This program is inspired by the above paper, using publicly available information. Any failure to replicate the results of the 
# paper should be read as a consequence of the limitations of my implementation.
# Paper abstract:
# "Aggregation means the organization of elements of a system into patterns that tend to put highly 
# compatible elements together and less compatible elements apart. Landscape theory predicts how 
# aggregation will lead to alignments among actors (such as nations), whose leaders are myopic in 
# their assessments and incremental in their actions. The predicted configurations are based upon 
# the attempts of actors to minimize their frustration based upon their pairwise propensities to align 
# with some actors and oppose others. These attempts lead to a local mini- mum in the energy landscape 
# of the entire system. The theory is supported by the results of two cases: the alignment of 
# seventeen European nations in the Second World War..."
#
# The following python implementation makes a number of simplifications but remains true to the general 
# outline of the model as presented in the paper, seeking to calculate the most stable aggregation in 1938
# The focus is on the method more than the data
#
# Possible aggregations are represented by two vectors representing the respective sides (A and B). 
# Pairwise propensities to cooperate are represented as a matrix, with data from:
#	The Penguin Atlas of World History Vol II
#	The Correlates of War Project (as in the original paper - supplemented with data from wikipedia if necessary (national religion data) - since only the dominant religion is recorded, and most countries were close to religious monocultures at that time, precision is not required)
#
# For each possible aggregation, a total energy is calculated as the sum of the total frustration on each 
# of the two opposing sides - the most stable (lowest energy) aggregation is recorded.
	

import numpy as np

year = 1938
n = 17 					# number of countries
nspace = 2**(n-1)			# number of distinct aggregations

class Country(object):			
	# defines a class for countries

	def __init__(self, number, name, ethnicity, religion, ideology, capability):
		self.number = number
		self.name = name
		self.ethnicity = ethnicity
		self.religion = religion
		self.ideology = ideology	
		self.capability = capability	

	def cname(self):
		return self.name
	def cethnicity(self):
		return self.ethnicity
	def creligion(self):
		return self.religion
	def cideology(self):
		return self.ideology
	def ccapability(self):
		return self.ideology

class Europe(Country):
	# creates a class containing countries

	def __init__(self, country):
        	self.countries = []

	def addcountry(self, country):
        	self.countries.append(country)


### initialising / data import ###


def populate():				# creates 'europe', an instance of the class Europe, containing countries (instances of class Country) considered in the model
	europe = Europe(Country)	# in the following program, matrix rows/columns and vector elements are numbered as below
	
	c1 = Country(1, "Britain", "-", "Protestant", "Democracy", 0.077787)
	europe.addcountry(c1)
	c2 = Country(2, "Czechoslovakia", "-", "Catholic", "Democracy", 0.013048)
	europe.addcountry(c2)
	c3 = Country(3, "Denmark", "-", "Protestant", "Democracy", 0.001991)
	europe.addcountry(c3)
	c4 = Country(4, "Estonia", "-", "Protestant", "Democracy", 0.000611)
	europe.addcountry(c4)
	c5 = Country(5, "Finland", "-", "Protestant", "Democracy", 0.001926)
	europe.addcountry(c5)
	c6 = Country(6, "France", "-", "Catholic", "Democracy", 0.045569)
	europe.addcountry(c6)
	c7 = Country(7, "Germany", "-", "Protestant", "Fascism/Authoritarianism", 0.154222)
	europe.addcountry(c7)
	c8 = Country(8, "Greece", "-", "Orthodox", "Fascism/Authoritarianism", 0.003725)
	europe.addcountry(c8)
	c9 = Country(9, "Hungary", "-", "Catholic", "Democracy", 0.004694)
	europe.addcountry(c9)
	c10 = Country(10, "Italy", "-", "Catholic", "Fascism/Authoritarianism", 0.031849)
	europe.addcountry(c10)
	c11 = Country(11, "Latvia", "-", "Protestant", "Fascism/Authoritarianism", 0.001261)
	europe.addcountry(c11)
	c12 = Country(12, "Lithuania", "-", "Catholic", "Fascism/Authoritarianism", 0.001132)
	europe.addcountry(c12)
	c13 = Country(13, "Poland", "-", "Catholic", "Fascism/Authoritarianism", 0.019573)
	europe.addcountry(c13)
	c14 = Country(14, "Portugal", "-", "Catholic", "Fascism/Authoritarianism", 0.002674)
	europe.addcountry(c14)
	c15 = Country(15, "Romania", "-", "Orthodox", "Democracy", 0.008126)
	europe.addcountry(c15)
	c16 = Country(16, "Soviet Union", "-", "Atheism", "Communism", 0.164359)
	europe.addcountry(c16)
	c17 = Country(17, "Yugoslavia", "-", "Orthodox", "Democracy", 0.006346)
	europe.addcountry(c17)

	return europe


### functions for computing propensity matrices ###


def hocmatrix():
	# Entries of +1 where pairs of countries have a history of conflict (war/border dispute) since 1900. Represented as a triangular matrix to avoid double counting

		#Br Cz De Es Fi Fr Ge Gr Hu It La Li Ro Pl Pr SU Y
	HOC =(([[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],	#Britain
		[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],	#Czechoslovakia
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],	#Denmark
		[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 	#Estonia
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],	#Finland
		[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],	#France
		[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],	#Germany
		[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],	#Greece
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],	#Hungary
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],	#Italy
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],	#Latvia
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],	#Lithuania
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],	#Romania
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],	#Poland
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],	#Portugal
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],	#Soviet Union
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))	#Yugoslavia
	HOCf = np.array(HOC, dtype = float)
	return HOCf

def sizematrix(world):			# Diagonal matrix with entries representing the 'size' (COW capabilities index) of each country as a percentage of global capability
	size = np.zeros((17, 17))
	for country in world.countries:
		i = country.number - 1
		size[i,i] = 100 * country.capability 
	sizea = np.array(size)
	return sizea

def ethnicitymatrix(world):
	ethnicitya = np.zeros((17, 17))
	for country1 in world.countries:
		for country2 in world.countries:
			if country1.number < country2.number:
				ethnicitya[country1.number - 1, country2.number - 1] = 1.0
				if country1.ethnicity == country2.ethnicity:
					ethnicitya[country1.number - 1, country2.number - 1] = -1.0
	return ethnicitya

def religionmatrix(world):
	religiona = np.zeros((17, 17))
	for country1 in world.countries:
		for country2 in world.countries:
			if country1.number < country2.number:
				religiona[country1.number - 1, country2.number - 1] = 1.0
				if country1.religion == country2.religion:
					religiona[country1.number - 1, country2.number - 1] = -1.0
	return religiona

def ideologymatrix(world):
	ideologya = np.zeros((17, 17))
	for country1 in world.countries:
		for country2 in world.countries:
			if country1.number < country2.number:
				ideologya[country1.number - 1, country2.number - 1] = 1.0
				if country1.ideology == country2.ideology:
					ideologya[country1.number - 1, country2.number - 1] = -1.0
	return ideologya


### program functions ###


def vecA(p): 				# given an integer (between 0 and (2^16) - 1), returns a numpy array representing an aggregation (side A) corresponding to that number
	agg = 2**16 + p
	agg_b = format(agg, '017b')
	ls = list(agg_b)
	vector = np.array([ int(x) for x in ls ])
	return vector

	#	e.g. given input number '3'
	#	16 bit binary representation = 0000000000000011
	#	add 2^16:
	#	binary representation gets one extra 1 on the left: 10000000000000011 (by definition, Britain is always on side A)
	#	convert to a vector (1d array)	
	#
	#					   Br Cz De Es Fi Fr Ge Gr Hu It La Li Ro Pl Pr SU Y
	# 	vector representation of side A = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
	# 	side A = [Britain, Soviet Union, Yugoslavia]
	

def otherside(side):			
	# given a vector representation of a side, returns the opposite side in the same form

	vecI = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])	# identity vector for n=17
	otherside = vecI - side
	return otherside
	

def aggregationlist(prop):

	# cycle through each possible aggregation and convert to two vectors representing respective sides
	Emin = 1000000
	Amin = np.zeros(17)

	for i in range(nspace - 1): 	# for each aggregation
		A = vecA(i) 		# countries on side A
		B = otherside(A)	# countries on side B
		PROP = prop
		Etot = calculateenergy(A, B, PROP)
		if Etot < Emin:
			Emin = Etot
			Amin = A
			Bmin = B
	print 'minimum energy =', Emin
	return Amin


def calculateenergy(a, b, prop):	
	# calculates the energy of a particular aggregation, given vectors representing opposing sides a and b, and the propensity matrix prop
	
	PROPxA = np.dot(prop, a)
	EA = np.dot(a.T, PROPxA)
	PROPxB = np.dot(prop, b)
	EB = np.dot(b.T, PROPxB)
	Etot = EA + EB
	return Etot
			
def returnsides(sideA, sideB, world):
	# prints final program output (optimum aggregation)
	
	lsA = []; lsB = []
	
	for i in range(17):
		if sideA[i] == 1:
			for country in world.countries:
				if country.number == i + 1:
					lsA.append(country.name)
	for j in range(17):
		if sideB[j] == 1:
			for country in world.countries:
				if country.number == j + 1:
					lsB.append(country.name)

	print 'stable/minimum energy aggregation:'
	print lsA
	print 'vs'
	print lsB
	


### MAIN PROGRAM ###


europe = populate()			# creates 'europe', an instance of the class 'Europe', containing countries (instances of class Country) considered in the model

#calculate relevant matrices to compute propensity matrix (pairwise propensities)

SIZE = sizematrix(europe)		# diagonal matrix with entries representing relative country sizes according to capabilities index (COW)
HOC = hocmatrix()			# triangular matrix representing whether or not pairs of countries have a history of conflict since 1900 (yes = 1, no = 0)
ETH = ethnicitymatrix(europe)		# triangular matrix comparing dominant ethnic groups b/w country pairs (+1 for different, -1 for same)
REL = religionmatrix(europe)		# as above, comparing dominant religions
IDE = ideologymatrix(europe)		# as above, comparing types of government

PROP = (2*HOC) + ETH + REL + IDE	# relative propensities, not adjusted by country size - HOC gets a multiplicative factor of 2 give it equal weights vs other parameters

PROPxS = np.dot(PROP, SIZE)
PROPabs = np.dot(SIZE, PROPxS) 		# size-adjusted propensity matrix

Amin = aggregationlist(PROPabs)		# cycles through all possible aggregations, computing the total energy for each, returning the most stable aggregation (with minimum energy)
Bmin = otherside(Amin)

returnsides(Amin, Bmin, europe)		# prints aggregation members for most stable configuration

print '\n', '\n', '------------------------------------------------'
print 'NB original model gives the following prediction:'
print '[Br, Cz, De, Fr, Gr, SU, Yu] vs [Es, Fi, Ge, Hu, It, La, Li, Pl, Pt, Ro]'
print '\n', 'actual aggregation at outbreak of war:'
print '[Br, Cz, De, Fr, Gr, Pl, Pt, SU, Yu] vs [Es, Fi, Ge, Hu, It, La, Li, Ro]'
