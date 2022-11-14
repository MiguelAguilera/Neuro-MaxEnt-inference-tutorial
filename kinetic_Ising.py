import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

class Ising:
	def __init__(self, netsize):	#Create ising model
	
		self.size=netsize
		self.H=np.zeros(netsize)
		self.J=np.zeros((netsize,netsize))
		self.randomize_state()
		self.Beta=1.0
	
	def randomize_state(self):
		self.s = np.random.randint(0,2,self.size)*2-1
		
	def GlauberStep(self,i=None):			#Execute step of Glauber algorithm
		if i is None:
			i = np.random.randint(self.size)
		h = self.H[i] + np.dot(self.J[i,:],self.s)
		self.s[i] = int(np.random.rand()*2-1 < np.tanh(self.Beta*h))*2-1   # Glauber

	def SequentialGlauberStep(self):
		for i in np.random.permutation(self.size):
			self.GlauberStep(i)
