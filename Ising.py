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
	
	def pdf(self):	#Get probability density function of ising model with parameters h, J
	
		self.P=np.zeros(2**self.size)
		for n in range(2**self.size):
			s=bitfield(n,self.size)*2-1
#			self.P[n]=np.exp(self.Beta*(np.dot(s,self.h) + np.dot(np.dot(s,self.J),s)))
			P1=np.exp(self.Beta*(np.dot(s,self.h) + np.dot(np.dot(s,self.J),s)))
			self.P.itemset(n,P1)
		self.Z=np.sum(self.P)
		self.P/=self.Z

	def MetropolisStep(self,i=None):	    #Execute step of Metropolis algorithm
		if i is None:
			i = np.random.randint(self.size)
		eDiff = self.deltaE(i)
		if eDiff <= 0 or np.log(np.random.rand())< -self.Beta*eDiff:    # Metropolis
			self.s[i] = -self.s[i]

	def deltaE(self,i):		#Compute energy difference between two states with a flip of spin i
		return 2*(self.s[i]*self.H[i] + np.sum(self.s[i]*(self.J[i,:]*self.s)+self.s[i]*(self.J[:,i]*self.s)))

	def SequentialMetropolisStep(self):
		for i in np.random.permutation(self.size):
			self.MetropolisStep(i)

		
	def GlauberStep(self,i=None):			#Execute step of Glauber algorithm
		if i is None:
			i = np.random.randint(self.size)
		h = 2*self.s[i]*(self.H[i] + np.dot(self.J[i,:]+self.J[:,i],self.s))
		self.s[i] = int(random.rand() < np.tanh(self.Beta*h))    # Glauber

	def SequentialGlauberStep(self):
		for i in np.random.permutation(self.size):
			self.GlauberStep(i)
	
def bool2int(x):				#Transform bool array into positive integer
    y = 0
    for i,j in enumerate(np.array(x)[::-1]):
        y += j*2**i
    return y
    
def bitfield(n,size):			#Transform positive integer into bit array
    x = [int(x) for x in bin(n)[2:]]
    x = [0]*(size-len(x)) + x
    return np.array(x)
