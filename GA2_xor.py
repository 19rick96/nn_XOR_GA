import numpy as np
import random
import operator
import cPickle as pickle

range_init = 3.0
mut_v = 0.75
pop = 160

outfile = file('chromosomes.txt','w+')

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

def feedforward(inp1,inp2,arr):
	l1 = sigmoid(arr[0]+ (inp1*arr[1]) + (inp2*arr[2]))
	l2 = sigmoid(arr[3]+ (inp1*arr[4]) + (inp2*arr[5]))
	out = sigmoid(arr[6] + (l1*arr[7]) + (l2*arr[8]))
	return out

def loss(arr):
	f1 = feedforward(0.0,0.0,arr)
	f2 = feedforward(0.0,1.0,arr)
	f3 = feedforward(1.0,0.0,arr)
	f4 = feedforward(1.0,1.0,arr)
	f = (f1-0.0)**2 + (f2-1.0)**2 + (f3-1.0)**2 + (f4-0.0)**2
	return f/2.0

def mutate_weights(arr):
	for i in range(0,len(arr)):
		a = random.randint(1,5)
		if a==1:
			mut = random.uniform(-1.0*mut_v,mut_v)
			arr[i] = arr[i] + mut
	return arr

def crossover_weights(arr1,arr2):
	arr3 = []
	for i in range(0,len(arr1)):
		a = random.randint(0,1)
		if a == 0:
			arr3.append(arr1[i])
		else :
			arr3.append(arr2[i])
	arr3 = np.asarray(arr3)
	return arr3

def parent_prob(loss,temp):
	pr = np.exp((-1.0*loss)/temp)
	num = random.uniform(0.0,1.0)
	if num<=pr:
		pr = 1
	else:
		pr = 0
	return pr

#initial population
population = np.random.uniform(-1.0*range_init,range_init,(pop,9))
loss_val = 4.0
itr = 0
T = 200

while loss_val>0.01:
	#calc fitness and sort
	itr = itr + 1
	parent = []
	for i in range(0,pop):
		l = loss(population[i])
		if l<loss_val:
			loss_val = l
		p = parent_prob(l,T)
		if p == 1:
			parent.append(population[i])
	if len(parent)%2 != 0:
		del parent[-1]
	parent = np.asarray(parent)
	print "%d   %f"%(itr,loss_val)
	children = []
	for j in range(0,10):
		for i in range(0,len(parent)/2):
			c1 = mutate_weights(parent[2*i])
			c2 = mutate_weights(parent[(2*i)+1])
			c3 = crossover_weights(c1,c2)
			f = loss(c3)
			a = []
			a.append(c3)
			a.append(f)
			children.append(a)
			c1 = mutate_weights(parent[2*i])
			c2 = mutate_weights(parent[(2*i)+1])
			c3 = crossover_weights(c1,c2)
			f = loss(c3)
			a = []
			a.append(c3)
			a.append(f)
			children.append(a)	
		random.shuffle(parent)
	top = []
	children = sorted(children,key=operator.itemgetter(1))
	pop_l = pop
	if pop_l>len(children):
		pop_l = len(children)
	for i in range(0,pop_l):
		top.append(children[i][0])
	top = np.asarray(top)
	population = top
	if T>1.0:
		T = T - 0.5	



























