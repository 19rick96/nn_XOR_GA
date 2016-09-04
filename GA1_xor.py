import numpy as np
import random
import operator
import cPickle as pickle

range_init = 1.0
mut_v = 1.0
pop = 60

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
	return f

#mutate nodes

def mutate_weights(arr):
	for i in range(0,len(arr)):
		a = random.randint(1,9)
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

#initial population 
population = np.random.uniform(-1.0*range_init,range_init,(pop,9))
loss_val = 4.0
itr = 0

while loss_val>0.02:
	#calc fitness and sort
	itr = itr + 1
	loss_arr = []
	for i in range(0,pop):
		fit = loss(population[i])
		elem = []
		elem.append(i)
		elem.append(fit)
		loss_arr.append(elem)
	loss_arr = sorted(loss_arr,key=operator.itemgetter(1))
	loss_val = loss_arr[0][1]
	print "%d   %f"%(itr,loss_val)
	np.savetxt("fit3.txt",population[loss_arr[0][0]])
	#separate top 8
	top_8 = []
	for i in range(0,40):
		top_8.append(population[loss_arr[i][0]])
	par = np.asarray(top_8)
	np.savetxt(outfile,par,fmt='%-7.4f')
	outfile.write("\n")
	#mating phase ( 40 children )
	children = []
	for j in range(0,3):
		for i in range(0,20):
			c1 = top_8[2*i]
			c2 = top_8[(2*i)+1]
			c3 = crossover_weights(c1,c2)
			c3 = mutate_weights(c3)
			children.append(c3)
		random.shuffle(top_8)
	children = np.asarray(children)
	population = children		

