# nn_XOR_GA

The aim here is to train a neural network through genetic algorithm to act as a XOR gate.

This problem has often been used as a benchmark to study evolving neural networks. Three types of genetic algorithm have been 
implemented here. The commonly used backpropagation training suffers from the problem of getting stuck at local optima and often
times there is no sense of using backprop. For example : "training a robot to walk" is actually more of a reinforcement learning
problem rather than supervised. On implementing properly, GAs can converge to a nearly global optima very efficiently and also 
comes with mathematical simplicity.

The GAs implemented here evolve only the weights of the neural network and not the neural network topology itself. 

The GA implemented in "GA1_xor.py" is not a very good one. The genes of the chromosome are real numbers and the mutation is just 
a random addition of a random value to a weight (gene). The future candidates are chosen through ranking and completely replaces
the parent generation. It often fails to converge at global optima. I believe this is because of the encoding technique of the
chromosome. Once it gets stuck at a local optima, performing mutations on the chromosomes are not very successfull in pulling it
out f the local optima.

"GA2_xor.py" was an attempt at combining the previous algorithm with Simulated Annealing. Though it showed some minor improvement, 
it is nothing considerable.

"GA3_xor.py" implements a VERY SUCCESSFULL model which converges at an acceptable value almost all of the time. The encoding here 
is much different than the previous cases. Here each gene of the chromosome is each digit of a weight. Each weight is represented 
by 5 digits. First digit determines the sigh of the weight and the later ones represent the value. This encoding is very efficient
in converging to good values as performing mutation on this kind of encoding might result in something completely different. Also
an elitist approach has been used here combined with Stochastic Universal Sampling.
