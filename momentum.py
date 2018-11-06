from util import transform_data, derivative_v,derivative_b1,derivative_w, derivative_b0, generate_Y, generate_T, cross_entropy,\
classification_rate
from sklearn.utils import shuffle
import numpy as np, matplotlib.pyplot as plt 

def batch(learning_rate):

	X,Y = transform_data()
	X,Y = shuffle(X,Y)
	N = len(X)//2
	Xtrain = X[:N]
	Ytrain = Y[:N]
	Ttrain = generate_T(Ytrain)
	Xtest = X[N:]
	Ytest = Y[N:]
	Ttest = generate_T(Ytest)
	N,D = Xtrain.shape
	M = 100
	K = len(set(Y))
	iterations = 50
	batch_N = 250
	batches = len(X)//batch_N
	v = np.random.randn(M,K)/np.sqrt(M+K)
	b_1 = np.random.randn(K)/np.sqrt(K)
	w = np.random.randn(D,M)/np.sqrt(D+M)
	b_0 = np.random.randn(M)/np.sqrt(M)
	batch_cost = []
	batch_cr = []
	best_batch = 0
	best_iteration = 0
	for i in range(iterations):
		for b in range(batches):
			X = Xtrain[b*batch_N:(b+1)*batch_N,:]
			T = Ttrain[b*batch_N:(b+1)*batch_N,:]
			Y,Z = generate_Y('tanh',X,w,b_0,v,b_1)
			Y_test,_ = generate_Y('tanh',Xtest,w,b_0,v,b_1)
			P_test = np.argmax(Y_test,axis=1)
			if b % batches == 0:
				batch_cost.append(cross_entropy(Y_test,Ttest))
				cr = classification_rate(P_test,Ytest)
				batch_cr.append(cr)
				if cr > best_batch:
					best_batch = cr
					best_iteration = i
			v -= learning_rate*derivative_v('tanh',Z,Y,T)
			b_1 -= learning_rate*derivative_b1('tanh',Y,T)
			w -= learning_rate*derivative_w('tanh',X,Y,Z,T,v)
			b_0 -= learning_rate*derivative_b0('tanh',Y,Z,T,v)
		if i % 100 == 0:
			print('Batch Cost: ',batch_cost[i],'Batch Classification: ',batch_cr[i])
	return batch_cost,batch_cr,best_batch,best_iteration

def momentum(learning_rate):

	X,Y = transform_data()
	X,Y = shuffle(X,Y)
	N = len(X)//2
	Xtrain = X[:N]
	Ytrain = Y[:N]
	Ttrain = generate_T(Ytrain)
	Xtest = X[N:]
	Ytest = Y[N:]
	Ttest = generate_T(Ytest)
	N,D = Xtrain.shape
	M = 100
	K = len(set(Y))
	iterations = 50
	batch_N = 250
	batches = N//batch_N
	v = np.random.randn(M,K)/np.sqrt(M+K)
	b_1 = np.random.randn(K)/np.sqrt(K)
	w = np.random.randn(D,M)/np.sqrt(D+M)
	b_0 = np.random.randn(M)/np.sqrt(M)
	mu = .9
	dv = 0
	db_1 = 0
	dw = 0
	db_0 = 0
	momentum_cost = []
	momentum_cr = []
	best_momentum = 0
	best_iteration = 0
	for i in range(iterations):
		for b in range(batches):
			X = Xtrain[b*batch_N:(b+1)*batch_N,:]
			T = Ttrain[b*batch_N:(b+1)*batch_N,:]
			Y,Z = generate_Y('tanh',X,w,b_0,v,b_1)
			Y_test,_ = generate_Y('tanh',Xtest,w,b_0,v,b_1)
			P_test = np.argmax(Y_test,axis=1)
			if b % batches == 0:
					momentum_cost.append(cross_entropy(Y_test,Ttest))
					cr = classification_rate(P_test,Ytest)
					momentum_cr.append(cr)
					if cr > best_momentum:
						best_momentum = cr
						best_iteration = i
			dv = mu*dv-learning_rate*derivative_v('tanh',Z,Y,T)
			db_1 = mu*db_1-learning_rate*derivative_b1('tanh',Y,T)
			dw = mu*dw-learning_rate*derivative_w('tanh',X,Y,Z,T,v)
			db_0 = mu*db_0-learning_rate*derivative_b0('tanh',Y,Z,T,v)
			v += dv
			b_1 += db_1
			w += dw
			b_0 += db_0
		if i % 100 == 0:
			print('Momentum Cost: ',momentum_cost[i],'Momentum Classification: ',momentum_cr[i])
	return momentum_cost,momentum_cr,best_momentum,best_iteration

def nesterov_momentum(learning_rate):

	X,Y = transform_data()
	X,Y = shuffle(X,Y)
	N = len(X)//2
	Xtrain = X[:N]
	Ytrain = Y[:N]
	Ttrain = generate_T(Ytrain)
	Xtest = X[N:]
	Ytest = Y[N:]
	Ttest = generate_T(Ytest)
	N,D = Xtrain.shape
	M = 100
	K = len(set(Y))
	iterations = 50
	batch_N = 250
	batches = N//batch_N
	v = np.random.randn(M,K)/np.sqrt(M+K)
	b_1 = np.random.randn(K)/np.sqrt(K)
	w = np.random.randn(D,M)/np.sqrt(D+M)
	b_0 = np.random.randn(M)/np.sqrt(M)
	mu = .9
	dv = 0
	db_1 = 0
	dw = 0
	db_0 = 0
	nesterov_cost = []
	nesterov_cr = []
	best_nesterov = 0
	best_iteration = 0
	for i in range(iterations):
		for b in range(batches):
			X = Xtrain[b*batch_N:(b+1)*batch_N,:]
			T = Ttrain[b*batch_N:(b+1)*batch_N,:]
			Y,Z = generate_Y('tanh',X,w,b_0,v,b_1)
			Y_test,_ = generate_Y('tanh',Xtest,w,b_0,v,b_1)
			P_test = np.argmax(Y_test,axis=1)
			if b % batches == 0:
				nesterov_cost.append(cross_entropy(Y_test,Ttest))
				cr = classification_rate(P_test,Ytest)
				nesterov_cr.append(cr)
				if cr > best_nesterov:
					best_nesterov = cr
					best_iteration = i
			dv = mu*dv-learning_rate*derivative_v('tanh',Z,Y,T)
			db_1 = mu*db_1-learning_rate*derivative_b1('tanh',Y,T)
			dw = mu*dw-learning_rate*derivative_w('tanh',X,Y,Z,T,v)
			db_0 = mu*db_0-learning_rate*derivative_b0('tanh',Y,Z,T,v)
			v += mu*dv-learning_rate*derivative_v('tanh',Z,Y,T)
			b_1 += mu*db_1-learning_rate*derivative_b1('tanh',Y,T)
			w += mu*dw+learning_rate*derivative_w('tanh',X,Y,Z,T,v)
			b_0 += mu*db_0-learning_rate*derivative_b0('tanh',Y,Z,T,v)
		if i % 100 == 0:
			print('Nesterov Cost: ',nesterov_cost[i],'Nesterov Classification: ',nesterov_cr[i])
	return nesterov_cost,nesterov_cr,best_nesterov,best_iteration

if __name__ == '__main__':

	learning_rate = 10e-4
	batch_cost,batch_cr,best_batch,batch_iteration = batch(learning_rate)
	momentum_cost,momentum_cr,best_momentum,momentum_iteration = momentum(learning_rate)
	nesterov_cost,nesterov_cr,best_nesterov,nesterov_iteration = nesterov_momentum(learning_rate)
	batch_momentum = np.mean(np.array(batch_cost) - np.array(momentum_cost))
	batch_nesterov = np.mean(np.array(batch_cost) - np.array(nesterov_cost))
	momentum_nesterov = np.mean(np.array(momentum_cost) - np.array(nesterov_cost))
	plt.plot(batch_cost,label='Batch')
	plt.plot(momentum_cost,label='Momentum')
	plt.plot(nesterov_cost,label='Nesterov')
	plt.legend()
	plt.show()
	plt.plot(batch_cr,label='Batch')
	plt.plot(momentum_cr,label='Momentum')
	plt.plot(nesterov_cr,label='Nesterov')
	plt.legend()
	plt.show()
	print('')
	print('Best Batch Classification Rate: ',best_batch,'Iteration: ',batch_iteration)
	print('Best Momentum Classification Rate: ',best_momentum,'Iteration: ',momentum_iteration)
	print('Best Nesterov Classification Rate: ',best_nesterov,'Iteration: ',nesterov_iteration)
	print('')
	if batch_momentum > 0:
		print('Batch w/ Standard Momentum error rate is ' + str(batch_momentum) + ' less per iteration on avg than w/o Standard Momentum')
	elif batch_momentum < 0:
		print('Batch w/o Standard Momentum error rate is ' + str(-batch_momentum) + ' less per iteration on avg than w/ Standard Momentum')
	else:
		print('Batch w/ and w/o Standard Momentum have equivalent error rates per iteration.')
	if batch_nesterov > 0:
		print('Batch w/ Nesterov Momentum error rate is ' + str(batch_nesterov) + ' less per iteration on avg than w/o Nesterov Momentum')
	elif batch_nesterov < 0:
		print('Batch w/o Nesterov Momentum error rate is ' + str(-batch_nesterov) + ' less per iteration on avg than w Nesterov Momentum')
	else:
		print('Batch w/ and w/o Nesterov Momentum have equivalent error rates per iteration.')
	if momentum_nesterov > 0:
		print('Nesterov Momentum error rate is ' + str(momentum_nesterov) + ' less per iteration on avg than Standard Momentum')
	elif momentum_nesterov < 0:
		print('Standard Momentum error rate is ' + str(-momentum_nesterov) + ' less per iteration on avg than Nesterov Momentum')
	else:
		print('Standard and Nesterov Momentum have equivalent error rates on avg per iteration.')