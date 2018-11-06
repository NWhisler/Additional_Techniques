from util import transform_data,derivative_v,derivative_b1,derivative_w, derivative_b0, generate_Y, generate_T,\
cross_entropy, classification_rate
from datetime import datetime
from sklearn.utils import shuffle
import numpy as np, matplotlib.pyplot as plt

def derivative_w(X,Y,T):

	return X.T.dot(Y - T)

def derivative_b(Y,T):

	return (Y - T).sum(axis=0)

def relu(a):

	return a * (a > 0)

def full(learning_rate):

	X,Y = transform_data()
	N = len(X)//2
	Xtrain = X[:N]
	Ytrain = Y[:N]
	Ttrain = generate_T(Ytrain)
	Xtest = X[N:]
	Ytest = Y[N:]
	Ttest = generate_T(Ytest)
	N,D = Xtrain.shape
	K = len(set(Y))
	w = np.random.randn(D,K)/np.sqrt(D+K)
	b = np.random.randn(K)/np.sqrt(K)
	cost = []
	full_cr = []
	best_cr = 0
	best_iteration = 0
	t0 = datetime.now()
	for i in range(100):
		A_train = relu(Xtrain.dot(w) + b)
		Y_train = np.exp(A_train)/np.exp(A_train).sum(axis=1,keepdims=True)
		A_test = relu(Xtest.dot(w) + b)
		Y_test = np.exp(A_test)/np.exp(A_test).sum(axis=1,keepdims=True)
		P_test = np.argmax(Y_test,axis=1)
		cost.append(cross_entropy(Y_test,Ttest))
		cr = classification_rate(P_test,Ytest)
		full_cr.append(cr)
		if cr > best_cr:
			best_cr = cr
			best_iteration = i
		w -= learning_rate*derivative_w(Xtrain,Y_train,Ttrain)
		b -= learning_rate*derivative_b(Y_train,Ttrain)
		if i % 100 == 0:
			print('Cost: ',cost[i])
	print('Time to fit: ',datetime.now() - t0)
	return cost,full_cr,best_cr,best_iteration

def stochastic(learning_rate):

	X,Y = transform_data()
	K = len(set(Y))
	N = len(X)//2
	Xtrain = X[:N]
	Ytrain = Y[:N]
	Ttrain = generate_T(Ytrain)
	Xtest = X[N:]
	Ytest = Y[N:]
	Ttest = generate_T(Ytest)
	N,D = Xtrain.shape
	w = np.random.randn(D,K)/np.sqrt(D+K)
	b = np.random.randn(K)/np.sqrt(K)
	cost = []
	stochastic_cr = []
	best_cr = 0
	best_iteration = 0
	t0 = datetime.now()
	for i in range(100):
		current_X,current_T = shuffle(Xtrain,Ttrain)
		for n in range(20):
			X = current_X[n,:].reshape(1,D)
			A_train = relu(X.dot(w) + b)
			Y_train = np.exp(A_train)/np.exp(A_train).sum(axis=1,keepdims=True)
			A_test = relu(Xtest.dot(w) + b)
			Y_test = np.exp(A_test)/np.exp(A_test).sum(axis=1,keepdims=True)
			T = current_T[n].reshape(1,10)
			P_test = np.argmax(Y_test,axis=1)
			w -= learning_rate*derivative_w(X,Y_train,T)
			b -= learning_rate*derivative_b(Y_train,T)
			if n % 500 == 0:
				cost.append(cross_entropy(Y_test,Ttest))
				cr = classification_rate(P_test,Ytest)
				stochastic_cr.append(cr)
				if cr > best_cr:
					best_cr = cr
					best_iteration = i
		if i % 100 == 0:
			print('Stochastic Cost: ',cost[i])
	print('Time to fit: ', datetime.now() - t0)
	return cost,stochastic_cr,best_cr,best_iteration

def batch(learning_rate):

	X,Y = transform_data()
	N = len(X)//2
	Xtrain = X[:N]
	Ytrain = Y[:N]
	Ttrain = generate_T(Ytrain)
	Xtest = X[N:]
	Ytest = Y[N:]
	Ttest = generate_T(Ytest)
	N,D = Xtrain.shape
	batch_N = 250
	batches = N//batch_N
	K = len(set(Y))
	w = np.random.randn(D,K)/np.sqrt(D+K)
	b = np.random.randn(K)/np.sqrt(K)
	cost = []
	batch_cr = []
	best_cr = 0
	best_iteration = 0
	t0 = datetime.now()
	for i in range(100):
		for b in range(batches):
			X = Xtrain[b*batch_N:(b+1)*batch_N,:]
			T = Ttrain[b*batch_N:(b+1)*batch_N,:]
			A_train = relu(X.dot(w) + b)
			Y_train = np.exp(A_train)/np.exp(A_train).sum(axis=1,keepdims=True)
			A_test = relu(Xtest.dot(w) + b)
			Y_test = np.exp(A_test)/np.exp(A_test).sum(axis=1,keepdims=True)
			P_test = np.argmax(Y_test,axis=1)
			if b % batches == 0:
				cost.append(cross_entropy(Y_test,Ttest))
				cr = classification_rate(P_test,Ytest)
				batch_cr.append(cr)
				if cr > best_cr:
					best_cr = cr
					best_iteration = i
			w -= learning_rate*derivative_w(X,Y_train,T)
			b -= learning_rate*derivative_b(Y_train,T)
		if i % 100 == 0:
			print('Batch Cost: ',cost[i])
	print('Time to fit: ',datetime.now() - t0)
	return cost,batch_cr,best_cr,best_iteration

if __name__ == '__main__':

	learning_rate = 1e-7
	full_cost,full_cr,best_full,full_iteration = full(learning_rate)
	stochastic_cost,stochastic_cr,best_stochastic,stochastic_iteration = stochastic(learning_rate)
	batch_cost,batch_cr,best_batch,batch_iteration = batch(learning_rate)
	full_stochastic = np.mean(np.array(full_cost) - np.array(stochastic_cost))
	full_batch = np.mean(np.array(full_cost) - np.array(batch_cost))
	stochastic_batch = np.mean(np.array(stochastic_cost) - np.array(batch_cost))
	plt.plot(full_cost,label='Full')
	plt.plot(stochastic_cost,label='Stochastic')
	plt.plot(batch_cost,label='Batch')
	plt.title('Cost')
	plt.legend()
	plt.show()
	plt.plot(full_cr,label='Full')
	plt.plot(stochastic_cr,label='Stochastic')
	plt.plot(batch_cr,label='Batch')
	plt.title('Classification Rate')
	plt.legend()
	plt.show()
	print('')
	if full_stochastic > 0:
		print('Stochastic error rate is ' + str(full_stochastic) + ' less per iteration on avg than Full.')
	elif full_stochastic < 0:
		print('Full error rate is ' + str(-full_stochastic) + ' less per iteration on avg than Stochastic.')
	else:
		print('Full and Stochastic have equivalent error rates on avg.')
	if full_batch > 0:
		print('Batch error rate is ' + str(full_batch) + ' less per iteration on avg than Full.')
	elif full_batch < 0:
		print('Full error rate is ' + str(-full_batch) + ' less per iteration on avg than Batch.')
	else:
		print('Full and Batch have equivalent error rates on avg.')
	if stochastic_batch > 0:
		print('Batch error rate is ' + str(stochastic_batch) + ' less per iteration on avg than Stochastic.')
	elif stochastic_batch < 0:
		print('Stochastic error rate is ' + str(-stochastic_batch) + ' less per iteration on avg than Batch.')
	else:
		print('Stochastic and Batch have equivalent error rates on avg.')
	print('')
	print('Best Full Classification Rate: ',best_full,'Iteration: ',full_iteration)
	print('Best Stochastic Classification Rate: ',best_stochastic,'Iteration: ',stochastic_iteration)
	print('Best Batch Classification Rate: ',best_batch,'Iteration: ',batch_iteration)