from util_adaptive import transform_data,derivative_v,derivative_b1,derivative_w,derivative_b0,cross_entropy,classification_rate,\
generate_Y,generate_T
from sklearn.utils import shuffle
import numpy as np, matplotlib.pyplot as plt 

def derivative_w(X,Y,T):

	return X.T.dot(Y - T)

def derivative_b(Y,T):

	return (Y - T).sum(axis=0)

def relu(a):

	return a * (a > 0)

def sigmoid(a):

	return 1/(1 + np.exp(-a))

def grid():

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
	K = len(set(Y))
	w0 = np.random.randn(D,K)/np.sqrt(D+K)
	b0 = np.random.randn(K)/np.sqrt(K)
	learning_rates = [10**i for i in range(-7,-3,1)]
	momentums = [1-10**i for i in sorted(list(range(-4,0)),reverse=True)]
	iterations = 2000
	best_lr = 0
	best_momentum = 0
	best_cr = 0
	cost = {}
	cr = {}
	for lr in learning_rates:
		learning_rate = lr
		for mu in momentums:
			dw = 0
			db = 0
			cost[(lr,mu)] = list()
			cr[(lr,mu)] = list()
			for i in range(iterations):
				if i == 0:
					A_train = relu(Xtrain.dot(w0) + b0)
					A_test = relu(Xtest.dot(w0) + b0)
				else:
					A_train = relu(Xtrain.dot(w) + b0)
					A_test = relu(Xtest.dot(w) + b0)	
				Y_train = np.exp(A_train)/np.exp(A_train).sum(axis=1,keepdims=True)
				Y_test = np.exp(A_test)/np.exp(A_test).sum(axis=1,keepdims=True)
				P_test = np.argmax(Y_test,axis=1)
				cost[(lr,mu)].append(cross_entropy(Y_test,Ttest))
				current_cr = classification_rate(P_test,Ytest)
				cr[(lr,mu)].append(current_cr)
				if current_cr > best_cr:
					best_cr = current_cr
					best_lr = lr
					best_mu = mu
				dw = mu*dw - (1-mu)*learning_rate*derivative_w(Xtrain,Y_train,Ttrain)
				db = mu*db - (1-mu)*learning_rate*derivative_b(Y_train,Ttrain)
				if i == 0:
					w = w0 + dw
					b = b0 + db
				else:
					w += dw
					b += db
				if i % 100 == 0:
					print('Learning Rate: ',lr,'Momentum: ',mu,'Cost: ',cost[(lr,mu)][i],'Classification Rate: ',cr[(lr,mu)][i])
				if i == (iterations - 1):
					print('')
	return cost,cr,best_lr,best_mu,best_cr

if __name__ == '__main__':

	cost,cr,best_lr,best_mu,best_cr = grid()
	for e,i in cost.items():
		plt.plot(i,label='lr={0},mu={1}'.format(e[0],e[1]))
	plt.title('Cost')
	plt.legend()
	plt.show()	
	for e,i in cr.items():
		plt.plot(i,label='lr={0},mu={1}'.format(e[0],e[1]))
	plt.title('Classification Rate')
	plt.legend()
	plt.show()
	print('Best Learning Rate: ',best_lr)
	print('Best Momentum: ',best_mu)
	print('Best Classification Rate: ',best_cr)