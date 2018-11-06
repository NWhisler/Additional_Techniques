from util import transform_data,derivative_v,derivative_b1,derivative_w,derivative_b0,cross_entropy,classification_rate,\
generate_Y,generate_T
from sklearn.utils import shuffle
import numpy as np, matplotlib.pyplot as plt 

class Adam(object):

	def __init__(self,activation,hidden_layer,learning_rate,iterations,decay_rates,gradient_descent='Full'):

		self.activation = activation
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.gradient_descent = gradient_descent
		self.hidden_layer = hidden_layer
		self.decay_0 = decay_rates[0]
		self.decay_1 = decay_rates[1]

	def fit(self):

		X,Y = transform_data()
		X,Y = shuffle(X,Y)
		N = len(X)//2
		self.Xtrain = X[:N]
		self.Ytrain = Y[:N]
		self.Ttrain = generate_T(self.Ytrain)
		self.Xtest = X[N:]
		self.Ytest = Y[N:]
		self.Ttest = generate_T(self.Ytest)
		N,D = self.Xtrain.shape
		M = self.hidden_layer
		K = len(set(Y))
		self.v = np.random.randn(M,K)/np.sqrt(M+K)
		self.b_1 = np.random.randn(K)/np.sqrt(K)
		self.w = np.random.randn(D,M)/np.sqrt(D+M)
		self.b_0 = np.random.randn(M)/np.sqrt(M)
		self.m_v = 0
		self.m_b1 = 0
		self.m_w = 0
		self.m_b0 = 0
		self.v_v = 0
		self.v_b1 = 0
		self.v_w = 0
		self.v_b0 = 0
		self.epsilon = 10e-8
		self.train_cost = []
		self.test_cost = []
		self.train_cr = []
		self.test_cr = []
		self.best_train = 0
		self.train_iteration = 0
		self.best_test = 0
		self.test_iteration = 0
		if self.gradient_descent == 'Full' or self.gradient_descent == 'full':
			self.full()
		elif self.gradient_descent == 'Stochastic' or self.gradient_descent == 'stochastic':
			samples = int(input('Sample Size: '))
			self.stochastic(samples)
		else:
			batch_N = int(input('Batch Size: '))
			self.batch(batch_N)

	def full(self):

		for i in range(self.iterations):
			Y_train,Z = generate_Y(self.activation,self.Xtrain,self.w,self.b_0,self.v,self.b_1)
			P_train = np.argmax(Y_train,axis=1)
			Y_test,_ = generate_Y(self.activation,self.Xtest,self.w,self.b_0,self.v,self.b_1)
			P_test = np.argmax(Y_test,axis=1)
			self.train_cost.append(cross_entropy(Y_train,self.Ttrain))
			self.test_cost.append(cross_entropy(Y_test,self.Ttest))
			train_cr = classification_rate(P_train,self.Ytrain)
			self.train_cr.append(train_cr)
			test_cr = classification_rate(P_test,self.Ytest)
			self.test_cr.append(test_cr)
			if train_cr > self.best_train:
				self.best_train = train_cr
				self.train_iteration = i			
			if test_cr > self.best_test:
				self.best_test = test_cr
				self.test_iteration = i
			self.m_v = self.decay_0*self.m_v + (1-self.decay_0)*derivative_v(self.activation,Z,Y_train,self.Ttrain)
			self.dm_v = self.m_v/(1-self.decay_0**(i+1))
			self.v_v = self.decay_1*self.v_v + (1-self.decay_1)*derivative_v(self.activation,Z,Y_train,self.Ttrain)**2
			self.dv_v = self.v_v/(1-self.decay_1**(i+1))
			self.m_b1 = self.decay_0*self.m_b1 + (1-self.decay_0)*derivative_b1(self.activation,Y_train,self.Ttrain)
			self.dm_b1 = self.m_b1/(1-self.decay_0**(i+1))
			self.v_b1 = self.decay_1*self.v_b1 + (1-self.decay_1)*derivative_b1(self.activation,Y_train,self.Ttrain)**2
			self.dv_b1 = self.v_b1/(1-self.decay_1**(i+1))
			self.m_w = self.decay_0*self.m_w + (1-self.decay_0)*derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,self.v)
			self.dm_w = self.m_w/(1-self.decay_0**(i+1))
			self.v_w = self.decay_1*self.v_w + (1-self.decay_1)*derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,self.v)**2
			self.dv_w = self.v_w/(1-self.decay_1**(i+1))
			self.m_b0 = self.decay_0*self.m_b0 + (1-self.decay_0)*derivative_b0(self.activation,Y_train,Z,self.Ttrain,self.v)
			self.dm_b0 = self.m_b0/(1-self.decay_0**(i+1))
			self.v_b0 = self.decay_1*self.v_b0 + (1-self.decay_1)*derivative_b0(self.activation,Y_train,Z,self.Ttrain,self.v)**2
			self.dv_b0 = self.v_b0/(1-self.decay_1**(i+1))
			self.v -= self.learning_rate*self.dm_v/(np.sqrt(self.dv_v+self.epsilon))
			self.b_1 -= self.learning_rate*self.dm_b1/(np.sqrt(self.dv_b1+self.epsilon))
			self.w -= self.learning_rate*self.dm_w/(np.sqrt(self.dv_w+self.epsilon))
			self.b_0 -= self.learning_rate*self.dm_b0/(np.sqrt(self.dv_b0+self.epsilon))
			if i % 100 == 0:
				print(i,'Train Cost: ',self.train_cost[i],'Train Classification Rate: ',self.train_cr[i])

	def stochastic(self,samples):

		for i in range(self.iterations):
			current_X,current_T = shuffle(self.Xtrain,self.Ttrain)
			for s in range(samples):
				X = current_X[s,:].reshape(1,current_X.shape[1])
				T = current_T[s,:].reshape(1,current_T.shape[1])
				Y,Z = generate_Y(self.activation,X,self.w,self.b_0,self.v,self.b_1)
				Y_train,_ = generate_Y(self.activation,self.Xtrain,self.w,self.b_0,self.v,self.b_1)
				P_train = np.argmax(Y_train,axis=1)
				Y_test,_ = generate_Y(self.activation,self.Xtest,self.w,self.b_0,self.v,self.b_1)
				P_test = np.argmax(Y_test,axis=1)
				self.train_cost.append(cross_entropy(Y_train,self.Ttrain))
				self.test_cost.append(cross_entropy(Y_test,self.Ttest))
				train_cr = classification_rate(P_train,self.Ytrain)
				self.train_cr.append(train_cr)
				test_cr = classification_rate(P_test,self.Ytest)
				self.test_cr.append(test_cr)
				if train_cr > self.best_train:
					self.best_train = train_cr
					self.train_iteration = i			
				if test_cr > self.best_test:
					self.best_test = test_cr
					self.test_iteration = i
				self.m_v = self.decay_0*self.m_v + (1-self.decay_0)*derivative_v(self.activation,Z,Y,T)
				self.dm_v = self.m_v/(1-self.decay_0**(i+1))
				self.v_v = self.decay_1*self.v_v + (1-self.decay_1)*derivative_v(self.activation,Z,Y,T)**2
				self.dv_v = self.v_v/(1-self.decay_1**(i+1))
				self.m_b1 = self.decay_0*self.m_b1 + (1-self.decay_0)*derivative_b1(self.activation,Y,T)
				self.dm_b1 = self.m_b1/(1-self.decay_0**(i+1))
				self.v_b1 = self.decay_1*self.v_b1 + (1-self.decay_1)*derivative_b1(self.activation,Y,T)**2
				self.dv_b1 = self.v_b1/(1-self.decay_1**(i+1))
				self.m_w = self.decay_0*self.m_w + (1-self.decay_0)*derivative_w(self.activation,X,Y,Z,T,self.v)
				self.dm_w = self.m_w/(1-self.decay_0**(i+1))
				self.v_w = self.decay_1*self.v_w + (1-self.decay_1)*derivative_w(self.activation,X,Y,Z,T,self.v)**2
				self.dv_w = self.v_w/(1-self.decay_1**(i+1))
				self.m_b0 = self.decay_0*self.m_b0 + (1-self.decay_0)*derivative_b0(self.activation,Y,Z,T,self.v)
				self.dm_b0 = self.m_b0/(1-self.decay_0**(i+1))
				self.v_b0 = self.decay_1*self.v_b0 + (1-self.decay_1)*derivative_b0(self.activation,Y,Z,T,self.v)**2
				self.dv_b0 = self.v_b0/(1-self.decay_1**(i+1))
				self.v -= self.learning_rate*self.dm_v/(np.sqrt(self.dv_v+self.epsilon))
				self.b_1 -= self.learning_rate*self.dm_b1/(np.sqrt(self.dv_b1+self.epsilon))
				self.w -= self.learning_rate*self.dm_w/(np.sqrt(self.dv_w+self.epsilon))
				self.b_0 -= self.learning_rate*self.dm_b0/(np.sqrt(self.dv_b0+self.epsilon))
			if i % 100 == 0:
				print(i,'Train Cost: ',self.train_cost[i],'Train Classification Rate: ',self.train_cr[i])

	def batch(self,batch_N):

		batches = len(self.Xtrain)//batch_N
		for i in range(self.iterations):
			for b in range(batches):
				X = self.Xtrain[b*batches:(b+1)*batches,:]
				T = self.Ttrain[b*batches:(b+1)*batches,:]
				Y,Z = generate_Y(self.activation,X,self.w,self.b_0,self.v,self.b_1)
				Y_train,_ = generate_Y(self.activation,self.Xtrain,self.w,self.b_0,self.v,self.b_1)
				P_train = np.argmax(Y_train,axis=1)
				Y_test,_ = generate_Y(self.activation,self.Xtest,self.w,self.b_0,self.v,self.b_1)
				P_test = np.argmax(Y_test,axis=1)
				self.train_cost.append(cross_entropy(Y_train,self.Ttrain))
				self.test_cost.append(cross_entropy(Y_test,self.Ttest))
				train_cr = classification_rate(P_train,self.Ytrain)
				self.train_cr.append(train_cr)
				test_cr = classification_rate(P_test,self.Ytest)
				self.test_cr.append(test_cr)
				if train_cr > self.best_train:
					self.best_train = train_cr
					self.train_iteration = i			
				if test_cr > self.best_test:
					self.best_test = test_cr
					self.test_iteration = i
				self.m_v = self.decay_0*self.m_v + (1-self.decay_0)*derivative_v(self.activation,Z,Y,T)
				self.dm_v = self.m_v/(1-self.decay_0**(i+1))
				self.v_v = self.decay_1*self.v_v + (1-self.decay_1)*derivative_v(self.activation,Z,Y,T)**2
				self.dv_v = self.v_v/(1-self.decay_1**(i+1))
				self.m_b1 = self.decay_0*self.m_b1 + (1-self.decay_0)*derivative_b1(self.activation,Y,T)
				self.dm_b1 = self.m_b1/(1-self.decay_0**(i+1))
				self.v_b1 = self.decay_1*self.v_b1 + (1-self.decay_1)*derivative_b1(self.activation,Y,T)**2
				self.dv_b1 = self.v_b1/(1-self.decay_1**(i+1))
				self.m_w = self.decay_0*self.m_w + (1-self.decay_0)*derivative_w(self.activation,X,Y,Z,T,self.v)
				self.dm_w = self.m_w/(1-self.decay_0**(i+1))
				self.v_w = self.decay_1*self.v_w + (1-self.decay_1)*derivative_w(self.activation,X,Y,Z,T,self.v)**2
				self.dv_w = self.v_w/(1-self.decay_1**(i+1))
				self.m_b0 = self.decay_0*self.m_b0 + (1-self.decay_0)*derivative_b0(self.activation,Y,Z,T,self.v)
				self.dm_b0 = self.m_b0/(1-self.decay_0**(i+1))
				self.v_b0 = self.decay_1*self.v_b0 + (1-self.decay_1)*derivative_b0(self.activation,Y,Z,T,self.v)**2
				self.dv_b0 = self.v_b0/(1-self.decay_1**(i+1))
				self.v -= self.learning_rate*self.dm_v/(np.sqrt(self.dv_v+self.epsilon))
				self.b_1 -= self.learning_rate*self.dm_b1/(np.sqrt(self.dv_b1+self.epsilon))
				self.w -= self.learning_rate*self.dm_w/(np.sqrt(self.dv_w+self.epsilon))
				self.b_0 -= self.learning_rate*self.dm_b0/(np.sqrt(self.dv_b0+self.epsilon))
			if i % 100 == 0:
				print(i,'Train Cost: ',self.train_cost[i],'Train Classification Rate: ',self.train_cr[i])

	def graph_cost(self):

		plt.plot(self.train_cost,label='Train')
		plt.plot(self.test_cost,label='Test')
		plt.title('Cost')
		plt.legend()
		plt.show()

	def graph_cr(self):

		plt.plot(self.train_cr,label='Train')
		plt.plot(self.test_cr,label='Test')
		plt.title('Classification Rate')
		plt.legend()
		plt.show()

if __name__ == '__main__':

	model_0 = Adam('tanh',100,10e-5,500,[.9,.9],'full')
	model_0.fit()
	model_1 = Adam('tanh',100,10e-5,500,[.9,.9],'stochastic')
	model_1.fit()
	model_2 = Adam('tanh',100,10e-5,500,[.9,.9],'batch')
	model_2.fit()
	model_0.graph_cost()
	model_0.graph_cr()
	model_1.graph_cost()
	model_1.graph_cr()
	model_2.graph_cost()
	model_2.graph_cr()
	print('')
	print('Final Train Classification Rate: ',model_0.train_cr[-1])
	print('Final Test Classification Rate: ',model_0.test_cr[-1])
	print('')
	print('Best Train Classification Rate: ',model_0.best_train,'Iteration: ',model_0.train_iteration)
	print('Best Test Classification Rate: ',model_0.best_test,'Iteration: ',model_0.test_iteration)
	print('')
	print('Final Train Classification Rate: ',model_1.train_cr[-1])
	print('Final Test Classification Rate: ',model_1.test_cr[-1])
	print('')
	print('Best Train Classification Rate: ',model_1.best_train,'Iteration: ',model_1.train_iteration)
	print('Best Test Classification Rate: ',model_1.best_test,'Iteration: ',model_1.test_iteration)
	print('')
	print('Final Train Classification Rate: ',model_2.train_cr[-1])
	print('Final Test Classification Rate: ',model_2.test_cr[-1])
	print('')
	print('Best Train Classification Rate: ',model_2.best_train,'Iteration: ',model_2.train_iteration)
	print('Best Test Classification Rate: ',model_2.best_test,'Iteration: ',model_2.test_iteration)