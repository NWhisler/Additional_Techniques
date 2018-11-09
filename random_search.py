from util import transform_data, derivative_v, derivative_b1, derivative_w, derivative_b0, cross_entropy, classification_rate, \
generate_Y, generate_T
from sklearn.utils import shuffle
import numpy as np, matplotlib.pyplot as plt 

class Random(object):

	def __init__(self,activation,iterations,hidden_dimension,learning_rates,momentums,\
		momentum_type=None,ada=None,rms=None,adaptive=None,gradient=None):

		self.activation = activation
		self.iterations = iterations
		self.hidden_dimension = hidden_dimension
		self.learning_rates = learning_rates
		self.momentums = momentums
		self.momentum_type = momentum_type
		self.adagrad = ada
		self.rmsprop = rms
		self.adaptive_moment = adaptive
		self.gradient = gradient

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
		self.cost = {}
		self.cr = {}
		self.best_cr = 0
		self.best_dimension = None
		self.best_lr = None
		self.best_mu = None
		self.samples = None
		self.batch_N = None
		self.N,self.D = self.Xtrain.shape
		self.K = len(set(Y))
		self.hd_sampling = 5
		self.lr_sampling = 4
		self.dimension_cost = []
		self.hd_samples = []
		self.current_samples = 0
		self.previous_samples = self.current_samples
		for i in range(self.hd_sampling):
			self.idx_hd = np.random.choice(range(self.hidden_dimension[0],self.hidden_dimension[1]))
			if i == 0:
				self.dimension_cost.append(self.idx_hd)
			while self.idx_hd in self.hd_samples:
				self.idx_hd = np.random.choice(range(self.hidden_dimension[0],self.hidden_dimension[1]))
				if len(self.hd_samples) == len(range(self.hidden_dimension[0],self.hidden_dimension[1])):
					break
			self.current_samples += 1
			self.hd_samples.append(self.idx_hd)
			self.M = self.idx_hd
			self.v_0 = np.random.randn(self.M,self.K)/np.sqrt(self.M+self.K)
			self.b1_0 = np.random.randn(self.K)/np.sqrt(self.K)
			self.w_0 = np.random.randn(self.D,self.M)/np.sqrt(self.D+self.M)
			self.b0_0 = np.random.randn(self.M)/np.sqrt(self.M)
			self.lr_samples = []
			for e in range(self.lr_sampling):
				idx_lr = np.random.choice(range(self.learning_rates[0],self.learning_rates[1]))
				while idx_lr in self.lr_samples:
					idx_lr = np.random.choice(range(self.learning_rates[0],self.learning_rates[1]))
					if len(self.lr_samples) == len(range(self.learning_rates[0],self.learning_rates[1])):
						break
				self.lr_samples.append(idx_lr)
				self.lr = 10**int(idx_lr)
				if self.momentum_type is None:
					self.mu = None
					self.cost[(self.idx_hd,self.lr,self.mu)] = list()
					self.cr[(self.idx_hd,self.lr,self.mu)] = list()	
					self.cache_v = 1
					self.cache_b1 = 1
					self.cache_w = 1
					self.cache_b0 = 1
					if self.gradient:
						if self.gradient == 'Full' or self.gradient == 'full':
							self.full()
						elif self.gradient == 'Stochastic' or self.gradient == 'stochastic':
							if self.samples:
								self.stochastic(self.samples)
							else:
								self.samples = int(input('Samples: '))
								self.stochastic(self.samples)
						else:
							if self.batch_N:
								self.batch(self.batch_N)
							else:
								self.batch_N = int(input('Batch Size: '))
								self.batch(self.batch_N)
					else:
						self.full()
				else:
					self.mu_sampling = 3
					self.mu_samples = []
					for m in range(self.mu_sampling):
						self.dv = 0
						self.db_1 = 0
						self.dw = 0
						self.db_0 = 0
						idx_mu = np.random.choice(range(self.momentums[0],self.momentums[1]))
						while idx_mu in self.mu_samples:
							idx_mu = np.random.choice(range(self.momentums[0],self.momentums[1]))
							if len(self.mu_samples) == len(range(self.momentums[0],self.momentums[1])):
								break
						self.mu_samples.append(idx_mu)
						self.mu = 1 - 10**int(idx_mu)
						self.mu_samples.append(self.mu)
						self.cost[(self.idx_hd,self.lr,self.mu)] = list()
						self.cr[(self.idx_hd,self.lr,self.mu)] = list()
						if self.gradient:
							if self.gradient == 'Full' or self.gradient == 'full':
								self.full()
							elif self.gradient == 'Stochastic' or self.gradient == 'stochastic':
								if self.samples:
									self.stochastic(self.samples)
								else:
									self.samples = int(input('Samples: '))
									self.stochastic(self.samples)
							else:
								if self.batch_N:
									self.batch(self.batch_N)
								else:
									self.batch_N = int(input('Batch Size: '))
									self.batch(self.batch_N)
						else:
							self.full()

	def full(self):

		for i in range(self.iterations):
			if i == 0:
				Y_train,Z = generate_Y(self.activation,self.Xtrain,self.w_0,self.b0_0,self.v_0,self.b1_0)
				Y_test,_ = generate_Y(self.activation,self.Xtest,self.w_0,self.b0_0,self.v_0,self.b1_0)
			else:
				Y_train,Z = generate_Y(self.activation,self.Xtrain,self.w,self.b_0,self.v,self.b_1)
				Y_test,_ = generate_Y(self.activation,self.Xtest,self.w,self.b_0,self.v,self.b_1)
			P_train = np.argmax(Y_train,axis=1)
			P_test = np.argmax(Y_test,axis=1)
			if self.momentum_type:
				if self.momentum_type == 'Standard' or self.momentum_type == 'standard':
					self.cost[(self.idx_hd,self.lr,self.mu)].append(cross_entropy(Y_test,self.Ttest))
					cr = classification_rate(self.Ytest,P_test)
					self.cr[(self.idx_hd,self.lr,self.mu)].append(cr)
					if cr > self.best_cr:
						self.best_cr = cr
						self.best_dimension = self.idx_hd
						self.best_lr = self.lr 
						self.best_mu = self.mu
					self.dv = self.mu*self.dv - (1-self.mu)*self.lr*derivative_v(self.activation,Z,Y_train,self.Ttrain)
					self.db_1 = self.mu*self.db_1 - (1-self.mu)*self.lr*derivative_b1(self.activation,Y_train,self.Ttrain)
					if i == 0:
						self.dw = self.mu*self.dw - (1-self.mu)*self.lr*\
						derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,self.v_0)
						self.db_0 = self.mu*self.db_0 - (1-self.mu)*self.lr*\
						derivative_b0(self.activation,Y_train,Z,self.Ttrain,self.v_0)
					else:
						self.dw = self.mu*self.dw - (1-self.mu)*self.lr*\
						derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,self.v)
						self.db_0 = self.mu*self.db_0 - (1-self.mu)*self.lr*\
						derivative_b0(self.activation,Y_train,Z,self.Ttrain,self.v)
					if i == 0:
						self.v = self.v_0 + self.dv
						self.b_1 = self.b1_0 + self.db_1
						self.w = self.w_0 + self.dw
						self.b_0 = self.b0_0 + self.db_0
					else:
						self.v += self.dv
						self.b_1 += self.db_1
						self.w += self.dw
						self.b_0 += self.db_0
				else:
					self.cost[(self.idx_hd,self.lr,self.mu)].append(cross_entropy(Y_test,self.Ttest))
					cr = classification_rate(self.Ytest,P_test)
					self.cr[(self.idx_hd,self.lr,self.mu)].append(cr)
					if cr > self.best_cr:
						self.best_cr = cr
						self.best_dimension = self.idx_hd
						self.best_lr = self.lr 
						self.best_mu = self.mu
					self.dv = self.mu*self.dv - (1-self.mu)*self.lr*derivative_v(self.activation,Z,Y_train,self.Ttrain)
					self.db_1 = self.mu*self.db_1 - (1-self.mu)*self.lr*derivative_b1(self.activation,Y_train,self.Ttrain)
					if i == 0:
						self.dw = self.mu*self.dw - (1-self.mu)*self.lr*derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,\
							self.v_0)
						self.db_0 = self.mu*self.db_0 - (1-self.mu)*self.lr*derivative_b0(self.activation,Y_train,Z,self.Ttrain,self.v_0)
					else:
						self.dw = self.mu*self.dw - (1-self.mu)*self.lr*derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,\
							self.v)
						self.db_0 = self.mu*self.db_0 - (1-self.mu)*self.lr*derivative_b0(self.activation,Y_train,Z,self.Ttrain,self.v)
					if i == 0:
						self.v = self.v_0 + self.mu*self.dv - (1-self.mu)*self.lr*derivative_v(self.activation,Z,Y_train,self.Ttrain)
						self.b_1 = self.b1_0 + self.mu*self.db_1 - (1-self.mu)*self.lr*derivative_b1(self.activation,Y_train,self.Ttrain)
						self.w = self.w_0 + self.mu*self.dw - (1-self.mu)*self.lr*derivative_w(self.activation,self.Xtrain,Y_train,Z,\
							self.Ttrain,self.v)
						self.b_0 = self.b0_0 + self.mu*self.db_0 - (1-self.mu)*self.lr*derivative_b0(self.activation,Y_train,Z,\
							self.Ttrain,self.v)
					else:
						self.v += self.mu*self.dv - (1-self.mu)*self.lr*derivative_v(self.activation,Z,Y_train,self.Ttrain)
						self.b_1 += self.mu*self.db_1 - (1-self.mu)*self.lr*derivative_b1(self.activation,Y_train,self.Ttrain)
						self.w += self.mu*self.dw - (1-self.mu)*self.lr*derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,self.v)
						self.b_0 += self.mu*self.db_0 - (1-self.mu)*self.lr*derivative_b0(self.activation,Y_train,Z,self.Ttrain,self.v)
			else:
				if self.adagrad:
					self.cost[(self.idx_hd,self.lr,self.mu)].append(cross_entropy(Y_test,self.Ttest))
					cr = classification_rate(P_test,self.Ytest)
					self.cr[(self.idx_hd,self.lr,self.mu)].append(cr)
					if cr > self.best_cr:
						self.best_cr = cr
						self.best_dimension = self.idx_hd
						self.best_lr = self.lr 
						self.best_mu = self.mu
					epsilon = 10e-10
					self.cache_v += derivative_v(self.activation,Z,Y_train,self.Ttrain)**2
					self.cache_b1 += derivative_b1(self.activation,Y_train,self.Ttrain)**2
					if i == 0:
						self.cache_w += derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,self.v_0)**2
						self.cache_b0 += derivative_b0(self.activation,Y_train,Z,self.Ttrain,self.v_0)**2
					else:
						self.cache_w += derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,self.v)**2
						self.cache_b0 += derivative_b0(self.activation,Y_train,Z,self.Ttrain,self.v)**2
					if i == 0:
						self.v = self.v_0 - self.lr*derivative_v(self.activation,Z,Y_train,self.Ttrain)/(np.sqrt(self.cache_v + epsilon))
						self.b_1 = self.b1_0 - self.lr*derivative_b1(self.activation,Y_train,self.Ttrain)/(np.sqrt(self.cache_b1 + epsilon))
						self.w = self.w_0 - self.lr*derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,\
							self.v_0)/(np.sqrt(self.cache_w + epsilon))
						self.b_0 = self.b0_0 - self.lr*derivative_b0(self.activation,Y_train,Z,self.Ttrain,\
							self.v_0)/(np.sqrt(self.cache_b0 + epsilon))
					else:
						self.v -= self.lr*derivative_v(self.activation,Z,Y_train,self.Ttrain)/(np.sqrt(self.cache_v + epsilon))
						self.b_1 -= self.lr*derivative_b1(self.activation,Y_train,self.Ttrain)/(np.sqrt(self.cache_b1 + epsilon))
						self.w -= self.lr*derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,self.v)/(np.sqrt(self.cache_w + \
							epsilon))
						self.b_0 -= self.lr*derivative_b0(self.activation,Y_train,Z,self.Ttrain,self.v)/(np.sqrt(self.cache_b0 + epsilon))
				elif self.rmsprop:
					epsilon = 10e-10
					self.cost[(self.idx_hd,self.lr,self.mu)].append(cross_entropy(Y_test,self.Ttest))
					cr = classification_rate(P_test,self.Ytest)
					self.cr[(self.idx_hd,self.lr,self.mu)].append(cr)
					if cr > self.best_cr:
						self.best_cr = cr
						self.best_dimension = self.idx_hd
						self.best_lr = self.lr 
						self.best_mu = self.mu
					decay = .9
					self.cache_v = decay*self.cache_v + (1-decay)*derivative_v(self.activation,Z,Y_train,self.Ttrain)**2
					self.cache_b1 = decay*self.cache_b1 + (1-decay)*derivative_b1(self.activation,Y_train,self.Ttrain)**2
					if i == 0:
						self.cache_w = decay*self.cache_w + (1-decay)*derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,\
							self.v_0)**2
						self.cache_b0 = decay*self.cache_b0 + (1-decay)*derivative_b0(self.activation,Y_train,Z,self.Ttrain,self.v_0)**2
					else:
						self.cache_w = decay*self.cache_w + (1-decay)*derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,\
							self.v)**2
						self.cache_b0 = decay*self.cache_b0 + (1-decay)*derivative_b0(self.activation,Y_train,Z,self.Ttrain,self.v)**2
					if i == 0:
						self.v = self.v_0 - self.lr*derivative_v(self.activation,Z,Y_train,self.Ttrain)/(np.sqrt(self.cache_v + epsilon))
						self.b_1 = self.b1_0 - self.lr*derivative_b1(self.activation,Y_train,self.Ttrain)/(np.sqrt(self.cache_b1 + epsilon))
						self.w = self.w_0 - self.lr*derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,\
							self.v)/(np.sqrt(self.cache_w + epsilon))
						self.b_0 = self.b0_0 - self.lr*derivative_b0(self.activation,Y_train,Z,self.Ttrain,\
							self.v)/(np.sqrt(self.cache_b0 + epsilon))
					else:
						self.v -= self.lr*derivative_v(self.activation,Z,Y_train,self.Ttrain)/(np.sqrt(self.cache_v + epsilon))
						self.b_1 -= self.lr*derivative_b1(self.activation,Y_train,self.Ttrain)/(np.sqrt(self.cache_b1 + epsilon))
						self.w -= self.lr*derivative_w(self.activation,self.Xtrain,Y_train,Z,self.Ttrain,self.v)/(np.sqrt(self.cache_w + \
							epsilon))
						self.b_0 -= self.lr*derivative_b0(self.activation,Y_train,Z,self.Ttrain,self.v)/(np.sqrt(self.cache_b0 + epsilon))
			if self.hd_sampling > 1:
				if self.current_samples != self.previous_samples and self.current_samples != 1:
					print('-' * 50)
					self.previous_samples = self.current_samples
			if i % 100 == 0:
				print('Iteration: ',i)
				print('Hidden layer dimensionality: ',(self.D,self.M))
				print('Learning rate: ',self.lr)
				print('Momentum: ',self.mu)
				print('Cost: ',self.cost[self.idx_hd,self.lr,self.mu][i])
				print('Classification Rate: ',self.cr[self.idx_hd,self.lr,self.mu][i])
				print('')

	def stochastic(self,samples):

		for i in range(self.iterations):
			current_X,current_T = shuffle(self.Xtrain,self.Ttrain)
			for s in range(samples):
				X = current_X[s,:].reshape(1,self.Xtrain.shape[1])
				T = current_T[s,:].reshape(1,self.Ttrain.shape[1])
				if i == 0 and s == 0:
					Y_train,Z = generate_Y(self.activation,X,self.w_0,self.b0_0,self.v_0,self.b1_0)
					Y_test,_ = generate_Y(self.activation,self.Xtest,self.w_0,self.b0_0,self.v_0,self.b1_0)
				else:
					Y_train,Z = generate_Y(self.activation,X,self.w,self.b_0,self.v,self.b_1)
					Y_test,_ = generate_Y(self.activation,self.Xtest,self.w,self.b_0,self.v,self.b_1)
				P_train = np.argmax(Y_train,axis=1)
				P_test = np.argmax(Y_test,axis=1)
				if self.momentum_type:
					if self.momentum_type == 'Standard' or self.momentum_type == 'standard':
						self.cost[(self.idx_hd,self.lr,self.mu)].append(cross_entropy(Y_test,self.Ttest))
						cr = classification_rate(self.Ytest,P_test)
						self.cr[(self.idx_hd,self.lr,self.mu)].append(cr)
						if cr > self.best_cr:
							self.best_cr = cr
							self.best_dimension = self.idx_hd
							self.best_lr = self.lr 
							self.best_mu = self.mu
						self.dv = self.mu*self.dv - (1-self.mu)*self.lr*derivative_v(self.activation,Z,Y_train,T)
						self.db_1 = self.mu*self.db_1 - (1-self.mu)*self.lr*derivative_b1(self.activation,Y_train,T)
						if i == 0 and s == 0:
							self.dw = self.mu*self.dw - (1-self.mu)*self.lr*\
							derivative_w(self.activation,X,Y_train,Z,T,self.v_0)
							self.db_0 = self.mu*self.db_0 - (1-self.mu)*self.lr*\
							derivative_b0(self.activation,Y_train,Z,T,self.v_0)
						else:
							self.dw = self.mu*self.dw - (1-self.mu)*self.lr*\
							derivative_w(self.activation,X,Y_train,Z,T,self.v)
							self.db_0 = self.mu*self.db_0 - (1-self.mu)*self.lr*\
							derivative_b0(self.activation,Y_train,Z,T,self.v)
						if i == 0 and s == 0:
							self.v = self.v_0 + self.dv
							self.b_1 = self.b1_0 + self.db_1
							self.w = self.w_0 + self.dw
							self.b_0 = self.b0_0 + self.db_0
						else:
							self.v += self.dv
							self.b_1 += self.db_1
							self.w += self.dw
							self.b_0 += self.db_0
					else:
						self.cost[(self.idx_hd,self.lr,self.mu)].append(cross_entropy(Y_test,self.Ttest))
						cr = classification_rate(self.Ytest,P_test)
						self.cr[(self.idx_hd,self.lr,self.mu)].append(cr)
						if cr > self.best_cr:
							self.best_cr = cr
							self.best_dimension = self.idx_hd
							self.best_lr = self.lr 
							self.best_mu = self.mu
						self.dv = self.mu*self.dv - (1-self.mu)*self.lr*derivative_v(self.activation,Z,Y_train,T)
						self.db_1 = self.mu*self.db_1 - (1-self.mu)*self.lr*derivative_b1(self.activation,Y_train,T)
						if i == 0 and s == 0:
							self.dw = self.mu*self.dw - (1-self.mu)*self.lr*derivative_w(self.activation,X,Y_train,Z,T,\
								self.v_0)
							self.db_0 = self.mu*self.db_0 - (1-self.mu)*self.lr*derivative_b0(self.activation,Y_train,Z,T,self.v_0)
						else:
							self.dw = self.mu*self.dw - (1-self.mu)*self.lr*derivative_w(self.activation,X,Y_train,Z,T,\
								self.v)
							self.db_0 = self.mu*self.db_0 - (1-self.mu)*self.lr*derivative_b0(self.activation,Y_train,Z,T,self.v)
						if i == 0 and s == 0:
							self.v = self.v_0 + self.mu*self.dv - (1-self.mu)*self.lr*derivative_v(self.activation,Z,Y_train,T)
							self.b_1 = self.b1_0 + self.mu*self.db_1 - (1-self.mu)*self.lr*derivative_b1(self.activation,Y_train,T)
							self.w = self.w_0 + self.mu*self.dw - (1-self.mu)*self.lr*derivative_w(self.activation,X,Y_train,Z,\
								T,self.v)
							self.b_0 = self.b0_0 + self.mu*self.db_0 - (1-self.mu)*self.lr*derivative_b0(self.activation,Y_train,Z,\
								T,self.v)
						else:
							self.v += self.mu*self.dv - (1-self.mu)*self.lr*derivative_v(self.activation,Z,Y_train,T)
							self.b_1 += self.mu*self.db_1 - (1-self.mu)*self.lr*derivative_b1(self.activation,Y_train,T)
							self.w += self.mu*self.dw - (1-self.mu)*self.lr*derivative_w(self.activation,X,Y_train,Z,T,self.v)
							self.b_0 += self.mu*self.db_0 - (1-self.mu)*self.lr*derivative_b0(self.activation,Y_train,Z,T,self.v)
				else:
					if self.adagrad:
						self.cost[(self.idx_hd,self.lr,self.mu)].append(cross_entropy(Y_test,self.Ttest))
						cr = classification_rate(P_test,self.Ytest)
						self.cr[(self.idx_hd,self.lr,self.mu)].append(cr)
						if cr > self.best_cr:
							self.best_cr = cr
							self.best_dimension = self.idx_hd
							self.best_lr = self.lr 
							self.best_mu = self.mu
						epsilon = 10e-10
						self.cache_v += derivative_v(self.activation,Z,Y_train,T)**2
						self.cache_b1 += derivative_b1(self.activation,Y_train,T)**2
						if i == 0 and s == 0:
							self.cache_w += derivative_w(self.activation,X,Y_train,Z,T,self.v_0)**2
							self.cache_b0 += derivative_b0(self.activation,Y_train,Z,T,self.v_0)**2
						else:
							self.cache_w += derivative_w(self.activation,X,Y_train,Z,T,self.v)**2
							self.cache_b0 += derivative_b0(self.activation,Y_train,Z,T,self.v)**2
						if i == 0 and s == 0:
							self.v = self.v_0 - self.lr*derivative_v(self.activation,Z,Y_train,T)/(np.sqrt(self.cache_v + epsilon))
							self.b_1 = self.b1_0 - self.lr*derivative_b1(self.activation,Y_train,T)/(np.sqrt(self.cache_b1 + epsilon))
							self.w = self.w_0 - self.lr*derivative_w(self.activation,X,Y_train,Z,T,\
								self.v_0)/(np.sqrt(self.cache_w + epsilon))
							self.b_0 = self.b0_0 - self.lr*derivative_b0(self.activation,Y_train,Z,T,\
								self.v_0)/(np.sqrt(self.cache_b0 + epsilon))
						else:
							self.v -= self.lr*derivative_v(self.activation,Z,Y_train,T)/(np.sqrt(self.cache_v + epsilon))
							self.b_1 -= self.lr*derivative_b1(self.activation,Y_train,T)/(np.sqrt(self.cache_b1 + epsilon))
							self.w -= self.lr*derivative_w(self.activation,X,Y_train,Z,T,self.v)/(np.sqrt(self.cache_w + \
								epsilon))
							self.b_0 -= self.lr*derivative_b0(self.activation,Y_train,Z,T,self.v)/(np.sqrt(self.cache_b0 + epsilon))
					elif self.rmsprop:
						epsilon = 10e-10
						self.cost[(self.idx_hd,self.lr,self.mu)].append(cross_entropy(Y_test,self.Ttest))
						cr = classification_rate(P_test,self.Ytest)
						self.cr[(self.idx_hd,self.lr,self.mu)].append(cr)
						if cr > self.best_cr:
							self.best_cr = cr
							self.best_dimension = self.idx_hd
							self.best_lr = self.lr 
							self.best_mu = self.mu
						decay = .9
						self.cache_v = decay*self.cache_v + (1-decay)*derivative_v(self.activation,Z,Y_train,T)**2
						self.cache_b1 = decay*self.cache_b1 + (1-decay)*derivative_b1(self.activation,Y_train,T)**2
						if i == 0 and s == 0:
							self.cache_w = decay*self.cache_w + (1-decay)*derivative_w(self.activation,X,Y_train,Z,T,\
								self.v_0)**2
							self.cache_b0 = decay*self.cache_b0 + (1-decay)*derivative_b0(self.activation,Y_train,Z,T,self.v_0)**2
						else:
							self.cache_w = decay*self.cache_w + (1-decay)*derivative_w(self.activation,X,Y_train,Z,T,\
								self.v)**2
							self.cache_b0 = decay*self.cache_b0 + (1-decay)*derivative_b0(self.activation,Y_train,Z,T,self.v)**2
						if i == 0 and s == 0:
							self.v = self.v_0 - self.lr*derivative_v(self.activation,Z,Y_train,T)/(np.sqrt(self.cache_v + epsilon))
							self.b_1 = self.b1_0 - self.lr*derivative_b1(self.activation,Y_train,T)/(np.sqrt(self.cache_b1 + epsilon))
							self.w = self.w_0 - self.lr*derivative_w(self.activation,X,Y_train,Z,T,\
								self.v)/(np.sqrt(self.cache_w + epsilon))
							self.b_0 = self.b0_0 - self.lr*derivative_b0(self.activation,Y_train,Z,T,\
								self.v)/(np.sqrt(self.cache_b0 + epsilon))
						else:
							self.v -= self.lr*derivative_v(self.activation,Z,Y_train,T)/(np.sqrt(self.cache_v + epsilon))
							self.b_1 -= self.lr*derivative_b1(self.activation,Y_train,T)/(np.sqrt(self.cache_b1 + epsilon))
							self.w -= self.lr*derivative_w(self.activation,X,Y_train,Z,T,self.v)/(np.sqrt(self.cache_w + \
								epsilon))
							self.b_0 -= self.lr*derivative_b0(self.activation,Y_train,Z,T,self.v)/(np.sqrt(self.cache_b0 + epsilon))
				if self.hd_sampling > 1:
					if self.current_samples != self.previous_samples and self.current_samples != 1:
						print('-' * 50)
						self.previous_samples = self.current_samples
			if i % 100 == 0:
				print('Iteration: ',i)
				print('Hidden layer dimensionality: ',(self.D,self.M))
				print('Learning rate: ',self.lr)
				print('Momentum: ',self.mu)
				print('Cost: ',self.cost[self.idx_hd,self.lr,self.mu][i])
				print('Classification Rate: ',self.cr[self.idx_hd,self.lr,self.mu][i])
				print('')

	def batch(self,batch_N):

		batches = len(self.Xtrain)//batch_N
		for i in range(self.iterations):
			for b in range(batches):
				X = self.Xtrain[b*batches:(b+1)*batches,:]
				T = self.Ttrain[b*batches:(b+1)*batches,:]
				if i == 0 and b == 0:
					Y_train,Z = generate_Y(self.activation,X,self.w_0,self.b0_0,self.v_0,self.b1_0)
					Y_test,_ = generate_Y(self.activation,self.Xtest,self.w_0,self.b0_0,self.v_0,self.b1_0)
				else:
					Y_train,Z = generate_Y(self.activation,X,self.w,self.b_0,self.v,self.b_1)
					Y_test,_ = generate_Y(self.activation,self.Xtest,self.w,self.b_0,self.v,self.b_1)
				P_train = np.argmax(Y_train,axis=1)
				P_test = np.argmax(Y_test,axis=1)
				if self.momentum_type:
					if self.momentum_type == 'Standard' or self.momentum_type == 'standard':
						self.cost[(self.idx_hd,self.lr,self.mu)].append(cross_entropy(Y_test,self.Ttest))
						cr = classification_rate(self.Ytest,P_test)
						self.cr[(self.idx_hd,self.lr,self.mu)].append(cr)
						if cr > self.best_cr:
							self.best_cr = cr
							self.best_dimension = self.idx_hd
							self.best_lr = self.lr 
							self.best_mu = self.mu
						self.dv = self.mu*self.dv - (1-self.mu)*self.lr*derivative_v(self.activation,Z,Y_train,T)
						self.db_1 = self.mu*self.db_1 - (1-self.mu)*self.lr*derivative_b1(self.activation,Y_train,T)
						if i == 0 and b == 0:
							self.dw = self.mu*self.dw - (1-self.mu)*self.lr*\
							derivative_w(self.activation,X,Y_train,Z,T,self.v_0)
							self.db_0 = self.mu*self.db_0 - (1-self.mu)*self.lr*\
							derivative_b0(self.activation,Y_train,Z,T,self.v_0)
						else:
							self.dw = self.mu*self.dw - (1-self.mu)*self.lr*\
							derivative_w(self.activation,X,Y_train,Z,T,self.v)
							self.db_0 = self.mu*self.db_0 - (1-self.mu)*self.lr*\
							derivative_b0(self.activation,Y_train,Z,T,self.v)
						if i == 0 and b == 0:
							self.v = self.v_0 + self.dv
							self.b_1 = self.b1_0 + self.db_1
							self.w = self.w_0 + self.dw
							self.b_0 = self.b0_0 + self.db_0
						else:
							self.v += self.dv
							self.b_1 += self.db_1
							self.w += self.dw
							self.b_0 += self.db_0
					else:
						self.cost[(self.idx_hd,self.lr,self.mu)].append(cross_entropy(Y_test,self.Ttest))
						cr = classification_rate(self.Ytest,P_test)
						self.cr[(self.idx_hd,self.lr,self.mu)].append(cr)
						if cr > self.best_cr:
							self.best_cr = cr
							self.best_dimension = self.idx_hd
							self.best_lr = self.lr 
							self.best_mu = self.mu
						self.dv = self.mu*self.dv - (1-self.mu)*self.lr*derivative_v(self.activation,Z,Y_train,T)
						self.db_1 = self.mu*self.db_1 - (1-self.mu)*self.lr*derivative_b1(self.activation,Y_train,T)
						if i == 0 and b == 0:
							self.dw = self.mu*self.dw - (1-self.mu)*self.lr*derivative_w(self.activation,X,Y_train,Z,T,\
								self.v_0)
							self.db_0 = self.mu*self.db_0 - (1-self.mu)*self.lr*derivative_b0(self.activation,Y_train,Z,T,self.v_0)
						else:
							self.dw = self.mu*self.dw - (1-self.mu)*self.lr*derivative_w(self.activation,X,Y_train,Z,T,\
								self.v)
							self.db_0 = self.mu*self.db_0 - (1-self.mu)*self.lr*derivative_b0(self.activation,Y_train,Z,T,self.v)
						if i == 0 and b == 0:
							self.v = self.v_0 + self.mu*self.dv - (1-self.mu)*self.lr*derivative_v(self.activation,Z,Y_train,T)
							self.b_1 = self.b1_0 + self.mu*self.db_1 - (1-self.mu)*self.lr*derivative_b1(self.activation,Y_train,T)
							self.w = self.w_0 + self.mu*self.dw - (1-self.mu)*self.lr*derivative_w(self.activation,X,Y_train,Z,\
								T,self.v)
							self.b_0 = self.b0_0 + self.mu*self.db_0 - (1-self.mu)*self.lr*derivative_b0(self.activation,Y_train,Z,\
								T,self.v)
						else:
							self.v += self.mu*self.dv - (1-self.mu)*self.lr*derivative_v(self.activation,Z,Y_train,T)
							self.b_1 += self.mu*self.db_1 - (1-self.mu)*self.lr*derivative_b1(self.activation,Y_train,T)
							self.w += self.mu*self.dw - (1-self.mu)*self.lr*derivative_w(self.activation,X,Y_train,Z,T,self.v)
							self.b_0 += self.mu*self.db_0 - (1-self.mu)*self.lr*derivative_b0(self.activation,Y_train,Z,T,self.v)
				else:
					if self.adagrad:
						self.cost[(self.idx_hd,self.lr,self.mu)].append(cross_entropy(Y_test,self.Ttest))
						cr = classification_rate(P_test,self.Ytest)
						self.cr[(self.idx_hd,self.lr,self.mu)].append(cr)
						if cr > self.best_cr:
							self.best_cr = cr
							self.best_dimension = self.idx_hd
							self.best_lr = self.lr 
							self.best_mu = self.mu
						epsilon = 10e-10
						self.cache_v += derivative_v(self.activation,Z,Y_train,T)**2
						self.cache_b1 += derivative_b1(self.activation,Y_train,T)**2
						if i == 0 and b == 0:
							self.cache_w += derivative_w(self.activation,X,Y_train,Z,T,self.v_0)**2
							self.cache_b0 += derivative_b0(self.activation,Y_train,Z,T,self.v_0)**2
						else:
							self.cache_w += derivative_w(self.activation,X,Y_train,Z,T,self.v)**2
							self.cache_b0 += derivative_b0(self.activation,Y_train,Z,T,self.v)**2
						if i == 0 and b == 0:
							self.v = self.v_0 - self.lr*derivative_v(self.activation,Z,Y_train,T)/(np.sqrt(self.cache_v + epsilon))
							self.b_1 = self.b1_0 - self.lr*derivative_b1(self.activation,Y_train,T)/(np.sqrt(self.cache_b1 + epsilon))
							self.w = self.w_0 - self.lr*derivative_w(self.activation,X,Y_train,Z,T,\
								self.v_0)/(np.sqrt(self.cache_w + epsilon))
							self.b_0 = self.b0_0 - self.lr*derivative_b0(self.activation,Y_train,Z,T,\
								self.v_0)/(np.sqrt(self.cache_b0 + epsilon))
						else:
							self.v -= self.lr*derivative_v(self.activation,Z,Y_train,T)/(np.sqrt(self.cache_v + epsilon))
							self.b_1 -= self.lr*derivative_b1(self.activation,Y_train,T)/(np.sqrt(self.cache_b1 + epsilon))
							self.w -= self.lr*derivative_w(self.activation,X,Y_train,Z,T,self.v)/(np.sqrt(self.cache_w + \
								epsilon))
							self.b_0 -= self.lr*derivative_b0(self.activation,Y_train,Z,T,self.v)/(np.sqrt(self.cache_b0 + epsilon))
					elif self.rmsprop:
						epsilon = 10e-10
						self.cost[(self.idx_hd,self.lr,self.mu)].append(cross_entropy(Y_test,self.Ttest))
						cr = classification_rate(P_test,self.Ytest)
						self.cr[(self.idx_hd,self.lr,self.mu)].append(cr)
						if cr > self.best_cr:
							self.best_cr = cr
							self.best_dimension = self.idx_hd
							self.best_lr = self.lr 
							self.best_mu = self.mu
						decay = .9
						self.cache_v = decay*self.cache_v + (1-decay)*derivative_v(self.activation,Z,Y_train,T)**2
						self.cache_b1 = decay*self.cache_b1 + (1-decay)*derivative_b1(self.activation,Y_train,T)**2
						if i == 0 and b == 0:
							self.cache_w = decay*self.cache_w + (1-decay)*derivative_w(self.activation,X,Y_train,Z,T,\
								self.v_0)**2
							self.cache_b0 = decay*self.cache_b0 + (1-decay)*derivative_b0(self.activation,Y_train,Z,T,self.v_0)**2
						else:
							self.cache_w = decay*self.cache_w + (1-decay)*derivative_w(self.activation,X,Y_train,Z,T,\
								self.v)**2
							self.cache_b0 = decay*self.cache_b0 + (1-decay)*derivative_b0(self.activation,Y_train,Z,T,self.v)**2
						if i == 0 and b == 0:
							self.v = self.v_0 - self.lr*derivative_v(self.activation,Z,Y_train,T)/(np.sqrt(self.cache_v + epsilon))
							self.b_1 = self.b1_0 - self.lr*derivative_b1(self.activation,Y_train,T)/(np.sqrt(self.cache_b1 + epsilon))
							self.w = self.w_0 - self.lr*derivative_w(self.activation,X,Y_train,Z,T,\
								self.v)/(np.sqrt(self.cache_w + epsilon))
							self.b_0 = self.b0_0 - self.lr*derivative_b0(self.activation,Y_train,Z,T,\
								self.v)/(np.sqrt(self.cache_b0 + epsilon))
						else:
							self.v -= self.lr*derivative_v(self.activation,Z,Y_train,T)/(np.sqrt(self.cache_v + epsilon))
							self.b_1 -= self.lr*derivative_b1(self.activation,Y_train,T)/(np.sqrt(self.cache_b1 + epsilon))
							self.w -= self.lr*derivative_w(self.activation,X,Y_train,Z,T,self.v)/(np.sqrt(self.cache_w + \
								epsilon))
							self.b_0 -= self.lr*derivative_b0(self.activation,Y_train,Z,T,self.v)/(np.sqrt(self.cache_b0 + epsilon))
				if self.hd_sampling > 1:
					if self.current_samples != self.previous_samples and self.current_samples != 1:
						print('-' * 50)
						self.previous_samples = self.current_samples
			if i % 100 == 0:
				print('Iteration: ',i)
				print('Hidden layer dimensionality: ',(self.D,self.M))
				print('Learning rate: ',self.lr)
				print('Momentum: ',self.mu)
				print('Cost: ',self.cost[self.idx_hd,self.lr,self.mu][i])
				print('Classification Rate: ',self.cr[self.idx_hd,self.lr,self.mu][i])
				print('')

	def graph_cost(self):

		for e,i in self.cost.items():
			plt.plot(i,label='hd=%s,lr=%s,mu=%s'%(e[0],e[1],e[2]))
		plt.title('Cost')
		plt.legend()
		plt.show()

	def graph_cr(self):

		for e,i in self.cr.items():
			plt.plot(i,label='hd=%s,lr=%s,mu=%s'%(e[0],e[1],e[2]))
		plt.title('Classification Rate')
		plt.legend()
		plt.show()

if __name__ == '__main__':

	# Random(activation,iterations,hidden_dimension,learning_rates,momentums,
	# momentum_type=None,ada=None,rms=None,adaptive=None,gradient=None)
	model = Random('relu',2000,(20,500),(-8,-4),(-4,-1),momentum_type='Standard',gradient='Full')
	model.fit()
	model.graph_cost()
	model.graph_cr()
	print('Best Classification Rate: ',model.best_cr)
	print('Best Hyperparamters: ')
	print('Hidden layer dimensionality: ',model.best_dimension)
	print('Learning rate: ',model.best_lr)
	print('Momentum: ',model.best_mu)