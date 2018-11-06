from util import transform_data, derivative_v,derivative_b1,derivative_w, derivative_b0, generate_Y, generate_T, cross_entropy,\
classification_rate
from sklearn.utils import shuffle
import numpy as np, matplotlib.pyplot as plt 

def step_decay(learning_rate):

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
	dv = 0
	d_b1 = 0
	dw = 0
	d_b0 = 0
	mu = .9
	v = np.random.randn(M,K)/np.sqrt(M+K)
	b_1 = np.random.randn(K)/np.sqrt(K)
	w = np.random.randn(D,M)/np.sqrt(D+M)
	b_0 = np.random.randn(M)/np.sqrt(M)
	learning_rate = learning_rate
	step_cost = []
	step_cr = []
	step_lr = []
	best_step = 0
	best_iteration = 0
	for i in range(iterations):
		if i != 0 and i % 10 == 0:
			learning_rate = learning_rate/2
		step_lr.append(learning_rate)
		for b in range(batches):
			X = Xtrain[b*batches:(b+1)*batches,:]
			T = Ttrain[b*batches:(b+1)*batches,:]
			Y,Z = generate_Y('tanh',X,w,b_0,v,b_1)
			Y_test,_ = generate_Y('tanh',Xtest,w,b_0,v,b_1)
			P_test = np.argmax(Y_test,axis=1)
			if b % batches == 0:
				step_cost.append(cross_entropy(Y_test,Ttest))
				cr = classification_rate(P_test,Ytest)
				step_cr.append(cr)
				if cr > best_step:
					best_step = cr
					best_iteration = i
			dv = mu*dv - learning_rate*derivative_v('tanh',Z,Y,T)
			d_b1 = mu*d_b1 - learning_rate*derivative_b1('tanh',Y,T)
			dw = mu*dw - learning_rate*derivative_w('tanh',X,Y,Z,T,v)
			d_b0 = mu*d_b0 - learning_rate*derivative_b0('tanh',Y,Z,T,v)
			v += dv
			b_1 += d_b1
			w += dw
			b_0 += d_b0
		if i % 10 == 0:
			print('Step Cost: ',step_cost[i],'Step Classification: ',step_cr[i])
	return step_cost,step_cr,step_lr,best_step,best_iteration

def exp_decay(learning_rate):

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
	dv = 0
	d_b1 = 0
	dw = 0
	d_b0 = 0
	mu = .9
	v = np.random.randn(M,K)/np.sqrt(M+K)
	b_1 = np.random.randn(K)/np.sqrt(K)
	w = np.random.randn(D,M)/np.sqrt(D+M)
	b_0 = np.random.randn(M)/np.sqrt(M)
	learning_rate = learning_rate
	exp_cost = []
	exp_cr = []
	exp_lr = []
	best_exp = 0
	best_iteration = 0
	for i in range(iterations):
		learning_rate = learning_rate*np.exp(-K*i)
		exp_lr.append(learning_rate)
		for b in range(batches):
			X = Xtrain[b*batches:(b+1)*batches,:]
			T = Ttrain[b*batches:(b+1)*batches,:]
			Y,Z = generate_Y('tanh',X,w,b_0,v,b_1)
			Y_test,_ = generate_Y('tanh',Xtest,w,b_0,v,b_1)
			P_test = np.argmax(Y_test,axis=1)
			if b % batches == 0:
				exp_cost.append(cross_entropy(Y_test,Ttest))
				cr = classification_rate(P_test,Ytest)
				exp_cr.append(cr)
				if cr > best_exp:
					best_exp = cr
					best_iteration = i
			dv = mu*dv - learning_rate*derivative_v('tanh',Z,Y,T)
			d_b1 = mu*d_b1 - learning_rate*derivative_b1('tanh',Y,T)
			dw = mu*dw - learning_rate*derivative_w('tanh',X,Y,Z,T,v)
			d_b0 = mu*d_b0 - learning_rate*derivative_b0('tanh',Y,Z,T,v)
			v += dv
			b_1 += d_b1
			w += dw
			b_0 += d_b0
		if i % 10 == 0:
			print('Exp Cost: ',exp_cost[i],'Exp Classification: ',exp_cr[i])
	return exp_cost,exp_cr,exp_lr,best_exp,best_iteration

def one_over(learning_rate):

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
	dv = 0
	d_b1 = 0
	dw = 0
	d_b0 = 0
	mu = .9
	v = np.random.randn(M,K)/np.sqrt(M+K)
	b_1 = np.random.randn(K)/np.sqrt(K)
	w = np.random.randn(D,M)/np.sqrt(D+M)
	b_0 = np.random.randn(M)/np.sqrt(M)
	learning_rate = learning_rate
	over_cost = []
	over_cr = []
	over_lr = []
	best_over = 0
	best_iteration = 0
	for i in range(iterations):
		learning_rate = learning_rate/(K*i+1)
		over_lr.append(learning_rate)
		for b in range(batches):
			X = Xtrain[b*batches:(b+1)*batches,:]
			T = Ttrain[b*batches:(b+1)*batches,:]
			Y,Z = generate_Y('tanh',X,w,b_0,v,b_1)
			Y_test,_ = generate_Y('tanh',Xtest,w,b_0,v,b_1)
			P_test = np.argmax(Y_test,axis=1)
			if b % batches == 0:
				over_cost.append(cross_entropy(Y_test,Ttest))
				cr = classification_rate(P_test,Ytest)
				over_cr.append(cr)
				if cr > best_over:
					best_over = cr
					best_iteration = i
			dv = mu*dv - learning_rate*derivative_v('tanh',Z,Y,T)
			d_b1 = mu*d_b1 - learning_rate*derivative_b1('tanh',Y,T)
			dw = mu*dw - learning_rate*derivative_w('tanh',X,Y,Z,T,v)
			d_b0 = mu*d_b0 - learning_rate*derivative_b0('tanh',Y,Z,T,v)
			v += dv
			b_1 += d_b1
			w += dw
			b_0 += d_b0
		if i % 10 == 0:
			print('Over Cost: ',over_cost[i],'Over Classification: ',over_cr[i])
	return over_cost,over_cr,over_lr,best_over,best_iteration

def adagrad(learning_rate):

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
	dv = 0
	d_b1 = 0
	dw = 0
	d_b0 = 0
	mu = .9
	v = np.random.randn(M,K)/np.sqrt(M+K)
	b_1 = np.random.randn(K)/np.sqrt(K)
	w = np.random.randn(D,M)/np.sqrt(D+M)
	b_0 = np.random.randn(M)/np.sqrt(M)
	cache_v = np.ones((M,K))
	cache_b1 = np.ones(K)
	cache_w = np.ones((D,M))
	cache_b0 = np.ones(M)
	epsilon = 10e-10
	adagrad_cost = []
	adagrad_cr = []
	best_ada = 0
	best_iteration = 0
	for i in range(iterations):
		for b in range(batches):
			X = Xtrain[b*batches:(b+1)*batches,:]
			T = Ttrain[b*batches:(b+1)*batches,:]
			Y,Z = generate_Y('tanh',X,w,b_0,v,b_1)
			Y_test,_ = generate_Y('tanh',Xtest,w,b_0,v,b_1)
			P_test = np.argmax(Y_test,axis=1)
			if b % batches == 0:
				adagrad_cost.append(cross_entropy(Y_test,Ttest))
				cr = classification_rate(P_test,Ytest)
				adagrad_cr.append(cr)
				if cr > best_ada:
					best_ada = cr
					best_iteration = i
			cache_v += derivative_v('tanh',Z,Y,T)**2
			cache_b1 += derivative_b1('tanh',Y,T)**2
			cache_w += derivative_w('tanh',X,Y,Z,T,v)**2
			cache_b0 += derivative_b0('tanh',Y,Z,T,v)**2
			dv = mu*dv - learning_rate*derivative_v('tanh',Z,Y,T)/(np.sqrt(cache_v + epsilon))
			d_b1 = mu*d_b1 - learning_rate*derivative_b1('tanh',Y,T)/(np.sqrt(cache_b1 + epsilon))
			dw = mu*dw - learning_rate*derivative_w('tanh',X,Y,Z,T,v)/(np.sqrt(cache_w + epsilon))
			d_b0 = mu*d_b0 - learning_rate*derivative_b0('tanh',Y,Z,T,v)/(np.sqrt(cache_b0 + epsilon))
			v += dv
			b_1 += d_b1
			w += dw
			b_0 += d_b0
		if i % 10 == 0:
			print('Adagrad Cost: ',adagrad_cost[i],'Adagrad Classification: ',adagrad_cr[i])
	return adagrad_cost,adagrad_cr,best_ada,best_iteration

def rmsprop(learning_rate):

	X,Y = transform_data()
	X,Y = shuffle(X,Y)
	N = len(X)//2
	Xtrain = X[:N]
	Ytrain = Y[:N]
	Ttrain = generate_T(Ytrain)
	Xtest = X[N:]
	Ytest = Y[N:]
	Ttest = generate_T(Ytest)
	Ttest = generate_T(Ytest)
	N,D = Xtrain.shape
	M = 100
	K = len(set(Y))
	iterations = 50
	batch_N = 250
	batches = N//batch_N
	dv = 0
	d_b1 = 0
	dw = 0
	d_b0 = 0
	mu = .9
	v = np.random.randn(M,K)/np.sqrt(M+K)
	b_1 = np.random.randn(K)/np.sqrt(K)
	w = np.random.randn(D,M)/np.sqrt(D+M)
	b_0 = np.random.randn(M)/np.sqrt(M)
	cache_v = np.ones((M,K))
	cache_b1 = np.ones(K)
	cache_w = np.ones((D,M))
	cache_b0 = np.ones(M)
	epsilon = 10e-10
	decay = .9
	rmsprop_cost = []
	rmsprop_cr = []
	best_rms = 0
	best_iteration = 0
	for i in range(iterations):
		for b in range(batches):
			X = Xtrain[b*batches:(b+1)*batches,:]
			T = Ttrain[b*batches:(b+1)*batches,:]
			Y,Z = generate_Y('tanh',X,w,b_0,v,b_1)
			Y_test,_ = generate_Y('tanh',Xtest,w,b_0,v,b_1)
			P_test = np.argmax(Y_test,axis=1)
			if b % batches == 0:
				rmsprop_cost.append(cross_entropy(Y_test,Ttest))
				cr = classification_rate(P_test,Ytest)
				rmsprop_cr.append(cr)
				if cr > best_rms:
					best_rms = cr
					best_iteration = i
			cache_v = decay*cache_v + (1-decay)*derivative_v('tanh',Z,Y,T)**2
			cache_b1 = decay*cache_b1 + (1-decay)*derivative_b1('tanh',Y,T)**2
			cache_w = decay*cache_w + (1-decay)*derivative_w('tanh',X,Y,Z,T,v)**2
			cache_b0 = decay*cache_b0 + (1-decay)*derivative_b0('tanh',Y,Z,T,v)**2
			dv = mu*dv - learning_rate*derivative_v('tanh',Z,Y,T)/(np.sqrt(cache_v + epsilon))
			d_b1 = mu*d_b1 - learning_rate*derivative_b1('tanh',Y,T)/(np.sqrt(cache_b1 + epsilon))
			dw = mu*dw - learning_rate*derivative_w('tanh',X,Y,Z,T,v)/(np.sqrt(cache_w + epsilon))
			d_b0 = mu*d_b0 - learning_rate*derivative_b0('tanh',Y,Z,T,v)/(np.sqrt(cache_b0 + epsilon))
			v += dv
			b_1 += d_b1
			w += dw
			b_0 += d_b0
		if i % 10 == 0:
			print('RMSProp Cost: ',rmsprop_cost[i],'RMSProp Classification: ',rmsprop_cr[i])
	return rmsprop_cost,rmsprop_cr,best_rms,best_iteration

if __name__ == '__main__':

	learning_rate = 10e-5
	step_cost,step_cr,step_lr,best_step,step_iteration = step_decay(learning_rate)
	exp_cost,exp_cr,exp_lr,best_exp,exp_iteration = exp_decay(learning_rate)
	over_cost,over_cr,over_lr,best_over,over_iteration = one_over(learning_rate)
	adagrad_cost,adagrad_cr,best_ada,ada_iteration = adagrad(learning_rate)
	rmsprop_cost,rmsprop_cr,best_rms,rms_iteration = rmsprop(learning_rate)
	step_exp = np.mean(np.array(step_cost) - np.array(exp_cost))
	step_over = np.mean(np.array(step_cost) - np.array(over_cost))
	step_ada = np.mean(np.array(step_cost) - np.array(adagrad_cost))
	step_rms = np.mean(np.array(step_cost) - np.array(rmsprop_cost))
	exp_over = np.mean(np.array(exp_cost) - np.array(over_cost))
	exp_ada = np.mean(np.array(exp_cost) - np.array(adagrad_cost))
	exp_rms = np.mean(np.array(exp_cost) - np.array(rmsprop_cost))
	ada_rms = np.mean(np.array(adagrad_cost) - np.array(rmsprop_cost))
	plt.plot(step_cost,label='Step Decay')
	plt.plot(exp_cost,label='Exp Decay')
	plt.plot(over_cost,label='One Over Decay')
	plt.plot(adagrad_cost,label='Adagrad')
	plt.plot(rmsprop_cost,label='RMSProp')
	plt.title('Cost')
	plt.legend()
	plt.show()
	plt.plot(step_cr,label='Step Decay')
	plt.plot(exp_cr,label='Exp Decay')
	plt.plot(over_cr,label='One Over Decay')
	plt.plot(adagrad_cr,label='Adagrad')
	plt.plot(rmsprop_cr,label='RMSProp')
	plt.title('Classification Rate')
	plt.legend()
	plt.show()
	plt.plot(step_lr,label='Step Decay')
	plt.title('Learning Rate')
	plt.legend()
	plt.show()
	plt.plot(exp_lr,label='Exp Decay')
	plt.title('Learning Rate')
	plt.legend()
	plt.show()
	plt.plot(over_lr,label='One Over Decay')
	plt.title('Learning Rate')
	plt.legend()
	plt.show()
	print('')
	print('Best Step Decay Classification Rate: ',best_step,'Iteration: ',step_iteration)
	print('Best Exponential Decay Classification Rate: ',best_exp,'Iteration: ',exp_iteration)
	print('Best Proportional Decay Classification Rate: ',best_over,'Iteration: ',over_iteration)
	print('Best AdaGrad Classification Rate: ',best_ada,'Iteration: ',ada_iteration)
	print('Best RMSProp Classification Rate: ',best_rms,'Iteration: ',rms_iteration)
	print('')
	if step_exp > 0:
		print('Exponential decay error rate is ' + str(step_exp) + ' less per iteration on avg than step decay.')
	elif step_exp < 0:
		print('Step decay error rate is ' + str(-step_exp) + ' less per iteration on avg than exponential decay.')
	else:
		print('Step and exponential decay have equivalent error rates on avg.')
	if step_over > 0:
		print('Proportional decay error rate is ' + str(step_over) + ' less per iteration on avg than step decay.')
	elif step_over < 0:
		print('Step decay error rate is ' + str(-step_over) + ' less per iteration on avg than proportional decay.')
	else:
		print('Step and proportional decay have equivalent error rates on avg.')
	if step_ada > 0:
		print('AdaGrad error rate is ' + str(step_ada) + ' less per iteration on avg than step decay.')
	elif step_ada < 0:
		print('Step decay error rate is ' + str(-step_ada) + ' less per iteration on avg than AdaGrad.')
	else:
		print('Step decay and AdaGrad have equivalent error rates on avg.')
	if step_rms > 0:
		print('RMSProp error rate is ' + str(step_rms) + ' less per iteration on avg than step decay.')
	elif step_rms < 0:
		print('Step decay error rate is ' + str(-step_rms) + ' less per iteration on avg than RMSProp.')
	else:
		print('Step decay and RMSProp have equivalent error rates on avg.')
	if exp_over > 0:
		print('Proportional decay error rate is ' + str(exp_over) + ' less per iteration on avg than exponential decay.')
	elif exp_over < 0:
		print('Exponential decay error rate is ' + str(-exp_over) + ' less per iteration on avg than proportional decay.')
	else:
		print('Exponential and proportional decay have equivalent error rates on avg.')
	if exp_ada > 0:
		print('AdaGrad error rate is ' + str(exp_ada) + ' less per iteration on avg than exponential decay.')
	elif exp_ada < 0:
		print('Exponential decay error rate is ' + str(-exp_ada) + ' less per iteration on avg than AdaGrad.')
	else:
		print('Exponential and AdaGrad have equivalent error rates on avg.')
	if exp_rms > 0:
		print('RMSProp error rate is ' + str(exp_rms) + ' less per iteration on avg than exponential decay.')
	elif exp_rms < 0:
		print('Exponential decay error rate is ' + str(-exp_rms) + ' less per iteration on avg than RMSProp.')
	else:
		print('Exponential and RMSProp have equivalent error rates on avg.')
	if ada_rms > 0:
		print('RMSProp error rate is ' + str(ada_rms) + ' less per iteration on avg than AdaGrad.')
	elif ada_rms < 0:
		print('AdaGrad error rate is ' + str(-ada_rms) + ' less per iteration on avg than RMSProp.')
	else:
		print('AdaGrad and RMSProp have equivalent error rates on avg.')