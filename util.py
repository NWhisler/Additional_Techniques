from sklearn.decomposition import PCA
import pandas as pd, numpy as np, matplotlib.pyplot as plt 

def transform_data():

	df = pd.read_csv('train.csv')
	data = df.as_matrix()
	X = data[:,1:]
	X = (X - X.mean())/X.std()
	pca = PCA(150)
	X = pca.fit_transform(X)
	Y = data[:,0]
	labels = {0:'Zero',1:'One',2:'Two',3:'Three',4:'Four',5:'Five',6:'Six',7:'Seven',8:'Eight',9:'Nine'}
	# while True:
	# 	N,D = X.shape
	# 	sample = np.random.choice(range(N))
	# 	plt.imshow(X[sample,:].reshape(28,28),cmap='gray')
	# 	plt.title('%s' % labels[Y[sample]])
	# 	plt.show()
	# 	if input('Continue: (Y/N)') != 'Y':
	# 		break
	return X,Y

def derivative_v(activation,Z,Y,T):

	return Z.T.dot(Y - T)

def derivative_b1(activation,Y,T):

	return (Y - T).sum(axis=0)

def derivative_w(activation,X,Y,Z,T,v):

	if activation == 'tanh':
		a = (Y - T).dot(v.T)
		b = (1 - Z * Z)
		return X.T.dot(a * b)
	elif activation == 'sigmoid':
		a = (Y - T).dot(v.T)
		b = Z * (1 - Z)
		return X.T.dot(a * b)
	else:
		a = (Y - T).dot(v.T)
		b = Z * (Z > 0)
		return X.T.dot(a * b)

def derivative_b0(activation,Y,Z,T,v):

	if activation == 'tanh':
		a = (Y - T).dot(v.T)
		b = (1 - Z * Z)
		return (a * b).sum(axis=0)
	elif activation == 'sigmoid':
		a = (Y - T).dot(v.T)
		b = Z * (1 - Z)
		return (a * b).sum(axis=0)
	else:
		a = (Y - T).dot(v.T)
		b = Z * (Z > 0)
		return (a * b).sum(axis=0)

def sigmoid(a):

	return 1/(1 + np.exp(-a))

def relu(a):

	return a * (a > 0)

def generate_Y(activation,X,w,b0,v,b1):

	a = X.dot(w) + b0
	if activation == 'tanh':
		Z = np.tanh(a)
	elif activation == 'sigmoid':
		Z = sigmoid(a)
	else:
		Z = relu(a)
	A = Z.dot(v) + b1
	return np.exp(A)/np.exp(A).sum(axis=1,keepdims=True),Z

def generate_T(Y):

	N = len(Y)
	K = len(set(Y))
	T = np.zeros((N,K))
	idx_row = np.arange(N)
	idx_col = Y.astype(int)
	T[idx_row,idx_col] = 1
	return T

def cross_entropy(Y,T):

	return -np.sum(T * np.log(Y))

def classification_rate(Y,T):

	return np.mean(Y == T)