"""
线性回归

"""

from numpy import *

# 加载数据
def loadData(filename):
	file = open(filename)
	fea_num = len(file.readline().split('\t'))-1 # 特征数
	data = []
	label = []
	file = open(filename)
	for line in file.readlines():
		lines = []
		cur_line = line.strip().split('\t')
		for i in range(fea_num):
			lines.append(float(cur_line[i]))
		data.append(lines)
		label.append(float(cur_line[-1]))
	return data, label

# 标准的线性回归
def reg(X, y):
	X = mat(X)
	y = mat(y).T
	XTX = X.T*X
	if linalg.det(XTX) == 0.0: # 行列式是否为0
		print("This matrix is singular, cannot do inverse")
		return
	weights = XTX.I*(X.T*y)
	return weights

# 绘图
def plotResult(X, y, weights):
	import matplotlib.pyplot as plt
	X = mat(X)
	y = mat(y)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(X[:,1].flatten().A[0], y.T[:,0].flatten().A[0])
	X_copy = X.copy()
	X_copy.sort(0) # 升序排序
	y_hat = X_copy*weights
	ax.plot(X_copy[:,1], y_hat)
	plt.show()


if __name__ == '__main__':
	X, y = loadData('ex0.txt')
	weights = reg(X, y)
	print("The regression weights is: ", weights)
	plotResult(X, y, weights)