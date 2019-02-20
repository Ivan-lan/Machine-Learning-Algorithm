"""
逻辑回归
逻辑回归的推导中，它假设样本服从伯努利分布（0-1分布）
"""
from numpy import *

# 加载数据
def loadData():
	data = []
	label = []
	file = open('testSet.txt')
	for line in file.readlines():
		line = line.strip().split()
		data.append([1.0, float(line[0]), float(line[1])])
		label.append(int(line[2]))
	return data, label

# Sigmoid 函数
def sigmoid(X):
	return 1.0/(1 + exp(-X))

# 梯度上升法
def gradientAscent(data,label):
	data = mat(data)
	label = mat(label).transpose() # 转为列向量
	row, col = shape(data)
	alpha = 0.001 # 步长
	iter_num = 500 # 迭代次数
	weights = ones((col, 1)) # 初始化权重向量
	for i in range(iter_num):
		y_ = sigmoid(data*weights) # 一次全部样本
		error = (label - y_) # 计算误差
		weights = weights + alpha*data.transpose()*error # 梯度上升的更新公式：w = w + alpha*[y_-y]*X^T 列向量乘矩阵
	return weights

# 简化的随机梯度上升法
"""
对批量梯度上升法的改进，一次仅用一个样本点来更新系数；
由于可在样本到来时对分类器进行增量式更新，
是在线学习算法。（一次处理所有数据的称为批处理）。计算量更少，更快！！
"""
def stoGraAscent0(data, label):
	row, col = shape(data)
	alpha = 0.01
	weights = ones(col)
	for i in range(row): # 仅遍历一次全部样本
		y_ = sigmoid(sum(data[i]*weights)) # 一次一个样本
		error = label[i] - y_
		weights = weights + alpha*error*data[i]
	return weights

# 改进的随机梯度下降法
"""
alpha 在每次迭代的时候都会调整：随着迭代次数的增加而减小，以使系数趋于收敛
通过随机选择样本来更新系数
增加对迭代次数的控制
"""
def stoGraAscent1(data, label, iter_num = 150):
	row, col = shape(data)
	weights = ones(col)
	for k in range(iter_num):
		idx = range(row) # 每次迭代都初始化一个索引列表(后面会删除操作)
		for i in range(row):
		alpha = 4/(1.0+j+i) + 0.0001 # alpha随迭代次数的增加而变小
		idx_random = int(random.uniform(0, len(idx)))  # 随机选取样本
		y_ = sigmoid(sum(data[idx_random]*weights))
		error = label[idx_random] - y_
		weights = weights + alpha*error*data[idx_random]
		del(list(idx)[idx_random]) # 删除所选样本的索引
	return weights

# 画决策边界
def plotResult(weights):
	import matplotlib.pyplot as plt
	data , label = loadData()
	data = array(data)
	row = shape(data)[0]
	X1 = []
	y1= []
	X2 = []
	y2 = []
	for i in range(row):
		if int(label[i]) == 1:
			X1.append(data[i, 1])
			y1.append(data[i, 2])
		else:
			X2.append(data[i, 1])
			y2.append(data[i, 2])
	plt.figure()
	x = arange(-3.0, 3.0, 0.1).reshape(1, -1)
	y = (-weights[0] - weights[1]*x)/weights[2]
	plt.plot(x,y,'o')
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.scatter(X1, y1, s=30, c='red', marker='s')
	plt.scatter(X2, y2, s=30, c='green')
	plt.show()

if __name__ == "__main__":
	data, label = loadData()
	weights = gradientAscent(data,label)
	print("The predicted weights is :", weights)
	plotResult(weights)