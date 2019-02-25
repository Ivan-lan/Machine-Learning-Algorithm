"""
树回归

回归树： 每个叶节点包含单个值
模型树： 每个叶节点包含一个线性方法

"""

from numpy import *

# 加载数据
def loadData(filename):
	data = []
	file = open(filename)
	for line in file.readlines():
		cur_line = line.strip().split('\t')
		flt_line = list(map(float, cur_line))
		data.append(flt_line)
	return data

# 根据给定特征和特征值二分数据集
"""
CART算法假设决策树是二叉树，采用二元切分法，
即每次将数据集切分成两份，如果特征值大于给定值就走左子树，否则走右子树。
"""
def binSplitData(data, fea, value):
	mat0 = data[nonzero(data[:, fea] > value)[0], :] # np.nonzero函数是numpy中用于得到数组array中非零元素的位置
	mat1 = data[nonzero(data[:, fea] <= value)[0], :]
	return mat0, mat1

# 生成叶节点和计算误差
"""
回归树假设叶节点是常数值，叶节点的最佳输出值是叶节点的所有实例的输出的均值。
对于连续的输出变量，在选择划分的特征和特征值时，以什么标准选择？
用平方误差来表示对训练数据的预测误差。即用平方误差最小的准则求解。（统计学习方法）

"""
def Leaf(data):
	return mean(data[:, -1])

def Error(data):
	return var(data[:,-1]) * shape(data)[0] # 总方差等于均方差x样本个数(预测值是均值，所以误差等于方差)


# 切分函数

"""
伪代码： 对每个特征：

对每个特征值：
	将数据集切分成两份
	计算切分的误差
	如果当前误差小于当前最小误差，那么将当前切分设定为最佳切分并更新最小误差
返回最佳切分的特征和阈值

遍历所有特征及其可能的取值来找到使误差最小化的切分阈值
"""

def choseBestSplit(data, leafType=Leaf, errType=Error, ops=(1,4)):
	tolS = ops[0] # 容许的误差下降值
	tolN = ops[1] # 切分的最小样本数
	# 切分的停止条件：
	if len(set(data[:, -1].T.tolist()[0])) == 1:
		return None, leafType(data)

	row, col = shape(data)
	Error = errType(data) # 计算数据集总方差
	error_best = inf
	idx_best = 0  # 最佳特征索引
	value_best = 0  # 最佳特征取值

	for idx in range(col-1): # 遍历所有特征
		for val in set(data[:, idx].T.tolist()[0]): # 遍历所有取值
			mat0, mat1 = binSplitData(data, idx, val)
			if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
				continue # 切分后的子集样本数小于指定值，不切
			error_new = errType(mat0) + errType(mat1)
			if error_new < error_best:
				idx_best = idx
				value_best = val
				error_best = error_new

	if (Error - error_best) < tolS: # 切分停止条件2
		return None, leafType(data) 

	mat0, mat1 = binSplitData(data, idx_best, value_best)

	if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
		return None, leafType(data)

	return idx_best, value_best  


# 构建树

"""
createTree()函数的伪代码：
找到最佳待切分特征：
	如果该节点不能再分，将该节点存为叶节点
	执行二元切分
	在右子树调用createTree()方法
	在左子树调用createTree()方法

在回归树中，createTree()函数的默认参数为：误差计算方式是调用Error()函数，即计算总方差。
叶节点的生成方式是调用Leaf()函数，即计算均值。在后面的模型树的构建中，共用这个函数，
只需传入模型树所需的误差计算函数（先线性拟合，再计算方差和）和叶节点生成函数（叶节点为线性模型，即返回其系数）。
"""

def createTree(data, leafType=Leaf, errType=Error, ops=(1,4)):
	fea, val = choseBestSplit(data, leafType, errType, ops)
	if fea == None: # 递归停止条件
		return val
	retTree = {}
	retTree['fea_idx'] = fea
	retTree['fea_val'] = val
	lSet, rSet = binSplitData(data, fea, val)
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree

if __name__ == "__main__":
	data = loadData('ex0.txt')
	data = mat(data)
	retTree = createTree(data)
	print("The tree is: ", retTree)

# -------------------------------------------------------------------

# 树剪枝
"""
一棵树的节点如果过多，表明模型可能对数据“过拟合”，如何判断过拟合呢？
可以用测试集上交叉验证来发现过拟合。

通过降低决策树的复杂度来避免过拟合的过程称为剪枝。

在函数chooseBestSplit()中的提前终止条件，实际上就是在进行预剪枝。
另一种形式的剪枝需要训练集和测试集，称为“后剪枝”。后剪枝更为有效。
"""

# 预剪枝
"""
chooseBestSplit()中的提前终止条件其实对用户指定的参数tolS 和 tolN比较敏感，取值会影响结果，
即树的结构会受参数的影响，手工调整最佳参数不现实。

后剪枝是利用测试集来对树进行剪枝，不用指定参数，更为理想。
"""

# 后剪枝

"""
使用后剪枝需要将数据集分为训练集和测试集。首先指定参数，使构建的树足够大足够复杂，便于剪枝。
接下来从上而下找到叶节点，用测试集来判断这些叶节点合并是否能够降低测试误差，如果是就合并。
伪代码：
基于已有的树切分测试数据：
	如果存在任一子集是一棵树，则在该子集递归剪枝过程
	计算将当前两个叶节点合并后的误差
	计算不合并的误差
	如果合并会降低误差的话，将叶节点合并
"""

# 判断是否是叶节点
def isTree(obj):
	return (type(obj).__name__ == "dict")

# 合并叶节点
def mergeNode(tree):
	if isTree(tree['right']):
		tree['right'] = mergeNode(tree['right'])
	if isTree(tree['left']):
		tree['left'] = mergeNode(tree['left'])
	return (tree['left'] + tree['right']/2)

# 剪枝函数
"""
递归遍历到叶节点，计算剪枝前后的误差，以决定是否剪枝。
"""
def prune(tree, testData):
	"""
	tree : 待剪枝决策树
	testData: 剪枝所需的测试集
	"""
	# 递归停止条件，测试集为空
	if shape(testData)[0] == 0:
		return mergeNode(tree)
	# 将测试集划分到树的分支上
	if (isTree(tree['right']) or isTree(tree['left'])):
		lSet, rSet = binSplitData(testData, tree['fea_idx'], tree['fea_val'])
	# 递归
	if isTree(tree['left']):
		tree['left'] = prune(tree['left'], lSet)
	if isTree(tree['right']):
		tree['right'] = prune(tree['right'], rSet)

	# 叶子节点，计算合并前后的误差，决定是否合并 
	if not isTree(tree['left']) and not isTree(tree['right']):
		lSet, rSet = binSplitData(testData, tree['fea_idx'], tree['fea_val'])
		error_noMerge = sum(power(lSet[:,-1]-tree['left'], 2)) + sum(power(rSet[:,-1]-tree['right'],2))
		merge_mean = (tree['left']+tree['right'])/2.0 # 合并节点
		error_merge = sum(power(testData[:,-1] - merge_mean, 2)) # 合并后的误差
		if error_merge < error_noMerge:
			print("Merging")
			return merge_mean # 返回合并后的预测结果
		else:
			return tree
	else:
		return tree

if __name__ == "__main__":
	data = loadData('ex2.txt')
	data = mat(data)
	tree = createTree(data, ops=(0,1))
	data_test = loadData('ex2test.txt')
	data_test = mat(data_test)
	tree = prune(tree, data_test)
	print("After merge: ", tree)


