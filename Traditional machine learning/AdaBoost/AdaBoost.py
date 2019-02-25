"""
AdaBoost算法（二分类）

"""

from numpy import *

def loadData(filename):
	fea_num = len(open(filename).readline().split('\t'))
	data = []
	label = []
	file = open(filename)
	for line in file.readlines():
		lines = []
		cur_line = line.strip().split('\t')
		for i in range(fea_num - 1):
			lines.append(float(cur_line[i]))
		data.append(lines)
		label.append(float(cur_line[-1]))
	return data, label
	
# 数据分类
def stumpClassify(data, dimen, threshVal, threshIneq):
	"""
	通过阈值比较来对数据进行分类，所有在阈值一遍的分到-1类，另一边为+1
	"""
	ret = ones((shape(data)[0], 1))
	if threshIneq == 'lt':
		ret[data[:, dimen] <= threshVal] = -1.0
	else:
		ret[data[:, dimen] > threshVal] = -1.0
	return ret

# 构建单个单层决策树

"""
这是一个基于加权输入值进行决策的分类器。
在一个加权数据集中循环，找到具有最低错误率的单层决策树，注意是分类树，不是回归树

伪代码：
将最小错误率minError设为∞
对数据集中的每一个特征（第一层循环）：
	对每个步长（第二层循环）：
	对每个不等号（第三层循环）:
		建立一颗单层决策树，并利用加权数据对它进行测试
		如果错误率低于minError,则将当前单层决策树设为最佳单层决策树
返回最佳单层决策树
"""
def buildStump(data, labels, weights):
	data = mat(data)
	labels = mat(labels).T
	row, col = shape(data)
	steps = 10.0
	stump_best = {}
	label_best = mat(zeros((row,1)))
	error_min = inf
	# 遍历每个特征，特征值，符号方向，寻找最小误差决策树结构
	for i in range(col):
		Min = data[:, i].min()
		Max = data[:, i].max()
		step_size = (Max - Min)/steps
		for j in range(-1, int(steps)+1):
			for inequal in ['lt', 'gt']:
				threshVal = (Min + float(j)*step_size)
				Val_pred = stumpClassify(data, i, threshVal, inequal)
				error = mat(ones((row, 1))) 
				error[Val_pred == labels] = 0
				error_weight = weights.T*error # 误差和样本权重相关

				if error_weight < error_min:
					error_min = error_weight
					label_best = Val_pred.copy()
					stump_best['dim'] = i
					stump_best['thresh'] = threshVal
					stump_best['ineq'] = inequal
	return stump_best, error_min, label_best

# 基于单层决策树的AdaBoost算法
"""
伪代码：
对每次迭代：
	利用buildStump()函数找到最佳的单层决策树
	将最佳单层决策树加入单层决策树组
	计算单个分类器的权重值alpha
	计算更新的特征权重向量D
	更新累计类别估计值
	如果错误率等于0，则退出循环
"""
def AdaBoost(data, labels, n_tree = 40):
	tree_set = []
	row = shape(data)[0]
	weights = mat(ones((row,1))/row)
	labels_agg = mat(zeros((row,1)))
	for i in range(n_tree):
		stump_best, error, label = buildStump(data, labels, weights)
		# 分类器权重
		alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
		stump_best['alpha'] = alpha
		tree_set.append(stump_best)
		# 更新样本权重
		expon = multiply(-1*alpha*mat(labels).T, label) # 对应元素相乘
		weights = multiply(weights, exp(expon))
		weights = weights/sum(weights)
		# 计算误差
		labels_agg += alpha*label
		error_agg = multiply(sign(labels_agg) != mat(labels).T, ones((row, 1))) # sign函数，大于0取1
		error_rate = error_agg.sum()/row
		print("Total error rate:", error_rate)
		if error_rate == 0.0:
			break
	return tree_set, labels_agg

# AdaBoost分类器
def AdaClassify(data, classifier):
	data = mat(data)
	row = shape(data)[0]
	labels_agg = mat(zeros((row, 1)))
	for i in range(len(classifier)):
		labels = stumpClassify(data, classifier[0][i]['dim'],\
								classifier[0][i]['thresh'],\
								classifier[0][i]['ineq'])
		labels_agg += classifier[0][i]['alpha'] * labels
		print(labels_agg)
	return sign(labels_agg)

if __name__ == "__main__":
	data, label = loadData('horseColicTraining2.txt')
	classifier = AdaBoost(data, label, 10)
	data_test, label_test = loadData('horseColicTest2.txt')
	pred = AdaClassify(data_test, classifier)
	print(pred)


