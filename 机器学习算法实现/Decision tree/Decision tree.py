from math import log
import operator

"""
决策树算法（ID3 算法）

参考：《机器学习实战》
"""

# 构造数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    fea_name = ['no surfacing','flippers']
    return dataSet, fea_name

# 计算给定数据集的熵
def calcEntropy(data):
	"""
	熵用来度量数据集的混乱度，熵越大，数据集的混乱度越大。
	先计算各类别的占比，作为概率，即可计算熵
	"""
	sample_num = len(data) # 样本数
	label_dict = {} # key类别：value样本数
	for Vec in data:
		label = Vec[-1] # 获取标签
		if label not in label_dict.keys():
			label_dict[label] = 0
		label_dict[label] += 1
	entropy = 0.0
	for key in label_dict:
		prob = float(label_dict[key])/sample_num # 类频率
		entropy -= prob * log(prob, 2)  # 累加计算熵
	return entropy

# 按给定特征的值抽取数据集
def splitData(data, fea_indx, fea_value):
	ret = []
	for Vec in data:
		if Vec[fea_indx] == fea_value: # 抽取满足给定特征值的样本
			Vec_new = Vec[: fea_indx]
			Vec_new.extend(Vec[fea_indx+1:]) # 删除该特征列
			ret.append(Vec_new)
	return ret

# 选择最好的划分数据集的特征(返回索引)
def chooseBestFea(data):
	"""
	遍历特征和特征的取值，计算信息增益，即划分前的熵-划分后的熵。
	两层循环，第一层循环遍历所有特征，第二层循环遍历特征的取值，计算信息增益，保存信息增益最大的特征的索引
	"""
	fea_num = len(data[0]) - 1
	entropy_pre = calcEntropy(data) # 划分前数据集的熵
	info_gain_max = 0.0
	best_fea = -1
	for i in range(fea_num):
		fea_list = [sample[i] for sample in data] # 特征i的取值构成的列表
		fea_val_unique = set(fea_list) # 特征取值的集合
		entropy_after = 0.0
		for value in fea_val_unique:
			subdata = splitData(data, i, value)
			prob = len(subdata)/float(len(data))  # 子集占比
			entropy_after  += prob * calcEntropy(subdata) # 累加计算划分后的全部子集的熵
		info_gain = entropy_pre - entropy_after # 计算信息增益
		if (info_gain > info_gain_max): # 寻找最大的信息增益
			info_gain_max = info_gain
			best_fea = i # 信息增益最大的特征的索引
		return best_fea 

# 投票器：叶节点样本数量最多的类别胜出
def voteLabel(label_list):
	"""
	划分数据集会消耗特征，即之前选择过的特征不再使用。
	当只剩一个特征时， 不能再划分，成为叶子节点，这时样本中若
	存在不同类别，采取少数服从多数的方法决定节点的类别。
	"""
	label_dict = {}
	for label in label_list:
		if label not in label_dict.keys():
			label_dict[label] = 0
		label_dict[label] += 1
	sort_label = sorted(label_dict.iteritems(), key=operator.itemgetter(1), reverse=True) # 根据样本数量排序元组
	return sort_label[0][0] 

# 构建树
def buildTree(data, fea_name):
	"""
	需先确定递归停止的条件：样本都属于同一类别或没有特征可以划分了。
	根据信息增益最大的原则，选择最佳特征划分数据集，
	再在子集上递归调用自身，划分子集。用字典储存树结构。
	"""
	label_list = [sample[-1] for sample in data] # 类标签列表
	if label_list.count(label_list[0]) == len(label_list): # 第一个标签的数量=样本数
		return label_list[0] # 全部样本的标签相同时，直接返回标签
	if len(data[0]) == 1: # 特征数等于1，不能再继续划分，投票决定类别
		return voteLabel(label_list)
	best_fea_idx = chooseBestFea(data) 
	best_fea = fea_name[best_fea_idx] # 根据信息增益最大原则选择特征
	Tree = {best_fea: {}} # 字典储存树结构
	subFea_name = fea_name.copy() # 下面会删除特征，在副本删除
	del(subFea_name [best_fea_idx]) # 特征选出来后删除
	fea_value = [sample[best_fea_idx] for sample in data] # 最佳特征的取值组成的列表
	fea_val_unique = set(fea_value)
	for value in fea_val_unique:
		# subFea_name = fea_name.copy()  # 复制特征名字列表，列表参数是引用的，防止调用函数改变原始列表
		# 递归构建树结构：不断地根据最佳特征划分数据集
		Tree[best_fea][value] = buildTree(splitData(data, best_fea_idx,value), subFea_name) # 先抽取子数据集
	return Tree

# 决策树分类函数
def TreeClassifier(Tree, fea_name, testdata):
	"""
	参数：
	Tree : 树结构，例如:{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}} 标签是yes no
	fea_name : 特征名称的列表  ['no surfacing', 'flippers']
	testdata : 测试数据

	"""
	node_1 = list(Tree.keys())[0] # 决策树的第一个节点特征
	sub_tree = Tree[node_1] # 子树
	fea_idx = fea_name.index(node_1) # 特征在特征名字列表的索引

	fea = testdata[fea_idx] # 在测试数据集中的特征
	fea_value = sub_tree[fea_idx] # 特征在树结构中对应的类别

	if isinstance(fea_value, dict): # 当类型是字典时，递归调用
		label = TreeClassifier(fea_value, fea_name, testdata)
	else:
		label = fea_value 
	return label

# 决策树的储存与加载
def storeTree(Tree, filename):
	import pickle
	file = open(filename, 'w')
	pickle.dump(Tree, file) # 序列化
	file.close()

def loadTree(filename):
	import pickle
	file = open(filename)
	return pickle.load(file) # 反序列化

if __name__ == '__main__':
	data, fea_name = createDataSet() # 数据集特征和标签
	Tree = buildTree(data, fea_name)
	testdata = [1, 0, 'no']
	label = TreeClassifier(Tree, fea_name, testdata)
	print ("The predicted label is:", label)