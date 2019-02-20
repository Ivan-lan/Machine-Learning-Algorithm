from numpy import *
import operator
from os import listdir

"""
KNNs算法

参考：《机器学习实战》
"""


# 构建数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# K-近邻算法
# 计算给定样本点与训练集各样本点的距离，从小到大排序，取前k个，k个中的最大类别为预测类别

def KnnClassifier(X, data, labels, k):
	"""
	参数：
	X : 待分类向量
	data : 训练样本集
	y ： 标签向量
	k : 近邻参数
	"""
	sample_num = data.shape[0]
	diff_Mat = tile(X, (sample_num, 1)) - data  # 矩阵相减. tile: 在列方向重复1次，行方向重复m次，行复制
	diff_Mat_square = diff_Mat**2 # 求平方
	distance_suqare = diff_Mat_square.sum(axis=1) # 每行的元素进行加总，返回m行1列
	distance = distance_suqare**0.5 # 开方，计算X到每个点的距离
	sortIndice = distance.argsort() # 返回排序后的原索引
	class_dict = {}
	for i in range(k):
		label = labels[sortIndice[i]]  # 一次取距离最近到第k近的点的标签
		class_dict[label] = class_dict.get(label, 0) + 1 # get 返回指定键的值，不存在返回0，再加1。计数
	sortClass = sorted(class_dict.items(), key = operator.itemgetter(1), reverse = True) # 元组排序，按第二个元素排
	return sortClass[0][0] # 返回数量最多的类别

if __name__ == "__main__":
	data, labels = createDataSet()
	label = KnnClassifier([0,0], data, labels, 3)
	print(label)
# ------------------------------------------------------------------------------------------
# 示例：使用k-近邻算法改进约会网站的配对效果
"""
数据集为海伦的潜在交往对象的数据，她曾交往过三种类型的人：不喜欢的人，魅力一般的人，极具魅力的人（即样本的标签）；
样本有三种特征：每年获得的飞行常客里程数，玩视视频游戏所消耗时间占比，每周消费的冰激凌公升数。
"""

# 文本解析
def fiel2matrix(filename):
	file = open(filename)
	line_num = len(file.readlines()) # 文本的行数
	res = zeros((line_num, 3)) # 初始化特征矩阵
	label = [] # 初始化标签向量
	row = 0
	file = open(filename)
	for line in file.readlines():
		line = line.strip() # 删除首尾空格
		line2list = line.split('\t') # 分割字符串为列表
		res[row, :] = line2list[0:3] # 储存三个特征
		label.append(int(line2list[-1])) # 储存标签
		row += 1
	return res, label

# 标准化: X - min/(max - min)
def norm(data):
	val_min = data.min(0)
	val_max = data.max(0)
	ranges = val_max - val_min
	data_norm = zeros(shape(data))
	sample_num = data.shape[0]
	data_norm = data - tile(val_min, (sample_num, 1)) 
	data_norm = data_norm/tile(ranges, (sample_num, 1))
	return data_norm, ranges, val_min

def Classifier_Test():
	hold_ratio = 0.10
	data, label = fiel2matrix('./datingTestSet2.txt')
	data, ranges, val_min = norm(data)
	sample_num = data.shape[0]
	sample_test_num = int(sample_num*hold_ratio)
	error_num = 0.0
	for i in range(sample_test_num):
		result = KnnClassifier(data[i, :],data[sample_test_num:sample_num, :],label[sample_test_num:sample_num],3)
		#  调用KNN算法，k=3,样本无序，前10%作为测试集，后90%作为训练集
		print ("The KnnClassifier predict label: %d, the real label: %d " %(result, label[i]))
		if (result != label[i]): error_num += 1.0 # 错误预测计数
	print ("The total error rate is : %f" % (error_num/float(sample_test_num)))
	print("Error predict number:", error_num)

if __name__ == "__main__":
	Classifier_Test()

	
	








