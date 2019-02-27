
"""
本文参考：https://zhuanlan.zhihu.com/p/39780305

"""

sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
                                       min_samples_leaf=1,min_weight_fraction_leaf=0.0, max_features=None, 
                                       random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                       min_impurity_split=None, class_weight=None, presort=False)


# 参数说明：

"""
criterion:
特征选择的标准，有信息增益和基尼系数两种，使用信息增益的是ID3和C4.5算法（使用信息增益比），使用基尼系数的CART算法，默认是gini系数。

splitter:
特征切分点选择标准，决策树是递归地选择最优切分点，spliter是用来指明在哪个集合上来递归，有“best”和“random”两种参数可以选择，
best表示在所有特征上递归，适用于数据集较小的时候，random表示随机选择一部分特征进行递归，适用于数据集较大的时候。

max_depth:
决策树最大深度，决策树模型先对所有数据集进行切分，再在子数据集上继续循环这个切分过程，max_depth可以理解成用来限制这个循环次数。

min_samples_split:
子数据集再切分需要的最小样本量，默认是2，如果子数据样本量小于2时，则不再进行下一步切分。如果数据量较小，使用默认值就可，
如果数据量较大，为降低计算量，应该把这个值增大，即限制子数据集的切分次数。

min_samples_leaf:
叶节点（子数据集）最小样本数，如果子数据集中的样本数小于这个值，那么该叶节点和其兄弟节点都会被剪枝（去掉），该值默认为1。

min_weight_fraction_leaf:
在叶节点处的所有输入样本权重总和的最小加权分数，如果不输入则表示所有的叶节点的权重是一致的。

max_features:
特征切分时考虑的最大特征数量，默认是对所有特征进行切分，也可以传入int类型的值，表示具体的特征个数；
也可以是浮点数，则表示特征个数的百分比；还可以是sqrt,表示总特征数的平方根；也可以是log2，表示总特征数的log个特征。

random_state:
随机种子的设置，与LR中参数一致。

max_leaf_nodes:
最大叶节点个数，即数据集切分成子数据集的最大个数。

min_impurity_decrease:
切分点不纯度最小减少程度，如果某个结点的不纯度减少小于这个值，那么该切分点就会被移除。

min_impurity_split:
切分点最小不纯度，用来限制数据集的继续切分（决策树的生成），如果某个节点的不纯度（可以理解为分类错误率）小于这个阈值，
那么该点的数据将不再进行切分。

class_weight:
权重设置，主要是用于处理不平衡样本，与LR模型中的参数一致，可以自定义类别权重，也可以直接使用balanced参数值进行不平衡样本处理。

presort:
是否进行预排序，默认是False，所谓预排序就是提前对特征进行排序，我们知道，决策树分割数据集的依据是，优先按照信息增益/基尼系数大的
特征来进行分割的，涉及的大小就需要比较，如果不进行预排序，则会在每次分割的时候需要重新把所有特征进行计算比较一次，如果进行了预排序以后，
则每次分割的时候，只需要拿排名靠前的特征就可以了。
"""

# 对象属性

"""
classes_:分类模型的类别，以字典的形式输出

clf.classes_  # array([0, 1, 2])#表示0，1，2类别

feature_importances_:特征重要性，以列表的形式输出每个特征的重要性max_features_:最大特征数

n_classes_:类别数，与classes_对应，classes_输出具体的类别

n_features_:特征数，当数据量小时，一般max_features和n_features_相等

n_outputs_:输出结果数

tree_:输出整个决策树,用于生成决策树的可视化

clf.tree_    # <sklearn.tree._tree.Tree at 0x241c20e5d30>
"""

# 方法

"""
decision_path(X):返回X的决策路径

fit(X, y):在数据集(X,y)上使用决策树模型

get_params([deep]):获取模型的参数

predict(X):预测数据值X的标签

predict_log_proba(X):返回每个类别的概率值的对数

predict_proba(X):返回每个类别的概率值（有几类就返回几列值）

score(X,y):返回给定测试集和对应标签的平均准确率
"""