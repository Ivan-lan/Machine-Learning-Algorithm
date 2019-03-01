"""
参考：https://zhuanlan.zhihu.com/p/39780508

"""

"""
SVM主要有LinearSVC、NuSVC和SVC三种方法，我们将具体介绍这三种分类方法都有哪些参数值以及不同参数值的含义。
"""

# LinearSVC   Linear Support Vector Classification.

class sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, 
	                        tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True,
	                         intercept_scaling=1, class_weight=None, verbose=0, 
	                         random_state=None, max_iter=1000)

"""
参数说明：

penalty:正则化参数，L1和L2两种参数可选，仅LinearSVC有。

loss:损失函数，有‘hinge’和‘squared_hinge’两种可选，前者又称L1损失，后者称为L2损失，默认是是’squared_hinge’，
其中hinge是SVM的标准损失，squared_hinge是hinge的平方。

dual:是否转化为对偶问题求解，默认是True。

tol:残差收敛条件，默认是0.0001，与LR中的一致。

C:惩罚系数，用来控制损失函数的惩罚系数，类似于LR中的正则化系数。

multi_class:负责多分类问题中分类策略制定，有‘ovr’和‘crammer_singer’ 两种参数值可选，默认值是’ovr’，
'ovr'的分类原则是将待分类中的某一类当作正类，其他全部归为负类，通过这样求取得到每个类别作为正类时的正确率，
取正确率最高的那个类别为正类；‘crammer_singer’ 是直接针对目标函数设置多个参数值，最后进行优化，得到不同类别的参数值大小。

fit_intercept:是否计算截距，与LR模型中的意思一致。

class_weight:与其他模型中参数含义一样，也是用来处理不平衡样本数据的，可以直接以字典的形式指定不同类别的权重，
也可以使用balanced参数值。

verbose:是否冗余，默认是False。

random_state:随机种子的大小。

max_iter:最大迭代次数，默认是1000。
-------------------------------------------------------------------------------
对象属性：

coef_:各特征的系数（重要性）

intercept_:截距的大小（常数值）
"""

################################################################################

# NuSVC

class sklearn.svm.NuSVC(nu=0.5, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                        shrinking=True, probability=False, tol=0.001, cache_size=200, 
                        class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', 
                        random_state=None)

"""
参数说明：

nu:训练误差部分的上限和支持向量部分的下限，取值在（0，1）之间，默认是0.5

kernel:核函数，核函数是用来将非线性问题转化为线性问题的一种方法，默认是“rbf”核函数，常用的核函数有以下几种：
linear线性核函数poly多项式核函数rbf高斯核函数sigmodsigmod核函数precomputed自定义核函数
关于不同核函数之间的区别，可以参考这篇文章：https://blog.csdn.net/batuwuhanpei/article/details/52354822

degree:当核函数是多项式核函数的时候，用来控制函数的最高次数。（多项式核函数是将低维的输入空间映射到高维的特征空间）

gamma:核函数系数，默认是“auto”，即特征维度的倒数。

coef0:核函数常数值(y=kx+b中的b值)，只有‘poly’和‘sigmoid’核函数有，默认值是0。

max_iter:最大迭代次数，默认值是-1，即没有限制。

probability:是否使用概率估计，默认是False。

decision_function_shape:与'multi_class'参数含义类似。

cache_size:缓冲大小，用来限制计算量大小，默认是200M。

对象
support_:以数组的形式返回支持向量的索引。

support_vectors_:返回支持向量。

n_support_:每个类别支持向量的个数。

dual_coef_:支持向量系数。

coef_:每个特征系数（重要性），只有核函数是LinearSVC的时候可用。

intercept_:截距值（常数值）。
"""

################################################################################

# SVC

class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, 
	                  shrinking=True, probability=False, tol=0.001, cache_size=200, 
	                  class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', 
	                  random_state=None)

"""
C:惩罚系数。
SVC和NuSVC方法基本一致，唯一区别就是损失函数的度量方式不同（NuSVC中的nu参数和SVC中的C参数）。


方法

三种分类方法的方法基本一致，所以就一起来说啦。

decision_function(X):获取数据集X到分离超平面的距离。

fit(X, y):在数据集(X,y)上使用SVM模型。

get_params([deep]):获取模型的参数。

predict(X):预测数据值X的标签。

score(X,y):返回给定测试集和对应标签的平均准确率
"""
