"""
scikit-learn GBDT 调参入门

参考：
https://www.cnblogs.com/pinard/p/6143927.html
https://blog.csdn.net/han_xiaoyang/article/details/52663170
https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

"""

"""
在sacikit-learn中，GradientBoostingClassifier为GBDT的分类类， 而GradientBoostingRegressor为GBDT的回归类。
两者的参数类型完全相同，当然有些参数比如损失函数loss的可选择项并不相同。
这些参数中，类似于Adaboost，我们把重要参数分为两类，第一类是Boosting框架的重要参数，第二类是弱学习器即CART回归树的重要参数。
"""

"""
# boosting框架参数

首先，我们来看boosting框架相关的重要参数。由于GradientBoostingClassifier和GradientBoostingRegressor的参数绝大部分相同，
我们下面会一起来讲，不同点会单独指出。

1) n_estimators: 也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说n_estimators太小，容易欠拟合，
   n_estimators太大，又容易过拟合，一般选择一个适中的数值。默认是100。在实际调参的过程中，我们常常将n_estimators
   和下面介绍的参数learning_rate一起考虑。

2) learning_rate: 即每个弱学习器的权重缩减系数$ν$，也称作步长，在原理篇的正则化章节我们也讲到了，加上了正则化项，
   我们的强学习器的迭代公式为$f_{k}(x) = f_{k-1}(x) + \nu h_k(x)$, ν的取值范围为 $0 < \nu \leq 1$, 
   对于同样的训练集拟合效果，较小的$ν$意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定
   算法的拟合效果。所以这两个参数n_estimators和learning_rate要一起调参。一般来说，可以从一个小一点的ν开始调参，默认是1。

3) subsample: 即我们在原理篇的正则化章节讲到的子采样，取值为(0,1]。注意这里的子采样和随机森林不一样，随机森林使用的是
放回抽样，而这里是不放回抽样。如果取值为1，则全部样本都使用，等于没有使用子采样。如果取值小于1，则只有一部分样本会去做
GBDT的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。推荐在[0.5, 0.8]
之间，默认是1.0，即不使用子采样。

4) init: 即我们的初始化的时候的弱学习器，拟合对应原理篇里面的$f_{0}(x)$，如果不输入，则用训练集样本来做样本集的初始化分
   类回归预测。否则用init参数提供的学习器做初始化分类回归预测。一般用在我们对数据有先验知识，或者之前做过一些拟合的时候，
   如果没有的话就不用管这个参数了。

5) loss: 即我们GBDT算法中的损失函数。分类模型和回归模型的损失函数是不一样的。

- 对于分类模型，有对数似然损失函数"deviance"和指数损失函数"exponential"两者输入选择。默认是对数似然损失函数"deviance"。
  在原理篇中对这些分类损失函数有详细的介绍。一般来说，推荐使用默认的"deviance"。它对二元分离和多元分类各自都有比较好的优化。
  而指数损失函数等于把我们带到了Adaboost算法。

- 对于回归模型，有均方差"ls", 绝对损失"lad", Huber损失"huber"和分位数损失“quantile”。默认是均方差"ls"。
 一般来说，如果数据的噪音点不多，用默认的均方差"ls"比较好。如果是噪音点较多，则推荐用抗噪音的损失函数"huber"。
  而如果我们需要对训练集进行分段预测的时候，则采用“quantile”。

6) alpha：这个参数只有GradientBoostingRegressor有，当我们使用Huber损失"huber"和分位数损失“quantile”时，
   需要指定分位数的值。默认是0.9，如果噪音点较多，可以适当降低这个分位数的值。

"""

"""
# 弱学习器参数

这里我们再对GBDT的类库弱学习器的重要参数做一个总结。由于GBDT使用了CART回归决策树，因此它的参数基本来源于决策树类，也就是说，
和DecisionTreeClassifier和DecisionTreeRegressor的参数基本类似。如果你已经很熟悉决策树算法的调参，那么这一节基本可以跳过。
不熟悉的朋友可以继续看下去。

1) 划分时考虑的最大特征数max_features: 可以使用很多种类型的值，默认是"None",意味着划分时考虑所有的特征数；
如果是"log2"意味着划分时最多考虑$log_2N$, 个特征；如果是"sqrt"或者"auto"意味着划分时最多考虑$\sqrt{N}$个特征。
如果是整数，代表考虑的特征绝对数。如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数。
一般来说，如果样本特征数不多，比如小于50，我们用默认的"None"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值
来控制划分时考虑的最大特征数，以控制决策树的生成时间。

2) 决策树最大深度max_depth: 默认可以不输入，如果不输入的话，默认值是3。一般来说，数据少或者特征少的时候可以不管这个值。
   如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。

3) 内部节点再划分所需最小样本数min_samples_split: 这个值限制了子树继续划分的条件，如果某节点的样本数少于min_samples_split，
  则不会继续再尝试选择最优特征来进行划分。 默认是2.如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

4) 叶子节点最少样本数min_samples_leaf: 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起
  被剪枝。 默认是1,可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。
  如果样本量数量级非常大，则推荐增大这个值。

5）叶子节点最小的样本权重和min_weight_fraction_leaf：这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，
则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别
偏差很大，就会引入样本权重，这时我们就要注意这个值了。

6) 最大叶子节点数max_leaf_nodes: 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。
   如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，
   可以加以限制，具体的值可以通过交叉验证得到。

7) 节点划分最小不纯度min_impurity_split:  这个值限制了决策树的增长，如果某节点的不纯度(基于基尼系数，均方差)小于这个阈值，
  则该节点不再生成子节点。即为叶子节点 。一般不推荐改动默认值1e-7。
"""

"""
# 调参实例

这里我们用一个二元分类的例子来讲解下GBDT的调参
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
%matplotlib inline

train = pd.read_csv('train_modified.csv')
target = 'Disbursed'
IDcol = 'ID'
train['Disbursed'].value_counts()

x_columns = [x for x in train.columns if x not in [target, IDcol]]
x = train[x_columns]
y = train[target]

gbm0 = GradientBoostingClassifier(random_state=10, verbose=1)
gbm0.fit(x,y)

y_pred = gbm0.predict(x)
y_predprob = gbm0.predict_proba(x)[:, 1]
print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

"""
首先我们从步长(learning rate)和迭代次数(n_estimators)入手。
一般来说,开始选择一个较小的步长来网格搜索最好的迭代次数。
这里，我们将步长初始值设置为0.1。对于迭代次数进行网格搜索如下：
"""
param1 = {'n_estimators': list(range(20,81,10))}
search1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300, min_samples_leaf=20, 
	                                                         max_depth=8, max_features='sqrt', subsample=0.8, random_state=10,
	                                                         verbose=1), param_grid=param1,scoring='roc_auc',iid=False, cv=5, 
                                                             verbose=1)
search1.fit(x,y)
search1.grid_scores_, search1.best_params_, search1.best_score_

"""
找到了一个合适的迭代次数，现在我们开始对决策树进行调参。
首先我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
"""
params2 = {'max_depth':list(range(3,14,2)), 'min_samples_split':list(range(100,801,200))}
search2 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf= 20,
                                                           max_features='sqrt', subsample=0.8,random_state=10,verbose=1),
                                                          param_grid=params2, scoring='roc_auc', iid=False, cv=5,verbose=1)
search2.fit(x,y)
search2.grid_scores_, search2.best_params_, search2.best_score_

"""
由于决策树深度7是一个比较合理的值，我们把它定下来，对于内部节点再划分所需最小样本数min_samples_split，
我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。
下面我们再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参。
"""

params3 = {'min_samples_split':list(range(800,1900,200)), 'min_samples_leaf':list(range(60,101,10))}
search3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=7, 
	                                                          max_features='sqrt',subsample=0.8, random_state=10,verbose=1),
                                                           param_grid=params3, scoring='roc_auc',iid=False, cv=5, verbose=1)
search3.fit(x,y)
search3.grid_scores_, search3.best_params_, search3.best_score_ 

"""
输出结果如下，可见这个min_samples_split在边界值，还有进一步调试小于边界60的必要。
由于这里只是例子，所以大家可以自己下来用包含小于60的网格搜索来寻找合适的值。

我们调了这么多参数了，终于可以都放到GBDT类里面去看看效果了。现在我们用新参数拟合数据：
"""

gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=7, 
	                               min_samples_leaf=60,min_samples_split=1200, 
                                   max_features='sqrt', subsample=0.8, random_state=10)
gbm1.fit(x,y)

y_pred = gbm1.predict(x)
y_predprob = gbm1.predict_proba(x)[:,1]
print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
print("AUC Score (Train) : %f" % metrics.roc_auc_score(y, y_predprob))

"""
 对比我们最开始完全不调参的拟合效果，可见精确度稍有下降，主要原理是我们使用了0.8的子采样，20%的数据没有参与拟合。

 现在我们再对最大特征数max_features进行网格搜索。
"""

params4 = {'max_features': list(range(7,20,2))}
search4 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=7, min_samples_leaf=60,
                                                         min_samples_split=1200, subsample=0.8, random_state=10, verbose=1),
                                                        param_grid=params4, scoring='roc_auc', iid=False, cv=5,verbose=1)
search4.fit(x,y)
search4.grid_scores_, search4.best_score_, search4.best_params_

"""
现在我们再对子采样的比例进行网格搜索：
"""
params5 = {'subsample': [0.6, 0.7,0.75, 0.8,0.85,0.9]}
search5 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=7, min_samples_leaf=60,
                                                           min_samples_split=1200, max_features=9, random_state=10,verbose=1),
                                                           param_grid=params5, scoring='roc_auc', iid=False, cv=5, verbose=1)
search5.fit(x,y)
search5.grid_scores_, search5.best_score_, search5.best_params_

"""
现在我们基本已经得到我们所有调优的参数结果了。这时我们可以减半步长，
最大迭代次数加倍来增加我们模型的泛化能力。再次拟合我们的模型：
"""
gbm2 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120,max_depth=7, min_samples_leaf=60,
                                  min_samples_split=1200, max_features=9,subsample=0.7, random_state=10)
gbm2.fit(x,y)

y_pred = gbm2.predict(x)
y_predprob = gbm2.predict_proba(x)[:, 1]
print("Accuracy: %.4g" % metrics.accuracy_score(y.values, y_pred))
print("AUC score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

"""
可以看到AUC分数比起之前的版本稍有下降，这个原因是我们为了增加模型泛化能力，为防止过拟合而减半步长，最大迭代次数加倍，
同时减小了子采样的比例，从而减少了训练集的拟合程度。

下面我们继续将步长缩小5倍，最大迭代次数增加5倍，继续拟合我们的模型：
"""

gbm3 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600,max_depth=7, min_samples_leaf =60, 
               min_samples_split =1200, max_features=9, subsample=0.7, random_state=10)
gbm3.fit(x,y)
y_pred = gbm3.predict(x)
y_predprob = gbm3.predict_proba(x)[:,1]
print ("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

"""
可见减小步长增加迭代次数可以在保证泛化能力的基础上增加一些拟合程度。

最后我们继续步长缩小一半，最大迭代次数增加2倍，拟合我们的模型：
"""

gbm4 = GradientBoostingClassifier(learning_rate=0.005, n_estimators=1200,max_depth=7, min_samples_leaf =60, 
               min_samples_split =1200, max_features=9, subsample=0.7, random_state=10)
gbm4.fit(x,y)
y_pred = gbm4.predict(x)
y_predprob = gbm4.predict_proba(x)[:,1]

print ("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

"""
此时由于步长实在太小，导致拟合效果反而变差，也就是说，步长不能设置的过小。
"""