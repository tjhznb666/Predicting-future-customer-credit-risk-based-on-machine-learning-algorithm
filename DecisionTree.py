import sklearn.tree as tree
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
import matplotlib.pyplot as plt
import math
from lift_curve import lift_curve
import numpy as np
from scipy.stats import scoreatpercentile
# 直接使用交叉网格搜索来优化决策树模型，边训练边优化
from sklearn.model_selection import GridSearchCV


# 导入csv文件
file = 'data.csv'
data = pd.read_csv(file)

# 选择训练集和测试集
y = data['y']
X = data.drop(['y'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

# 不平衡问题处理
# SMOTE法过采样
smote = SMOTE(random_state=0)  # random_state为0（此数字没有特殊含义，可以换成其他数字）使得每次代码运行的结果保持一致
X_smo, y_smo = smote.fit_resample(X_train, y_train)  # 使用原始数据的特征变量和目标变量生成过采样数据集
print('SMOTE过采样后标签数据统计：', Counter(y_smo))

# 随机过采样
ros = RandomOverSampler(random_state=0, sampling_strategy='auto')
X_ros, y_ros = ros.fit_resample(X_train, y_train)
print('随机过采样后标签数据统计：', Counter(y_ros))


#  网格搜索的参数：正常决策树建模中的参数 - 评估指标，树的深度，
#  最小拆分的叶子样本数与树的深度
# 通常来说，十几层的树已经是比较深了
param_grid = {'criterion': ['entropy', 'gini'],
             'max_depth': [2, 3, 4, 5, 6, 7, 8],
             'min_samples_split': [4, 8, 12, 16, 20, 24, 28]}


#  传入模型，网格搜索的参数，评估指标，cv交叉验证的次数.这里也只是定义，还没有开始训练模型
clf = tree.DecisionTreeClassifier()  # 定义一棵树
clf_cv = GridSearchCV(estimator=clf, param_grid=param_grid,
                            scoring='roc_auc', cv=4)


#  训练决策树
clf_cv.fit(X=X_smo, y=y_smo)

print(clf_cv.best_params_, clf_cv.best_score_)  # 输出最优参数及最优模型得分


# 使用决策树对测试集进行预测
# 得到测试集每个样本为正例的概率（在这里用的是predict_proba，其他模型中也可以用decision_function，
# 要看GridSearchCV套用的模型的类中有哪些方法）
y_score = clf_cv.predict_proba(X_test)[:, 1]
y_pre = clf_cv.predict(X_test)
print('决策树精确度...')
print(metrics.classification_report(y_test, y_pre))
print('决策树 AUC...')
fpr, tpr, th = metrics.roc_curve(y_test, y_score)  # 构造 roc 曲线，第三个输出为阈值（每个阈值对应一个DPR和TPR）
ks = max(tpr - fpr)
print('KS=',ks)
print('AUC = %.4f' %metrics.auc(fpr, tpr))  # 求AUC值(ROC曲线下方面积)

# 画ROC曲线
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

# 画lift曲线
lift_curve(y_test, y_score, 10)
