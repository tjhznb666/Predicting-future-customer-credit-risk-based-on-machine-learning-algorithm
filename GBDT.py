import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
import sklearn.ensemble as ensemble  # ensemble learning: 集成学习
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from lift_curve import lift_curve


# 导入csv文件
file = 'data.csv'
data = pd.read_csv(file)


# 选择训练集和测试集
y = data['y']
X = data.drop(['y','FILE_TYPE','IS_INSUUANCE','EDUCATION','MOBILE_TIME_SPAN'], axis=1)
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


# 实现代价敏感学习
sample_weight = []
for i in range(len(y_smo)):
    if y_smo[i] == 0:
        sample_weight.append(0.2)
    else:
        sample_weight.append(0.8)

clf = ensemble.GradientBoostingClassifier()
gbdt_model = clf.fit(X_smo, y_smo, sample_weight=sample_weight)


# 使用GBDT对测试集进行预测
# 得到测试集每个样本为正例的概率（在这里用的是predict_proba，其他模型中也可以用decision_function）
y_score = gbdt_model.predict_proba(X_test)[:, 1]
y_pre = gbdt_model.predict(X_test)
print('GBDT精确度...')
print(metrics.classification_report(y_test, y_pre))
print('GBDT AUC...')
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
