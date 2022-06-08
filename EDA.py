import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# 导入csv文件
file = 'data.csv'
data = pd.read_csv(file)

# 数据预处理
print(data.isnull().sum())  # 查看是否有缺失值

# 对标签中的变量进行计数统计
y = data['y']
X = data.drop(['y'], axis=1)
print('原始标签数据统计：', Counter(y))

# 相关性分析
correlations = data.corr()  # 求特征间的样本相关阵
cor = correlations.apply(lambda x: round(x, 2))  # 保留两位小数
figure, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(cor, square=True, annot=True, ax=ax, cmap='YlOrRd')
plt.show()
