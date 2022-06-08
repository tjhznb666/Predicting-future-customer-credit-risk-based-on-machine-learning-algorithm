from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题


class KS_CURVE:
    def __init__(self):
        pass

    def ComuTF(self, lst1, lst2):
        # 计算TPR和FPR
        # lst1为真实值,lst2为预测值
        TP = sum([1 if a == b == 1 else 0 for a, b in zip(lst1, lst2)])  # 正例被预测为正例
        FN = sum([1 if a == 1 and b == 0 else 0 for a, b in zip(lst1, lst2)])  # 正例被预测为反例
        TPR = TP / (TP + FN)
        TN = sum([1 if a == b == 0 else 0 for a, b in zip(lst1, lst2)])  # 反例被预测为反例
        FP = sum([1 if a == 0 and b == 1 else 0 for a, b in zip(lst1, lst2)])  # 反例被预测为正例
        FPR = FP / (TN + FP)
        return TPR - FPR

    def Getps_ks(self, real_data, data):
        # real_data为真实值，data为原数据
        d = []
        for i in data:
            pre_data = [1 if line >= i else 0 for line in data]
            d.append(self.ComuTF(real_data, pre_data))
        return max(d), data[d.index(max(d))]

    def GetKS(self, y_test, y_pred_prob):
        '''
        功能: 计算KS值，输出对应分割点和累计分布函数曲线图
        输入值:
        y_pred_prob: 一维数组或series，代表模型得分（一般为预测正类的概率）
        y_test: 真实值，一维数组或series，代表真实的标签（{0,1}或{-1,1}）
        '''
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        ks = max(tpr - fpr)
        # 画ROC曲线
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
        # 画ks曲线
        plt.plot(tpr)
        plt.plot(fpr)
        plt.plot(tpr - fpr)
        plt.show()
        return fpr, tpr, thresholds, ks
