# 示例代码：绘制一个AUC-ROC曲线

# 导入必要的库
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics








# 假设有以下标签和模型预测得分
# 实际标签
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])

# 模型预测的概率得分（通常来自模型的predict_proba方法）
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.65, 0.2, 0.9, 0.5, 0.3, 0.6])

# 计算ROC曲线的坐标点
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)

# 计算AUC值
roc_auc = metrics.auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
plt.savefig("test.png")
