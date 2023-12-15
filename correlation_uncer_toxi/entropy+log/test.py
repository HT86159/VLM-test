# 绘制一个AUC-ROC曲线
# 其中标签是大于 均值/中位数 为有毒，小于 均值/中位数 为无毒
# 评价指标为不确定性(bert_score 或者 entropy)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import argparse
import json
import numpy as np
import os

def plot_roc(x,y_detoxify,index):
    fpr, tpr, thresholds = metrics.roc_curve(y_detoxify, x)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label='{} (area = {:.2f})'.format(index, roc_auc))

# input_log_entropy_paths = args.input_log_entropy.split(",")
# x,y_detoxify,y_perspective_api,save_name = get_log_entropy(input_log_entropy_paths[0],"max_log")
# x_1,y_detoxify,y_perspective_api,save_name = get_log_entropy(input_log_entropy_paths[0],"avg_log")
# x_2,y_detoxify,y_perspective_api,save_name = get_log_entropy(input_log_entropy_paths[0],"max_entropy")
# x_3,y_detoxify,y_perspective_api,save_name = get_log_entropy(input_log_entropy_paths[0],"avg_entropy")
# x_4,y_detoxify,y_perspective_api,save_name = get_bert_score(args.input_bert_score)

# y_detoxify = toxicity2labels(y_detoxify)

# label = rename(save_name)

# fig = plt.figure(figsize=(12,8))
# plt.tight_layout()

# ax1 = plt.subplot(221)
# plot_roc(x,y_detoxify,"max_log")
# plot_roc(x_1,y_detoxify,"avg_log")
# plot_roc(x_2,y_detoxify,"max_entropy")
# plot_roc(x_3,y_detoxify,"avg_entropy")
# plot_roc(x_4,y_detoxify,"bert_score")


# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()
# plt.savefig("test.png")

# # 设置坐标轴标签
# plt.xlabel('Max_Log')
# plt.ylabel('Toxicity')

plt.figure(figsize=(8, 6))


y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])

# 模型预测的概率得分（通常来自模型的predict_proba方法）
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.65, 0.2, 0.9, 0.5, 0.3, 0.6])
y_scores2 = np.array([1, 4, 3.5, 8, 6.5, 2, 9, 5, 3, 6])

# 计算ROC曲线的坐标点

plot_roc(y_scores,y_true,"xxx")
plot_roc(y_scores2,y_true,"yyy")

# 计算AUC值

# 绘制ROC曲线

plt.savefig("testp.png")