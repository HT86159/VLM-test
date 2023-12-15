import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os
import seaborn as sns
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--toxicity_index",type=str,default="perspective_api")
    parser.add_argument("--normalization_method",type=str,default="no_normalization")
    args = parser.parse_args()
    return args
args = parse_args()

def read_jsonl_file(toxicity_path):
    toxicity_list = []
    with open(toxicity_path, 'r') as file:
        for line in file:
            # 解析 JSON 字符串并添加到列表中
            toxicity_data = json.loads(line)
            toxicity_list.append(toxicity_data)
        return toxicity_list
def minmax_normalization(data):
    min_val = min(data)
    max_val = max(data)

    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data
def Standardization_normalization(data):
    mean = np.mean(data)
    std = np.std(data)

    normalized_data = [(x - mean) / std for x in data]
    return normalized_data
def split_data(x,x_red,y_detoxify,y_detoxify_red,split):
    x_former = []
    y_former = []
    x_later = []
    y_later = []
    x = x+x_red
    y = y_detoxify+y_detoxify_red
    # 使用循环遍历 x 和 y，筛选出大于中位数的 x 值，并将对应的 y 值存入列表
    import pdb;pdb.set_trace()
    for i in range(len(x)):
        if y[i] > split:
            x_former.append(x[i])
            y_former.append(y[i])
        else:
            x_later.append(x[i])
            y_later.append(y[i])
    return x_former,y_former,x_later,y_later

def plot_box(data,beam,color):
    bins = [np.percentile(data, (i+1)/beam * 100) for i in range(beam-1)]
    di=np.digitize(data,bins)
    out = [[] for _ in range(beam)]
    for i in range(len(di)):
        out[di[i]].append(data[i])
    
    boxplot = plt.boxplot(out, labels=['25%', '50%', "75%","100%"], patch_artist=True)
    colors = [color]*4
    for box, color in zip(boxplot['boxes'], colors):
        box.set(facecolor=color)
    for flier in boxplot['fliers']:
        flier.set(marker='o', markerfacecolor=color, markersize=8)

def main(toxicity_path):
    x = []
    y_detoxify = []
    y_perspective_api = []
    # toxicity dir
    toxicity_list = read_jsonl_file(toxicity_path)
    self_berscore_dir ="/data/huangtao/projects/llm-safty/MyVLMTest/uncertainty/bert_score_results/"
    save_name = os.path.basename(toxicity_path)
    save_name = save_name.replace("_perspective_detoxify.jsonl","")
    self_berscore_file_name = os.path.basename(toxicity_path).replace("_perspective_detoxify","bert_score")
    import pdb;pdb.set_trace()
    self_berscore_path = os.path.join(self_berscore_dir,os.path.basename(os.path.dirname(toxicity_path)),self_berscore_file_name)
    self_berscore_list = read_jsonl_file(self_berscore_path)
    tqdm_length = min(len(toxicity_list),len(self_berscore_list))


    from tqdm import tqdm
    for i in tqdm(range(tqdm_length),desc="Processing"):
        # import pdb;pdb.set_trace()
        print(i)
        # import pdb;pdb.set_trace()
        assert self_berscore_list[i]["continuation"] == toxicity_list[i]["continuation"], "Not the same examples"
        x.append(self_berscore_list[i]["bert_score"][-1])
        # if self_berscore_list[i]["bert_score"][-1]>1:
        # import pdb;pdb.set_trace()
        sto_length = len(toxicity_list[i]["stochastic_response"])
        if type(toxicity_list[i]["continuation_metric"]['detoxify']) == list:
            # y_detoxify.append(sum(toxicity_list[i]["continuation_metric"]['detoxify'])/(sto_length+1)+sum(toxicity_list[i]["stochastic_response_metric"]["detoxify"])*sto_length/(sto_length+1))
            # y_perspective_api.append( sum(toxicity_list[i]["continuation_metric"]['perspective_api'])/(sto_length+1)+sum(toxicity_list[i]["stochastic_response_metric"]["perspective_api"])*sto_length/(sto_length+1))
            y_detoxify.append(sum(toxicity_list[i]["continuation_metric"]['detoxify']))
            y_perspective_api.append( sum(toxicity_list[i]["continuation_metric"]['perspective_api']))
        else:
            # y_detoxify.append(sum(toxicity_list[i]["continuation_metric"]['detoxify'].values())/(sto_length+1)+sum(toxicity_list[i]["stochastic_response_metric"]["detoxify"])*sto_length/(sto_length+1))
            # y_perspective_api.append( sum(toxicity_list[i]["continuation_metric"]['perspective_api'].values())/5+sum(toxicity_list[i]["stochastic_response_metric"]["perspective_api"])*4/5)
            y_detoxify.append(sum(toxicity_list[i]["continuation_metric"]['detoxify'].values()))
            y_perspective_api.append( sum(toxicity_list[i]["continuation_metric"]['perspective_api'].values()))
    return x,y_detoxify,y_perspective_api,save_name
# normalization
# if args.normalization_method == "standardization_normalization":
#     x = Standardization_normalization(x)
#     y_detoxify = Standardization_normalization(y_detoxify)
#     y_perspective_api = Standardization_normalization(y_perspective_api)
# elif args.normalization_method == "minmax_normalization":
#     x = minmax_normalization(x)
#     y_detoxify = minmax_normalization(y_detoxify)
#     y_perspective_api = minmax_normalization(y_perspective_api)

# import pdb;pdb.set_trace()


fig = plt.figure(figsize=(12,8))
ax1 = plt.subplot(221)


# 调整子图之间的间距
plt.tight_layout()


input_paths = args.input.split(",")
x,y_detoxify,y_perspective_api,save_name1 = main(input_paths[0])
x_red,y_detoxify_red,y_perspective_api_red,save_name2 = main(input_paths[1])
label1 = "Imagenet" if "nips" in save_name1 else "Hatememe"
label1 = label1 + "-Instrction" if "instru" in save_name1 else label1 + "-Redteam"
label2 = "Imagenet" if "nips" in save_name2 else "Hatememe" 
label2 = label2 + "-Instrction" if "instru" in save_name2 else label2 + "-Redteam"
# label1= "imagenet-redteam"

label1 = "Imagenet-Redteam" if "red_team1k" in save_name1 else label1
label2 = "Imagenet-Redteam" if "red_team1k" in save_name2 else label2
third_quartile = np.percentile(y_detoxify+y_detoxify_red, 50)
# x,y_detoxify,x_red,y_detoxify_red = split_data(x,x_red,y_detoxify,y_detoxify_red,split=third_quartile)

# import pdb;pdb.set_trace()
# plt.scatter(x, y_detoxify, c="blue",alpha=0.5, cmap='viridis',ax=ax1)  # 绘制散点图
# plt.scatter(x_red, y_detoxify_red, c="orange",alpha=0.5, cmap='viridis',ax=ax1)  # 绘制散点图
import pdb;pdb.set_trace()
# plt.boxplot(y_detoxify) #绘制箱图




plt.scatter(x, y_detoxify, c="blue",alpha=0.5,s=10, cmap='viridis',label =label1)  # 绘制散点图
plt.scatter(x_red, y_detoxify_red, c="orange",alpha=0.5,s= 10, cmap='viridis',label =label2)  # 绘制散点图
plt.xlabel('Self-bert-score')
plt.ylabel('Toxicity')
plt.legend()
plt.title("SelfBert-Score VS Toxicity")

ax2 = plt.subplot(222)
sns.kdeplot(data=x, fill=True,label =label1)
sns.kdeplot(data=x_red, fill=True,label=label2)


plt.legend()
# 设置坐标轴标签
plt.xlabel('Self-bert-score')
plt.ylabel('Frequency')
# 设置标题
# plt.xlim(0, 1.4)
# plt.ylim(0, 3.5)
import pdb;pdb.set_trace()
plt.title("SelfBert-Score VS Frequency")
plt.gcf().subplots_adjust(left=0.05,top=0.88,bottom=0.12,wspace=0.15, hspace=0.15) 
ax3 = plt.subplot(223)
plot_box(x,4,(0.53, 0.81, 0.92, 0.4))
plot_box(x_red,4,(1.0, 0.6, 0.0, 0.4))
plt.xlabel('Bins')
plt.ylabel('Self-bert-score')
ax3.yaxis.grid(True) 

ax4 = plt.subplot(224)

boxplot = plt.boxplot([x,x_red], labels=[label1, label2], patch_artist=True)
colors = [(0.53, 0.81, 0.92, 0.4),(1.0, 0.6, 0.0, 0.4)]
for box, color in zip(boxplot['boxes'], colors):
    box.set(facecolor=color)
# for i, flier in enumerate(boxplot['fliers']):
#     import pdb;pdb.set_trace()
#     if i ==0:
#         flier.set(marker='o', markerfacecolor=color[0], markersize=8)
#     else:
        
#         flier.set(marker='o', markerfacecolor=color[1], markersize=8)
plt.xlabel('image-text-pair')
plt.ylabel('Self-bert-score')
ax4.yaxis.grid(True) 
plt.suptitle(label1+"  VS  "+label2)
# 添加颜色条
# plt.colorbar()
print("sava_name:{}".format(label1+">><<"+label2))
# plt.savefig(label1+">><<"+label2+".png")
plt.savefig("test.png")