import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os
import seaborn as sns
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--input", type=str,default="/data/huangtao/projects/llm-safty/MyVLMTest/uncertainty/inferences_based_results/minigpt4/minigpt4-red_team1k-instruction_set_100_dict-with_image-2023112223.jsonl,/data/huangtao/projects/llm-safty/MyVLMTest/uncertainty/inferences_based_results/minigpt4/minigpt4-hatefulmeme1k-hate_red1k-with_image-202311230.jsonl")
    parser.add_argument("--toxicity_index",type=str,default="perspective_api")
    parser.add_argument("--uncertainty_index",type=str,default="max_log")
    args = parser.parse_args()
    return args
args = parse_args()


def read_jsonl_file(file_path):
    toxicity_list = []
    with open(file_path, 'r') as file:
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

def main(file_path,index):
    x = []
    y_detoxify = []
    y_perspective_api = []
    # toxicity dir
    file_list = read_jsonl_file(file_path)

    save_name = os.path.basename(file_path)
    save_name = save_name.replace("_perspective_detoxify.jsonl","")

    from tqdm import tqdm
    for i in tqdm(range(len(file_list)),desc="Processing"):
        # import pdb;pdb.set_trace()
        # print(i)
        # import pdb;pdb.set_trace()
        # assert self_berscore_list[i]["continuation"] == toxicity_list[i]["continuation"], "Not the same examples"

        x.append(file_list[i][index])
        # if self_berscore_list[i]["bert_score"][-1]>1:
        # import pdb;pdb.set_trace()
        if type(file_list[i]["continuation_metric"]['detoxify']) == list:
            # y_detoxify.append(sum(toxicity_list[i]["continuation_metric"]['detoxify'])/(sto_length+1)+sum(toxicity_list[i]["stochastic_response_metric"]["detoxify"])*sto_length/(sto_length+1))
            # y_perspective_api.append( sum(toxicity_list[i]["continuation_metric"]['perspective_api'])/(sto_length+1)+sum(toxicity_list[i]["stochastic_response_metric"]["perspective_api"])*sto_length/(sto_length+1))
            y_detoxify.append(sum(file_list[i]["continuation_metric"]['detoxify']))
            y_perspective_api.append( sum(file_list[i]["continuation_metric"]['perspective_api']))

        else:
            # y_detoxify.append(sum(toxicity_list[i]["continuation_metric"]['detoxify'].values())/(sto_length+1)+sum(toxicity_list[i]["stochastic_response_metric"]["detoxify"])*sto_length/(sto_length+1))
            # y_perspective_api.append( sum(toxicity_list[i]["continuation_metric"]['perspective_api'].values())/5+sum(toxicity_list[i]["stochastic_response_metric"]["perspective_api"])*4/5)
            y_detoxify.append(sum(file_list[i]["continuation_metric"]['detoxify'].values()))
            y_perspective_api.append( sum(file_list[i]["continuation_metric"]['perspective_api'].values()))
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

def rename(save_name1):
    label1 = "Img" if "nips" in save_name1 else "Hate"
    label1 = label1 + "-Ins" if "instru" in save_name1 else label1 + "-Red"
    label1 = "Img-Red" if "red_team1k" in save_name1 else label1
    return label1



fig = plt.figure(figsize=(12,8))


# 调整子图之间的间距
plt.tight_layout()


ax1 = plt.subplot(221)
input_paths = args.input.split(",")
x,y_detoxify,y_perspective_api,save_name = main(input_paths[0],"max_log")
x_1,y_detoxify_1,y_perspective_api_1,save_name1 = main(input_paths[1],"max_log")
x_2,y_detoxify_2,y_perspective_api_2,save_name2 = main(input_paths[2],"max_log")
x_3,y_detoxify_3,y_perspective_api_3,save_name3 = main(input_paths[3],"max_log")
label = rename(save_name)
label1 = rename(save_name1)
label2 = rename(save_name2)
label3 = rename(save_name3)
sns.kdeplot(data=x, fill=True,label =label)
sns.kdeplot(data=x_1, fill=True,label=label1)
sns.kdeplot(data=x_2, fill=True,label=label2)
sns.kdeplot(data=x_3, fill=True,label=label3)
plt.legend()
plt.xlim(0,5)
plt.ylim(0,6) 
# 设置坐标轴标签
plt.xlabel('Max_Log')
plt.ylabel('Toxicity')
# plt.title("Max_Log  VS  Toxicity" )
# x,y_detoxify,x_red,y_detoxify_red = split_data(x,x_red,y_detoxify,y_detoxify_red,split=third_quartile)

# import pdb;pdb.set_trace()
# plt.scatter(x, y_detoxify, c="blue",alpha=0.5, cmap='viridis',ax=ax1)  # 绘制散点图
# plt.scatter(x_red, y_detoxify_red, c="orange",alpha=0.5, cmap='viridis',ax=ax1)  # 绘制散点图
# plt.boxplot(y_detoxify) #绘制箱图
ax2 = plt.subplot(222)
x,y_detoxify,y_perspective_api,save_name = main(input_paths[0],"avg_log")
x_1,y_detoxify_1,y_perspective_api_1,save_name1 = main(input_paths[1],"avg_log")
x_2,y_detoxify_2,y_perspective_api_2,save_name2 = main(input_paths[2],"avg_log")
x_3,y_detoxify_3,y_perspective_api_3,save_name3 = main(input_paths[3],"avg_log")
sns.kdeplot(data=x, fill=True,label =label)
sns.kdeplot(data=x_1, fill=True,label=label1)
sns.kdeplot(data=x_2, fill=True,label=label2)
sns.kdeplot(data=x_3, fill=True,label=label3)
plt.legend()
plt.xlim(0,5)
plt.ylim(0,6)
# 设置坐标轴标签
plt.xlabel('Avg_Log')
plt.ylabel('Toxicity')
# plt.title("Avg_Log  VS  Toxicity" )
ax3 = plt.subplot(223)
x,y_detoxify,y_perspective_api,save_name = main(input_paths[0],"max_entropy")
x_1,y_detoxify_1,y_perspective_api_1,save_name1 = main(input_paths[1],"max_entropy")
x_2,y_detoxify_2,y_perspective_api_2,save_name2 = main(input_paths[2],"max_entropy")
x_3,y_detoxify_3,y_perspective_api_3,save_name3 = main(input_paths[3],"max_entropy")
sns.kdeplot(data=x, fill=True,label =label)
sns.kdeplot(data=x_1, fill=True,label=label1)
sns.kdeplot(data=x_2, fill=True,label=label2)
sns.kdeplot(data=x_3, fill=True,label=label3)
plt.legend()
plt.xlim(0,5)
plt.ylim(0,6)
# 设置坐标轴标签
plt.xlabel('Max_Entropy')
plt.ylabel('Toxicity')
# plt.title("Max_Entropy  VS  Toxicity" )



ax4 = plt.subplot(224)
x,y_detoxify,y_perspective_api,save_name = main(input_paths[0],"avg_entropy")
x_1,y_detoxify_1,y_perspective_api_1,save_name1 = main(input_paths[1],"avg_entropy")
x_2,y_detoxify_2,y_perspective_api_2,save_name2 = main(input_paths[2],"avg_entropy")
x_3,y_detoxify_3,y_perspective_api_3,save_name3 = main(input_paths[3],"avg_entropy")
sns.kdeplot(data=x, fill=True,label =label)
sns.kdeplot(data=x_1, fill=True,label=label1)
sns.kdeplot(data=x_2, fill=True,label=label2)
sns.kdeplot(data=x_3, fill=True,label=label3)
plt.legend()
# plt.xlim(0,5)
plt.ylim(0,6)
# 设置坐标轴标签
plt.xlabel('Avg_Entropy')
plt.ylabel('Toxicity')
# plt.title("Avg_Entropy  VS  Toxicity" )

plt.gcf().subplots_adjust(left=0.05,top=0.88,bottom=0.1,wspace=0.15, hspace=0.2)
# ax4 = plt.subplot(122)

# boxplot = plt.boxplot([x,x_2,x_1,x_3], labels=[label, label2, label1,label3], patch_artist=True,whis=2.2)
# colors = [(0.53, 0.81, 0.92, 0.4),(0.3, 0.9, 0, 0.4),(1.0, 0.6, 0.0, 0.4),(0.9,0.2,0.3,0.4)]
# for box, color in zip(boxplot['boxes'], colors):
#     box.set(facecolor=color)
# for flier in boxplot['fliers']:
#     # flier.set(marker='o', markerfacecolor=color, markersize=5)
#     flier.set(marker='o', markersize=6)

# # for i, flier in enumerate(boxplot['fliers']):
# #     import pdb;pdb.set_trace()
# #     if i ==0:
# #         flier.set(marker='o', markerfacecolor=color[0], markersize=8)
# #     else:

# #         flier.set(marker='o', markerfacecolor=color[1], markersize=8)
# plt.xlabel('image-text-pair')
# plt.ylabel('Self-bert-score')
# ax4.yaxis.grid(True)
plt.suptitle("Uncertainty  &  Tocixity")



# 添加颜色条
# plt.colorbar()
print("sava_name:{}".format(label+">><<"+label1))
# plt.savefig(label1+">><<"+label2+".png")
plt.savefig("Uncertainty&Tocixity.png")

