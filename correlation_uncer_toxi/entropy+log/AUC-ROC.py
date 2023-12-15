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
import pdb

# minigpt4-hatefulmeme1k-hate_red1k-with_image-202311230_bert_score_perspective_detoxify
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--data_dir", type=str,default="/data/huangtao/projects/llm-safty/MyVLMTest/results/toxicity_results/minigpt4-hate_red1k-with_image-2023121220_perspective_detoxify_gptannotated.jsonl")
    parser.add_argument("--toxicity_index",type=str,default="perspective_api")
    args = parser.parse_args()
    return args
args = parse_args()

def get_data(folder_path):
    merged_data = []
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
        # 检查文件是否以"_bert_score_perspective_detoxify.jsonl"结尾
            if filename.endswith("_bert_score_perspective_detoxify_gptannotated.jsonl"):
                file_path = os.path.join(folder_path, filename)
                # if "_red1k-" in file_path or "red_team1k-red_team1k" in file_path:
                # 打开文件并读取其内容
                print(file_path)
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_data = [json.loads(line.strip()) for line in file]
                    # 将文件数据添加到总数据集中
                merged_data.extend(file_data)
    elif os.path.isfile(folder_path):
        file_path = folder_path
        # if "_red1k-" in file_path or "red_team1k-red_team1k" in file_path:
        # 打开文件并读取其内容
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            file_data = [json.loads(line.strip()) for line in file]
            merged_data.extend(file_data)
    # import pdb;pdb.set_trace()
    return merged_data
data = get_data(args.data_dir)

# def read_jsonl_file(file_path):
#     toxicity_list = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             # 解析 JSON 字符串并添加到列表中
#             toxicity_data = json.loads(line)
#             toxicity_list.append(toxicity_data)
#         return toxicity_list

# def get_log_entropy(file_path,index):
#     x = []
#     y_detoxify = []
#     y_perspective_api = []
#     # toxicity dir
#     file_list = read_jsonl_file(file_path)
#     save_name = os.path.basename(file_path)
#     save_name = save_name.replace("_perspective_detoxify.jsonl","")

#     from tqdm import tqdm
#     for i in tqdm(range(len(file_list)),desc="Processing"):
#         x.append(file_list[i][index])
#         if type(file_list[i]["continuation_metric"]['detoxify']) == list:
#             # y_detoxify.append(sum(toxicity_list[i]["continuation_metric"]['detoxify'])/(sto_length+1)+sum(toxicity_list[i]["stochastic_response_metric"]["detoxify"])*sto_length/(sto_length+1))
#             # y_perspective_api.append( sum(toxicity_list[i]["continuation_metric"]['perspective_api'])/(sto_length+1)+sum(toxicity_list[i]["stochastic_response_metric"]["perspective_api"])*sto_length/(sto_length+1))
#             y_detoxify.append(sum(file_list[i]["continuation_metric"]['detoxify']))
#             y_perspective_api.append( sum(file_list[i]["continuation_metric"]['perspective_api']))

#         else:
#             # y_detoxify.append(sum(toxicity_list[i]["continuation_metric"]['detoxify'].values())/(sto_length+1)+sum(toxicity_list[i]["stochastic_response_metric"]["detoxify"])*sto_length/(sto_length+1))
#             # y_perspective_api.append( sum(toxicity_list[i]["continuation_metric"]['perspective_api'].values())/5+sum(toxicity_list[i]["stochastic_response_metric"]["perspective_api"])*4/5)
#             y_detoxify.append(sum(file_list[i]["continuation_metric"]['detoxify'].values()))
#             y_perspective_api.append( sum(file_list[i]["continuation_metric"]['perspective_api'].values()))
#     return x,y_detoxify,y_perspective_api,save_name

# def get_bert_score(toxicity_path):
#     x = []
#     y_detoxify = []
#     y_perspective_api = []
#     # toxicity dir

#     toxicity_list = read_jsonl_file(toxicity_path)
#     self_berscore_dir ="/data/huangtao/projects/llm-safty/MyVLMTest/uncertainty/bert_score_results/"
#     save_name = os.path.basename(toxicity_path)
#     save_name = save_name.replace("_perspective_detoxify.jsonl","")
#     self_berscore_file_name = os.path.basename(toxicity_path).replace("_perspective_detoxify","bert_score")
#     # import pdb;pdb.set_trace()
#     self_berscore_path = os.path.join(self_berscore_dir,os.path.basename(os.path.dirname(toxicity_path)),self_berscore_file_name)
#     self_berscore_list = read_jsonl_file(self_berscore_path)
#     tqdm_length = min(len(toxicity_list),len(self_berscore_list))


#     from tqdm import tqdm
#     for i in tqdm(range(tqdm_length),desc="Processing"):
#         # import pdb;pdb.set_trace()
#         assert self_berscore_list[i]["continuation"] == toxicity_list[i]["continuation"], "Not the same examples"
#         x.append(self_berscore_list[i]["bert_score"][-1])
#         # if self_berscore_list[i]["bert_score"][-1]>1:
#         # import pdb;pdb.set_trace()
#         sto_length = len(toxicity_list[i]["stochastic_response"])
#         if type(toxicity_list[i]["continuation_metric"]['detoxify']) == list:
#             # y_detoxify.append(sum(toxicity_list[i]["continuation_metric"]['detoxify'])/(sto_length+1)+sum(toxicity_list[i]["stochastic_response_metric"]["detoxify"])*sto_length/(sto_length+1))
#             # y_perspective_api.append( sum(toxicity_list[i]["continuation_metric"]['perspective_api'])/(sto_length+1)+sum(toxicity_list[i]["stochastic_response_metric"]["perspective_api"])*sto_length/(sto_length+1))
#             y_detoxify.append(sum(toxicity_list[i]["continuation_metric"]['detoxify']))
#             y_perspective_api.append( sum(toxicity_list[i]["continuation_metric"]['perspective_api']))
#         else:
#             # y_detoxify.append(sum(toxicity_list[i]["continuation_metric"]['detoxify'].values())/(sto_length+1)+sum(toxicity_list[i]["stochastic_response_metric"]["detoxify"])*sto_length/(sto_length+1))
#             # y_perspective_api.append( sum(toxicity_list[i]["continuation_metric"]['perspective_api'].values())/5+sum(toxicity_list[i]["stochastic_response_metric"]["perspective_api"])*4/5)
#             y_detoxify.append(sum(toxicity_list[i]["continuation_metric"]['detoxify'].values()))
#             y_perspective_api.append( sum(toxicity_list[i]["continuation_metric"]['perspective_api'].values()))
#     return x,y_detoxify,y_perspective_api,save_name
def key_exist_append(obj, lst, index):
    if index in obj.keys():
        lst.append(obj[index])
    else:
        pass

def get_uncer_tox(data):
    data_dict = dict()
    max_log = []
    max_entropy = []
    avg_log = []
    avg_entropy = []
    length = []
    sum_log= []
    sum_entropy = []
    max_detoxify = []
    avg_detoxify = []
    bert_score = []
    max_conflict = []
    max_ig = []
    max_theta = []
    avg_conflict = []
    avg_ig = []
    avg_theta = []
    sum_conflict = []
    sum_ig = []
    sum_theta = []
    # pdb.set_trace()
    for obj in data:
        # max_log.append(obj["contiuation_metric"][0]["max_log"])
        # max_entropy.append(obj["contiuation_metric"][0]["max_entropy"])
        # avg_log.append(obj["contiuation_metric"][0]["avg_log"])
        # avg_entropy.append(obj["contiuation_metric"][0]["avg_entropy"])
        # leng = len(obj["continuation"].split(" "))
        # length.append(leng)
        # sum_log.append(leng * obj["contiuation_metric"][0]["avg_log"])
        # sum_entropy.append(leng * obj["contiuation_metric"][0]["avg_entropy"])
        # max_detoxify.append(max(obj["continuation_metric"]["detoxify"]))
        # avg_detoxify.append(sum(obj["continuation_metric"]["detoxify"])/len(obj["continuation_metric"]["detoxify"]))
        # bert_score.append(max(obj["bert_score"]))
        leng = len(obj["continuation"].split(" "))
        length.append(leng)
        max_detoxify.append(max(obj["continuation_metric"]["detoxify"]))
        avg_detoxify.append(sum(obj["continuation_metric"]["detoxify"])/len(obj["continuation_metric"]["detoxify"]))
        key_exist_append(obj, max_log,"max_log")
        key_exist_append(obj, max_conflict,"max_conflict")
        key_exist_append(obj, max_ig,"max_ignorance_log")
        key_exist_append(obj, max_theta,"max_theta")
        key_exist_append(obj, max_entropy,"max_entropy")
        key_exist_append(obj, avg_log,"avg_log")
        key_exist_append(obj, avg_conflict,"avg_conflict")
        key_exist_append(obj, avg_ig,"avg_ignorance_log")
        key_exist_append(obj, avg_theta,"avg_theta")
        key_exist_append(obj, avg_entropy,"avg_entropy")
        sum_log.append([leng * avg_log[-1][0]])
        sum_entropy.append([leng * avg_entropy[-1][0]])
        sum_conflict.append([leng * avg_conflict[-1]])
        sum_ig.append([leng * avg_ig[-1]])
        sum_theta.append([leng * avg_theta[-1]])


    data_dict["max_log"] = max_log
    data_dict["max_entropy"] = max_entropy
    data_dict["max_detoxify"] = max_detoxify
    data_dict["avg_log"] = avg_log
    data_dict["avg_entropy"] = avg_entropy
    data_dict["avg_detoxify"] = avg_detoxify
    data_dict["length"] = length
    data_dict["sum_log"] = sum_log
    data_dict["sum_entropy"] = sum_entropy
    data_dict["bert_score"] = bert_score

    data_dict["max_conflict"] = max_conflict
    # data_dict["max_ig"] = max_ig
    data_dict["max_theta"] = max_theta
    data_dict["avg_conflict"] = avg_conflict
    # data_dict["avg_ig"] = avg_ig
    data_dict["avg_theta"] = avg_theta
    data_dict["sum_conflict"] = sum_conflict
    # data_dict["sum_ig"] = sum_ig
    data_dict["sum_theta"] = sum_theta
    # pdb.set_trace()
    return data_dict

def rename(save_name1):
    label1 = "Img" if "nips" in save_name1 else "Hate"
    label1 = label1 + "-Ins" if "instru" in save_name1 else label1 + "-Red"
    label1 = "Img-Red" if "red_team1k" in save_name1 else label1
    return label1

def toxicity2labels(y, method="mean"):
    # import pdb;pdb.set_trace()
    y_mean = sum(y) / len(y) if method=="mean" else np.percentile(y, [method])[-1]
    label = np.array(y) - y_mean
    label = np.sign(label)
    label[label==-1]=0
    print(sum(label))
    return label

def toxicity2labels_acctodataset():
    label = [0]*1000 + [1]*2000 + [0]*1000
    return label

def labels_from_chatgpt(data):
    labels = []
    for obj in data:
        labels.append(int(obj["gpt_annotated"]))
    labels = np.array(labels)
    labels[labels==-1]=0
    print(sum(labels))
    return labels

def plot_roc(x,y_detoxify,index):
    # pdb.set_trace()
    # print(len(y_detoxify),len(x))
    if len(x)==0:
        print("Pass {}".format(index))
        pass
    else:
        # print("Process {}".format(index))
        x = np.array(x).flatten()
        # pdb.set_trace()
        fpr, tpr, thresholds = metrics.roc_curve(y_detoxify, x)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='{} (area = {:.2f})'.format(index, roc_auc))

data_dict = get_uncer_tox(data)

# label = rename(save_name)
print("here")
fig = plt.figure(figsize=(20,12))


ax1 = plt.subplot(221)
y_detoxify = toxicity2labels(data_dict["max_detoxify"])
#['max_log', 'max_entropy', 'max_detoxify', 'avg_log', 'avg_entropy', 'avg_detoxify', 'length', 'sum_log', 'sum_entropy', 'bert_score', 'max_conflict', 'max_theta', 'avg_conflict', 'avg_theta', 'sum_conflict', 'sum_theta']
# for index in [ "sum_theta","sum_conflict","length"]:
for index in data_dict.keys():
    plot_roc(data_dict[index],y_detoxify,index)
print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<")

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("split_by_max_detoxcify-mean")

ax2 = plt.subplot(222)
y_detoxify = toxicity2labels(data_dict["max_detoxify"],50)
for index in data_dict.keys():
    plot_roc(data_dict[index],y_detoxify,index)


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("split_by_max_detoxcify-percentile50")

ax3 = plt.subplot(223)
y_detoxify = toxicity2labels(data_dict["max_detoxify"],90)
for index in data_dict.keys():
    plot_roc(data_dict[index],y_detoxify,index)


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("split_by_avg_detoxcify-percentile90")


ax4 = plt.subplot(224)
y_detoxify = labels_from_chatgpt(data)
print(data_dict.keys())
for index in data_dict.keys():
    plot_roc(data_dict[index],y_detoxify,index)


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("split_by_chatgpt")
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# # plt.legend(loc="lower right")
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.title("split_by_redteam")

# ax1 = plt.subplot(224)
# y_detoxify = labels_from_chatgpt(data)
# plot_roc(data_dict["max_log"],y_detoxify,"max_log")
# plot_roc(data_dict["avg_log"],y_detoxify,"avg_log")
# plot_roc(data_dict["max_entropy"],y_detoxify,"max_entropy")
# plot_roc(data_dict["avg_entropy"],y_detoxify,"avg_entropy")
# plot_roc(data_dict["bert_score"],y_detoxify,"bert_score")
# plot_roc(data_dict["sum_log"],y_detoxify,"sum_log")
# plot_roc(data_dict["sum_entropy"],y_detoxify,"sum_entropy")
# plot_roc(data_dict["length"],y_detoxify,"length")

# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# # plt.legend(loc="lower right")
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.title("split_by_chatgpt")

plt.tight_layout()


plt.show()
plt.savefig("test-1214.png")