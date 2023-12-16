
import json
import argparse
import json
import os
import ssl
import sys
import numpy as np
# 添加一个目录到模块搜索路径
sys.path.append('/data/huangtao/projects/llm-safty/MyVLMTest/metric')
from perspective_api import PerspectiveApiScorer
from detoxify_api import DetoxifyScorer

ssl._create_default_https_context = ssl._create_unverified_context
# api_key = open(".perspective_api_key").read().strip()
# api_key = "AIzaSyBIOW4DEoZjXE6gnQLsIvbbid2Rx6ZYfkw"
api_key = "AIzaSyASiqjBGYvBJ-AXB7MtwFN1HCZylcOYMB8"
perspective_scorer = PerspectiveApiScorer(api_key)
detoxify_scorer = DetoxifyScorer()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    return args
def read_red_team(file_path):
    datasets = []
    image_idsets = []
    jsol_file = open(file_path,"r")
    # import pdb;pdb.set_trace()
    for i in jsol_file:
        if "\"instruction\"" in i:
            instruction = i[i.find("instruction")+len("instruction")+3:-3]
            datasets.append(instruction)
        if "\"image_id\"" in i:
            image_id = i[i.find("image_id")+len("image_id")+3:i.find("png")+len("png")]
            image_idsets.append(image_id)
    return image_idsets,datasets


args = parse_args()
input_file = args.input#/data/huangtao/datasets/red_team1k/red_team1k.json
output_dir = "/data/huangtao/projects/llm-safty/MyVLMTest/results/dataset"
output_file_name = os.path.basename(input_file).split('.json')[0] + '_perspective_detoxify.jsonl'
output_file = os.path.join(output_dir,output_file_name)
save_path1 = os.path.join(os.path.dirname(output_file),args.input.split("/")[-1].split(".")[0]+'_perspective_api.txt')
assert not os.path.exists(save_path1),"File alreay exists"
save_path2 = os.path.join(os.path.dirname(output_file),args.input.split("/")[-1].split(".")[0]+'_detoxify.txt')
assert not os.path.exists(save_path2),"File alreay exists"

def float_it(score):
    for key in score.keys():
        score[key] = float(score[key])
    return score


# import pdb;pdb.set_trace()
# inputs = open(input_file).read().strip().split('\n')
image_idsets,datasets = read_red_team(input_file)

with open(output_file, 'w') as f:

    for image_id,instruction in zip(image_idsets,datasets):
        out = dict()
        res = {'detoxify': float_it(detoxify_scorer.get_scores(instruction)),
                'perspective_api': float_it(perspective_scorer.get_scores(instruction))}
        out["image_id"] = image_id
        out["instruction"] = instruction
        out['instruction_metric'] = res
        f.write(json.dumps(out))
        f.write('\n')
        # import pdb;pdb.set_trace()
f.close()

detoxify_keys = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
perspective_api_keys = ['toxicity', 'severe_toxicity', 'sexually_explicit', 'threat', 'profanity', 'identity_attack']
with open(output_file, "r") as jsonl_file: 
    i = 0
    num = 0
    detoxify = 0
    perspective_api = 0
    for line in jsonl_file:
        # 解析当前行的 JSON 对象
        data = json.loads(line)

        if 'continuation_metric' not in data.keys():
            continue
        # pdb.set_trace()
        if i !=0:
            detoxify = detoxify + np.array([data['instruction_metric']['detoxify'][j] for j in detoxify_keys])
            perspective_api = perspective_api + np.array([data['instruction_metric']['perspective_api'][k] for k in perspective_api_keys])
            num = num +1
        i = i + 1
# pdb.set_trace()
write1 = open(save_path1,'w')
perspective_api_dict = dict(zip(perspective_api_keys,perspective_api/num*100))
write1.write(str(perspective_api_dict))
for index in perspective_api_dict.keys():
    print("{}:{}".format(index,perspective_api_dict[index]))

detoxify_dict = dict(zip(detoxify_keys,detoxify/num*100))
write2 = open(save_path2,'w')
write2.write(str(detoxify_dict))
for index in detoxify_dict.keys():
    print("{}:{}".format(index,detoxify_dict[index]))

