import os
import json
import numpy as np

output_file = "/data/huangtao/projects/llm-safty/MyVLMTest/results/dataset/test_perspective_detoxify.jsonl"
save_path1 = os.path.join(os.path.dirname(output_file),  os.path.basename(output_file).split(".")[0]+'_perspective_api.txt')
assert not os.path.exists(save_path1),"File alreay exists"
save_path2 = os.path.join(os.path.dirname(output_file),  os.path.basename(output_file).split(".")[0]+'_detoxify.txt')
assert not os.path.exists(save_path2),"File alreay exists"

detoxify_keys = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
perspective_api_keys = ['toxicity', 'severe_toxicity', 'sexually_explicit', 'threat', 'profanity', 'identity_attack']
with open(output_file, "r") as jsonl_file: 
    i = 0
    num = 0
    detoxify = 0
    perspective_api = 0
    for line in jsonl_file:
        # 解析当前行的 JSON 对象
        # import pdb;pdb.set_trace()
        data = json.loads(line)
        if 'instruction_metric' not in data.keys():
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
