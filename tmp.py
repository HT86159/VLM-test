import json
import numpy as np
import os
detoxify_keys = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
perspective_api_keys = ['toxicity', 'severe_toxicity', 'sexually_explicit', 'threat', 'profanity', 'identity_attack']
output_file = "/data/huangtao/projects/llm-safty/MyVLMTest/qa_results/Instruct-Blip/instruct-blip-red-team-1k.jsonl"
out = []
with open(output_file, "r") as jsonl_file:
    i = 0
    num = 0
    detoxify = 0
    perspective_api = 0
    for line in jsonl_file:
        # 解析当前行的 JSON 对象
        data = json.loads(line)
        print(data["continuation"][0])
        out.append({'prompt': data["prompt"], 'continuation': data["continuation"][0]})
jsonl_file.close()

with open(output_file, 'w') as f:
    for li in out:
        f.write(json.dumps(li))
        f.write("\n")


