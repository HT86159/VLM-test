# The Python file uses ChatGPT to annotate continuations, with labels being -1 or 1, where -1 indicates non-toxic and 1 indicates toxic.
import json
import argparse
import time
import time
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--input", type=str, default="/data/huangtao/projects/llm-safty/MyVLMTest/results/toxicity_results/minigpt4-hate_red1k-with_image-2023121220_perspective_detoxify_gptannotated.jsonl")
    args = parser.parse_args()
    return args
import os
args = parse_args()
input_file = args.input

output_file = os.path.splitext(input_file)[0]+"_modified.jsonl"

print(">>>>>>>>>>>>>>>>>>>Begin!<<<<<<<<<<<<<<<<<<<<<")
print(">>>>>>>>>>>>{}<<<<<<<<<<<".format(os.path.dirname(output_file)))

inputs = open(input_file).read().strip().split('\n')
i = 0
outputs = []
for li in tqdm(inputs,ncols=180,position=0 ,leave=True):
    i = i+1
    obj = json.loads(li)
    if 'continuation' in obj and obj['continuation'] != "":
        if obj['gpt_annotated'] != "1" and obj['gpt_annotated'] != "-1":
            print(obj["continuation"])
            print(obj["gpt_annotated"])
            score = input("Please enter a score: ")
            obj["gpt_annotated"] = str(score)
            outputs.append(obj)
        else:
            outputs.append(obj)

with open(output_file, 'w') as f:
    for obj in outputs:
        f.write(json.dumps(obj))
        f.write('\n')
            # time.sleep(20)
f.close()






# import datetime
# current_time = datetime.datetime.now()
# year = current_time.year
# month = current_time.month
# day = current_time.day
# hour = current_time.hour
# data = str(year)+str(month)+str(day)+str(hour)

























# print("ID:", completion["id"])
# print("Object:", completion["object"])
# print("Created:", completion["created"])
# print("Model:", completion["model"])
# print("Choice Index:", completion["choices"][0]["index"])
# print("Role:", completion["choices"][0]["message"]["role"])
# print("Prompt Tokens:", completion["usage"]["prompt_tokens"])
# print("Completion Tokens:", completion["usage"]["completion_tokens"])
# print("Total Tokens:", completion["usage"]["total_tokens"])

