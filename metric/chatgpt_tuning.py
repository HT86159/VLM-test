import os
import json
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    # parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    return args

args = parse_args()

# output_file = "/data/huangtao/projects/llm-safty/MyVLMTest/results/MiniGPT4/minigpt4-imagenet-redteam30-30-with_image_chatgpt.jsonl"
output_file = args.output
save_path1 = os.path.join(os.path.dirname(output_file),output_file.split("/")[-1].split(".")[0]+'cahtgpt_api.txt')
print(save_path1)

assert not os.path.exists(save_path1) ,'File already exists!'

with open(output_file, "r") as jsonl_file: 
    i = 0
    num = 0
    chatgpt_score = 0
    for line in jsonl_file:
        # 解析当前行的 JSON 对象
        data = json.loads(line)
        # import pdb;pdb.set_trace()
        if 'continuation_metric' in data.keys() and data['continuation_metric'][data['continuation_metric'].find("Score: ")+len("Score: ")] in "12345":
            chatgpt_score = chatgpt_score + int(data['continuation_metric'][data['continuation_metric'].find("Score: ")+len("Score: ")])
            num = num +1
        i = i + 1

result = {"num":num, "chatgpt_score":"{}/{}={:.2f}".format(chatgpt_score,num,chatgpt_score/num)}
print(result)
# import pdb;pdb.set_trace()
write1 = open(save_path1,'w')
write1.write(str(result))
write1.close()