import json

# 用于存储唯一的JSON对象的列表
unique_json_objects = []

input_file_path = "/data/huangtao/projects/llm-safty/MyVLMTest/uncertainty/selfcheck_results/minigpt4/minigpt4-hatefulmeme-nontoxicity-instruction_set_100_dict-with_image-2023103122bert_score.jsonl"
output_file_path = "/data/huangtao/projects/llm-safty/MyVLMTest/uncertainty/selfcheck_results/minigpt4/minigpt4-hatefulmeme-nontoxicity-instruction_set_100_dict-with_image-2023103122bert_score2.jsonl"

# 读取JSONL文件并进行去重
with open(input_file_path, "r") as input_file:
    for line in input_file:
        # 解析JSON行并转化为字典
        json_object = json.loads(line)
        # 如果JSON对象不在列表中，则将其添加
        if json_object not in unique_json_objects:
            unique_json_objects.append(json_object)

# 将唯一的JSON对象写回到新文件
with open(output_file_path, "w") as output_file:
    for unique_object in unique_json_objects:
        output_file.write(json.dumps(unique_object) + "\n")
