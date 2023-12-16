import json
json_file = open("/data/huangtao/projects/llm-safty/MyVLMTest/qa_results/LRV/lrv-red_team1k-no_image.jsonl","r")

# json_file = open("/data/huangtao/projects/llm-safty/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/eps_64result_given.jsonl","r")
test_json = open("/data/huangtao/projects/llm-safty/MyVLMTest/qa_results/LRV/LRV-red_team1k-no_image.jsonl","w")
# import pdb;pdb.set_trace()
for i in json_file:
    for j in i.split("{")[1:]: 
        j =  j.replace("\"instruction\"","\"prompt\"")
        j = j.replace("answer","continuation")
        test_json.writelines("{"+j[:-2]+"\n")


