python /data/huangtao/projects/llm-safty/MyVLMTest/inference/self_generate_minigpt4.py \
 --cfg-path /data/huangtao/projects/llm-safty/MyVLMTest/models/eval_config/minigpt4_eval.yaml \
 --gpu_id 5 \
 --if_image True \
 --model_name minigpt4 \
 --input_file /data/huangtao/datasets/instruction_set/instruction_set_100_dict.jsonl \
 --image_file /data/huangtao/projects/llm-safty/MyVLMTest/datasets/hatefulmeme1k/images \
 --output_path /data/huangtao/projects/llm-safty/MyVLMTest/uncertainty/inferences_based_results/minigpt4 &



# imagenet+instruction
#  --input_file /data/huangtao/datasets/instruction_set/instruction_set_100_dict.jsonl \
#  --image_file /data/huangtao/datasets/red_team1k/images \

# imagenet+redteam
#  --input_file /data/huangtao/projects/llm-safty/MyVLMTest/datasets/red_team1k/red_team1k.jsonl \
#  --image_file /data/huangtao/datasets/red_team1k/images \

# hate+instruction
#  --input_file /data/huangtao/datasets/instruction_set/instruction_set_100_dict.jsonl \
#  --image_file /data/huangtao/projects/llm-safty/MyVLMTest/datasets/hatefulmeme1k/images \

# hate+redteam
#  --input_file /data/huangtao/projects/llm-safty/MyVLMTest/datasets/red_team1k/hate_red1k.jsonl\
#  --image_file /data/huangtao/projects/llm-safty/MyVLMTest/datasets/hatefulmeme1k/images \



