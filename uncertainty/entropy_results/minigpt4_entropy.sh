
python /data/huangtao/projects/llm-safty/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/inference_with_entropy.py \
 --cfg-path /data/huangtao/projects/llm-safty/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/eval_configs/minigpt4_eval.yaml \
 --gpu_id 3 \
 --if_image True \
 --model_name minigpt4 \
 --input_file /data/huangtao/datasets/instruction_set/instruction_set_100_dict.jsonl \
 --image_file /data/huangtao/datasets/hatefulmeme1k/images \
 --output_path /data/huangtao/projects/llm-safty/MyVLMTest/uncertainty/entropy_results/minigpt4

#   --input_file /data/huangtao/datasets/hatefulmeme-toxicity/hatefulmeme_toxicity1k_redteam.jsonl \
#  --image_file /data/huangtao/datasets/hatefulmeme-toxicity \
#  --input_file /data/huangtao/datasets/red_team1k/red_team1k.jsonl \
#  --image_file /data/huangtao/datasets/red_team1k/images \
#  --input_file /data/huangtao/datasets/instruction_set/instruction_set_100_dict.jsonl \
#  --image_file /data/huangtao/datasets/red_team1k/images \