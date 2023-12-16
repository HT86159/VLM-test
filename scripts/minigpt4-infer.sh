python /data/huangtao/projects/llm-safty/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/inference.py \
 --cfg-path /data/huangtao/projects/llm-safty/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/eval_configs/minigpt4_eval.yaml \
 --gpu_id 3 \
 --if_image True \
 --model_name minigpt4 \
 --input_file /data/huangtao/datasets/facebook-hateful-meme/test.jsonl \
 --image_file /data/huangtao/datasets/facebook-hateful-meme/img  \
 --output_path /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/MiniGPT4  


# python /data/huangtao/projects/llm-safty/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/inference.py \
#  --cfg-path /data/huangtao/projects/llm-safty/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/eval_configs/minigpt4_eval.yaml \
#  --gpu_id 4 \
#  --if_image False \
#  --model_name minigpt4 \
#  --input_file /data/huangtao/datasets/facebook-hateful-meme/test.jsonl \
#  --image_file /data/huangtao/datasets/facebook-hateful-meme/img  \
#  --output_path /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/MiniGPT4/ 

# python /data/huangtao/projects/llm-safty/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/inference.py \
#  --cfg-path /data/huangtao/projects/llm-safty/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/eval_configs/minigpt4_eval.yaml \
#  --gpu_id 5 \
#  --if_image False \
#  --model_name minigpt4 \
#  --input_file /data/huangtao/datasets/red_team1k/red_team1k.json \
#  --image_file /data/huangtao/datasets/red_team1k/images  \
#  --output_path /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/MiniGPT4/ &

# python /data/huangtao/projects/llm-safty/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/inference.py \
#  --cfg-path /data/huangtao/projects/llm-safty/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/eval_configs/minigpt4_eval.yaml \
#  --gpu_id 6 \
#  --if_image True \
#  --model_name minigpt4 \
#  --input_file /data/huangtao/datasets/red_team1k/red_team1k.json \
#  --image_file /data/huangtao/datasets/red_team1k/images  \
#  --output_path /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/MiniGPT4/ 