# python /data/huangtao/projects/llm-safty/LRV-Instruction/MiniGPT-4/inference.py \
#  --cfg-path /data/huangtao/projects/llm-safty/LRV-Instruction/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
#  --gpu-id 2 \
#  --if_image Ture \
#  --model_name lrv \
#  --input_file /data/huangtao/datasets/imagenet-redteam_30-30/imagenet-redteam30-30.json \
#  --image_file /data/huangtao/datasets/imagenet-redteam_30-30/images  \
#  --output_dir /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/LRV &
 
#  python /data/huangtao/projects/llm-safty/LRV-Instruction/MiniGPT-4/inference.py \
#  --cfg-path /data/huangtao/projects/llm-safty/LRV-Instruction/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
#  --gpu-id 2 \
#  --if_image True \
#  --model_name lrv \
#  --input_file /data/huangtao/datasets/imagenet-redteam_30-30/imagenet-redteam30-30.json \
#  --image_file /data/huangtao/datasets/imagenet-redteam_30-30/images  \
#  --output_dir /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/LRV


  python /data/huangtao/projects/llm-safty/LRV-Instruction/MiniGPT-4/inference.py \
 --cfg-path /data/huangtao/projects/llm-safty/LRV-Instruction/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
 --gpu-id 6 \
 --if_image True \
 --model_name lrv \
 --input_file /data/huangtao/datasets/facebook-hateful-meme/test.jsonl \
 --image_file /data/huangtao/datasets/facebook-hateful-meme/img  \
 --output_dir /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/LRV 

 
#   python /data/huangtao/projects/llm-safty/LRV-Instruction/MiniGPT-4/inference.py \
#  --cfg-path /data/huangtao/projects/llm-safty/LRV-Instruction/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
#  --gpu-id 5 \
#  --if_image False \
#  --model_name lrv \
#  --input_file /data/huangtao/datasets/facebook-hateful-meme/test.jsonl \
#  --image_file /data/huangtao/datasets/facebook-hateful-meme/img  \
#  --output_dir /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/LRV