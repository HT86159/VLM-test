  python /data/huangtao/projects/llm-safty/LRV-Instruction/MiniGPT-4/selfcheck_generate.py \
 --cfg-path /data/huangtao/projects/llm-safty/LRV-Instruction/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
 --gpu-id 6 \
 --if_image True \
 --model_name lrv \
 --input_file /data/huangtao/datasets/red_team1k/red_team1k.json\
 --image_file /data/huangtao/datasets/red_team1k/images  \
 --output_dir /data/huangtao/projects/llm-safty/MyVLMTest/uncertainty/selfcheck_results/lrv

 