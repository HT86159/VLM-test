 torchrun --nproc_per_node 1 --master_port 25641 /data/huangtao/projects/llm-safty/LLaMA-Adapter/selfcheck-_generate.py \
 --llama_path  /data/public/models/llama-hf/LLaMA-7B \
 --tokenizer_path /data/public/models/llama-hf/LLaMA-7B/tokenizer.model \
 --gpu-id 5 \
  --if_image False \
 --model_name llama-adapter \
 --input_file /data/huangtao/datasets/red_team1k/red_team1k.json \
 --image_file /data/huangtao/datasets/red_team1k/images  \
 --output_dir /data/huangtao/projects/llm-safty/MyVLMTest/uncertainty/selfcheck_results\
 --adapter_path /data/huangtao/projects/llm-safty/LLaMA-Adapter/llama_adapter_len10_layer30_release.pth

#   --input_file /data/huangtao/datasets/facebook-hateful-meme/test.jsonl \
#  --image_file /data/huangtao/datasets/facebook-hateful-meme/img  \

