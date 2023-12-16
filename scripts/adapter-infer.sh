 torchrun --nproc_per_node 1 /data/huangtao/projects/llm-safty/LLaMA-Adapter/infer.py \
 --llama_path  /data/public/models/llama-hf/LLaMA-7B \
 --tokenizer_path /data/public/models/llama-hf/LLaMA-7B/tokenizer.model \
 --gpu-id 7 \
  --if_image False \
 --model_name llama-adapter \
 --input_file /data/huangtao/datasets/facebook-hateful-meme/test.jsonl \
 --image_file /data/huangtao/datasets/facebook-hateful-meme/img  \
 --output_dir /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/LLaMA-adapter\
 --adapter_path /data/huangtao/projects/llm-safty/LLaMA-Adapter/llama_adapter_len10_layer30_release.pth

#   --input_file /data/huangtao/datasets/facebook-hateful-meme/test.jsonl \
#  --image_file /data/huangtao/datasets/facebook-hateful-meme/img  \
