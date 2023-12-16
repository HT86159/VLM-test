torchrun --nproc_per_node 1 /data/huangtao/projects/llm-safty/llama/selfcheck_generate.py \
 --gpu-id 7 \
 --llama_path /data/public/models/llama_ori/7B \
 --tokenizer_path /data/public/models/llama_ori/tokenizer.model \
 --input_file /data/huangtao/datasets/facebook-hateful-meme/test.jsonl \
 --image_file /data/huangtao/datasets/facebook-hateful-meme/img  \
 --output_dir /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/LLaMA


#  --input_file /data/huangtao/datasets/imagenet-redteam_30-30/imagenet-redteam30-30.json \
#  --image_file /data/huangtao/datasets/imagenet-redteam_30-30/images \