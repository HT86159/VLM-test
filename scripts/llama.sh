torchrun --nproc_per_node 1 /data/huangtao/projects/llm-safty/llama/selfcheck_generate.py\
 --gpu-id 2 \
 --llama_path /data/public/models/llama_ori/7B \
 --tokenizer_path /data/public/models/llama_ori/tokenizer.model \
 --input_file /data/huangtao/datasets/red_team1k/red_team1k.json \
 --image_file /data/huangtao/datasets/red_team1k/images  \
 --output_dir /data/huangtao/projects/llm-safty/MyVLMTest/uncertainty/selfcheck_results/llama


#  --input_file /data/huangtao/datasets/imagenet-redteam_30-30/imagenet-redteam30-30.json \
#  --image_file /data/huangtao/datasets/imagenet-redteam_30-30/images \
