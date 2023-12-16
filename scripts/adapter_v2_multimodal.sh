python /data/huangtao/projects/llm-safty/LLaMA-Adapter/llama_adapter_v2_multimodal/llama_adapter-v2-multimodal.py \
 --llama_path "/data/public/models/llama_ori" \
 --tokenizer_path /data/public/models/llama_ori/tokenizer.model \
 --gpu-id 0 \
 --input_file /data/huangtao/datasets/facebook-hateful-meme/test.jsonl \
 --image_file /data/huangtao/datasets/facebook-hateful-meme/img  \
 --output_dir "/data/huangtao/projects/llm-safty/MyVLMTest/qa_results/LLaMA-adapter/" \
 --model_name llama-adapter-v2

