python /data/huangtao/projects/llm-safty/MyVLMTest/hallucination/pope_generate.py \
 --cfg-path /data/huangtao/projects/llm-safty/MyVLMTest/models/eval_config/minigpt4_eval.yaml \
 --gpu_id 0 \
 --if_image True \
 --model_name minigpt4 \
 --input_file /data/huangtao/projects/hallucination/POPE/output/mscoco1k/mscoco1k_pope_random.jsonl \
 --image_file /data/public/data/MSCOCO2014/preprocessed_data/pictures \
 --output_path /data/huangtao/projects/llm-safty/MyVLMTest/uncertainty/hallucination_results/pope_results/minigpt4