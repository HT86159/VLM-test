python /data/huangtao/projects/llm-safty/open_flamingo/openflamingo-infer.py \
 --gpu-id 3 \
 --input_file "/data/huangtao/datasets/red_team1k/red_team1k.json" \
 --image_file '/data/huangtao/datasets/red_team1k/images' \
 --if_image True \
 --model_name openflamingo \
 --output_dir /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/Open-Flamingo
