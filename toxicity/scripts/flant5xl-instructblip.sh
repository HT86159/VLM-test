python /data/huangtao/projects/llm-safty/LAVIS/projects/instructblip/instruct_blip.py \
 --gpu-id 5 \
 --if_image True \
 --input_file /data/huangtao/datasets/toxicity_split/hatefulmeme_redteam_v1.jsonl \
 --image_file /data/huangtao/datasets/facebook-hateful-meme/img \
 --model_name instruct-blip \
 --model_type vicuna7b \
 --output_dir /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/Instruct-Blip 


python /data/huangtao/projects/llm-safty/LAVIS/projects/instructblip/instruct_blip.py \
 --gpu-id 5 \
 --if_image True \
 --input_file /data/huangtao/datasets/toxicity_split/hatefulmeme_redteam_v2.jsonl \
 --image_file /data/huangtao/datasets/facebook-hateful-meme/img \
 --model_name instruct-blip \
 --model_type vicuna7b \
 --output_dir /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/Instruct-Blip 


python /data/huangtao/projects/llm-safty/LAVIS/projects/instructblip/instruct_blip.py \
 --gpu-id 5 \
 --if_image True \
 --input_file /data/huangtao/datasets/toxicity_split/hatefulmeme_redteam_v3.jsonl \
 --image_file /data/huangtao/datasets/facebook-hateful-meme/img \
 --model_name instruct-blip \
 --model_type vicuna7b \
 --output_dir /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/Instruct-Blip 


python /data/huangtao/projects/llm-safty/LAVIS/projects/instructblip/instruct_blip.py \
 --gpu-id 5 \
 --if_image True \
 --input_file /data/huangtao/datasets/toxicity_split/hatefulmeme_redteam_v4.jsonl \
 --image_file /data/huangtao/datasets/facebook-hateful-meme/img \
 --model_name instruct-blip \
 --model_type vicuna7b \
 --output_dir /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/Instruct-Blip 

#  --model_type flant5xl \
#   --input_file /data/huangtao/datasets/facebook-hateful-meme/test.jsonl \
#  --image_file /data/huangtao/datasets/facebook-hateful-meme/img  \
#  --input_file /data/huangtao/datasets/imagenet-redteam_30-30/imagenet-redteam30-30.json \
#  --image_file /data/huangtao/datasets/imagenet-redteam_30-30/images \

