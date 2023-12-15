python /data/huangtao/projects/llm-safty/MyVLMTest/metric/chatgpt_api.py \
 --input /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/MiniGPT4/minigpt4-test-no_image-202383111.jsonl &
python /data/huangtao/projects/llm-safty/MyVLMTest/metric/chatgpt_api.py \
 --input /data/huangtao/projects/llm-safty/MyVLMTest/qa_results/MiniGPT4/minigpt4-test-with_image-202383111.jsonl &
# #============ get the file name ===========
# Folder_A="/data/huangtao/projects/llm-safty/MyVLMTest/qa_results/LLaMA"
# for file_a in ${Folder_A}/*; do
#     json_file=`basename $file_a`
#     if [ "${json_file##*.}"x = "jsonl"x ]||[ "${json_file##*.}"x = "json"x ]; then
#         echo $Folder_A/$json_file
#         python /data/huangtao/projects/llm-safty/MyVLMTest/metric/chatgpt_api.py \
#         --input $Folder_A/$json_file &
    
#     fi
# done  
