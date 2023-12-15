from tqdm import tqdm
import torch
from PIL import Image
import json
import sys
from inference_utils import *
import os
sys.path.append("../models")
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from utils import prompt_wrapper, generator



print('>>> Initializing Models')


args = parse_args()
cfg = Config(args)
input_file = open(args.input_file,"r",encoding='utf-8')
# import pdb;pdb.set_trace()
file_length = len(input_file.readlines())
if file_length!= 1000:
    image_idsets, datasets = get_all_path(args.image_file,args.input_file)
else:
    image_idsets, datasets = read_jsonl_file(args.input_file)
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
model = model_cls.from_config(model_config).to('cuda:{}'.format(str(args.gpu_id)))
# import pdb;pdb.set_trace()

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
print('Initialization Finished')

my_generator = generator.Generator(model=model, device='cuda')

# my_generator = generator.Generator(model=model)


# ========================================
#             Inference
# ========================================

##  TODO: expose interface.
# import pdb;pdb.set_trace()

# datasets = rtp_read('rtp_prompts.jsonl')
# import pdb;pdb.set_trace()


# for i in datasets:
#     print(i)
# import pdb;pdb.set_trace()

import datetime
current_time = datetime.datetime.now()
year = current_time.year
month = current_time.month
day = current_time.day
hour = current_time.hour
data = str(year)+str(month)+str(day)+str(hour)

if_image = 'with_image' if args.if_image=='True' else "no_image"
output_file = os.path.join(args.output_path,"{}-{}-{}-{}-{}.jsonl".format(args.model_name,args.image_file.split("/")[-2].split('.')[0],args.input_file.split('/')[-1].split('.')[0],if_image,data))
# import pdb;pdb.set_trace()
print("output_file:>>>>>>>>>>>>>>>>>>")
print(output_file)
out = []
with open(output_file, 'a') as f:
    with torch.no_grad():
        i = 0
        for img_name, user_message in tqdm(zip(image_idsets,datasets)):
            i = i + 1
            stochastic_response = []
            stochastic_metric = []
            prefix = prompt_wrapper.minigpt4_chatbot_prompt
            img_path = os.path.join(args.image_file,os.path.basename(img_name))
            if args.if_image=='True':#有图
                # import pdb;pdb.set_trace()
                img = Image.open(img_path).convert('RGB')
                img_prompt = [vis_processor(img).unsqueeze(0).to(model.device)]
                text_prompt = prefix % ('Please answer the following sentences: %s')
                prompt = prompt_wrapper.Prompt(model=model, img_prompts=[img_prompt])
                prompt.update_text_prompt([text_prompt % user_message])
                response, _, probs= my_generator.generate(prompt,1)
                max_log,avg_log,max_entropy,avg_entropy = get_log_entropy(probs)
                continuation_metric = [{"max_log" : max_log, "avg_log" : avg_log, "max_entropy" : max_entropy, "avg_entropy" : avg_entropy}]
                for _ in range(5):
                    sto_resp, _ ,prob= my_generator.generate(prompt,10)
                    max_log,avg_log,max_entropy,avg_entropy = get_log_entropy(prob)
                    stochastic_response.append(sto_resp)
                    stochastic_metric.append({"max_log" : max_log, "avg_log" : avg_log, "max_entropy" : max_entropy, "avg_entropy" : avg_entropy})
            else:
                prefix = ""
                prefix += "###Human:%s ###Assistant:"
                img_prompt = []
                # import pdb;pdb.set_trace()
                text_prompt = prefix % ('Please extend the following sentence: %s')
                prompt = prompt_wrapper.Prompt(model=model, img_prompts=[img_prompt])
                prompt.update_text_prompt([text_prompt % user_message])
                response, _ = my_generator.generate_no_image(prompt,1)
                for _ in range(5):
                    sto_resp, _ = my_generator.generate_no_image(prompt,10)
                    stochastic_response.append(sto_resp)
            if i %10==0:
                print(f" ----- {i} ----")
                print(" -- prompt: ---")
                print(text_prompt % user_message)
                print(" -- continuation: ---")
                print(response)
                print(" -- stochastic_response: ---")
                print(stochastic_response)
            obj = {'image_id':img_name,'prompt': user_message, 'continuation': response, "contiuation_metric": continuation_metric,"stochastic_response" : stochastic_response, "stochastic_metric": stochastic_metric}
            # import pdb;pdb.set_trace()
            out.append(obj)
            f.write(json.dumps(obj))
            f.write("\n")


