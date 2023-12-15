import argparse
import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from PIL import Image
import json
import torch.nn as nn
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
def get_all_path(image_folder,instruction_path):
    file_paths = []
    datasets = []
    json_file = open(instruction_path,"r")
    # import pdb;pdb.set_trace()
    json_dict = json.load(json_file)
    for root, directories, files in os.walk(image_folder):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(os.path.basename(filepath))
            random_key = random.choice(list(json_dict.keys()))
            random_value = json_dict[random_key]
            datasets.append(random_value)
    return file_paths,datasets

def rtp_read(text_file):
    dataset = []
    lines = open(text_file).read().split("\n")
    for li in lines:
        obj = json.loads(li)
        if obj['challenging']:
            dataset.append(obj['prompt']['text'])
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=1, help="specify the gpu to load the model.")
    parser.add_argument("--if_image", type=str, help="if there is image")
    parser.add_argument("--model_name", type=str, help="name of model")
    parser.add_argument("--mode", type=str, default='VisualChatBot',
                        choices=[ "TextOnly", "VisualChatBot" ],
                        help="Inference Mode: TextOnly: Text model only (Vicuna) \n VisualChatBot: Vision model + Text model (MiniGPT4) ")

    parser.add_argument("--input_file", type=str, default='')

    parser.add_argument("--image_file", type=str, default='./image.bmp',
                        help="Image file")
    parser.add_argument("--output_path", type=str, default='./result.jsonl',
                        help="Output file.")

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================
def read_jsonl_file(file_path):
    datasets = []
    image_idsets = []
    jsol_file = open(file_path,"r")
    lines = jsol_file.readlines()
    for i in lines:
        line_in_json = json.loads(i)
        datasets.append(line_in_json["instruction"])
        image_idsets.append(line_in_json["image_id"])
    return image_idsets,datasets
print('>>> Initializing Models')




args = parse_args()
cfg = Config(args)
input_file = open(args.input_file,"r",encoding='utf-8')
file_length = len(input_file.readlines())
if file_length==1:
    image_idsets, datasets = get_all_path(args.image_file,args.input_file)
else:
    image_idsets, datasets = read_jsonl_file(args.input_file)
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(str(args.gpu_id)))
# import pdb;pdb.set_trace()

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
print('Initialization Finished')

my_generator = generator.Generator(model=model)

import datetime
current_time = datetime.datetime.now()
year = current_time.year
month = current_time.month
day = current_time.day
hour = current_time.hour
data = str(year)+str(month)+str(day)+str(hour)

if_image = 'with_image' if args.if_image=='True' else "no_image"
output_file = os.path.join(args.output_path,"{}-{}-{}-{}.jsonl".format(args.model_name,args.input_file.split('/')[-1].split('.')[0],if_image,data))
# import pdb;pdb.set_trace()
print("output_file:>>>>>>>>>>>>>>>>>>")
print(output_file)
out = []
f = open(output_file, 'w')
f.close()
with open(output_file, 'a') as f:
    with torch.no_grad():
        i = 0
        for img_name, user_message in tqdm(zip(image_idsets,datasets)):
            # import pdb;pdb.set_trace()
            i = i + 1
            stochastic_response = []
            prefix = prompt_wrapper.minigpt4_chatbot_prompt
            img_path = os.path.join(args.image_file,os.path.basename(img_name))
            # import pdb;pdb.set_trace()
            if args.if_image=='True':#有图
                # import pdb;pdb.set_trace()
                img = Image.open(img_path).convert('RGB')
                img_prompt = [vis_processor(img).unsqueeze(0).to(model.device)]
                text_prompt = prefix % ('Please answer the following sentences, and you must answer ”yes" or "no": %s')

                prompt = prompt_wrapper.Prompt(model=model, img_prompts=[img_prompt])
                prompt.update_text_prompt([text_prompt % user_message])
                j = 0
                response = ""
                while j==0 or response not in ["yes","no","yes.","no.","Yes","Yes.","No","No."]:
                    response, _ = my_generator.generate(prompt,1)
                    j = j+1
                    if j==5:
                        print(">>>i={}<<<,fail!!".format(i))
                for k in range(5):
                    m = 0
                    response, _ = my_generator.generate(prompt,10)
                    stochastic_response.append(response)
            if i %10==0:
                print(f" ----- {i} ----")
                print(" -- prompt: ---")
                print(text_prompt % user_message)
                print(" -- continuation: ---")
                print(response)
                print(" -- stochastic_response: ---")
                print(stochastic_response)
            obj = {'image_id':img_name,'prompt': user_message, 'continuation': response, "stochastic_response" : stochastic_response}
            out.append(obj)
            f.write(json.dumps(obj))
            f.write("\n")


