import argparse
import numpy as np
import random
import os
import torch.backends.cudnn as cudnn
import gradio as gr
import torch.nn as nn
import sys
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
import json
import math




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


def get_max_prob(prob):
    # for a distribution
    return -math.log2(prob.max())

def get_entropy(prob):
    # for a distribution
    non_zero_indices = (prob != 0)
    prob = prob[non_zero_indices]
    return sum([-p*math.log2(p) if p>0 else 0 for p in prob])

def get_max_log(probs):
    return max([get_max_prob(prob) for prob in probs])

def get_avg_log(probs):
    return sum([get_max_prob(prob) for prob in probs]) / len(probs)

def get_max_entropy(probs):
    return max([get_entropy(prob) for prob in probs])

def get_avg_entropy(probs):
    return sum([get_entropy(prob) for prob in probs]) / len(probs)
def get_log_entropy(probs):
    probs = [prob[0] for prob in probs]
    max_log = get_max_log(probs)
    avg_log = get_avg_log(probs)
    max_entropy = float(get_max_entropy(probs))
    avg_entropy = float(get_avg_entropy(probs))
    return max_log,avg_log,max_entropy,avg_entropy