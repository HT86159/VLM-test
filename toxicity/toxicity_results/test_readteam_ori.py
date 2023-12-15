
import json
from perspective_api import PerspectiveApiScorer
from detoxify_api import DetoxifyScorer
import argparse
import json
import numpy as np
import os
import ssl
import tqdm
import spacy
ssl._create_default_https_context = ssl._create_unverified_context
detoxify_scorer = DetoxifyScorer()

print(">>>>>>>>>>>>>>>API_FINISHED<<<<<<<<<<<<<<<<<")
# import pdb;pdb.set_trace()

input_file = "/data/huangtao/datasets/red_team1k/red_team_all.jsonl"
output_file = "/data/huangtao/datasets/red_team1k/red_team_all_detoxify.jsonl"


def float_it(score):
    for key in score.keys():
        score[key] = float(score[key])
    return score


inputs = open(input_file).read().strip().split('\n')
# import pdb;pdb.set_trace()
with open(output_file, 'a') as f:
    flag = 1
    for li in tqdm.tqdm(inputs):
        obj = json.loads(li)
        flag = 0
        # import pdb;pdb.set_trace()
        if 'instruction' in obj and obj['instruction'] != "":
            # import pdb;pdb.set_trace()
            passage = obj["instruction"]
            detoxify = list(float_it(detoxify_scorer.get_scores(passage)).values())
            detoxify.append(sum(detoxify)/len(detoxify))
            res = {
                'detoxify': detoxify}
            obj['continuation_metric'] = res
        f.write(json.dumps(obj))
        f.write('\n')
        # f.close()
f.close()

