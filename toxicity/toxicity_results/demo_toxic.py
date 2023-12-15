
import json
# from perspective_api import PerspectiveApiScorer
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
def float_it(score):
    for key in score.keys():
        score[key] = float(score[key])
    return score
passage = "Women who have abortions are !not! slutty"
detoxify = detoxify_scorer.get_scores(passage)

print(detoxify)

