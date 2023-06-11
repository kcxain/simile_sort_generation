import os
import logging
from typing import List, Dict
from collections import OrderedDict

import numpy as np
import prettytable as pt
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json

def load_from_json(data_path: str) -> list:
        """Loads simile data from json file.

        Args:
            data_path: The file path of the raw simile data.

        Returns:
            simile_data: A list containing the raw simile data.
        """
        try:
            with open(data_path, 'r') as f:
                simile_data = json.load(f)
        except json.JSONDecodeError:
            # print('Fail to load {} with json.load().'.format(data_path))
            # print('Try to load it line by line with json.loads()...')
            with open(data_path, 'r') as f:
                simile_data = [json.loads(line) for line in f.readlines()]
        return simile_data

class Evaluator:
    """#TODO: adds docstring
    """

    def __init__(self,
                 checkpoint_dir_path,
                 eval_data_dir_path='evaluation/dev.json',
                 result_file_name='mlr_eval_results.txt',
                 console_output=True):
        self.checkpoint_dir_path = checkpoint_dir_path
        self.eval_data_dir_path = eval_data_dir_path
        self.console_output = console_output
        self.result_file_path = os.path.join(
            checkpoint_dir_path, result_file_name)
        self.dev_data = dev_data = load_from_json(eval_data_dir_path)
        self.metric_model = None
        self.additional_eval_info = None

    def evaluate(self, metric_model):
        self.metric_model = metric_model
        ans = torch.tensor([0,1,2,3]).to('cuda:0')
        cnt = 0
        acc = 0
        hits = 0
        recall2 = 0
        for line in self.dev_data:
            send = [line['positive'][0], line['vehicle_isa_replace'][0], line['single_random_replace'][0], line['both_random_replace'][0]]
            scores = metric_model.get_score(send).flatten()
            # print(scores)
            preds = torch.argsort(scores,descending=True)
            # print(line['positive'][0])
            # print(line['vehicle_isa_replace'][0])
            # print(preds)
            # print(ans)
            # print(scores)
            # input('>')
            acc += torch.eq(ans,preds.squeeze(dim=-1)).int().sum()
            # print(acc)
            if int(preds[0]) == 0:
                hits += 1
            if int(preds[0]) == 0 or int(preds[1]) == 0:
                recall2 += 1
            # if int(preds[0]) != 0:
            #     print(scores)
            #     print(send)
            #     input('>')
            # print(hits)
            # input('>')
            cnt += 1
        print(f'r@1:{hits / cnt}')
        print(f'r@2:{recall2 / cnt}')
        print(f'accuracy:{acc  / (cnt * 4)}')
        return hits / cnt, acc  / (cnt * 4)


if __name__ == '__main__':
    evale = Evaluator('../output/model_best_dual_mlr_loss.ckpt')
    print(evale.evaluate('../model/bert_metric.py'))