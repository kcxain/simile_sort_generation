import os
from typing import List, Dict
import json

import numpy as np
from tqdm import tqdm
import prettytable as pt
from transformers import AutoTokenizer
from torch.utils.data import random_split

class Processor:
    """Base class for all simile data processors. A simile data processor loads
    and processes the simile data of a specific simile dataset.

    Attributes:
        dataset_name: The dataset name.
        split_file_paths: A Dict whose keys are split names
            and values are the corresponding split file paths.
        output_dir_path: The output directory path of the processed simile data.
        max_seq_length: The max sequence length of the simile sentence.
    """

    def __init__(self,
                 dataset_name: str,
                 split_file_paths: Dict[str, str],
                 output_dir_path: str,
                 max_seq_length: int
                 ):
        self.dataset_name = dataset_name
        self.split_file_paths = split_file_paths
        self.output_dir_path = output_dir_path
        self.max_seq_length = max_seq_length
        self.cur_split_name = None
        self.cur_file_path = None

    def prepare(self) -> None:
        for split_name, file_path in self.split_file_paths.items():
            self.cur_split_name = split_name
            self.cur_file_path = file_path
            self._load_data()
            self._process_data()
            self._save_data()

    def analyze(self) -> None:
        for split_name, file_path in self.split_file_paths.items():
            print('------ analyze {} data in {} ------'.format(split_name,
                                                               file_path))
            self.cur_split_name = split_name
            self.cur_file_path = file_path
            self._load_data()
            self._analyze_data()

    def _load_data(self):
        raise NotImplementedError

    def _process_data(self):
        raise NotImplementedError

    def _save_data(self):
        raise NotImplementedError

    def _analyze_data(self):
        raise NotImplementedError

    @property
    def cur_split_dir_path(self):
        return os.path.join(self.output_dir_path, self.cur_split_name)

    @staticmethod
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

    @staticmethod
    def maybe_create_dir(dir_path: str) -> bool:
        """Creates directory if it does not exist.

        Args:
            dir_path: The path of the directory needed to be created.

        Returns:
            bool: Whether a new directory is created.
        """
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
            return True
        return False


class SimileDataProcessor(Processor):
    """simile data processor of the Simile dataset.

    Attributes:
        input_dir_path: The input directory path of the raw simile data.
        output_dir_path: The output directory path of the processed simile data.
        simile_types: A list of desired simile types.
        max_seq_length: The max sequence length of the simile sentence.
        pretrained_model_name: A str indicating the pretrained model adopted
            for initializing the tokenizer.
    """

    def __init__(self,
                 input_dir_path: str,
                 output_dir_path: str,
                 simile_types: List[str],
                 max_seq_length: int,
                 pretrained_model_name='bert-base-uncased'):
        split_file_paths = {
            'train': os.path.join(input_dir_path, 'train.json'),
            'validation': os.path.join(input_dir_path, 'dev.json'),
            'test': os.path.join(input_dir_path, 'test.json'),
        }
        super().__init__(dataset_name='SPGC',
                         split_file_paths=split_file_paths,
                         output_dir_path=output_dir_path,
                         max_seq_length=max_seq_length)

        self.simile_types = simile_types
        self.num_per_simile = 3
        self.raw_simile_data = None
        self.processed_simile_data = None
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def prepare(self):
        super().prepare()
        self._save_feature_types()

    def encode_simile_sentence(self, sentence: str):
        """Encodes the given simile sentence into ids.
        """
        tokenizer_outputs = self.tokenizer(
            sentence, truncation=True,
            padding='max_length', max_length=self.max_seq_length)
        input_ids = tokenizer_outputs['input_ids']
        token_type_ids = tokenizer_outputs['token_type_ids']
        attention_mask = tokenizer_outputs['attention_mask']
        return input_ids, token_type_ids, attention_mask

    def _load_data(self):
        print('Loading data ({})......'.format(self.cur_file_path))
        self.raw_simile_data = self.load_from_json(self.cur_file_path)

    def _process_data(self):
        print('Processing data......')
        self.processed_simile_data = []
        for simile in tqdm(self.raw_simile_data):
            processed_simile = {}
            for res_type in self.simile_types:
                processed_simile[res_type] = simile[res_type]
            self.processed_simile_data.append(processed_simile)

    def _save_data(self):
        print('Saving data......')
        self._create_output_dir()
        self._save_simile_data_in_text_form()
        self._save_simile_data_in_binary_form()

    def _save_feature_types(self):
        output_feature_type_path = os.path.join(
            self.output_dir_path, 'feature_types.json')
        with open(output_feature_type_path, 'w') as f:
            json.dump(self.feature_types, f, indent=4)

    def _create_output_dir(self):
        raise NotImplementedError

    def _save_simile_data_in_text_form(self):
        raise NotImplementedError

    def _save_simile_data_in_binary_form(self):
        raise NotImplementedError

    @property
    def feature_types(self):
        raise NotImplementedError
