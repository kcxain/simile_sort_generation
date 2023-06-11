import os
from typing import List

import torch
import logger
import warnings
from torch import nn
import texar.torch as tx
# from texar.torch.modules import BERTEncoder
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel, BertLayer, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling, MaskedLMOutput
from transformers.modeling_outputs import MaskedLMOutput

from util.config_base import Config


class CondenserForPretraining(nn.Module):
    def __init__(
            self,
            bert: BertModel,
            n_head_layers=4,
            skip_from=12,
            late_mlm=True
    ):
        super(CondenserForPretraining, self).__init__()
        self.late_mlm = late_mlm
        self.skip_from = skip_from
        self.lm = bert
        self.c_head = nn.ModuleList(
            [BertLayer(bert.config) for _ in range(n_head_layers)]
        )
        self.c_head.apply(self.lm._init_weights)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, model_input, labels):
        attention_mask = self.lm.get_extended_attention_mask(
            model_input['attention_mask'],
            model_input['attention_mask'].shape,
            model_input['attention_mask'].device
        )

        lm_out: MaskedLMOutput = self.lm(
            **model_input,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        cls_hiddens = lm_out.hidden_states[-1][:, :1]
        skip_hiddens = lm_out.hidden_states[self.skip_from]

        hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)

        for layer in self.c_head:
            layer_out = layer(
                hiddens,
                attention_mask,
            )
            hiddens = layer_out[0]

        loss = self.mlm_loss(hiddens, labels)
        if self.late_mlm:
            loss += lm_out.loss

        return MaskedLMOutput(
            loss=loss,
            logits=lm_out.logits,
            hidden_states=lm_out.hidden_states,
            attentions=lm_out.attentions,
        )

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss

    @classmethod
    def from_pretrained(cls, model_name_path):
        hf_model = AutoModelForMaskedLM.from_pretrained(model_name_path)
        model = cls(hf_model)
        path = model_name_path
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = model.load_state_dict(model_dict, strict=False)
        return model

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('lm')]
        warnings.warn(f'omiting {len(hf_weight_keys)} transformer weights')
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))


class BERTMetric(nn.Module):
    NAME = 'bert_metric'

    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name)
        self.max_seq_length = args.max_seq_length

        self.backbone = AutoModel.from_pretrained(args.pretrained_model_name)

        bert_hidden_size = self.backbone.config.hidden_size
        mlp_hidden_size_1 = int(bert_hidden_size / 2)
        mlp_hidden_size_2 = int(mlp_hidden_size_1 / 2)
        self.mlp = nn.Sequential(
            nn.Linear(bert_hidden_size, mlp_hidden_size_1),
            nn.ELU(),
            nn.Linear(mlp_hidden_size_1, mlp_hidden_size_2),
            nn.ELU(),
            nn.Linear(mlp_hidden_size_2, 1),
            nn.Sigmoid())

        if args.gpu:
            self.device = torch.device("cuda:{}".format(args.gpu))
            self.to(self.device)
            map_location = 'cuda:{}'.format(args.gpu)
        else:
            self.device = None
            map_location = None

        if hasattr(args, 'checkpoint_file_name'):
            # loads checkpoint
            checkpoint_file_path = os.path.join(
                args.checkpoint_dir_path, args.checkpoint_file_name)
            state_dict = torch.load(
                checkpoint_file_path,
                map_location=map_location)
            self.load_state_dict(state_dict)
            print('loading checkpoint from: {}'.format(checkpoint_file_path))

        if hasattr(args, 'pretrain_checkpoint_file_name'):
            # loads checkpoint
            checkpoint_file_path = os.path.join(
                args.pretrain_checkpoint_dir_path,
                args.pretrain_checkpoint_file_name)
            state_dict = torch.load(
                checkpoint_file_path,
                map_location=map_location)
            self.load_state_dict(state_dict)
            print('loading checkpoint from: {}'.format(checkpoint_file_path))

    def get_hidden_size(self):
        return self.backbone.config.hidden_size

    def forward(self, input_ids, token_type_ids, attention_mask):
        output_dict = self.backbone(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True)
        pooled_output = output_dict['pooler_output']
        score = self.mlp(pooled_output)
        return output_dict, score

    @torch.no_grad()
    def get_score(self, simile_sentence):
        self.eval()
        input_ids, token_type_ids, attention_mask = self.encode_simile_sentence(
            simile_sentence)
        _, score = self.forward(input_ids, token_type_ids, attention_mask)
        # print(score)
        return score

    def encode_simile_sentence(self, simile_sentence: str):
        """Encodes the given context-response pair into ids.
        """
        tokenizer_outputs = self.tokenizer(
            simile_sentence,
            return_tensors='pt', truncation=True,
            padding='max_length', max_length=self.max_seq_length)
        input_ids = tokenizer_outputs['input_ids']
        token_type_ids = tokenizer_outputs['token_type_ids']
        attention_mask = tokenizer_outputs['attention_mask']

        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        assert input_ids.size() == torch.Size([4, self.max_seq_length])
        assert token_type_ids.size() == torch.Size([4, self.max_seq_length])
        assert attention_mask.size() == torch.Size([4, self.max_seq_length])
        # tokenized_text = self.tokenizer.convert_ids_to_tokens(input_ids)
        # print('context: ', context)
        # print('response: ', response)
        # print('tokenizer_outputs: ', tokenizer_outputs)
        # print('tokenized_text: ', ' '.join(tokenized_text))
        # print('length: ', len(tokenized_text))
        # exit()
        return input_ids, token_type_ids, attention_mask
