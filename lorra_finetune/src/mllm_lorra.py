# Usage: deepspeed train_lora.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import logging
import pathlib
import random
import typing
import os
import json
import gc
from typing import Dict, List, Optional, Sequence

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import Trainer, BitsAndBytesConfig
# from transformers import deepspeed
import torch
import pickle
from mllm_utils import auto_configure_device_map
from accelerate import dispatch_model

# from custom import YerongTrainer
from args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)
from finetune.data_mix import Mix_dataset

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    max_length: int = field(
        default=8192,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
    # use_lora: bool = False
    fix_vit: bool = True
    fix_sampler: bool = False
    label_names: List[str] = field(default_factory=lambda: ['samples'])
@dataclass
class DataArguments:
    data_path: str = field(
        default='data.txt', metadata={'help': 'Path to the training data.'})
    given_num: bool = False
    batch_size: int = 4
    resolution: int = 560
    hd_num: int = 18
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = [instance['samples'] for instance in instances]
        for i, instance in enumerate(instances):
            print('====== Instance {} ======'.format(i+1))
            for key in instance.keys():
                print(key)
            print('=========================')

        
        text_input, data_type = tuple(
            [instance[key] for instance in instances]
            for key in ('text_input', 'data_type'))
        if 'image' not in instances[0]:
            text_input = [instance['text_input'][0] for instance in instances]
        batch = dict(
            text_input=text_input,
            data_type=data_type,
        )
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            batch['image'] = images

        return dict(samples=batch)


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
#                                    output_dir: str,
#                                    bias='none'):
#     """Collects the state dict and dump to disk."""
#     # check if zero3 mode enabled
#     # if deepspeed.is_deepspeed_zero3_enabled():
#     if False:
#         state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict(
#         )
#     else:
#         if trainer.args.use_lora:
#             state_dict = get_peft_state_maybe_zero_3(
#                 trainer.model.named_parameters(), bias)
#         else:
#             state_dict = trainer.model.state_dict()
#     if trainer.args.should_save and trainer.args.local_rank == 0:
#         trainer._save(output_dir, state_dict=state_dict)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    rank0_print('Loading data...')
    if data_args.data_path.endswith('json'):
        train_json = json.load(open(data_args.data_path))
    elif data_args.data_path.endswith('txt'):
        train_json = {}
        with open(data_args.data_path) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            line = line.split(' ')
            with open(line[0]) as f:
                temp = json.load(f)
            if data_args.given_num:
                assert len(line) == 2
                num = int(float(line[1]) * 1000)
                if len(temp) > num:
                    temp = random.sample(temp, num)
                else:
                    ex_temp = []
                    for i in range(num - len(temp)):
                        ex_temp.append(random.choice(temp))
                    temp.extend(ex_temp)
            else:
                if len(line) == 2:
                    ratio = float(line[1])
                    new_len = int(len(temp) * ratio)
                    # assert False, f"{ratio} ratio {new_len}"

                    if ratio < 1:
                        temp = random.sample(temp, new_len)
                    elif ratio > 1:
                        ex_temp = []
                        for i in range(new_len - len(temp)):
                            ex_temp.append(random.choice(temp))
                        temp.extend(ex_temp)
            rank0_print(f'Load {len(temp)} samples from {line}')
            train_json[line[0]] = temp
    train_dataset = Mix_dataset(
        train_json,
        data_args.batch_size,
        resolution=data_args.resolution,
        hd_num=data_args.hd_num,
        local_rank=local_rank)
    print(str(len(train_dataset)) + ' samples is loaded')
    eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset()
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

# def compute_loss(self, model, inputs, target_layers, alpha, beta, max_res_len=64, return_outputs=False, **kwargs):

#     input_ids = inputs.get("input_ids")
#     attention_mask = inputs.get("attention_mask")

#     assert input_ids.shape[1] == 3

#     orig_input_ids = input_ids[:, 0]
#     pos_input_ids = input_ids[:, 1]
#     neg_input_ids = input_ids[:, 2]

#     orig_attention_mask = attention_mask[:, 0]
#     pos_attention_mask = attention_mask[:, 1]
#     neg_attention_mask = attention_mask[:, 2]

#     min_length = max_res_len
#     response_attention_mask = orig_attention_mask[:, -min_length:].repeat(len(target_layers), 1, 1).unsqueeze(-1)

#     module = 'past_key_values' # 'hidden_states
#     with model.disable_adapter():
#         model.eval()
#         with torch.no_grad():
#             orig_outputs = model(
#                 input_ids=orig_input_ids,
#                 attention_mask=orig_attention_mask,
#                 output_hidden_states=True
#             )['hidden_states']
#             orig_hidden = [orig_outputs[l][:, -min_length:].detach() for l in target_layers]
#             pos_outputs = model(
#                 input_ids=pos_input_ids,
#                 attention_mask=pos_attention_mask,
#                 output_hidden_states=True
#             )['hidden_states']
#             neg_outputs = model(
#                 input_ids=neg_input_ids,
#                 attention_mask=neg_attention_mask,
#                 output_hidden_states=True
#             )['hidden_states']
#             direction_hidden = [pos_outputs[l][:, -min_length:].detach() - \
#                                 neg_outputs[l][:, -min_length:].detach() \
#                                 # + beta * torch.tensor(pca_directions[l - len(pca_directions)], device=model.device, dtype=torch.float16) \
#                                                 for l in target_layers]
#             target_hidden = torch.stack([orig_hidden[i] + alpha * direction_hidden[i] for i in range(len(target_layers))]) * response_attention_mask

#             del orig_outputs, pos_outputs, neg_outputs, orig_hidden, direction_hidden
#             gc.collect()
#             torch.cuda.empty_cache()

#     model.train()
#     lora_outputs = model(
#         input_ids=orig_input_ids,
#         attention_mask=orig_attention_mask,
#         output_hidden_states=True
#     )['hidden_states']
#     lora_hidden = torch.stack([lora_outputs[l][:, -min_length:] for l in target_layers]) * response_attention_mask

#     loss_fct = torch.nn.MSELoss()
#     loss = torch.norm(lora_hidden - target_hidden, dim=-1, p=2, dtype=torch.float).nanmean()
#     return (loss, lora_hidden) if return_outputs else loss


            
def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def train():
    global local_rank
    # parser = transformers.HfArgumentParser(
    #     (ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    # (
    #     model_args,
    #     data_args,
    #     training_args,
    #     lora_args,
    # ) = parser.parse_args_into_dataclasses()
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments,DataArguments, LoraArguments, LorraArguments)
    )
    (
        model_args,
        training_args,
        data_args,
        lora_args,
        lorra_args,
    ) = parser.parse_args_into_dataclasses()

    if False:
    # if getattr(training_args, 'deepspeed', None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    config.max_length = training_args.max_length

    # Load model and tokenizer
    print(f'Load model from: {model_args.model_name_or_path}')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side='right',
        use_fast=False,
        trust_remote_code=True,
    )
    model.tokenizer = tokenizer

    # if training_args.fix_vit:
    #     model.vit.requires_grad_(False)
    # else:
    #     model.vit.requires_grad_(True)
    #     model.vit.vision_tower.vision_model.post_layernorm = torch.nn.Identity(
    #     )

    # if training_args.fix_sampler:
    #     model.vision_proj.requires_grad_(False)
    # else:
    #     model.vision_proj.requires_grad_(True)

    if model_args.use_lora:
        for name, param in model.model.named_parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type='CAUSAL_LM',
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args)
    print(transformers.processing_utils.logging.is_progress_bar_enabled())
    transformers.processing_utils.logging.enable_progress_bar()

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module)

    trainer.train()
    def countdown(t):
        while t:
            mins, secs = divmod(t, 60)
            timer = 'TRAINER finished {:02d}:{:02d}'.format(mins, secs)
            print(timer, end="\r")
            time.sleep(1)
            t -= 1
        print('Countdown Over!')

    # Start countdown for 5 minutes (or 300 seconds)
    countdown(300)
    # trainer.save_state()

    # safe_save_model_for_hf_trainer(
    #     trainer=trainer,
    #     output_dir=training_args.output_dir,
    #     bias=lora_args.lora_bias)


    
if __name__ == "__main__":
    train()