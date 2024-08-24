# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import torch
import transformers
from accelerate.utils import DistributedType
from data_mix import Mix_dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model
from transformers import Trainer, deepspeed
from transformers.trainer_pt_utils import LabelSmoother

from functools import partial
from re_utils.ixc import custom_interleav_wrap

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from mllm_data_utils import AlpacaSupervisedDataset
from functools import partial
# from mllm_utils import custom_interleav_wrap
# from mllm_utils import custom_forward
# from mllm_utils import check_right_padding_with_embeddings
# from mllm_utils import check_left_padding_with_embeddings
# from math_utils import ChartQA

from args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)

class RETrainer(Trainer):
    def __init__(self, *args, lorra_args=None,lora_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.assistant_tag = lorra_args.assistant_tag
        self.target_layers = [int(layer) for layer in lorra_args.target_layers.split(",")] # target representations
        self.lorra_args = lorra_args
        self.lora_args = lora_args
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        # Original compute_loss functions
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        samples = inputs.get('samples', None)
        if samples:
            infer_mode = samples.get('infer_mode', 'base')
            if samples['data_type'][0] == 'text':
                has_img = False
            elif samples['data_type'][0] == 'multi':
                has_img = True
            else:
                raise NotImplementedError
    
            orig_s = samples['orig_s']
            pos_s = samples['pos_s']
            neg_s = samples['neg_s']
            
            # Print the elements
            # print(f"==== orig_s[0][0]: {orig_s[0][0]} ====")
            # print(f"==== pos_s[0][0]: {pos_s[0][0]} ====")
            # print(f"==== neg_s[0][0]: {neg_s[0][0]} ====")
            # print(Yerong)
            if has_img:
                image = samples['image'][0]
                bs = len(samples['orig_s'][0])
                # assert len(samples['orig_s']) == 1, 'self.per_device_train_batch_size'
                image_nums = []
                temp_image = []
                for im in image:
                    if type(im) is list:
                        image_nums.append(len(im))
                        temp_image.extend(im)
                    else:
                        image_nums.append(1)
                        temp_image.append(im)
                image = temp_image

                q_to_regress_embeds = [None, None, None]
                q_attention_mask = [None, None, None]
                q_targets = [None, None, None]
                q_im_mask = [None, None, None]
                
    
                
                for i, input_str in enumerate([orig_s, pos_s, neg_s]):
                    q_to_regress_embeds[i], q_attention_mask[i], q_targets[i], q_im_mask[i] = model.interleav_wrap(
                        image, [[e.split(self.assistant_tag)[0] for e in input_str[0]]], image_nums, 'right', set_length=self.lorra_args.query_max_len
                    )
    
                # Initialize the lists to store the outputs
                r_to_regress_embeds = [None, None, None]
                r_attention_mask = [None, None, None]
                r_targets = [None, None, None]
                r_im_mask = [None, None, None]
                # Loop through the input strings and store the outputs
                for i, input_str in enumerate([orig_s, pos_s, neg_s]):
                    if i == 0:
                        r_to_regress_embeds[0], r_attention_mask[0], r_targets[0], r_im_mask[0] = model.interleav_wrap(
                            image, [[e.split(self.assistant_tag)[1] for e in input_str[0]]], image_nums, 'left', set_length=self.lorra_args.response_max_len
                        )
                    else:
                        r_to_regress_embeds[i] = r_to_regress_embeds[0]
                        r_attention_mask[i] = r_attention_mask[0]
                        r_targets[i] = r_targets[0]
                        r_im_mask[i] = r_im_mask[0]
    
    
                # Assuming that q_to_regress_embeds and r_to_regress_embeds have been populated as per the previous logic
                
                to_regress_embeds = [
                    torch.cat((q_to_regress_embeds[i], r_to_regress_embeds[i]), dim=1)
                    for i in range(3)
                ]
                
                
                # Concatenate q_attention_mask and r_attention_mask
                to_attention_mask = [
                    torch.cat((q_attention_mask[i], r_attention_mask[i]), dim=1)
                    for i in range(3)
                ]
                
    
                # Concatenate q_targets and r_targets
                # to_targets = [
                #     torch.cat((q_targets[i], r_targets[i]), dim=1)
                #     for i in range(3)
                # ]
    
    
                # Concatenate q_im_mask and r_im_mask
                to_im_mask = [
                    torch.cat((q_im_mask[i], r_im_mask[i]), dim=1).bool()
                    for i in range(3)
                ]
                module = 'past_key_values' # 'hidden_states
                alpha = 16
                with model.disable_adapter():
                    model.eval()
                    with torch.no_grad():
                        # outputs = self.model(
                        #     input_ids=input_ids,
                        #     attention_mask=attention_mask,
                        #     position_ids=position_ids,
                        #     past_key_values=past_key_values,
                        #     inputs_embeds=inputs_embeds,
                        #     use_cache=use_cache,
                        #     output_attentions=output_attentions,
                        #     output_hidden_states=output_hidden_states,
                        #     return_dict=return_dict,
                        #     im_mask=im_mask,
                        #     infer_mode=infer_mode,
                        # )
                        self.min_length = self.lorra_args.response_max_len
                        response_attention_mask = to_attention_mask[0][:, -self.min_length:].repeat(len(self.target_layers), 1, 1).unsqueeze(-1)
                        # print(' === to_regress_embeds[0] =====')
                        # print(to_regress_embeds[0].shape)
                        # print(to_regress_embeds[1].shape)
                        # print(to_regress_embeds[2].shape)
                        orig_outputs = model(
                            input_ids=None,
                            attention_mask=to_attention_mask[0],
                            inputs_embeds=to_regress_embeds[0],
                            im_mask=to_im_mask[0],
                            output_hidden_states=True,
                            infer_mode=infer_mode,
                        )['hidden_states']
    
                        orig_hidden = [orig_outputs[l][:, -self.min_length:].detach() for l in self.target_layers]
                        # Generate positive outputs
                        pos_outputs = model(
                            input_ids=None,
                            attention_mask=to_attention_mask[1],
                            inputs_embeds=to_regress_embeds[1],
                            im_mask=to_im_mask[1],
                            output_hidden_states=True,
                            infer_mode=infer_mode,
                        )['hidden_states']
                        
                        
                        # Generate negative outputs
                        neg_outputs = model(
                            input_ids=None,
                            attention_mask=to_attention_mask[2],
                            inputs_embeds=to_regress_embeds[2],
                            im_mask=to_im_mask[2],
                            output_hidden_states=True,
                            infer_mode=infer_mode,
                        )['hidden_states']
                        direction_hidden = [pos_outputs[l][:, -self.min_length:].detach() - \
                                            neg_outputs[l][:, -self.min_length:].detach() \
                                            # + beta * torch.tensor(pca_directions[l - len(pca_directions)], device=model.device, dtype=torch.float16) \
                                                            for l in self.target_layers]
                        target_hidden = torch.stack([orig_hidden[i] + alpha * direction_hidden[i] for i in range(len(self.target_layers))]) * response_attention_mask
            
                        del orig_outputs, pos_outputs, neg_outputs, orig_hidden, direction_hidden
                        gc.collect()
                        torch.cuda.empty_cache()

        model.train()
        lora_outputs = model(
            input_ids=None,
            attention_mask=to_attention_mask[0],
            inputs_embeds=to_regress_embeds[0],
            im_mask=to_im_mask[0],
            output_hidden_states=True,
            infer_mode=infer_mode,
        )['hidden_states']
        lora_hidden = torch.stack([lora_outputs[l][:, -self.min_length:] for l in self.target_layers]) * response_attention_mask
    
        loss_fct = torch.nn.MSELoss()
        loss = torch.norm(lora_hidden - target_hidden, dim=-1, p=2, dtype=torch.float).nanmean()
        return (loss, lora_hidden) if return_outputs else loss
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, sanity_check=False, **kwargs):
        # print(eval_dataset)
        print(self.args.output_dir)
        print(f"Query Max Length: {self.lorra_args.query_max_len}")
        print(f"Response Max Length: {self.lorra_args.response_max_len}")
        print(f"Response MIN Length: {self.min_length}")
        # for dataset in eval_dataset:
        #     print(dataset.evaluate())
@dataclass
class LorraArguments:
    user_tag: str = field(metadata={"help": "User tag for chat models (eg: `USER:` or `[INST]`)"})
    assistant_tag: str = field(metadata={"help": "Assistant tag for chat models (eg: `ASSISTANT:` or `[\INST]`)"})
    pos_type: str = field(metadata={"help": "Concept/Function to be optimized towards (eg: 'a truthful')"})
    neg_type: str = field(metadata={"help": "vice versa of pos_type (eg: 'an untruthful')"})
    target_layers: str = field(metadata={"help": "Layers for Representation. Layers are seperate by `,` eg: `10,12,14,16,18,20` "})
    control_template: str = field(metadata={"help": "Control template for Representation setting (eg: Give a {type} answer)"})
    lorra_alpha: float = field(default=5, metadata={"help": "vice versa of pos_type (eg: 'an untruthful')"}) # LoRRA Hyperparameters
    lorra_beta: float = field(default=0, metadata={"help": "vice versa of pos_type (eg: 'an untruthful')"}) # LoRRA Hyperparameters
    query_max_len: int = field(default=64, metadata={"help": "truncated length for getting generated ouputs from lorra pos/neg exampels"}) # LoRRA Hyperparameters
    response_max_len: int = field(default=64, metadata={"help": "truncated length for getting generated ouputs from lorra pos/neg exampels"}) # LoRRA Hyperparameters


# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default='adamw_torch')
#     max_length: int = field(
#         default=8192,
#         metadata={
#             'help':
#             'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
#         },
#     )
#     use_lora: bool = False
#     fix_vit: bool = True
#     fix_sampler: bool = False
#     label_names: List[str] = field(default_factory=lambda: ['samples'])
# @dataclass
# class DataArguments:
#     data_path: str = field(
#         default='data.txt', metadata={'help': 'Path to the training data.'})
#     given_num: bool = False
#     batch_size: int = 7
#     resolution: int = 560
#     hd_num: int = 18

# @dataclass
# class ModelArguments:
#     model_name_or_path: Optional[str] = field(default='')


@dataclass
class DataArguments:
    data_path: str = field(
        default='data.txt', metadata={'help': 'Path to the training data.'})
    given_num: bool = False
    batch_size: int = 4
    resolution: int = 560
    hd_num: int = 18


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    max_length: int = field(
        default=8192,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    fix_sampler: bool = False
    label_names: List[str] = field(default_factory=lambda: ['samples'])


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        'attention.wqkv',
        'attention.wo',
        'feed_forward.w1',
        'feed_forward.w2',
        'feed_forward.w3',
    ])
    lora_weight_path: str = ''
    lora_bias: str = 'none'

class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = [instance['samples'] for instance in instances]
        # for i, instance in enumerate(instances):
        #     print('====== Instance {} ======'.format(i+1))
        #     for key in instance.keys():
        #         print(key)
        #     print('=========================')

        
        # text_input, data_type = tuple(
        #     [instance[key] for instance in instances]
        #     for key in ('text_input', 'data_type'))
        orig_s, pos_s, neg_s, data_type = tuple(
            [instance[key] for instance in instances]
            for key in ('orig_s', 'pos_s', 'neg_s', 'data_type')
        )

        # Image always exists
        # if 'image' not in instances[0]:
        #     text_input = [instance['text_input'][0] for instance in instances]
        
        batch = dict(
            orig_s=orig_s,
            pos_s=pos_s,
            neg_s=neg_s,
            data_type=data_type,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            batch['image'] = images

        return dict(samples=batch)


local_rank = None
# @dataclass
# class ModelArguments:
#     model_name_or_path: Optional[str] = field(default='')


# @dataclass
# class DataArguments:
#     data_path: str = field(
#         default='data.txt', metadata={'help': 'Path to the training data.'})
#     given_num: bool = False
#     batch_size: int = 4
#     resolution: int = 560
#     hd_num: int = 18


# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default='adamw_torch')
#     max_length: int = field(
#         default=8192,
#         metadata={
#             'help':
#             'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
#         },
#     )
#     use_lora: bool = False
#     fix_vit: bool = True
#     fix_sampler: bool = False
#     label_names: List[str] = field(default_factory=lambda: ['samples'])


# @dataclass
# class LoraArguments:
#     lora_r: int = 64
#     lora_alpha: int = 64
#     lora_dropout: float = 0.05
#     lora_target_modules: List[str] = field(default_factory=lambda: [
#         'attention.wqkv',
#         'attention.wo',
#         'feed_forward.w1',
#         'feed_forward.w2',
#         'feed_forward.w3',
#     ])
#     lora_weight_path: str = ''
#     lora_bias: str = 'none'


def maybe_zero_3(param):
    if hasattr(param, 'ds_id'):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == 'none':
        to_return = {k: t for k, t in named_params if 'lora_' in k}
    elif bias == 'all':
        to_return = {
            k: t
            for k, t in named_params if 'lora_' in k or 'bias' in k
        }
    elif bias == 'lora_only':
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if 'lora_' in k:
                to_return[k] = t
                bias_name = k.split('lora_')[0] + 'bias'
                lora_bias_names.add(bias_name)
            elif 'bias' in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str,
                                   bias='none'):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict(
        )
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias)
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = [instance['samples'] for instance in instances]
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
    print(str(len(train_dataset)) + 'samples is loaded')
    eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset()
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, LorraArguments))
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        lorra_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None):
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
    model.interleav_wrap = partial(custom_interleav_wrap, model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side='right',
        use_fast=False,
        trust_remote_code=True,
    )
    model.tokenizer = tokenizer

    if training_args.fix_vit:
        model.vit.requires_grad_(False)
    else:
        model.vit.requires_grad_(True)
        model.vit.vision_tower.vision_model.post_layernorm = torch.nn.Identity(
        )

    if training_args.fix_sampler:
        model.vision_proj.requires_grad_(False)
    else:
        model.vision_proj.requires_grad_(True)

    if training_args.use_lora:
        if hasattr(training_args, 'resume_from_checkpoint') and training_args.resume_from_checkpoint:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, training_args.resume_from_checkpoint)
            model = model.merge_and_unload()
            print(f" ==== Model merged successfully from checkpoint: {training_args.resume_from_checkpoint}")
        for name, param in model.model.named_parameters():
            param.requires_grad = False
        lorra_target_layers = [10,12,14,16,18,20] # target representations
        lora_layers_to_transform = list(range(lorra_target_layers[-1] + 1)) # LoRA layers
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            layers_to_transform=lora_layers_to_transform,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type='CAUSAL_LM',
        )
        print(lora_config)

        # lora_config = LoraConfig(
        #     r=lora_args.lora_r,
        #     lora_alpha=lora_args.lora_alpha,
        #     target_modules=lora_args.lora_target_modules,
        #     # layers_to_transform=lora_layers_to_transform,
        #     lora_dropout=lora_args.lora_dropout,
        #     bias=lora_args.lora_bias,
        #     task_type='CAUSAL_LM',
        # )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args)
    print(transformers.processing_utils.logging.is_progress_bar_enabled())
    transformers.processing_utils.logging.enable_progress_bar()

    # # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module)

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias)


if __name__ == '__main__':
    train()