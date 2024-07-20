from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
# from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
# from transformers_modules.internlm-xcomposer2-vl-7b.modeling_internlm_xcomposer2 import InternLMXComposer2ForCausalLM




from .modeling_internlm_xcomposer2 import InternLMXComposer2ForCausalLM
# from .modeling_internlm_xcomposer2 import InternLM2Model
from .build_mlp import build_vision_projector, build_vision_tower
from .configuration_internlm_xcomposer2 import InternLMXcomposer2Config
from .modeling_internlm2 import (InternLM2_INPUTS_DOCSTRING, InternLM2Model,
                                 InternLM2PreTrainedModel)
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode




from transformers.models.mistral.modeling_mistral import MistralModel, MistralForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, validate_stopping_criteria
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation import GreedySearchDecoderOnlyOutput
from repe.rep_control_reading_vec import WrappedBlock
import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from functools import partial

# === should work for Llama and Mistral ===
def contrast_greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GreedySearchDecoderOnlyOutput, torch.LongTensor]:
    # print('Entering constrastive greedy search')
    # assert False, 'See Routing'
    # print('input_ids', input_ids) # DEBUG
    # ===== pop repe/contrast control args ====
    # If 'get_params' is True in model_kwargs, return all input parameters as a dictionary
    if model_kwargs.get('get_params', False):
        params = {
            'input_ids': input_ids,
            'logits_processor': logits_processor,
            'stopping_criteria': stopping_criteria,
            'max_length': max_length,
            'pad_token_id': pad_token_id,
            'eos_token_id': eos_token_id,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'output_scores': output_scores,
            'return_dict_in_generate': return_dict_in_generate,
            'synced_gpus': synced_gpus,
            'streamer': streamer,
            'model_kwargs': model_kwargs,
        }
        return params

    alpha = model_kwargs.pop('alpha', None)
    contrast_tokens = model_kwargs.pop('contrast_tokens', None)
    compute_contrast = model_kwargs.pop('compute_contrast', None)
    pos_inputs_embeds = model_kwargs.pop('pos_inputs_embeds', None)
    pos_img_mask = model_kwargs.pop('pos_img_mask', None)
    neg_inputs_embeds = model_kwargs.pop('neg_inputs_embeds', None)
    neg_img_mask = model_kwargs.pop('neg_img_mask', None)
    control_layer_ids = model_kwargs.pop('control_layer_ids', None)
    # print('self.generation_config', self.generation_config) # DEBUG

    # assert not compute_contrast or not model_kwargs.get('use_cache', False), "Contrast Greedy Search not yet support generate with use_cache, please set model.generate(**kwargs, use_cache=False)" # DEBUG
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

# show inputs None 
# show generation_config.bos_token_id 1 
# show model_kwargs {'inputs_embeds': tensor([[[ 0.0009,  0.0001,  0.0056,  ..., -0.0002,  0.0017, -0.0022],
#          [ 0.0008,  0.0005, -0.0012,  ..., -0.0016, -0.0001,  0.0011],
#          [ 0.0011, -0.0013, -0.0125,  ...,  0.0175, -0.0038, -0.0222],
#          ...,
#          [-0.0126,  0.0248, -0.0103,  ..., -0.0117, -0.0082, -0.0093],
#          [-0.0199, -0.0079,  0.0151,  ...,  0.0035,  0.0123,  0.0461],
#          [-0.0030, -0.0014, -0.0103,  ..., -0.0109,  0.0173,  0.0079]]],
#        device='cuda:0', grad_fn=<SliceBackward0>), 'im_mask': tensor([[False, False, False,  ..., False, False, False]], device='cuda:0'), 'pos_inputs_embeds': tensor([[[ 0.0009,  0.0001,  0.0056,  ..., -0.0002,  0.0017, -0.0022],
#          [ 0.0008,  0.0005, -0.0012,  ..., -0.0016, -0.0001,  0.0011],
#          [ 0.0011, -0.0013, -0.0125,  ...,  0.0175, -0.0038, -0.0222],
#          ...,
#          [-0.0126,  0.0248, -0.0103,  ..., -0.0117, -0.0082, -0.0093],
#          [-0.0199, -0.0079,  0.0151,  ...,  0.0035,  0.0123,  0.0461],
#          [-0.0030, -0.0014, -0.0103,  ..., -0.0109,  0.0173,  0.0079]]],
#        device='cuda:0', grad_fn=<SliceBackward0>), 'pos_img_mask': tensor([[False, False, False,  ..., False, False, False]], device='cuda:0'), 'neg_inputs_embeds': tensor([[[ 0.0009,  0.0001,  0.0056,  ..., -0.0002,  0.0017, -0.0022],
#          [ 0.0008,  0.0005, -0.0012,  ..., -0.0016, -0.0001,  0.0011],
#          [ 0.0011, -0.0013, -0.0125,  ...,  0.0175, -0.0038, -0.0222],
#          ...,
#          [-0.0126,  0.0248, -0.0103,  ..., -0.0117, -0.0082, -0.0093],
#          [-0.0199, -0.0079,  0.0151,  ...,  0.0035,  0.0123,  0.0461],
#          [-0.0030, -0.0014, -0.0103,  ..., -0.0109,  0.0173,  0.0079]]],
#        device='cuda:0', grad_fn=<SliceBackward0>), 'neg_img_mask': tensor([[False, False, False,  ..., False, False, False]], device='cuda:0'), 'contrast_tokens': -8, 'compute_contrast': True, 'alpha': 0, 'control_layer_ids': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], 'input_ids': tensor([[1]], device='cuda:0')}
# input_ids tensor([[1]], device='cuda:0')
# input_ids in model_kwargs False
    # ====== REPE


    inputs_p = {'inputs_embeds' : pos_inputs_embeds}

    params_p = self.generate(
        **inputs_p,
        max_new_tokens=1024,
        eos_token_id=self.generation_config.eos_token_id,
        im_mask=pos_img_mask,
        do_sample=False,
        **{'get_params' : True},
    )

    inputs_n = {'inputs_embeds' : neg_inputs_embeds}

    params_n = self.generate(
        **inputs_n,
        max_new_tokens=1024,
        im_mask=neg_img_mask,
        do_sample=False,
        **{'get_params' : True},
    )
    # ====== REPE

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # print('input embeds shape', model_kwargs['inputs_embeds'].shape, params_p['model_kwargs']['inputs_embeds'].shape)

        model_inputs_p = self.prepare_inputs_for_generation(params_p['input_ids'], **params_p['model_kwargs'])
        model_inputs_n = self.prepare_inputs_for_generation(params_n['input_ids'], **params_n['model_kwargs'])
        # print('model_inputs att shape', model_inputs['attention_mask'].shape, model_inputs_p['attention_mask'].shape)
        # forward pass to get next token
        # print('model_inputs.keys()', model_inputs.keys()) # DEBUG dict_keys(['input_ids', 'position_ids', 'past_key_values', 'use_cache', 'attention_mask', 'im_mask'])
        # print('input_ids should be None', input_ids) # DEBUG not None
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            model_inputs_p=model_inputs_p,
            model_inputs_n=model_inputs_n,
            contrast_tokens=contrast_tokens,
            compute_contrast=compute_contrast,
            alpha=alpha,
            control_layer_ids=control_layer_ids,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.attentions,)
                )

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.hidden_states,)
                )

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        return GreedySearchDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )
    else:
        return input_ids

def postprocess(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                **kwargs
               ):
        # layer_outputs = decoder_layer(
        #     hidden_states,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_value=past_key_value,
        #     output_attentions=output_attentions,
        #     use_cache=use_cache,
        #     im_mask=im_mask,
        # )   
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds')
    
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
    
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device)
            position_ids = position_ids.unsqueeze(0)
    
        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)
            im_mask = torch.zeros(inputs_embeds.shape[:2]).to(
                inputs_embeds.device).bool()
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past),
                                        dtype=torch.bool,
                                        device=inputs_embeds.device)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds,
            past_key_values_length)
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'inputs_embeds': inputs_embeds,
            'use_cache': use_cache,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'batch_size': batch_size,
            'seq_length': seq_length,
            'seq_length_with_past': seq_length_with_past,
            'past_key_values_length': past_key_values_length,
        }
        
        if 'im_mask' in locals():
            result['im_mask'] = im_mask
        
        return result


def forward_contrast_vector(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                # ==== repe coeff ====
                alpha = None,
                contrast_tokens: int = None,
                compute_contrast: bool = False,
                model_inputs_p = None,
                model_inputs_n = None ,
                control_layer_ids: List[int] = [],
                pad_right: int = 0,
                **kwargs) -> Union[Tuple, BaseModelOutputWithPast]:
 
        im_mask = kwargs.get('im_mask', None)
        # print('model_inputs_p')
        # print(model_inputs_p)
        # print('model_inputs_n')
        # print(model_inputs_n)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError(
        #         'You cannot specify both input_ids and inputs_embeds at the same time'
        #     )
        # elif input_ids is not None:
        #     batch_size, seq_length = input_ids.shape[:2]
        # elif inputs_embeds is not None:
        #     batch_size, seq_length = inputs_embeds.shape[:2]
        # else:
        #     raise ValueError(
        #         'You have to specify either input_ids or inputs_embeds')

        # seq_length_with_past = seq_length
        # past_key_values_length = 0
        # if past_key_values is not None:
        #     past_key_values_length = past_key_values[0][0].shape[2]
        #     seq_length_with_past = seq_length_with_past + past_key_values_length

        # if position_ids is None:
        #     device = input_ids.device if input_ids is not None else inputs_embeds.device
        #     position_ids = torch.arange(
        #         past_key_values_length,
        #         seq_length + past_key_values_length,
        #         dtype=torch.long,
        #         device=device)
        #     position_ids = position_ids.unsqueeze(0)

        # if inputs_embeds is None:
        #     inputs_embeds = self.tok_embeddings(input_ids)
        #     im_mask = torch.zeros(inputs_embeds.shape[:2]).to(
        #         inputs_embeds.device).bool()
        # # embed positions
        # if attention_mask is None:
        #     attention_mask = torch.ones((batch_size, seq_length_with_past),
        #                                 dtype=torch.bool,
        #                                 device=inputs_embeds.device)
        # attention_mask = self._prepare_decoder_attention_mask(
        #     attention_mask, (batch_size, seq_length), inputs_embeds,
        #     past_key_values_length)

        # embed positions
        result=postprocess(self,
                input_ids = input_ids,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_values = past_key_values,
                inputs_embeds = inputs_embeds,
                use_cache = use_cache,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states
               )
        input_ids = result.get('input_ids')
        attention_mask = result.get('attention_mask')
        position_ids = result.get('position_ids')
<<<<<<< HEAD
=======
        # past_key_values = result.get('past_key_values')
>>>>>>> 962a80fe5ee3dd0486e4f65d8ba8fef35ddecb29
        inputs_embeds = result.get('inputs_embeds')
        batch_size = result.get('batch_size')
        seq_length = result.get('seq_length')

        if 'im_mask' in result:
            im_mask = result.get('im_mask')
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        # print('pos_inputs_embeds.shape', pos_inputs_embeds.shape)
        # print('inputs_embeds.shape', inputs_embeds.shape)
        # inputs_embeds.shape torch.Size([1, 1, 4096])
        # pos_inputs_embeds.shape torch.Size([1, 1272, 4096])
        # inputs_embeds.shape torch.Size([1, 1, 4096])
        # pos_inputs_embeds.shape torch.Size([1, 1272, 4096])
        # inputs_embeds.shape torch.Size([1, 1, 4096])
        # pos_inputs_embeds.shape torch.Size([1, 1272, 4096])
        # inputs_embeds.shape torch.Size([1, 1, 4096])
        # pos_inputs_embeds.shape torch.Size([1, 1272, 4096])
        # inputs_embeds.shape torch.Size([1, 1, 4096])
        # pos_inputs_embeds.shape torch.Size([1, 1272, 4096])
        # inputs_embeds.shape torch.Size([1, 1, 4096])



        activations = None
        if compute_contrast:
            # ======== REPE Compute repe =========    
            hidden_states_p = model_inputs_p['inputs_embeds']  
            hidden_states_n = model_inputs_n['inputs_embeds']  

            # results = postprocess(self, **model_inputs_p)
            # input_ids_p = result.get('input_ids')
            # attention_mask_p = result.get('attention_mask')
            # position_ids_p = result.get('position_ids')
            # past_key_values_p = result.get('past_key_values')
            # inputs_embeds_p = result.get('inputs_embeds')
            # use_cache_p = result.get('use_cache')
            # output_attentions_p = result.get('output_attentions')
            # output_hidden_states_p = result.get('output_hidden_states')
            # batch_size_p = result.get('batch_size')
            # seq_length_p = result.get('seq_length')
            # seq_length_with_past_p = result.get('seq_length_with_past')
            # past_key_values_length_p = result.get('past_key_values_length')
            
            # results = postprocess(self, **model_inputs_n)
            # input_ids_n = result.get('input_ids')
            # attention_mask_n = result.get('attention_mask')
            # position_ids_n = result.get('position_ids')
            # past_key_values_n = result.get('past_key_values')
            # inputs_embeds_n = result.get('inputs_embeds')
            # use_cache_n = result.get('use_cache')
            # output_attentions_n = result.get('output_attentions')
            # output_hidden_states_n = result.get('output_hidden_states')
            # batch_size_n = result.get('batch_size')
            # seq_length_n = result.get('seq_length')
            # seq_length_with_past_n = result.get('seq_length_with_past')
            # past_key_values_length_n = result.get('past_key_values_length')
            # # print('attention Shape', attention_mask.shape, attention_mask_p.shape)
            # position_ids_p = model_inputs_p['position_ids']
            # position_ids_n = model_inputs_n['position_ids']
            # ======== REPE Compute repe ========= 
    
        for idx, decoder_layer in enumerate(self.layers):
            # print('hidden_states')
            # print(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            past_key_value = past_key_values[
                idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                print('gradient_checkpointing branch')
                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None,
                                      im_mask)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    im_mask=im_mask,
                )

            hidden_states = layer_outputs[0]
            
            # 1. if layer in target layer, we recompute activations and add
            # 2. else will add previous computed activations
            if compute_contrast and False:
                # ======== Compute activations for POS/NEG layers =========  
                if self.gradient_checkpointing and self.training:
                    print('gradient_checkpointing branch')
                    def create_custom_forward(module):
    
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, None,
                                          im_mask)
    
                        return custom_forward
    
                    layer_outputs_p = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states_p,
                        attention_mask_p,
                        position_ids_p,
                        None,
                    )
                else:
                    layer_outputs_p = decoder_layer(
                        hidden_states_p,
                        attention_mask=attention_mask_p,
                        position_ids=position_ids_p,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        im_mask=model_inputs_p['im_mask'],
                    )
    
                hidden_states_p = layer_outputs_p[0]
                
                if self.gradient_checkpointing and self.training:
                    print('gradient_checkpointing branch')
                    def create_custom_forward(module):
    
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, None,
                                          im_mask)
    
                        return custom_forward
    
                    layer_outputs_n = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states_n,
                        attention_mask_n,
                        position_ids_n,
                        None,
                    )
                else:
                    layer_outputs_n = decoder_layer(
                        hidden_states_n,
                        attention_mask=attention_mask_n,
                        position_ids=position_ids_n,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        im_mask=model_inputs_n['im_mask'],
                    )
    
                hidden_states_n = layer_outputs_n[0]  
                # ======== Compute activations for POS/NEG layers =========  
                
                # ======== Perturbate the Layers =========    
                if idx in control_layer_ids:
                    pass
                # ======== Perturbate the Layers =========    

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1], )

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

def _always_true(self, *args, **kwargs):
    return True
    
# class ContrastVecMistralForCausalLM(MistralForCausalLM):
#     def __init__(self, config):
#         nn.Module.__init__(self)
#         self.config = config
#         self.model = MistralModel(config)
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()
        

#         # override current forward and generation functions
#         self.model.forward = partial(forward_contrast_vector, self.model)
#         self.greedy_search = partial(contrast_greedy_search, self)
#         # turn off _validate_model_kwargs, TODO: implement _validate_model_kwargs
#         self._validate_model_kwargs = partial(_always_true, self)

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         # ===== repe ====
#         **repe_kwargs
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
    
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             **repe_kwargs,
#         )

#         hidden_states = outputs[0]
#         logits = self.lm_head(hidden_states)
#         logits = logits.float()

#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#TODO
# InternLMXComposer2ForCausalLM
# class ContrastVecInternlmForCausalLM(LlamaForCausalLM):

class ContrastVecInternlmForCausalLM(InternLMXComposer2ForCausalLM):
    _auto_class = 'AutoModelForCausalLM'

    _tied_weights_keys = ['output.weight']
    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        self.tokenizer = None

        self.max_length = config.max_length
        print(f'Set max length to {self.max_length}')
        # Initialize weights and apply final processing
        self.post_init()

        self.vit = build_vision_tower()
        self.vision_proj = build_vision_projector()

        self.vis_processor = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        """ Initialization"""
        # Initialize weights and apply final processing
        self.post_init()

        # override current forward and generation functions
        self.model.forward = partial(forward_contrast_vector, self.model)
        self.greedy_search = partial(contrast_greedy_search, self)
        # turn off _validate_model_kwargs, TODO: implement _validate_model_kwargs
        self._validate_model_kwargs = partial(_always_true, self)

    def forward(self,
                    input_ids: torch.LongTensor = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_values: Optional[List[torch.FloatTensor]] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    labels: Optional[torch.LongTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None,
                    
                # ==== repe ====
                    **repe_kwargs
               ) -> Union[Tuple, CausalLMOutputWithPast]:
            r"""
            Args:
                labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                    Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                    config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                    (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            Returns:
            """
            # print('attention_mask input', attention_mask) # DEBUG attention_mask input tensor([[1, 1, 1,  ..., 1, 1, 1]], device='cuda:0')
            # assert False, 'see routing'
            samples = repe_kwargs.get('samples', None)
            if samples:
                if samples['data_type'][0] == 'text':
                    has_img = False
                elif samples['data_type'][0] == 'multi':
                    has_img = True
                else:
                    raise NotImplementedError
    
                # encode text
                text = samples['text_input']
                # encode image
                if has_img:
                    image = samples['image']
                    to_regress_embeds, attention_mask, targets, im_mask = self.interleav_wrap(
                        image, text)
                else:
                    to_regress_tokens, targets = self.text2emb(
                        text, add_special=True)
                    to_regress_embeds = self.model.tok_embeddings(
                        to_regress_tokens.input_ids)
                    print('attention_mask = to_regress_tokens.attention_mask')
                    attention_mask = to_regress_tokens.attention_mask
                    im_mask = torch.zeros(to_regress_embeds.shape[:2]).cuda()
    
                inputs_embeds = to_regress_embeds[:, :self.max_length]
                attention_mask = attention_mask[:, :self.max_length]
                targets = targets[:, :self.max_length]
                im_mask = im_mask[:, :self.max_length].bool()
                labels = targets
            else:
                im_mask = repe_kwargs.get('im_mask', None)
                if im_mask is None and inputs_embeds is not None:
                    im_mask = torch.zeros(inputs_embeds.shape[:2]).to(
                        inputs_embeds.device)
                    im_mask = im_mask.bool()
    
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else
                self.config.output_hidden_states)
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

  
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                # im_mask=im_mask,
                **repe_kwargs
            )
    
            hidden_states = outputs[0]
            logits = self.output(hidden_states)
            logits = logits.float()
    
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
    
            if not return_dict:
                output = (logits, ) + outputs[1:]
                return (loss, ) + output if loss is not None else output
    
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )