from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from PIL import Image
import numpy as np
import random
import torch.nn.functional as F

from torch import nn

from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import (add_start_docstrings_to_model_forward,
                                replace_return_docstrings)
def custom_interleav_wrap(self, img_list, text_list, image_nums, padding='right'):
    '''
    @image_nums is an input list that indicates the number of images associated with each text entry in the text_list. 
    This list helps the function understand how many images correspond to each piece of text and manage the interleaving of images and text accordingly.
    '''
    temp_embeds = []
    temp_im_mask = []
    temp_tars = []

    
    # Encode images using the vision transformer (ViT) model and project the embeddings
    img_embeds, img_split = self.vit(img_list, self.plora_glb_GN, self.plora_sub_GN)
    img_embeds = self.vision_proj(img_embeds)

    
    # Only process the first element of text_list (assuming it's a list of lists or a nested list)
    text_list = text_list[0]
    for idx, text in enumerate(text_list):
        # print(' **** ')
        # print(text)
        # print(' **** ')

        image_num = image_nums[idx]
        im_id = int(np.sum(image_nums[:idx]))
        images = []
        for i in range(image_nums[idx]):
            st = int(np.sum(img_split[:im_id + i]))
            sp = img_split[im_id + i]
            temp_img = img_embeds[:, st:st+sp]
            images.append(temp_img)
        atts_img = torch.ones((len(images), images[0].shape[1]), dtype=torch.long).to(self.device)
        img_target = torch.ones(
            (len(images), images[0].shape[1]), dtype=torch.long).to(
                self.device) * -100

        if image_num == 1 and text.find('<ImageHere>') == -1:
            text = '<ImageHere>' + text
        parts = text.split('<ImageHere>')

        wrap_tokens, wrap_embeds, wrap_im_mask = [], [], []
        temp_len = 0
        need_bos = True
        for idx, part in enumerate(parts):
            if len(part) > 0:
                part_tokens = self.tokenizer(part, return_tensors='pt', padding='longest',
                                             add_special_tokens=need_bos).to(self.device)
                if need_bos:
                    need_bos = False
                wrap_tokens.append(part_tokens.input_ids)
                part_embeds = self.model.tok_embeddings(part_tokens.input_ids)
                wrap_embeds.append(part_embeds)
                wrap_im_mask.append(torch.zeros(part_embeds.shape[:2]).to(self.device))
                temp_len += part_embeds.shape[1]
            if idx < image_num:
                wrap_embeds.append(images[idx])
                wrap_token = torch.ones(images[idx].shape[:2], dtype=torch.long).to(self.device) * -100
                wrap_tokens.append(wrap_token)
                wrap_im_mask.append(torch.ones(images[idx].shape[:2]).to(self.device))
                temp_len += images[idx].shape[1]
            if temp_len > self.max_length:
                break
        wrap_tokens = torch.cat(wrap_tokens, dim=1)
        wrap_embeds = torch.cat(wrap_embeds, dim=1)
        wrap_im_mask = torch.cat(wrap_im_mask, dim=1)

        wrap_target = self.mask_human_targets(wrap_tokens).to(self.device)

        temp_embeds.append(wrap_embeds)
        temp_im_mask.append(wrap_im_mask)
        temp_tars.append(wrap_target)

    temp_max_len = np.max([i.shape[1] for i in temp_embeds])
    # Max length
    temp_max_len = min(temp_max_len, self.max_length)

    final_input, final_atts, final_tars, final_mask = [], [], [], []
    pad = torch.ones([1, 1]) * self.tokenizer.pad_token_id
    pad = pad.long().to(self.device)
    pad_emb = self.model.tok_embeddings(pad)
    for idx in range(len(temp_embeds)):
        temp_len = temp_embeds[idx].shape[1]
        if temp_len >= temp_max_len:
            final_input.append(temp_embeds[idx][:, :temp_max_len])
            final_atts.append(torch.ones(1, temp_max_len).to(wrap_target.dtype).to(self.device))
            final_tars.append(temp_tars[idx][:, :temp_max_len])
            final_mask.append(temp_im_mask[idx][:, :temp_max_len])
            
        else:
            if padding == 'right':
                # Right padding
                final_input.append(torch.cat([temp_embeds[idx], pad_emb.repeat(1, temp_max_len-temp_len, 1)], dim=1))
                final_atts.append(torch.cat([torch.ones(1, temp_len), torch.zeros(1, temp_max_len-temp_len)], dim=1).to(wrap_target.dtype).to(self.device))
                final_tars.append(torch.cat([temp_tars[idx], (torch.ones(1, temp_max_len-temp_len)*-100).to(wrap_target.dtype).to(self.device)], dim=1))
                final_mask.append(torch.cat([temp_im_mask[idx], (torch.zeros(1, temp_max_len-temp_len)).to(wrap_target.dtype).to(self.device)], dim=1))
            elif padding == 'left':
                # Left padding
                final_input.append(torch.cat([pad_emb.repeat(1, temp_max_len-temp_len, 1), temp_embeds[idx]], dim=1))
                final_atts.append(torch.cat([torch.zeros(1, temp_max_len-temp_len), torch.ones(1, temp_len)], dim=1).to(wrap_target.dtype).to(self.device))
                final_tars.append(torch.cat([(torch.ones(1, temp_max_len-temp_len)*-100).to(wrap_target.dtype).to(self.device), temp_tars[idx]], dim=1))
                final_mask.append(torch.cat([(torch.zeros(1, temp_max_len-temp_len)).to(wrap_target.dtype).to(self.device), temp_im_mask[idx]], dim=1))



    inputs_embeds = torch.cat(final_input, dim=0)
    attention_mask = torch.cat(final_atts, dim=0)
    targets = torch.cat(final_tars, dim=0)
    im_mask = torch.cat(final_mask, dim=0)

    return inputs_embeds, attention_mask, targets, im_mask
    
def check_right_padding_with_embeddings(self, to_regress_embeds, attention_mask):
    """
    Check if padding in `to_regress_embeds` matches the right side padding in `attention_mask` using cosine similarity.
    """
    pad_token_id = self.tokenizer.pad_token_id
    pad_token_embedding = self.model.tok_embeddings(torch.tensor([pad_token_id]).to(to_regress_embeds.device)).squeeze(0)
    
    batch_size = to_regress_embeds.shape[0]
    
    for idx in range(batch_size):
        embeds = to_regress_embeds[idx]
        mask = attention_mask[idx]
        
        # Calculate cosine similarity between embeddings and pad_token_embedding
        pad_token_embedding_expanded = pad_token_embedding.unsqueeze(0).expand_as(embeds)
        similarities = F.cosine_similarity(embeds, pad_token_embedding_expanded, dim=-1)
        
        # Determine padding in embeddings
        padding_threshold = 0.99  # Threshold for considering an embedding as padding
        is_padding = similarities > padding_threshold
        is_padding_int = is_padding.int()  # Convert boolean tensor to integer tensor
        
        # Find the first location where padding starts
        if is_padding.any():
            padding_start_embed = torch.argmax(is_padding_int).item()
        else:
            padding_start_embed = len(similarities)
            
        padding_length_embed = len(similarities) - padding_start_embed
        
        # Determine padding from the attention mask
        mask_list = mask.tolist()
        padding_start_mask = mask_list.index(0) if 0 in mask_list else len(mask_list)
        padding_length_mask = len(mask_list) - padding_start_mask
        
        # Assertions to ensure padding consistency
        if padding_length_mask == 0:
            assert padding_length_embed == 0 or padding_length_embed == 1, f"Expected padding length in embeddings to be 0 but got {padding_length_embed}."
        else:
            assert padding_start_embed == padding_start_mask - 1, \
                f"Expected padding start in embeddings to be {padding_start_mask - 1} but got {padding_start_embed}."
            assert padding_length_embed == padding_length_mask + 1, \
                f"Expected padding length in embeddings to be {padding_length_mask + 1} but got {padding_length_embed}."

def check_left_padding_with_embeddings(self, to_regress_embeds, attention_mask):
    """
    Check if left-side non-padding in `to_regress_embeds` matches the left-side non-padding in `attention_mask` using cosine similarity.
    """
    pad_token_id = self.tokenizer.pad_token_id
    pad_token_embedding = self.model.tok_embeddings(torch.tensor([pad_token_id]).to(to_regress_embeds.device)).squeeze(0)
    
    batch_size = to_regress_embeds.shape[0]
    
    for idx in range(batch_size):
        embeds = to_regress_embeds[idx]
        mask = attention_mask[idx]
        
        # Calculate cosine similarity between embeddings and pad_token_embedding
        pad_token_embedding_expanded = pad_token_embedding.unsqueeze(0).expand_as(embeds)
        similarities = F.cosine_similarity(embeds, pad_token_embedding_expanded, dim=-1)
        
        # Determine padding in embeddings
        padding_threshold = 0.99  # Threshold for considering an embedding as padding
        is_padding = similarities < padding_threshold
        is_padding_int = is_padding.int()  # Convert boolean tensor to integer tensor
        
        # Find the first location where non-padding starts
        if not is_padding_int.all():
            padding_start_embed = torch.argmax(is_padding_int).item()
        else:
            padding_start_embed = len(similarities)  # All are padding
        
        # Determine the first non-padding location from the attention mask
        mask_list = mask.tolist()
        padding_start_mask = mask_list.index(1) if 1 in mask_list else len(mask_list)
        
        # Assertions to ensure non-padding consistency
        assert padding_start_embed == padding_start_mask, \
            f"Expected non-padding start in embeddings to be {padding_start_mask} but got {padding_start_embed}."
        
        # Check padding lengths
        padding_length_embed = len(similarities) - padding_start_embed
        padding_length_mask = len(mask_list) - padding_start_mask
        assert padding_length_embed == padding_length_mask, \
            f"Expected padding length in embeddings to be {padding_length_mask} but got {padding_length_embed}."

# @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
# @replace_return_docstrings(
        # output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def custom_forward(self,
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
            **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    Returns:
    """

    samples = kwargs.get('samples', None)
    if samples:
        infer_mode = samples.get('infer_mode', 'base')
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
            image = samples['image'][0]
            bs = len(samples['text_input'][0])
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
            assert type(image) is list and len(image_nums) == bs
            padding_side='left'
            to_regress_embeds, attention_mask, targets, im_mask = self.interleav_wrap(
                image, text, image_nums, padding_side)
            # self.check_right_padding_with_embeddings(to_regress_embeds, attention_mask)
            # self.check_left_padding_with_embeddings(to_regress_embeds, attention_mask)

        else:
            to_regress_tokens, targets = self.text2emb(
                text, add_special_tokens=True)
            to_regress_embeds = self.model.tok_embeddings(
                to_regress_tokens.input_ids)
            attention_mask = to_regress_tokens.attention_mask
            im_mask = torch.zeros(to_regress_embeds.shape[:2]).cuda()

        inputs_embeds = to_regress_embeds[:, :self.max_length]
        attention_mask = attention_mask[:, :self.max_length]
        targets = targets[:, :self.max_length]
        im_mask = im_mask[:, :self.max_length].bool()
        labels = targets

    else:
        im_mask = kwargs.get('im_mask', None)
        infer_mode = kwargs.get('infer_mode', 'base')
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
        im_mask=im_mask,
        infer_mode=infer_mode,
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