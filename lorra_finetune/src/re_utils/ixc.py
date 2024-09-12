import torch
import torch.nn.functional as F
import numpy as np

import gc
import logging

# Configure logging
logging.basicConfig(filename='logging.txt', level=logging.WARNING, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def custom_interleav_wrap(self, img_list, text_list, padding_side='left', set_length=2048):
    wrap_embeds_list, wrap_atts_list = [], []
    wrap_target_list, wrap_im_mask_list = [], []
    
    for image, text in zip(img_list, text_list):
        img_embeds, atts_img, img_target = self.img2emb(image)
        text = text[0]
        parts = text.split('<ImageHere>')
        # print('text ==== ')

        # print(text)
        # print(parts)

        wrap_tokens, wrap_embeds, wrap_atts, wrap_im_mask = [], [], [], []
        temp_len = 0
        image_nums, im_len = img_embeds.shape[:2]
        need_bos = True
        for idx, part in enumerate(parts):
            if len(part) > 0:
                part_tokens = self.tokenizer(
                    part,
                    return_tensors='pt',
                    padding='longest',
                    add_special_tokens=need_bos).to(self.device)
                if need_bos:
                    need_bos = False
                wrap_tokens.append(part_tokens.input_ids)
                part_embeds = self.model.get_input_embeddings()(
                    part_tokens.input_ids)
                wrap_embeds.append(part_embeds)
                wrap_atts.append(part_tokens.attention_mask)
                wrap_im_mask.append(
                    torch.zeros(part_embeds.shape[:2]).to(self.device))

                temp_len += part_embeds.shape[1]
            if idx < image_nums:
                wrap_tokens.append(img_target[idx].unsqueeze(0))
                wrap_embeds.append(img_embeds[idx].unsqueeze(0))
                wrap_atts.append(atts_img[idx].unsqueeze(0))
                wrap_im_mask.append(
                    torch.ones_like(atts_img[idx].unsqueeze(0)))

                temp_len += im_len
            if temp_len > set_length:
                logging.warning(f"set_length is too small. set_length: {set_length}, temp_len: {temp_len}")
                
                break

        wrap_tokens = torch.cat(wrap_tokens, dim=1)
        wrap_embeds = torch.cat(wrap_embeds, dim=1)
        wrap_atts = torch.cat(wrap_atts, dim=1)
        wrap_im_mask = torch.cat(wrap_im_mask, dim=1)

        wrap_target = self.mask_human_targets(wrap_tokens).to(self.device)

        wrap_embeds = wrap_embeds[:, :set_length].to(self.device)
        wrap_atts = wrap_atts[:, :set_length].to(self.device)
        wrap_target = wrap_target[:, :set_length].to(self.device)
        wrap_im_mask = wrap_im_mask[:, :set_length].to(self.device)


        if wrap_embeds.shape[1] < set_length:
            reslen = set_length - wrap_embeds.shape[1]
            # padding
            pad = torch.ones([1, 1]) * self.tokenizer.pad_token_id
            pad = pad.long().to(self.device)
            pad_emb = self.model.get_input_embeddings()(pad)
    
            if padding_side == 'right':
                # Right padding
                wrap_embeds = torch.cat([wrap_embeds, pad_emb.repeat(1, reslen, 1)], dim=1)
                wrap_atts = torch.cat([torch.ones(1, temp_len), torch.zeros(1, reslen)], dim=1).to(wrap_atts.dtype).to(self.device)
                wrap_target = torch.cat([wrap_target, (torch.ones(1, reslen)*-100).to(wrap_target.dtype).to(self.device)], dim=1)
                wrap_im_mask = torch.cat([wrap_im_mask, (torch.zeros(1, reslen)).to(wrap_target.dtype).to(self.device)], dim=1)
            elif padding_side == 'left':
                # Left padding
                wrap_embeds = torch.cat([pad_emb.repeat(1, reslen, 1), wrap_embeds], dim=1)
                wrap_atts = torch.cat([torch.zeros(1, reslen), torch.ones(1, temp_len)], dim=1).to(wrap_atts.dtype).to(self.device)
                wrap_target = torch.cat([(torch.ones(1, reslen)*-100).to(wrap_target.dtype).to(self.device), wrap_target], dim=1)
                wrap_im_mask = torch.cat([(torch.zeros(1, reslen)).to(wrap_target.dtype).to(self.device), wrap_im_mask], dim=1)
            
        wrap_embeds_list.append(wrap_embeds)
        wrap_atts_list.append(wrap_atts)
        wrap_target_list.append(wrap_target)
        wrap_im_mask_list.append(wrap_im_mask)

    wrap_embeds = torch.cat(wrap_embeds_list)
    wrap_atts = torch.cat(wrap_atts_list)
    wrap_target = torch.cat(wrap_target_list)
    wrap_im_mask = torch.cat(wrap_im_mask_list)
    # print(' ====== ')
    # print("Shape of wrap_embeds:", wrap_embeds.shape)
    # print("Shape of wrap_atts:", wrap_atts.shape)
    # print("Shape of wrap_target:", wrap_target.shape)
    # print("Shape of wrap_im_mask:", wrap_im_mask.shape)
    # assert wrap_embeds.shape[1] == set_length, " Shape does not match"
    # assert wrap_atts.shape[1] == set_length, " Shape does not match"
    # assert wrap_target.shape[1] == set_length, " Shape does not match"
    # assert wrap_im_mask.shape[1] == set_length, " Shape does not match"
    return wrap_embeds, wrap_atts, wrap_target, wrap_im_mask

def custom_interleav_wrap25(self, img_list, text_list, image_nums, padding='right', set_length=None):
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
                part_embeds = self.model.model.get_input_embeddings()(part_tokens.input_ids)
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
    

    if set_length is not None:
        if set_length < temp_max_len:
            logging.warning(f"set_length is too small. set_length: {set_length}, temp_max_len: {temp_max_len}")
        temp_max_len = set_length
        
    final_input, final_atts, final_tars, final_mask = [], [], [], []
    pad = torch.ones([1, 1]) * self.tokenizer.pad_token_id
    pad = pad.long().to(self.device)
    pad_emb = self.model.model.get_input_embeddings()(pad)
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
    pad_token_embedding = self.model.get_input_embeddings()(torch.tensor([pad_token_id]).to(to_regress_embeds.device)).squeeze(0)
    
    batch_size = to_regress_embeds.shape[0]
    
    for idx in range(batch_size):
        embeds = to_regress_embeds[idx]
        mask = attention_mask[idx]
        
        # Calculate cosine similarity between embeddings and pad_token_embedding
        pad_token_embedding_expanded = pad_token_embedding.unsqueeze(0).expand_as(embeds)
        similarities = F.cosine_similarity(embeds, pad_token_embedding_expanded, dim=-1)
        
        # Determine padding in embeddings
        padding_threshold = 0.98  # Threshold for considering an embedding as padding
        is_padding = similarities > padding_threshold
        is_padding_int = is_padding.int()  # Convert boolean tensor to integer tensor
        
        # Find the first location where padding starts
        if is_padding.any():
            padding_start_embed = (is_padding_int.nonzero(as_tuple=False).min() if is_padding.any() else len(similarities)).item()
        else:
            padding_start_embed = len(similarities)
            
        padding_length_embed = len(similarities) - padding_start_embed
        
        # Determine padding from the attention mask
        mask_list = mask.tolist()
        padding_start_mask = mask_list.index(0) if 0 in mask_list else len(mask_list)
        padding_length_mask = len(mask_list) - padding_start_mask

        # IXC does not have a </s> as End-of-Sentenve token
        # Assertions to ensure padding consistency
        if padding_length_mask == 0:
            assert padding_length_embed == 0, f"Expected padding length in embeddings to be 0 or 1 but got {padding_length_embed}."
        else:
            assert padding_start_embed == padding_start_mask, \
                f"Expected padding start in embeddings to be {padding_start_mask - 1} but got {padding_start_embed}."

def check_left_padding_with_embeddings(self, to_regress_embeds, attention_mask):
    """
    Check if left-side non-padding in `to_regress_embeds` matches the left-side non-padding in `attention_mask` using cosine similarity.
    """
    pad_token_id = self.tokenizer.pad_token_id
    pad_token_embedding = self.model.get_input_embeddings()(torch.tensor([pad_token_id]).to(to_regress_embeds.device)).squeeze(0)
    
    batch_size = to_regress_embeds.shape[0]
    
    for idx in range(batch_size):
        embeds = to_regress_embeds[idx]
        mask = attention_mask[idx]
        
        # Calculate cosine similarity between embeddings and pad_token_embedding
        pad_token_embedding_expanded = pad_token_embedding.unsqueeze(0).expand_as(embeds)
        similarities = F.cosine_similarity(embeds, pad_token_embedding_expanded, dim=-1)
        
        # Determine non-padding in embeddings
        padding_threshold = 0.99  # Threshold for considering an embedding as padding
        is_padding = similarities > padding_threshold
        is_padding_int = is_padding.int()  # Convert boolean tensor to integer tensor
        
        if 0 in is_padding_int:
            first_non_padding_idx_embed = torch.argmin(is_padding_int).item()
        else:
            first_non_padding_idx_embed = len(is_padding_int)  # All are padding
        
        # Determine the first non-padding location from the attention mask
        mask_list = mask.tolist()
        first_non_padding_idx_mask = mask_list.index(1) if 1 in mask_list else len(mask_list)

        
        # Assertions to ensure non-padding consistency
        assert first_non_padding_idx_embed == first_non_padding_idx_mask, \
            f"Expected non-padding start in embeddings to be {first_non_padding_idx_mask} but got {first_non_padding_idx_embed}."
        
        # Check padding lengths
        padding_length_embed = len(similarities) - first_non_padding_idx_embed
        padding_length_mask = len(mask_list) - first_non_padding_idx_mask
        assert padding_length_embed == padding_length_mask, \
            f"Expected padding length in embeddings to be {padding_length_mask} but got {padding_length_embed}."