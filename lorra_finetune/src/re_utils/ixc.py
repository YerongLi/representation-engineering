import torch
import torch.nn.functional as F

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