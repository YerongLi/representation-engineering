import torch
import gc
import logging
import numpy as np
# Configure logging
logging.basicConfig(filename='logging.txt', level=logging.WARNING, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def custom_interleav_wrap(self, img_list, text_list, image_nums, padding_side='left', set_length=1024):
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
                part_embeds = self.model.get_input_embeddings()(part_tokens.input_ids)
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
    pad_emb = self.model.get_input_embeddings()(pad)
    for idx in range(len(temp_embeds)):
        temp_len = temp_embeds[idx].shape[1]
        if temp_len >= temp_max_len:
            final_input.append(temp_embeds[idx][:, :temp_max_len])
            final_atts.append(torch.ones(1, temp_max_len).to(wrap_target.dtype).to(self.device))
            final_tars.append(temp_tars[idx][:, :temp_max_len])
            final_mask.append(temp_im_mask[idx][:, :temp_max_len])
            
        else:
            if padding_side == 'right':
                # Right padding
                final_input.append(torch.cat([temp_embeds[idx], pad_emb.repeat(1, temp_max_len-temp_len, 1)], dim=1))
                final_atts.append(torch.cat([torch.ones(1, temp_len), torch.zeros(1, temp_max_len-temp_len)], dim=1).to(wrap_target.dtype).to(self.device))
                final_tars.append(torch.cat([temp_tars[idx], (torch.ones(1, temp_max_len-temp_len)*-100).to(wrap_target.dtype).to(self.device)], dim=1))
                final_mask.append(torch.cat([temp_im_mask[idx], (torch.zeros(1, temp_max_len-temp_len)).to(wrap_target.dtype).to(self.device)], dim=1))
            elif padding_side == 'left':
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
