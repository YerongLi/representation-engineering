import torch
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
                # print(padding_side)
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
