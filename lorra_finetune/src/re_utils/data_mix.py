import random

import numpy as np
import torch
from ixc_utils import HD_transform
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import transformers
from .templates import template_dict
# ## Templates
# orig_template = "{user_tag}{user_prefix}{instruction}{user_end}{bot_prefix}{assistant_tag}{response}{bot_end}"
# # control_templates = [
# #     # "Pretend you're a {type} person giving a response.", 
# #     # "Make your response as {type} as possible.",
# #     # "Give a response that is {type}.",
# #     # "Generate a response in a {type} way.",
# # ]
# # user_tag should be empty
# pos_template = "[UNUSED_TOKEN_146]system\n{type}[UNUSED_TOKEN_145]\n{user_tag}{user_prefix}{instruction}{user_end}{bot_prefix}{assistant_tag}{response}{bot_end}"
# neg_template = "[UNUSED_TOKEN_146]system\n{type}[UNUSED_TOKEN_145]\n{user_tag}{user_prefix}{instruction}{user_end}{bot_prefix}{assistant_tag}{response}{bot_end}"

# # max_res_len = 64
# USR_PREFIX = '[UNUSED_TOKEN_146]user\n'
# BOT_PREFIX = '[UNUSED_TOKEN_146]assistant\n'

# END_HUMAN = '[UNUSED_TOKEN_145]\n'
# END_BOT = '[UNUSED_TOKEN_145]\n'

def get_truncated_outputs(all_outputs, prefixes, num_examples, user_tag, assistant_tag, pos_type, neg_type, control_template, template_system='ixc_system'):
    orig_s, pos_s, neg_s = [], [], []
    env = template_dict[template_system]

    for s, p in zip(all_outputs, prefixes):
        orig_s.append(env['orig_template'].format(
            user_tag=user_tag, user_prefix=env['USR_PREFIX'],
            assistant_tag=assistant_tag, bot_prefix=env['BOT_PREFIX'],
            instruction=p, response=s,
            user_end=env['END_HUMAN'], bot_end=env['END_BOT']))

        pos_s.append(env['pos_template'].format(
            user_tag=user_tag, user_prefix=env['USR_PREFIX'],
            assistant_tag=assistant_tag, bot_prefix=env['BOT_PREFIX'],
            instruction=p, type=control_template.format(type=pos_type), response=s,
            user_end=env['END_HUMAN'], bot_end=env['END_BOT']))

        neg_s.append(env['neg_template'].format(
            user_tag=user_tag, user_prefix=env['USR_PREFIX'],
            assistant_tag=assistant_tag, bot_prefix=env['BOT_PREFIX'],
            instruction=p, type=control_template.format(type=neg_type), response=s,
            user_end=env['END_HUMAN'], bot_end=env['END_BOT']))

    return orig_s, pos_s, neg_s

    # for s, p in zip(all_outputs, prefixes):
    #     orig_s.append(orig_template.format(
    #         user_tag=user_tag, user_prefix=USR_PREFIX,
    #         assistant_tag=assistant_tag, bot_prefix=BOT_PREFIX,
    #         instruction=p, response=s,
    #         user_end=END_HUMAN, bot_end=END_BOT))

    #     pos_s.append(pos_template.format(
    #         user_tag=user_tag, user_prefix=USR_PREFIX,
    #         assistant_tag=assistant_tag, bot_prefix=BOT_PREFIX,
    #         instruction=p, type=control_template.format(type=pos_type), response=s,
    #         user_end=END_HUMAN, bot_end=END_BOT))

    #     neg_s.append(neg_template.format(
    #         user_tag=user_tag, user_prefix=USR_PREFIX,
    #         assistant_tag=assistant_tag, bot_prefix=BOT_PREFIX,
    #         instruction=p, type=control_template.format(type=neg_type), response=s,
    #         user_end=END_HUMAN, bot_end=END_BOT))

    # return orig_s, pos_s, neg_s

class Mix_dataset(Dataset):

    def __init__(self,
                 json_datas,
                 batch_size=1,
                 img_size=224,
                 local_rank=0,
                 hd_num=-1):
        """vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file."""
        super().__init__()
        print(f'init mix data at rank {local_rank}')
        self.datasets_text, self.datasets_multi = [], []
        self.data_num_text, self.data_num_multi = [], []

        self.batch_size = batch_size
        self.set_seed = False
        self.local_rank = local_rank
        for _, d in json_datas.items():
            if 'image' in d[0].keys():
                has_img = True
            else:
                has_img = False
            sub_data_set = Sample_dataset(
                d,
                batch_size,
                has_img=has_img,
                img_size=img_size,
                hd_num=hd_num)
            if has_img:
                self.datasets_multi.append(sub_data_set)
                self.data_num_multi.append(len(sub_data_set))
            else:
                self.datasets_text.append(sub_data_set)
                self.data_num_text.append(len(sub_data_set))

        self.data_ratio_multi = [
            float(ratio) / sum(self.data_num_multi)
            for ratio in self.data_num_multi
        ]
        self.data_ratio_text = [
            float(ratio) / sum(self.data_num_text)
            for ratio in self.data_num_text
        ]
        self.data_num = np.sum(self.data_num_multi) + np.sum(
            self.data_num_text)
        self.use_multi = 0

    def __len__(self):
        return int(np.sum(self.data_num) / self.batch_size)

    def __getitem__(self, index):
        if not self.set_seed:
            random.seed(index)
            self.set_seed = True
            print(f'Set seed {index} for rank {self.local_rank}')

        if len(self.datasets_multi) == 0 and len(self.datasets_text) == 0:
            raise ValueError(
                'Both _multi and _text are empty. Cannot sample any data.')

        if len(self.datasets_multi) > 0 and (self.use_multi < self.batch_size
                                             or len(self.datasets_text) == 0):
            data_idx = random.choices(
                range(len(self.data_ratio_multi)),
                weights=self.data_ratio_multi,
                k=1)[0]
            sample = self.datasets_multi[data_idx].get_item()
        elif len(self.datasets_text) > 0:
            data_idx = random.choices(
                range(len(self.data_ratio_text)),
                weights=self.data_ratio_text,
                k=1)[0]
            sample = self.datasets_text[data_idx].get_item()
        else:
            raise ValueError('Unable to select a dataset for sampling.')

        self.use_multi += 1
        if self.use_multi > self.batch_size * 2:
            self.use_multi = 0
        return dict(samples=sample)
        

class AlpacaSupervisedDataset(Mix_dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                json_datas,
                tokenizer: transformers.PreTrainedTokenizer, 
                num_examples,
                lorra_args,
                 batch_size=1,
                 local_rank=0,
                 img_size=490,
                 hd_num=18,
                ):
        super(AlpacaSupervisedDataset, self).__init__(
                    json_datas=json_datas,
                    batch_size=batch_size,
                    local_rank=local_rank,
                    img_size=img_size,
                    hd_num=hd_num
                )
        # if len(self.datasets_multi) == 0 and len(self.datasets_text) == 0:
        for proto in (self.datasets_multi, self.datasets_text):
            for ds in proto:
                conversations = [ds.text_processor(ds.raw_data[i]['conversations']) for i in range(len(ds.raw_data))]

                instructions = [entry[0] for entry in conversations]
                outputs = [entry[1] for entry in conversations]
                self.user_tag = lorra_args.user_tag
                self.assistant_tag = lorra_args.assistant_tag
                # Load the JSON data
                
                ds.orig_s, ds.pos_s, ds.neg_s = get_truncated_outputs(outputs, 
                                                            instructions, 
                                                            num_examples, 
                                                            self.user_tag,
                                                            self.assistant_tag, 
                                                            lorra_args.pos_type, 
                                                            lorra_args.neg_type,
                                                            lorra_args.control_template,
                                                            lorra_args.template_system,
                                                        )
                # del ds.raw_data
        
                print("===== Original String (ds.orig_s[0]) =====")
                print(ds.orig_s[0])
                
                print("===== Positive String (ds.pos_s[0]) =====")
                print(ds.pos_s[0])
                
                print("===== Negative String (ds.neg_s[0]) =====")
                print(ds.neg_s[0])
        # self.max_res_len = lorra_args.max_res_len

        self.tokenizer = tokenizer
        
def conv2text(sources):
    # I only do one turn conversation
    query, response = '', ''

    # for idx, sentence in enumerate(sources):
    for idx, sentence in enumerate(sources[:2]):
        BEGIN_SIGNAL = ''

        from_str = sentence['from']
        if from_str.lower() == 'human' or from_str.lower() == 'user':

            temp = (
                BEGIN_SIGNAL + sentence['value'].strip())
            query+= temp
        else:
            temp = (
                BEGIN_SIGNAL + sentence['value'].strip())
            response+= temp

    return (query, response)

class ImageProcessor:

    def __init__(self, image_size=224):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, item):
        item = Image.open(item).convert('RGB')
        return self.transform(item)


class ImageProcessorHD:

    def __init__(self, image_size=224, hd_num=-1):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms.Normalize(mean, std)
        self.hd_num = hd_num

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, item):
        item = Image.open(item).convert('RGB')
        return self.transform(HD_transform(item, hd_num=self.hd_num))


class Mix_dataset(Dataset):

    def __init__(self,
                 json_datas,
                 batch_size=1,
                 img_size=224,
                 local_rank=0,
                 hd_num=-1):
        """vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file."""
        super().__init__()
        print(f'init mix data at rank {local_rank}')
        self.datasets_text, self.datasets_multi = [], []
        self.data_num_text, self.data_num_multi = [], []

        self.batch_size = batch_size
        self.set_seed = False
        self.local_rank = local_rank
        for _, d in json_datas.items():
            if 'image' in d[0].keys():
                has_img = True
            else:
                has_img = False
            sub_data_set = Sample_dataset(
                d,
                batch_size,
                has_img=has_img,
                img_size=img_size,
                hd_num=hd_num)
            if has_img:
                self.datasets_multi.append(sub_data_set)
                self.data_num_multi.append(len(sub_data_set))
            else:
                self.datasets_text.append(sub_data_set)
                self.data_num_text.append(len(sub_data_set))

        self.data_ratio_multi = [
            float(ratio) / sum(self.data_num_multi)
            for ratio in self.data_num_multi
        ]
        self.data_ratio_text = [
            float(ratio) / sum(self.data_num_text)
            for ratio in self.data_num_text
        ]
        self.data_num = np.sum(self.data_num_multi) + np.sum(
            self.data_num_text)
        self.use_multi = 0

    def __len__(self):
        return int(np.sum(self.data_num) / self.batch_size)

    def __getitem__(self, index):
        if not self.set_seed:
            random.seed(index)
            self.set_seed = True
            print(f'Set seed {index} for rank {self.local_rank}')

        if len(self.datasets_multi) == 0 and len(self.datasets_text) == 0:
            raise ValueError(
                'Both _multi and _text are empty. Cannot sample any data.')

        if len(self.datasets_multi) > 0 and (self.use_multi < self.batch_size
                                             or len(self.datasets_text) == 0):
            data_idx = random.choices(
                range(len(self.data_ratio_multi)),
                weights=self.data_ratio_multi,
                k=1)[0]
            sample = self.datasets_multi[data_idx].get_item()
        elif len(self.datasets_text) > 0:
            data_idx = random.choices(
                range(len(self.data_ratio_text)),
                weights=self.data_ratio_text,
                k=1)[0]
            sample = self.datasets_text[data_idx].get_item()
        else:
            raise ValueError('Unable to select a dataset for sampling.')

        self.use_multi += 1
        if self.use_multi > self.batch_size * 2:
            self.use_multi = 0
        return dict(samples=sample)


class Sample_dataset(Dataset):

    def __init__(self,
                 raw_data,
                 batch_size,
                 has_img=True,
                 img_size=224,
                 hd_num=16):
        self.raw_data = raw_data
        print(f'load {len(self.raw_data)} data')
        self.batch_size = batch_size
        
        if hd_num == -1:
            self.vis_processor = ImageProcessor(image_size=img_size)
        else:
            # for 4khd model
            self.vis_processor = ImageProcessorHD(
                image_size=img_size, hd_num=hd_num)
        self.text_processor = conv2text
        self.has_img = has_img

    def __len__(self):
        return len(self.raw_data)

    def __get_item__(self, i):
        # conv_text = conv2text(self.raw_data[i]['conversations'])
        # sample = dict(text_input=conv_text, )
        orig_s = self.orig_s[i]
        pos_s = self.pos_s[i]
        neg_s = self.neg_s[i]
        sample = dict(orig_s=orig_s, pos_s=pos_s, neg_s=neg_s)
        
        if self.has_img:
            image_file = self.raw_data[i]['image']
            image = [self.vis_processor(i) for i in image_file]
            sample['image'] = torch.stack(image)
        else:
            sample['image'] = None

        return sample

    def get_item(self, ):
        # text_input = []
        orig_s, pos_s, neg_s = [], [], []
        images = []
        for i in range(self.batch_size):
            idx = random.randrange(len(self.raw_data))
            sample = self.__get_item__(idx)
            # text_input.append(sample['text_input'])
            orig_s.append(sample['orig_s'])
            pos_s.append(sample['pos_s'])
            neg_s.append(sample['neg_s'])
            
            images.append(sample['image'])
        # sample = {
        #     'text_input': text_input,
        #     'data_type': 'multi' if self.has_img else 'text',
        # }
        sample = {
        'orig_s': orig_s,
        'pos_s': pos_s,
        'neg_s': neg_s,
        'data_type': 'multi' if self.has_img else 'text',
        }
        if self.has_img:
            sample['image'] = torch.cat(images)
        # print(' === get_item ====')
        # print(sample['image'].shape)
        # sample['image'] has shape ( # of images in this batch ) x 3 x height x width
        #  === get_item ====
        # torch.Size([2, 3, 490, 490])
        #  === get_item ====
        # torch.Size([2, 3, 490, 490])
        return sample
