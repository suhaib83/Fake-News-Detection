import os
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from transformers import BertTokenizer
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])


class FakedditImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.csv_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.csv_frame.loc[idx, 'id'] + '.jpg'
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            label = self.csv_frame.loc[idx, '6_way_label']
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            return None


class FakedditHybridDataset(FakedditImageDataset):
    def __init__(self, csv_file, root_dir, transform=None):

        super(FakedditHybridDataset, self).__init__(csv_file, root_dir, transform)
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:

            sent = self.csv_frame.loc[idx, 'clean_title']
            bert_encoded_dict = self.bert_tokenizer.encode_plus(
                sent,  
                add_special_tokens=True,  
                max_length=120, 
                padding='max_length',
                truncation=True,
                return_attention_mask=True,  
                return_tensors='pt',  
            )
            bert_input_id = bert_encoded_dict['input_ids']
            bert_attention_mask = bert_encoded_dict['attention_mask']
            img_name = self.csv_frame.loc[idx, 'id'] + '.jpg'
            img_path = os.path.join(self.root_dir, img_name)
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            label = self.csv_frame.loc[idx, '6_way_label']
            if self.transform:
                image = self.transform(image)
            return {'bert_input_id': bert_input_id, 
                    'bert_attention_mask': bert_attention_mask, 
                    'image': image,
                    'label': label}
          
        except Exception as e:
            return None


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch) 
