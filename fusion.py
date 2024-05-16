import sys
import os
VGG19_dir = os.path.join(os.path.dirname(__file__), '../VGG19/')
sys.path.append(VGG19_dir)

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms
from transformers import BertForSequenceClassification
from my_VGG19 import VGG19_6way
from FakedditDataset import FakedditHybridDataset, my_collate
from HybridModel import LateFusionModel
from training import ModelTrainer
import warnings
warnings.filterwarnings('ignore')

import multiprocessing
    
if __name__ == '__main__':
    multiprocessing.freeze_support()
    bert_classifier = BertForSequenceClassification.from_pretrained('../bert_save_dir')

    VGG19_model = VGG19_6way(pretrained=False)
    VGG19_dict = torch.load('../VGG19/fakeddit_VGG19_full_train.pt')
    VGG19_model.load_state_dict(VGG19_dict)

    hybrid_model = LateFusionModel(VGG19_model, bert_classifier)
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    hybrid_model = hybrid_model.to(device)
    
    # Prepare datesets
    csv_dir = "../../Data/"
    img_dir = "../../Data/Imag_processing/"
    l_datatypes = ['train', 'validate', 'test']
    csv_fnames = {
        'train': 'multimodal_train.csv',
        'validate': 'multimodal_valid.csv',
        'test': 'multimodal_test.csv'
    }
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    
    hybrid_datasets = {x: FakedditHybridDataset(os.path.join(csv_dir, csv_fnames[x]), img_dir, transform=data_transforms)
                       for x in l_datatypes}
    dataset_sizes = {x: len(hybrid_datasets[x]) for x in l_datatypes}
    
    # Dataloader
    dataloaders = {x: torch.utils.data.DataLoader(hybrid_datasets[x], batch_size=64, shuffle=True, num_workers=6,
                                                  collate_fn=my_collate) for x in l_datatypes}
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(hybrid_model.parameters(), lr=1e-4)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    # Trainer isntance
    trainer = ModelTrainer(l_datatypes, hybrid_datasets, dataloaders, hybrid_model)
    trainer.train_model(criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2, report_len=1000)
    trainer.save_model('hybrid_model.pt')
    trainer.generate_eval_report()

