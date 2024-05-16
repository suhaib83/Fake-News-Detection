import torch
from torch import nn
from torch.nn import Module
from transformers import BertModel, BertForSequenceClassification
from torchvision.models import VGG

class LateFusionModel(Module):

    def __init__(self, VGG19_model, bert_model):
        super(LateFusionModel, self).__init__()
        assert isinstance(bert_model, BertForSequenceClassification)
        assert isinstance(VGG19_model, VGG), "VGG19 model must be a VGG instance!"
        VGG19_feature_size = VGG19_model.classifier[6].in_features 
        VGG19_model.classifier[6] = nn.Linear(VGG19_feature_size, 6)
        self._VGG19 = VGG19_model
        self._VGG19.fc = nn.Identity()
      
        for param in self._VGG19.parameters():
            param.requires_grad = False
     
        bert_model.config.output_hidden_states = True
        self._bert = bert_model.bert
        bert_feature_size = bert_model.classifier.in_features
        self._bert.eval()
        for param in self._bert.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(bert_feature_size + VGG19_feature_size, 6)

    def forward(self, batch_in:dict):
        batch_in = {x: batch_in[x].to(next(self.parameters()).device) for x in batch_in}
        bert_output = self._bert(batch_in['bert_input_id'].squeeze(), attention_mask=batch_in['bert_attention_mask'].squeeze())
        cls_vector = bert_output.pooler_output
        VGG19_feature = self._VGG19(batch_in['image'])
        concatenated_features = torch.cat((cls_vector, VGG19_feature), dim=1)
        in_features = concatenated_features.size(1)
        
        self.linear = nn.Sequential( nn.Linear(in_features=in_features, out_features=1000),
                                    nn.ReLU(),
                                    nn.Linear(in_features=1000, out_features=6)
                                    )
        
        return self.linear(concatenated_features)
