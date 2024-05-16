import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from my_VGG19 import VGG19_6way
from FakedditDataset import FakedditImageDataset, my_collate
import os, time, copy
from tqdm import tqdm
from collections import deque
from statistics import mean
import ssl
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings('ignore')

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

csv_dir = "../../Data/"
img_dir = "../../Data/Imag_processing/"
l_datatypes = ['train', 'validate']
csv_fnames = {'train': 'multimodal_train.csv', 'validate': 'multimodal_valid.csv'}
image_datasets = {x: FakedditImageDataset(os.path.join(csv_dir, csv_fnames[x]), img_dir, transform=data_transforms) for x in
                  l_datatypes}
# Dataloader
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, collate_fn=my_collate) for x in l_datatypes}
dataset_sizes = {x: len(image_datasets[x]) for x in l_datatypes}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Note: corrupted images will be skipped in training")

train_losses = []
val_losses = []
train_accs = []
val_accs = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=1, report_len=500):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in l_datatypes:
            print(f'{phase} phase')
            if phase == 'train':
                model.train()  
            else:
                model.eval() 

            running_loss = 0.0
            running_corrects = 0
            loss_q = deque(maxlen=report_len)
            acc_q = deque(maxlen=report_len)
            counter = 0
            for inputs, labels in tqdm(dataloaders[phase]):
                counter += 1
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, t_pred = torch.max(outputs, 1)
                    #print(t_pred)
                    acc = (t_pred == labels).float().sum().item()
                    acc_q.append(acc)
                    loss = criterion(outputs, labels)
                    loss_q.append(loss.item())
                    if counter % report_len == 0:
                       print(f"Iter {counter}, loss: {mean(loss_q)}, accuracy:{sum(acc_q) / (report_len * len(labels))}")
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(t_pred.squeeze() == labels)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            train_losses.append(epoch_loss) 
            train_accs.append(epoch_acc.item())
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

         
            if phase == 'validate':
                val_losses.append(epoch_loss) 
                val_accs.append(epoch_acc.item())
            
            if phase == 'validate' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, "VGG19_best_model_full_train")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = True


model_ft =  VGG19_6way(pretrained=True)
set_parameter_requires_grad(model_ft, True)  
model_ft = model_ft.to(device)

criterion= nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=5)


# Plot training and validation losses
plt.figure()
plt.plot(train_losses[:len(val_losses)], label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracies
plt.figure()
plt.plot(train_accs[:len(val_accs)], label='Training accuracy')
plt.plot(val_accs, label='Validation accuracy')
plt.title('Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# save model
torch.save(model_ft.state_dict(), 'fakeddit_VGG19_full_train.pt')

torch.save(model_ft, "VGG19_model_save_full_train")
