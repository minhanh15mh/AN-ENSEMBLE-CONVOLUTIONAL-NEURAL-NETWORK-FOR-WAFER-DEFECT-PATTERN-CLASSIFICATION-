from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.ultis import *


class model_trainer(SettingConfig):
    def __init__(self, model_name, model, **args):
        super(model_trainer, self).__init__(**args)
        self.model_name = model_name
        self.model = model
    
    def getting_dataset(self):
        img_train_list,label_train_list = make_datapath_list(self.data_train_path)
        img_val_list,label_val_list = make_datapath_list(self.data_val_path) 

        train_dataloader = DataLoader(ImageDataset(img_train_list,label_train_list),self.batch_size, shuffle = True)
        val_dataloader = DataLoader(ImageDataset(img_val_list,label_val_list),self.batch_size, shuffle = False)
        dataloader_dict = {"train": train_dataloader,'val': val_dataloader}
        return dataloader_dict

    def training(self):

        model = self.model
        optimizer = optim.SGD(params = model.parameters(), lr = 0.001, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)
        criterion = nn.CrossEntropyLoss()
        dataloader_dict = self.getting_dataset()


        for epoch in range(self.num_epochs):


            print('Epoch {}/{}'.format(epoch, self.num_epochs))

     
            for phase in ['train', 'val']:

                running_loss = 0
                running_corrects = 0


                if phase == 'train':
                   model.train()
                   with torch.enable_grad():
                        for inputs, labels in tqdm(dataloader_dict[phase]):


                            optimizer.zero_grad()
                            outputs = model(inputs)
                            _,preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            preds = outputs.argmax(dim=1)                        
                        
                            loss.backward()
                            optimizer.step()

                            running_loss += loss.item() * inputs.size(0)
                            running_corrects += torch.sum(preds == labels.data)
                    
                        epoch_loss_train = running_loss / len(dataloader_dict[phase].dataset)
                        epoch_acc_train = running_corrects.double() / len(dataloader_dict[phase].dataset)
                        scheduler.step()
                        print('{} Loss: {:.4f} Accuracy: {:.4f}%'.format(phase, epoch_loss_train, epoch_acc_train*100))



                elif phase == 'val':
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in tqdm(dataloader_dict[phase]):                                           
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            preds = outputs.argmax(dim=1) 

                            running_loss += loss.item() * inputs.size(0)
                            running_corrects += torch.sum(preds == labels.data)    
                        

                        epoch_loss_val = running_loss / len(dataloader_dict[phase].dataset)
                        epoch_acc_val = running_corrects.double() / len(dataloader_dict[phase].dataset)
                        print('{} Loss: {:.4f} Accuracy: {:.4f}%'.format(phase, epoch_loss_val, epoch_acc_val*100))
                        print('-' * 10)
        save_model_path = self.save_model_path + self.model_name +'_'+ str(self.num_epochs) + 'epochs.pth'
        torch.save(model.state_dict(), save_model_path)
        return model 




