import torch
import torch.nn as nn
from torchvision import models, transforms

def load_model(): 
    ensembleCNN_model.load_state_dict(torch.load(r'.\SavedModel\ensemble_20epoch.pth'))
    return ensembleCNN_model

def predict_WM(model, img_input):
    label = ['Center', 'Donut', 'Edge Loc', 'Edge Ring', 'Loc', 'Random', 'Scratch', 'NearFull', 'None']
    trans = transforms.Compose([transforms.ToTensor()])
    trans_input = trans(img_input)
    trans_input = trans_input.reshape(1,3,224,224)
    model.eval()
    with torch.no_grad():
       pred = model(trans_input)
       pred_indice = pred.argmax(dim=1)
    return label[pred_indice]
