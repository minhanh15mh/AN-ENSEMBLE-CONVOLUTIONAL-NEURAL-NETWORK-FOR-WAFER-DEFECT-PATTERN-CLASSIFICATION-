from src.Training import *
from src.Evaluation import *
import torch
import json
from torchvision import models
import torch.nn as nn

class EnsembleModel(nn.Module):   
    def __init__(self, modelA, modelB, modelC):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.classifier = nn.Linear(9 * 3, 9)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.classifier(x)
        return out
#Load params
FILE_TRAIN_CONFIG = os.path.join('config', 'TrainConfig.json')
params = json.load(open(FILE_TRAIN_CONFIG))

FILE_TEST_CONFIG = os.path.join('config', 'TestConfig.json')
test_params = json.load(open(FILE_TEST_CONFIG))
# Load_model    
resnet18_model= models.resnet18(pretrained = True)
resnet18_model.fc = nn.Linear(resnet18_model.fc.in_features, out_features=9)
resnet18_model.load_state_dict(torch.load(r'E:\University\Thesis\Thesis_Code\SavedModel\Resnet18_SGD_20epoch.pth'))

googlenet_model= models.googlenet(pretrained = True)
googlenet_model.fc = nn.Linear(googlenet_model.fc.in_features, out_features=9)
googlenet_model.load_state_dict(torch.load(r'E:\University\Thesis\Thesis_Code\SavedModel\googlenet_SGD_20epoch.pth'))

mobilenetV2_model= models.mobilenet_v2(pretrained = True)
mobilenetV2_model.classifier[1] = nn.Linear(mobilenetV2_model.classifier[1].in_features, out_features=9)
mobilenetV2_model.load_state_dict(torch.load(r'E:\University\Thesis\Thesis_Code\SavedModel\mobilenetv2_SGD_20epoch.pth'))

# ensemble training
ensembleCNN_model = EnsembleModel(resnet18_model, googlenet_model, mobilenetV2_model)
ensembleCNN_training = model_trainer('Ensemble CNN',ensembleCNN_model,**params)

ensembleCNN_trained_model = ensembleCNN_training.training() 
ensembleCNN_eval = evaluation(ensembleCNN_trained_model,**test_params)
ensembleCNN_eval.visualize_result()

