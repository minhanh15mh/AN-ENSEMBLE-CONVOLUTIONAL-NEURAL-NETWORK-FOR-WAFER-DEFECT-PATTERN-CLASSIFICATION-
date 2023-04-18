from src.Training import *
import json
import os
from torchvision import models
from src.Evaluation import *

FILE_TRAIN_CONFIG = os.path.join('config', 'TrainConfig.json')
params = json.load(open(FILE_TRAIN_CONFIG))

FILE_TEST_CONFIG = os.path.join('config', 'TestConfig.json')
test_params = json.load(open(FILE_TEST_CONFIG))

resnet18_model= models.resnet18(pretrained = True)
resnet18_model.fc = nn.Linear(resnet18_model.fc.in_features, out_features=9)

resnet18_training = model_trainer('Resnet18',resnet18_model,**params)
def main():
    resnet_trained_model = resnet18_training.training()
    resnet18_eval = evaluation(resnet_trained_model,**test_params)
    resnet18_eval.visualize_result()

if __name__ == "__main__":
    main()
