from src.Training import *
import json
import os
from torchvision import models
from src.Evaluation import *

FILE_TRAIN_CONFIG = os.path.join('config', 'TrainConfig.json')
params = json.load(open(FILE_TRAIN_CONFIG))

FILE_TEST_CONFIG = os.path.join('config', 'TestConfig.json')
test_params = json.load(open(FILE_TEST_CONFIG))

googlenet_model= models.googlenet(pretrained = True)
googlenet_model.fc = nn.Linear(googlenet_model.fc.in_features, out_features=9)

googlenet_training = model_trainer('Googlenet',googlenet_model,**params)
def main():
    goooglenet_trained_model = googlenet_training.training()
    googlenet_eval = evaluation(goooglenet_trained_model,**test_params)
    googlenet_eval.visualize_result()

if __name__ == "__main__":
    main()
