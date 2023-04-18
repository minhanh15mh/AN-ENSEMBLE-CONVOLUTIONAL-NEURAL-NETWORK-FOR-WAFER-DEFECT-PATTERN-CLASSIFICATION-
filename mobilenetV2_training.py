from src.Training import *
import json
import os
from torchvision import models
from src.Evaluation import *

FILE_TRAIN_CONFIG = os.path.join('config', 'TrainConfig.json')
params = json.load(open(FILE_TRAIN_CONFIG))

FILE_TEST_CONFIG = os.path.join('config', 'TestConfig.json')
test_params = json.load(open(FILE_TEST_CONFIG))

mobilenetV2_model= models.mobilenet_v2(pretrained = True)
mobilenetV2_model.fc = nn.Linear(mobilenetV2_model.fc.in_features, out_features=9)

mobilenetV2_training = model_trainer('MobilenetV2',mobilenetV2_model,**params)
def main():
    mobilenetV2_trained_model = mobilenetV2_training.training()
    mobilenetV2_eval = evaluation(mobilenetV2_trained_model,**test_params)
    mobilenetV2_eval.visualize_result()

if __name__ == "__main__":
    main()
