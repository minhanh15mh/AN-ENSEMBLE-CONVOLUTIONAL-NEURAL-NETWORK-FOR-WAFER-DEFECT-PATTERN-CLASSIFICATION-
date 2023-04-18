import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns
from torch.utils.data import  DataLoader
import torch
from src.ultis import *

class evaluation(SettingConfig):
    def __init__(self, model_training, **args):
        super(evaluation, self).__init__(**args)
        self.model_training = model_training
    
    def test_dataset(self):
        img_test_list,label_test_list = make_datapath_list(self.data_test_path)
        test_dataset = ImageDataset(img_test_list,label_test_list)
        test_dataloader = DataLoader(test_dataset, batch_size = len(test_dataset) )
        return test_dataloader
    
    def prediction(self):
        test_dataloader = self.test_dataset()
        pred = torch.tensor([])
        self.model_training.eval()
        with torch.no_grad():
           for img, label_count in test_dataloader:
               output = self.model_training(img)
               pred = torch.concat((pred, output.argmax(dim = 1)))
        return pred
    
    def visualize_result(self):
        img_test, label_test = next(iter(self.test_dataset()))
        cnf_matrix=confusion_matrix(self.prediction(),label_test, normalize= 'true')
        plt.subplots(figsize=(8,8))
        sns.heatmap(cnf_matrix, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

        print(classification_report(self.prediction(), label_test))
