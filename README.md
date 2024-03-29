# The need of this project
Semiconductor manufacturing is one of the most important manufacturing industries for many countries. Semiconductor devices have become an essential component of many modern gadgets, as well as social infrastructure, and positive support for daily living. Semiconductors are the fundamental components of digital devices, consisting of tiny and ultra-compact chips in electrical components used in gadgets such as smartphones, self-driving cars, and even both artificial intelligence and data centers, etc. 


In particular, the semiconductor sector is progressively proving to have a vital role in providing exceptional steps in the process of growing worldwide science and technology. The development of semiconductor technology has aided in system optimization, increasingly miniature device sizes, energy savings, environmental protection, and providing a safe and quality life. The semiconductor industry, also known as electronic microphones, is valued at hundreds of billions of dollars and has become a key industry for many countries around the world. Some exceptional semiconductor applications:
-	The temperature sensor is constructed of semiconductor air conditioners. The rice cooker's precise temperature control mechanism, which employs semiconductors, allows it to cook rice to perfection. CPU computers' processors are also constructed of semiconductor materials.
-	Semiconductors are used in a wide range of digital consumer items, including mobile phones, cameras, televisions, washing machines, refrigerators, and LED lights.
-	In addition to consumer gadgets, semiconductors are critical in the operation of ATMs, railroads, the internet, media, and a variety of other equipment in social infrastructures, such as in hospitals. 


In semiconductor manufacturing, wafer is one of the most important components. According to Laplante (2005), a wafer is a thin slice of semiconductor material on which semiconductor devices are made. It is also called a "slice" or "substrate," and is used in electronics for the making of integrated circuits. The quality of the wafer directly affects the quality of finished product. The wafer map is used to present important information about the defect location on wafer for engineers. In semiconductor fabrication, the defect on wafer has been divided into two kind, systematic and random. The systematic defect on wafer can result from process, design, or test problems Nishi et al. (2000). Thus, wafer defect pattern classification can help to detect root causes of failure in a semiconductor manufacturing process and determine the stage of manufacturing at which wafer pattern failure occurs. The improvement of wafer map defect pattern classification can help businesses yield enhancement, save time, and reduce production costs.

# Dataset introduction
The dataset WM – 811k was used in this study which has been published since 2015, by Wu et.al (2015). The wafer maps in the dataset were divided into abnormality and normality classes. There are eight classes of defective wafers: Center, Donut, Edge-Loc, Edge-Ring, Local, Random, Scratch, Near-Full. The normality wafers were classified as class None. Link dataset: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map. 

The picture below illustrates the total classes in dataset.
![image](https://github.com/minhanh15mh/AN-ENSEMBLE-CONVOLUTIONAL-NEURAL-NETWORK-FOR-WAFER-DEFECT-PATTERN-CLASSIFICATION-/assets/86044915/5dedbbaa-47a7-4bf2-a32b-7408ebbb5f90)

First, since the dataset is highly imbalanced, the vast majority of the None class will be dropped randomly until it balances with the samples of minorities. In this experiment, the total samples of each class equal to 300. Then the balanced training dataset is applied to three individual CNN models (Resnet 18, Googlenet, MobilenetV2) for classification. To enhance the performance of classification, the result of three individual CNN models will be grouped in ensemble model.

![Ensemble](https://github.com/minhanh15mh/AN-ENSEMBLE-CONVOLUTIONAL-NEURAL-NETWORK-FOR-WAFER-DEFECT-PATTERN-CLASSIFICATION-/assets/86044915/568b890b-1494-4ada-90c8-b1b874c90ff7)

# Experiment result
These pictures below illustrate the confusion maxtrix of three pretrained CNN models (Resnet 18, GoogleNet, MobilenetV2) used in this project. The experiment indicates that the MobilenetV2 model has the highest performance among three models.


![resnet18](https://github.com/minhanh15mh/CONVOLUTIONAL-NEURAL-NETWORK-FOR-WAFER-DEFECT-PATTERN-CLASSIFICATION-/assets/86044915/1a76b922-a155-41fc-a107-82bdc69ad404)

The confusion matrix of Resnet 18 model

![googlenet](https://github.com/minhanh15mh/CONVOLUTIONAL-NEURAL-NETWORK-FOR-WAFER-DEFECT-PATTERN-CLASSIFICATION-/assets/86044915/200554fa-2bf2-4b7c-82a0-c7ed1cf1f466)

The confusion matrix of GoogleNet model

![mobilenetv2](https://github.com/minhanh15mh/CONVOLUTIONAL-NEURAL-NETWORK-FOR-WAFER-DEFECT-PATTERN-CLASSIFICATION-/assets/86044915/c3c121f0-6648-4272-8877-542ae211f8f3)

The confusion matrix of MobilenetV2 model

# Deploy model using streamlit

https://github.com/minhanh15mh/CONVOLUTIONAL-NEURAL-NETWORK-FOR-WAFER-DEFECT-PATTERN-CLASSIFICATION-/assets/86044915/180e1425-cb64-45ba-a771-c381f12ffea1
