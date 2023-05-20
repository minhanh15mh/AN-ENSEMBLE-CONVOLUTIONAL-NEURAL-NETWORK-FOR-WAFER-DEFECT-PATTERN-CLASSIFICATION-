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

# Methodology 
Diagram below shows proposed framework for study and is described in more detail as follows. First, since the dataset is highly imbalanced, the vast majority of the None class will be dropped randomly until it balances with the samples of minorities. In this experiment, the total samples of each class equal to 300. Then the balanced training dataset is applied to three individual CNN models (Resnet 18, Googlenet, MobilenetV2) for classification. To enhance the performance of classification, the result of three individual CNN models will be grouped in ensemble model. Finally, the model performance is evaluated by three popular measures for imbalanced problems: recall, precision, and F1- score.  

<img width="465" alt="Screenshot 2023-05-20 212354" src="https://github.com/minhanh15mh/AN-ENSEMBLE-CONVOLUTIONAL-NEURAL-NETWORK-FOR-WAFER-DEFECT-PATTERN-CLASSIFICATION-/assets/86044915/35b2be8a-0682-4bd7-a1fc-b0e12f4a123b">



