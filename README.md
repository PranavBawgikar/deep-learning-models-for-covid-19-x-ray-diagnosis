# deep learning models for covid-19 x-ray diagnosis
this repo contains a comprehensive analysis of ten models, including eight deep learning models and two traditional machine learning models, on a covid-19 x-ray image dataset. the investigation encompassed diverse feature extraction techniques, utilization of pre-trained models, and a meticulous evaluation of each model's performance. the outcomes contributed valuable insights into the effectiveness of distinct approaches for covid-19 x-ray image classification.
# 
in a bird's eye view, our feature extraction process involved the following key steps:
1. _preprocessing_: before extracting features, we preprocess the x-ray images to standardize their format, enhancing the model's ability to learn consistent patterns.
2.	_selection of discriminative features_: we carefully select features that capture distinctive aspects related to pulmonary conditions. these features are chosen to enhance the model's sensitivity to subtle abnormalities associated with pneumonia and covid-19.
3.	_utilization of convolutional neural networks (CNNs)_: a class of neural networks well-suited for image-related tasks are employed. these networks automatically learn hierarchical features, allowing the model to discern complex patterns in x-ray images.
4.	_transfer learning_: to capitalize on the knowledge embedded in pre-trained models, architectures such as VGG, Xception, ResNet, InceptionV3, InceptionResNetV2, DenseNet, and EfficientNet are utilized. this facilitated the extraction of high-level features without the need to train the models from scratch.
### dataset
you can acccess the dataset <a href="https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia">here</a>.

dataset is organized into 2 folders (train, test) and both train and test contain 3 subfolders (COVID19, PNEUMONIA, NORMAL). dataset contains total 6432 x-ray images and test data have 20% of total images.
<br /><br />
<img src="https://github.com/user-attachments/assets/9afd6c73-8c18-446f-9ce8-0f851653b6c6" alt="drawing" width="500"/>

the following images serve as tangible illustrations of the data variety incorporated into our model training process.

<img src="https://github.com/user-attachments/assets/79c2bb53-a70c-42f5-a4d7-4736db3363ea" alt="drawing" width="600"/> <br /><br />
<img src="https://github.com/user-attachments/assets/4edb39cf-ee5f-46e6-adcd-088a5a2c4ea7" alt="drawing" width="600"/>
<br /><br />
_performance of different models through binary cross-entropy classification (10 models were considered)._

| Classifier Group                | Classifier Type                    | Accuracy | Loss   |
|---------------------------------|------------------------------------|----------|--------|
| Convolutional Neural Networks   | Convolutional Neural Network (CNN) | 0.9500   | 0.1959 |
|                                 | VGG                                | 0.9750   | 0.0503 |
|                                 | Xception                           | 0.9750   | 0.0729 |
|                                 | ResNet50                           | 0.7750   | 0.4920 |
|                                 | InceptionV3                        | 0.8500   | 0.3183 |
|                                 | InceptionResNetV2                  | 0.9250   | 0.1051 |
|                                 | DenseNet201                        | 0.9500   | 0.1080 |
|                                 | EfficientNetB3                     | 0.5000   | 0.6848 |
| Support Vector Machine Classifier | Support Vector Machine (SVM)    | 0.9750   | -      |
| Random Forest Classifier        | Random Forest                      | 1.0      | -      |
# 
the evaluation results highlight the performance of various classifiers on the covid-19 x-ray image dataset. among convolutional neural networks (CNNs), the VGG model was the top performer with an accuracy of 97.50% and a loss of 0.0503. the SVM classifier also demonstrated strong performance with an accuracy of 97.50%. notably, the random forest classifier achieved perfect accuracy (100%). these results underscore the effectiveness of deep learning models, particularly VGG, in classifying covid-19 x-ray images, with high accuracy and low loss metrics.

