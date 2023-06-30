# Malaria-Cell-Detection

<p align="justify">This is a sample paragraph of text that will be justified.</p>

The primary objective of this project is to develop an accurate and efficient classification system capable of distinguishing between normal and abnormal malaria cells. To achieve this, the researcher aimed to create an algorithm that leverages the power of both machine learning (ML) and deep learning (DL) models. By analyzing cell images, the goal is to automatically detect and identify malaria-infected cells, enabling early and reliable diagnosis.

Malaria, a febrile illness caused by Plasmodium parasites transmitted through mosquito bites, poses a significant global health concern as reported by the World Health Organization (WHO). In their worldwide malaria report for 2021, the WHO documented over 240 million cases of malaria in 2020. The impact of this rise in malaria cases is particularly severe in developing nations, where healthcare systems often face overwhelming challenges (Murray & Bennett, 2009).

To address these challenges and expedite malaria detection, the healthcare sector can leverage the potential of advanced deep learning algorithms and machine learning techniques. By harnessing the power of these cutting-edge technologies, it becomes possible to enhance the speed and accuracy of malaria diagnosis.

Images below show samples of parasitized or infected malaria cells:

![parasitized_cells](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/cb2449d2-1b2f-4818-831c-d4291859906e)

Images below show samples of normal or uninfected malaria cells:

![uninfected_cells](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/2b62f276-8f51-453c-a69e-29fede1afa7e)

Converting RGB cell images to grayscale helps reduce computational complexities:

![grey_parasitized copy](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/8d6ddf00-fd9b-441c-ab80-dcf77475f94c)

![grey_uninfected](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/b34ac748-d89e-4472-b3dc-9f742acc783c)


The dataset consists of a total of 27,558 malaria cell images, comprising 13,779 parasitized cell images and 13,779 uninfected cell images. This balanced dataset provides an equal representation of both classes. However, the raw malaria cell images have varying sizes. Due to the limitations of the laptop used for this project, all images will be resized to 64 by 64 dimensions using a batch size of 48.

The Malaria dataset can be found and downloaded from this link: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria. Below figure shows the distribution of each label or set: 

![bar plots](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/5e02078e-d28a-4447-b1c3-abaa0f728d5c)

**ML/DL Algorithms**

1.	Convolutional Neural Networks (CNN)
   
<p align="justify">CNN, also known as ConvNet, is widely recognized as one of the most popular and powerful deep learning algorithms for image classification tasks. It utilizes convolution, a mathematical technique that combines two functions to derive a third function representing their interplay (Mandal, 2021). The models created by the researcher generally comprise multiple convolutional layers, kernels or filters (which generate feature maps), Rectified Linear Unit (ReLU) layers (to retain positive values and convert non-positive values to zero in the input data), and max pooling layers (for downsampling and reducing the dimensionality of feature maps to expedite calculations). Additionally, fully connected layers are employed to produce class scores from the activated features for classification purposes.</p>

Basic CNN model summary:

<img width="512" alt="model1_summary" src="https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/e5e3c9b5-0c21-4e6c-be6d-b05517df0ebf"> <br>

Basic CNN model performance:

![cnn_model1](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/20c99336-419b-4879-938a-5f9857a0ecb4)

Confusion matric of the basic CNN model:

![conf_matrix_model1](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/be33f685-22fa-41d0-a101-3f578fa712aa)

ROC_AUC:

![basic_cnn_roc](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/a9cd2a9e-b984-4a10-b8a6-65bcde1db8cb)

Precision and Recall:

![basic_cnn_precision_recall](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/2109ee4b-b72a-4bd7-a5e9-b3a1cc13643a)


2. Support Vector Machine (SVM)
   
<p align="justify">SVM is a powerful supervised machine learning model commonly used for classification tasks, both for binary and multi-class problems (Telrandhe et al., 2016). It works by separating data points in a plotted n-dimensional space into distinct classes using a straight line or hyperplane.Numerous studies have demonstrated the effectiveness of SVM in automatically detecting Malaria cells based on image data. By utilizing SVM, researchers have been able to train models that can accurately classify Malaria-infected cells from healthy cells.SVM's ability to handle complex decision boundaries and its robustness against overfitting make it well-suited for this task. The model learns from a labeled dataset, extracting key features from the images and creating a decision boundary that maximizes the separation between different classes.</p>

To reduce the dimensionality of the input before feeding into the model,  Principal Component Analysis (PCA). As shown in Figure 3, 95% of the variance was explained by 7th principal component. This will be used later for grid search.

![PCA](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/9029a96b-7184-4e67-94cd-1d59f7a566de)

After perform grid search, the best combination of cost and gamma are 100 and 0.01, respectively. These hyperparameters will be used to evaluate the model using the test set. Accuracy of 95% confidence interval is between 83.1% to 84.8%. 

Confusion matrix of SVM model:

![conf_matrix_svm](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/65fec157-6b8f-4a00-81c1-a92c55309bec)

ROC_AUC curve:

![SVM_ROC](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/695b2706-d208-4a5a-9acb-aff32f232dc3)

Precision and Recall curve:

![SVM_precision_recall](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/d9a34405-eaa9-4651-8a50-bbed3bb4bcdb)

**Model Evaluation**

Because relying solely on accuracy may not provide a comprehensive evaluation of the model's performance, additional metrics such as Precision (positive predictive value), Recall (true positive rate), and F1-score (a metric that combines Precision and Recall) were calculated. The following table presents the model metrics for the basic CNN and SVM:

<img width="925" alt="image" src="https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/7738cfe8-9408-495f-8d8b-894f250d6323">

These metrics provide a more detailed assessment of the model's ability to correctly identify positive and negative instances, taking into account both false positives and false negatives.

The results clearly indicate that the basic CNN algorithm outperformed the SVM algorithm across all the evaluated metrics. Based on these findings, it is evident that CNN is a more suitable choice for the task at hand. In the upcoming section, the researcher focuses on further enhancing the performance and accuracy of the CNN model through parameter tuning. This approach aims to fine-tune the model's hyperparameters and optimize its configuration to achieve even better results in terms of accuracy and overall model metrics.

**Hyperparameter Tuning**

Below is the summary architecture of the tuned CNN model:

<img width="603" alt="model2_summary" src="https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/48f36c33-a3fa-4928-918b-2f5a8563fd0f"><br>

Figure 5 presents the performance of tuned model. It is evident that the implementation of batch normalization and the addition of dropout layers in the convolutional layers effectively mitigated the overfitting issue encountered in basic CNN. These techniques have successfully improved the model's generalization capability and reduced the discrepancy between training and validation performance.<br>

![cnn_model2](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/53f9ad42-de4c-4b1e-a382-183260783ca5)<br>

Basic CNN vs tuned CNN performance comparison:

<img width="701" alt="image" src="https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/c5c2d912-7ff1-42db-920a-6e18d5edd9a2">









   
