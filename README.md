# Malaria-Cell-Detection



Summary of the project.

<p align="justify">The primary objective of this project is to develop an accurate and efficient classification system capable of distinguishing between normal and abnormal malaria cells. To achieve this, the researcher aimed to create an algorithm that leverages the power of both machine learning (ML) and deep learning (DL) models. By analyzing cell images, the goal is to automatically detect and identify malaria-infected cells, enabling early and reliable diagnosis.</p>

<p align="justify">Malaria, a febrile illness caused by Plasmodium parasites transmitted through mosquito bites, poses a significant global health concern as reported by the World Health Organization (WHO). In their worldwide malaria report for 2021, the WHO documented over 240 million cases of malaria in 2020. The impact of this rise in malaria cases is particularly severe in developing nations, where healthcare systems often face overwhelming challenges (Murray & Bennett, 2009).</p>

<p align="justify">To address these challenges and expedite malaria detection, the healthcare sector can leverage the potential of advanced deep learning algorithms and machine learning techniques. By harnessing the power of these cutting-edge technologies, it becomes possible to enhance the speed and accuracy of malaria diagnosis.</p>

Images below show samples of parasitized or infected malaria cells:

![parasitized_cells](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/15c0fb26-de1d-4025-ac9f-a9722ec8a8d9)

Images below show samples of normal or uninfected malaria cells:

![uninfected_cells](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/c3e8e14d-a2d5-4073-ab3d-005396173ee6)

Converting RGB cell images to grayscale helps reduce computational complexities:

![grey_parasitized copy](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/86de7ce9-523d-430f-845a-70bf251147b2) <br>

![grey_uninfected](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/d14c7c17-8e2d-49b5-a041-daa30c8cdde9) <br>

<p align="justify">The dataset consists of a total of 27,558 malaria cell images, comprising 13,779 parasitized cell images and 13,779 uninfected cell images. This balanced dataset provides an equal representation of both classes. However, the raw malaria cell images have varying sizes. Due to the limitations of the laptop used for this project, all images will be resized to 64 by 64 dimensions using a batch size of 48.</p>

The Malaria dataset can be found and downloaded from this link: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria. Below figure shows the distribution of each label or set: 

![bar_plots](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/f3897541-dc7e-4f7c-b966-bd6e5fa92eb4) <br>

**ML/DL Algorithms**

1.	Convolutional Neural Networks (CNN)
   
<p align="justify">CNN, also known as ConvNet, is widely recognized as one of the most popular and powerful deep learning algorithms for image classification tasks. It utilizes convolution, a mathematical technique that combines two functions to derive a third function representing their interplay (Mandal, 2021). The models created by the researcher generally comprise multiple convolutional layers, kernels or filters (which generate feature maps), Rectified Linear Unit (ReLU) layers (to retain positive values and convert non-positive values to zero in the input data), and max pooling layers (for downsampling and reducing the dimensionality of feature maps to expedite calculations). Additionally, fully connected layers are employed to produce class scores from the activated features for classification purposes.</p>

Basic CNN model summary:

<img width="512" alt="model1_summary" src="https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/7ad89549-cd38-4b5f-8304-e0c4d5b357d2"> <br>

Confusion matric of the basic CNN model:

![conf_matrix_model1](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/51deddee-1f9a-46e5-814a-df1d8c616bd5) <br>

ROC_AUC (Sensitivity and Specificity) :

![ROC_auc_model1](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/61038f8f-2e19-418b-b9be-f7c38468ec8d) <br>

ROC_AUC (Precision and Recall):

![basic_cnn_precision_recall](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/83878507-6f8b-462e-ae82-cc77968acb6e) <br>

2. Support Vector Machine (SVM)
   
<p align="justify">SVM is a powerful supervised machine learning model commonly used for classification tasks, both for binary and multi-class problems (Telrandhe et al., 2016). It works by separating data points in a plotted n-dimensional space into distinct classes using a straight line or hyperplane.Numerous studies have demonstrated the effectiveness of SVM in automatically detecting Malaria cells based on image data. By utilizing SVM, researchers have been able to train models that can accurately classify Malaria-infected cells from healthy cells.SVM's ability to handle complex decision boundaries and its robustness against overfitting make it well-suited for this task. The model learns from a labeled dataset, extracting key features from the images and creating a decision boundary that maximizes the separation between different classes.</p>

<p align="justify">To reduce the dimensionality of the input before feeding into the model,  Principal Component Analysis (PCA). As shown in Figure 3, 95% of the variance was explained by 7th principal component. This will be used later for grid search.</p>

<img width="792" alt="image" src="https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/ec36ccec-6476-4711-8039-c31a5013ae71"> <br>

<p align="justify">After perform grid search, the best combination of cost and gamma are 100 and 0.01, respectively. These hyperparameters will be used to evaluate the model using the test set. Accuracy of 95% confidence interval is between 83.1% to 84.8%. </p>

Confusion matrix of SVM model:

<img width="711" alt="image" src="https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/48d03fb5-d521-4aa8-bf89-0caafe5fa58f"> <br>

ROC_AUC curve:

<img width="523" alt="image" src="https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/a0345ab1-771b-491e-a73f-faff23f115b0"> <br>

Precision and Recall curve:

<img width="451" alt="image" src="https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/ac8c1f41-ab33-4e19-8d8b-1c013b2a6261"> <br>

**Model Evaluation**

<p align="justify">Because relying solely on accuracy may not provide a comprehensive evaluation of the model's performance, additional metrics such as Precision (positive predictive value), Recall (true positive rate), and F1-score (a metric that combines Precision and Recall) were calculated. The following table presents the model metrics for the basic CNN and SVM:</p>

<img width="696" alt="image" src="https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/f5e7a397-6cf9-4b86-b7be-2f7855e7be5e"> <br>

<p align="justify">These metrics provide a more detailed assessment of the model's ability to correctly identify positive and negative instances, taking into account both false positives and false negatives.</p>

<p align="justify">The results clearly indicate that the basic CNN algorithm outperformed the SVM algorithm across all the evaluated metrics. Based on these findings, it is evident that CNN is a more suitable choice for the task at hand. In the upcoming section, the researcher focuses on further enhancing the performance and accuracy of the CNN model through parameter tuning. This approach aims to fine-tune the model's hyperparameters and optimize its configuration to achieve even better results in terms of accuracy and overall model metrics.</p>

**Hyperparameter Tuning**

Below is the summary architecture of the tuned CNN model:

<img width="603" alt="model2_summary" src="https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/d80b9810-9335-467a-94b2-a62f8c133520"> <br>

<p align="justify">Figure 5 presents the performance of tuned model. It is evident that the implementation of batch normalization and the addition of dropout layers in the convolutional layers effectively mitigated the overfitting issue encountered in basic CNN. These techniques have successfully improved the model's generalization capability and reduced the discrepancy between training and validation performance.</p><br>

![model1_performance](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/1fff7bf7-a91e-4dab-a12b-abf30382b474) <br>


Basic CNN vs tuned CNN performance comparison:

<img width="696" alt="image" src="https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/b1b42b3a-febe-4a0b-a623-411c0412932f"> <br>

<br>

Sources:<br>
Murray, C. K. & Bennett, J. W., 2009. Rapid Diagnosis of Malaria. Murray, Clinton K, and Jason W Bennett. “Rapid Diagnosis of Malaria.” Interdisciplinary perspectives on infectious diseases.
Mandal, M., 2021. Analytics Vidhya. [Online] 
Available at: https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/
Chan, C., 2022. DisplayR Blog. [Online] 
Available at: https://www.displayr.com/what-is-a-roc-curve-how-to-interpret-it/
Telrandhe, S. R., Pimpalkar, A. & Kendhe, A., 2016. Detection of brain tumor from MRI images by using segmentation & SVM. 2016 World Conference on Futuristic Trends in Research and Innovation for Social Welfare (Startup Conclave).
Brownlee, J., 2020. Machine Learning Mastery. [Online] 
Available at: https://machinelearningmastery.com/confusion-matrix-machine-learning/
[Accessed 20 May 2022].









   
