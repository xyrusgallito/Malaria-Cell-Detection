# Malaria-Cell-Detection
Creating an algorithm to automatically detect malaria from cell images using ML and DL models

Malaria, a febrile illness caused by Plasmodium parasites transmitted through mosquito bites, poses a significant global health concern as reported by the World Health Organization (WHO). In their worldwide malaria report for 2021, the WHO documented over 240 million cases of malaria in 2020. The impact of this rise in malaria cases is particularly severe in developing nations, where healthcare systems often face overwhelming challenges (Murray & Bennett, 2009).

To address these challenges and expedite malaria detection, the healthcare sector can leverage the potential of advanced deep learning algorithms and machine learning techniques. By harnessing the power of these cutting-edge technologies, it becomes possible to enhance the speed and accuracy of malaria diagnosis.

Images below show samples of parasitized or infected malaria cells:

![parasitized_cells](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/cb2449d2-1b2f-4818-831c-d4291859906e)

Images below show samples of normal or uninfected malaria cells:

![uninfected_cells](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/2b62f276-8f51-453c-a69e-29fede1afa7e)

The objective of this project is to develop a classification system for distinguishing between normal and abnormal malaria cells.

The dataset consists of a total of 27,558 malaria cell images, comprising 13,779 parasitized cell images and 13,779 uninfected cell images. This balanced dataset provides an equal representation of both classes. However, the raw malaria cell images have varying sizes. Due to the limitations of the laptop used for this project, all images will be resized to 64 by 64 dimensions using a batch size of 48.

The Malaria dataset can be found and downloaded from this link: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria. Below figure shows the distribution of each label or set: 

![bar plots](https://github.com/xyrusgallito/Malaria-Cell-Detection/assets/32282729/5e02078e-d28a-4447-b1c3-abaa0f728d5c)
