# Bird-Classification
This project showcases a ResNet-34 model trained on 450 bird species images, achieving 98.6% test accuracy. The model was fine-tuned rom a pre-trained ResNet-34 model on the CIFAR-10 dataset. The ResNet-34 model was first trained from scratch on CIFAR-10 Dataset and achieved test accuracy of 94% it is available [here](https://github.com/Moddy2024/ResNet-34) using this as a backbone and an extra max-pooling layer in the network the model was trained for 30 epochs which took 8 hours on a P100 GPU with Adam Optimizer and COSINE Annealing. The model was trained for 7 epochs more with different combination data augmentations so it can classify birds correctly in different places in an image with a lower learning rate on AWS Sagemaker on a NVIDIA V100 GPU for about 5 hours. This was done so a higher testing accuracy can be achieved. 
# Dependencies
* [PyTorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/)
* [PIL](https://pypi.org/project/Pillow/)
* [Numpy](https://numpy.org/)
* [OS](https://docs.python.org/3/library/os.html)
* [Torchvision](https://pytorch.org/vision/stable/index.html)
* [torchinfo](https://github.com/TylerYep/torchinfo)

Once you have these dependencies installed, you can clone the Bird Classification repository from GitHub:
```bash
https://github.com/Moddy2024/Bird-Classification.git
```
# Key Files
* [training.ipynb](https://github.com/Moddy2024/Bird-Classification/blob/main/training.ipynb) - This file shows how the dataset has been downloaded, how the data looks like, the transformations, data augmentations, architecture of the ResNet the training and the validation and test accuracy.
* [training-sagemaker.ipynb](https://github.com/Moddy2024/Bird-Classification/blob/main/training-sagemaker.ipynb) - In this file I did even more data augmentations and trained for 7 more epochs with a lower learing rate to achieve higher accuracy.
* [prediction.ipynb](https://github.com/Moddy2024/Bird-Classification/blob/main/prediction.ipynb) - This file loads the trained model file and shows how to do predictions on single images, multiple images contained in a folder and images(multiple or single) that can be uploaded to google colab temporarily to perform the prediction.
* [trained-models](https://github.com/Moddy2024/Bird-Classification/tree/main/trained-models) - This directory contains the best trained model and the trained model saved after the last epoch.
* [test-data](https://github.com/Moddy2024/Bird-Classification/tree/main/test-data) - This directory contains test images collected randomly from the internet of different categories, sizes and shape for performing the predictions and seeing the results.
# Performance Evaluation
The model achieved an accuracy of 98.6% on the test set.
# Training and Validation Image Statistics
The bird classification dataset consists of 450 bird species with 70,626 training images, 2,250 test images (5 images per species) and 2,250 validation images (5 images per species). The images are high quality, with only one bird in each image and the bird taking up at least 50% of the image. The images are all 224 X 224 X 3 color images in JPG format, and are structured into train, test, and validation sets with 450 sub-directories, one for each bird species. The dataset also includes a CSV file, birds.csv, which contains information such as file paths, labels, scientific names, and class IDs. The images were gathered from internet searches by species name, checked for duplicates, cropped to ensure the bird takes up at least 50% of the image, and resized to 224 X 224 X 3 in JPG format. Please note that the training set is not balanced, with a varying number of files per species, and each species has at least 130 training image files. The test and validation images were hand-selected to be the "best" images, so a model's accuracy score is likely to be highest using these datasets.

A significant shortcoming of the dataset is the imbalance of male to female images, with 80% of the images being of males and 20% of females. Males tend to be more diverse in color, while females tend to be bland, meaning male and female images may look entirely different. As a result, almost all test and validation images are taken from males, and the classifier may not perform as well on female images.
# Dataset
The BIRDS 450 SPECIES dataset can be downloaded from [here](https://www.kaggle.com/datasets/gpiosenka/100-bird-species).

In this repository the dataset has been downloaded using wget command in the terminal.
```bash
!wget --no-check-certificate \
    "https://storage.googleapis.com/kaggle-data-sets/534640/4269088/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230118%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230118T114015Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=74e20b8164ad4532398558fa350cab86e2f138b95274ef7e6a8f2084f8ff847cc4d907f1a86e879ea255e3ecabb3985b729b2826e6b307a2dfbd73b22f6b4249070581333089cc3c048d14ee21f030ad3f7f2d3850f2774300b8dea9861f69b7072c38d48cb566c9d824adee801e87a2612c1b686e145341ccdadb0b252746ec7a6e7c5d89036717c9ff104a58e29c4580e6897290d8954baff56b8ba04e830e3cbb3bd31cb66b5dcf98a66ed2ccc40d8c338de6a323a997144756aaf91529c61f3db84e0f7a82a294345c4c27d2ae4a8165a536a47d3ae2b55563cbd38a3af52ca05a45e5d1065fa03e16a6d1220b624aa6b9ac8cbc6313e7da3037259c93be" \
    -O "/home/ec2-user/SageMaker/archive.zip"

local_zip = '/home/ec2-user/SageMaker/archive.zip'
zip_ref   = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/home/ec2-user/SageMaker/')
zip_ref.close()
os.remove(local_zip)
```
