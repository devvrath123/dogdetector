# dogdetector
Streamlit-based web-app to detect dog breeds from input images using a convolutional neural network backend implemented with PyTorch. Trained on the Stanford Dogs dataset. 
Webapp URL: https://huggingface.co/spaces/devvrath123/dogdetector

## Frontend & Deployment

The frontend was implemented using the Streamlit library. The interactive webapp allows the user to upload an image and receive a prediction. There are also FAQ and Prediction Metrics pages. The FAQ page contains important information and the metrics page displays the most popular prediction along with a chart containing the top 10 predictions. The webapp was containerized using Docker and deployed to HuggingFace spaces.

## Architecture, Data Processing + Training & Overall Workflow

The convolutional neural network uses the ConvNext Small architecture developed at Meta in 2020. It is a modern pure convolutional neural network designed to outperform traditional CNN architectures and match and sometimes even outperform vision transformers. The network was trained using **Transfer Learning** with pre-trained ImageNetV1 weights. Transfer Learning is a technique used to adapt a network with weights trained on one dataset to perform well on another, but related dataset. In this case, the Stanford Dogs dataset, a subset of ImageNet is used. It has 120 classes. 

### Architecture

The ConvNext Small consists of an input convolutional layer (224x224 input image and a 4x4 kernel with stride 4 to downsample the image, followed by LayerNorm. Output resolution is 56x56) followed by 4 'stages' and then a classification head. Each stage consists of a series of convolutional blocks. LayerNorm normalizes across the feature dimension for each sample, while BatchNorm normalizes across the batch for each feature.

**Stages:**
1. **Stage 1:** 56x56 input, 96 filters, 3 blocks
2. **Stage 2:** 28x28 input, 192 filters, 3 blocks
3. **Stage 3:** 14x14 input, 384 filters, 27 blocks
4. **Stage 4:** 7x7 input, 768 filters, 3 blocks

There are also downsampling layers which occur before stages 2, 3 and 4. They consist of a LayerNorm followed by a convolution with 2x2 kernel and stride 2.

Each convolutional block follows the following structure:

1. **Depthwise Convolution:** 7x7 kernel, 3 groups (each input channel is convolved separately), padding = 3
2. **Layer Normalisation** (applied to channel dimensions)
3. **Pointwise Convolution 1:** 1x1 kernel (expands no. of filters 4x)
4. **GELU** (Gaussian Error Linear Unit) activation function
5. **Pointwise Convolution 2:** 1x1 kernel (reduces no. of filters back from 4x to 1x)
6. **LayerScale:** Learnable scaling factor (learnable diagonal matrix) for each channel. Multiplied to each channel before being added to the skip connection below
7. **Residual Connection:** The output is added back to the original input of the block

The classification head has the following structure:

1. **Average Pooling:** Reduces the 7x7 input into 1x1
2. **LayerNorm**
3. **Linear Layer:** 768 neurons to no. of classes

### Data Processing + Training

Data processing and training were implemented in PyTorch. Training was conducted on a **v5-e8 TPU** (Tensor Processing Unit) with multi-core processing for superior training speed relative to CPU/GPU training. TPU training was made possible using the ```torch_xla``` package.

The 1st phase of transfer learning involves training only the last classification layer of the network to adapt the weights to the specific classification task at hand. After, there is a fine tuning phase which involves training all parameters of the network, however with a very small learning rate so as to not completely destroy the learnt weights and further improve accuracy. Models trained with transfer learning may exhibit higher validation accuracy than training accuracy; this is due to the use of pre-trained weights which were trained a larger dataset. Both phases of training use a learning rate scheduler which reduces learning rate if validation accuracy does not improve for a certain no. of epochs.

Since the Stanford Dogs dataset is structured as a series of subfolders for each class which contain the corresponding sample images, the ```ImageFolder``` class provided by PyTorch was used to handle the reading of the dataset. The dataset also has XML annotations provided for each sample image. These annotations specify the bounding box for the dog in each image so that it can be cropped to the corresponding bounding box, since the images contain other subjects/objects as well. Separate training and validation transforms were applied to the train/test sets using image transforms from ```torchvision```. Specifically, data augmentation was applied to the training set to improve model performance. Refer to the individual training notebooks for more info.

### Model Performance

**3 types of models were trained:**
1. **For the webapp:** For the purpose of the webapp, an 80/20 train-test split was used to improve model quality. The 'Eskimo Dog' class was merged with the 'Siberian Husky' class because a majority of the images contained in this class were of Siberian Huskies. Therefore the no. of classes for this model is 119. This model achieves 89.64% training accuracy and 96.14% validation accuracy and is by far the best model.
2. **Using the official 12000:8580 split + annotations applied:** The authors of the datasets specify this split ratio, therefore the network was trained and evaluated on this specific split. Annotations were applied here. This model achieves 86.27% training accuracy and 93.64% validation accuracy.
3. **Using the official 12000:8580 split without annotations:** The network was trained on the same split, but without the annotations applied to the data this time. This model achieves 91.62% training accuracy and 95.78% validation accuracy.

The neural network was trained on these 3 versions of the dataset to demonstrate the legitimacy and quality of the training workflow and network architecture, and also to ensure that the user experience of the webapp is good. All 3 models are available in the repository for testing.

The webapp is capable of predicting the [following breeds](https://github.com/devvrath123/dogdetector/blob/main/Breeds.md).

### Overall Workflow

Uploaded image -> test image transforms applied -> image fed to model -> top prediction and probability returned -> prediction displayed on webapp if probability threshold met (35%), else message displayed informing user that the image doesn't contain a dog

The threshold mentioned above is to avoid false predictions when the input image does not contain a dog.
