# dogdetector
Streamlit-based web-app to detect dog breeds from input images using a convolutional neural network backend implemented with PyTorch. Trained on the Stanford Dogs dataset. Webapp URL: https://huggingface.co/spaces/devvrath123/DogDetector

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

Since the Stanford Dogs dataset is structured as a series of subfolders for each class which contain the corresponding sample images, the ```ImageFolder``` class provided by PyTorch was used to handle the reading of the dataset. The dataset also has XML annotations provided for each sample image. These annotations specify the bounding box for the dog in each image so that it can be cropped to the corresponding bounding box, since the images contain other subjects/objects as well. Separate training and validation transforms were applied to the train/test sets using image transforms from ```torchvision```. Refer to the individual training notebooks for more info.

### Model Performance

**3 types of models were trained:**
1. **For the webapp:** For the purpose of the webapp, an 80/20 train-test split was used to improve model quality. The 'Eskimo Dog' class was merged with the 'Siberian Husky' class because a majority of the images contained in this class were of Siberian Huskies. Therefore the no. of classes for this model is 119. This model achieves 89.64% training accuracy and 96.14% validation accuracy and is by far the best model.
2. **Using the official 12000:8580 split + annotations applied:** The authors of the datasets specify this split ratio, therefore the network was trained and evaluated on this specific split. Annotations were applied here. This model achieves 86.27% training accuracy and 93.64% validation accuracy.
3. **Using the official 12000:8580 split without annotations:** The network was trained on the same split, but without the annotations applied to the data this time. This model achieves 91.62% training accuracy and 95.78% validation accuracy.

The neural network was trained on these 3 versions of the dataset to demonstrate the legitimacy and quality of the training workflow and network architecture, and also to ensure that the user experience of the webapp is good. All 3 models are available in the repository for testing.

The webapp is capable of predicting the following breeds:

1. Chihuahua
2. Japanese Spaniel
3. Maltese Dog
4. Pekinese
5. Shih-Tzu
6. Blenheim Spaniel
7. Papillon
8. Toy Terrier
9. Rhodesian Ridgeback
10. Afghan Hound
11. Basset
12. Beagle
13. Bloodhound
14. Bluetick
15. Black-And-Tan Coonhound
16. Walker Hound
17. English Foxhound
18. Redbone
19. Borzoi
20. Irish Wolfhound
21. Italian Greyhound
22. Whippet
23. Ibizan Hound
24. Norwegian Elkhound
25. Otterhound
26. Saluki
27. Scottish Deerhound
28. Weimaraner
29. Staffordshire Bullterrier
30. American Staffordshire Terrier
31. Bedlington Terrier
32. Border Terrier
33. Kerry Blue Terrier
34. Irish Terrier
35. Norfolk Terrier
36. Norwich Terrier
37. Yorkshire Terrier
38. Wire-Haired Fox Terrier
39. Lakeland Terrier
40. Sealyham Terrier
41. Airedale
42. Cairn
43. Australian Terrier
44. Dandie Dinmont
45. Boston Bull
46. Miniature Schnauzer
47. Giant Schnauzer
48. Standard Schnauzer
49. Scotch Terrier
50. Tibetan Terrier
51. Silky Terrier
52. Soft-Coated Wheaten Terrier
53. West Highland White Terrier
54. Lhasa
55. Flat-Coated Retriever
56. Curly-Coated Retriever
57. Golden Retriever
58. Labrador Retriever
59. Chesapeake Bay Retriever
60. German Short-Haired Pointer
61. Vizsla
62. English Setter
63. Irish Setter
64. Gordon Setter
65. Brittany Spaniel
66. Clumber
67. English Springer
68. Welsh Springer Spaniel
69. Cocker Spaniel
70. Sussex Spaniel
71. Irish Water Spaniel
72. Kuvasz
73. Schipperke
74. Groenendael
75. Malinois
76. Briard
77. Kelpie
78. Komondor
79. Old English Sheepdog
80. Shetland Sheepdog
81. Collie
82. Border Collie
83. Bouvier Des Flandres
84. Rottweiler
85. German Shepherd
86. Doberman
87. Miniature Pinscher
88. Greater Swiss Mountain Dog
89. Bernese Mountain Dog
90. Appenzeller
91. Entlebucher
92. Boxer
93. Bull Mastiff
94. Tibetan Mastiff
95. French Bulldog
96. Great Dane
97. Saint Bernard
98. Malamute
99. Siberian Husky
100. Affenpinscher
101. Basenji
102. Pug
103. Leonberg
104. Newfoundland
105. Great Pyrenees
106. Samoyed
107. Pomeranian
108. Chow
109. Keeshond
110. Brabancon Griffon
111. Pembroke
112. Cardigan
113. Toy Poodle
114. Miniature Poodle
115. Standard Poodle
116. Mexican Hairless
117. Dingo
118. Dhole
119. African Hunting Dog

Additionally, the models trained on the 12000:8580 split can also predict the Eskimo Dog class.

### Overall Workflow

Uploaded image -> test image transforms applied -> image fed to model -> top prediction and probability returned -> prediction displayed on webapp if probability threshold met (35%), else message displayed informing user that the image doesn't contain a dog

The threshold mentioned above is to avoid false predictions when the input image does not contain a dog.
