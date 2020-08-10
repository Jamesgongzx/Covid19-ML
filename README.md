# Team
1. James Gong
2. Tanraj Saran
3. Tarandeep Natt

# Introduction
COVID-19 (Coronavirus) has become a global pandemic. Around the world, scientists and epidemiologists are working day and night with the goal of developing a vaccine and prevent future infections of the disease. Currently, there is a steep shortage of test kits. To effectively allocate resources, one approach is to use x-ray imaging of the chest to determine to narrow down on whether or not a patient is infected with COVID-19.

The COVID-19 X-ray classification problem is a binary classification problem where we attempt to classify whether the image is of a patient with COVID-19 or not. The problems with this task is that it's difficult to come up with distinct features in the images that guarantee that the image is of a COVID-19 patient. So using a CNN that will learn which features are more important via weights would be a good way to solve this problem.

# Method
To get around the small training data problem, we looked into using transfer learning. Instead of building and training a CNN from scratch, we’ll use a pre-trained models provided by PyTorch. We decided to use ResNet152 as fixed feature extractor by freezing the early convolutional layers of the network except the final layer to make predictions. 

Initially, we started by using the ImageNet dataset, but decided to move towards potentially more useful data provided in the ChestX-ray14 dataset. However, we were ultimately unable to complete an implementation that pre-trained with the ChestX-ray14 dataset and noted it as one of the key points holding back our accuracy (discussing in the conclusion).

20% of training data is split into validation data. A seed was chosen for the randomizer to ensure at least one 0 label in the validation dataset. Preprocessing of the data was done to recenter and resize the images into sizes of 224 x 224. Training images are further preprocessed with random horizontal flips. Lastly, the CNN is fine-tuned by training it for 25 epochs on our training data using SGD and Cross Entropy Loss.

# Experiments
## feature extractors
We experimented with ResNet with 18, 50, 101 and 152 layers pretrained on the ImageNet dataset, CheXNet pretrained on the ChestX-ray14 dataset, and SqueezeNet pretrained on the ImageNet dataset.

## Optimizers
The Adam optimizer is generally considered to be better performing when compared with GD and SGD. However, using Adam optimizer didn't contribute to a higher accuracy for us compared to SGD.

## Pre-trained Data
We started off with the ImageNet dataset, but thought that switching to pre-training on the ["...ChestX-ray14 dataset, which contains 112,120 frontal view X-ray images from 30,805 unique patients"](https://github.com/brucechou1983/CheXNet-Keras) would provide more valuable pre-training data (on top of more data) so we also experimented with trying this approach.

# Results and Conclusion
ResNet152 produced the highest validation accuracy of 79% and training accuracy of 87%. An ensemble of SqueezeNet, CheXNet and multiple runs of ResNet152 produced the final prediction of all trues.

In our implementation, a seed is chosen for the randomizer to ensure at least one 0 label in the validation dataset. To improve our predictions, K-fold cross-validation could have been used to make use of all the data in training. 

No time was spent on hyperparameter tuning for our models. Hyperparameter searches on models, learning rates, momentum, batch sizes, criterions and optimizers could have futher improved our final predictions. 

Our aim was to pre-train with the ChestX-ray14 dataset, but we couldn't find any implementations that allowed us to easily select the data to pre-train with. We did find resources to help with manually pre-training using the ChestX-ray14 dataset but within the time constraints we had, were ultimately unsuccessful in making an architecture that pre-trains with ChestX-ray14, fine tunes with the Covid-19 training set and outputs labels for the Covid-19 test set.

Although we were able to solve the problem of the lack of data to pre-train on, we were not able to solve the problem of lack of quality data to pre-train on (this hinges on the assumption that pre-training with the ChestX-ray14 dataset would provide more valuable data than ImageNet, which is what we pre-trained with). This is a leading factor for why we believe we were not able to fully utilize ChexNet and why it was beat by ResNet152.

One way to improve the problem could be to include the medical history (or other parameters) relevant to each patient, along with the "...x-ray imaging of the chest..." as this could provide a greater degree of certainty (help accuracy) in determining "...whether or not a patient is infected with COVID-19."

The solution could be more valuable if it were more than a label output. For instance, [this video](https://www.youtube.com/watch?v=VJRCj-4E2iU) on ChexNet talks about how the areas of the image the model is looking at to base its decisions on are also outputted (as a heatmap).

As one final suggestion to make the solution more valuable, we could try turning the problem into a multi-class classification problem where each classification is different. This would help us more precisely interpret results of the classification which would allow us to tweak the neural networks more accurately.
