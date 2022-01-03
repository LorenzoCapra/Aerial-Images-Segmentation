# Aerial-Images-Segmentation
Pre-processing and semantic segmentation of aerial satellite imagery employing U-Net. The dataset is from Kaggle and can be downloaded at the following link: https://www.kaggle.com/humansintheloop/semantic-segmentation-of-aerial-imagery.
It contains multiple sets of Dubai aerial imagery obtained by MBRSC satellites and annotated with pixel-wise semantic segmentation in 6 classes. The total volume of the dataset is 72 images grouped into 6 larger tiles. The classes are:

1. Building: #3C1098
2. Land (unpaved area): #8429F6
3. Road: #6EC1E4
4. Vegetation: #FEDD3A
5. Water: #E2A929
6. Unlabeled: #9B9B9B

Notice that the amount of available images is quite small, making the training challenging. Moreover images are large in size, so the Python library **patchify** is used to get tinier arrays to train the neural network on. 
An example of the content of a patch (256x256) is here shown:

![img_mask](https://user-images.githubusercontent.com/91314465/147938698-c5245250-357b-4c38-a55d-0ec7d34ab2c2.png)

After training they are unpatchified applying smooth blending (https://github.com/Vooban/Smoothly-Blend-Image-Patches), to recover the predicted mask of original sized images. 
A tailored version for semantic segmentation of a Convolutional Neural Network - **U-Net** - is coded with Keras and then trained, after dividing the dataset in training and testing. After training the ML model can be tested, providing quite accurate results:

![prediction](https://user-images.githubusercontent.com/91314465/147939190-2b088a03-5937-41a2-be55-278b13cc9b7c.png)

The first and second picture represents the patch and its mask respectively. The third one is the prediction coming from the Net.

This project was developed following the guidelines from https://www.youtube.com/watch?v=jvZm8REF2KY&t=1313s and huge inspiration came from his code.

# Critical Issues #
The main problem when training on platforms like Google Colab is the limited RAM, which might lead to the kernel crashing and automatically restarting. This is a common issue when dealing with images and other large memory-eaters files. 

