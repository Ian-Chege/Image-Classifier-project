# Image-Classifier-project

This project consists of two subsections:

  1. Development Notebook
  2. Command Line Application
  
 In the first section, all the necessary packages and modules are imported. This includes:
    
    numpy
    json
    tensorflow
    
The data- Oxford Flowers 102 is loaded using TensorFlow Datasets. A pre-trained network, MobileNet to be used, is loaded using TensorFlow Hub and its parameters are frozen and then a new neural network is created using transfer learning. The number of neurons in the output layer should correspond to the number of classes of the dataset.

In the image processing sub-section; the function *process_image* successfully normalizes and resizes the input image, and the function should return a NumPy array with shape (224,224,3).

The function *predict* successfully takes the path to an image and a saved model, and then returns the top K most probably classes for that image. This is extended to a *matplotlib* figure created displaying an image and its associated top 5 most probable classes with actual flower names.

For the second part; the *predict.py* script successfully reads in an image and a saved Keras model and then prints the most likely image class and it's associated probability.
The *predict.py* script allows users to print out the top K classes along with associated probabilities. This script allows users to load a JSON file that maps the class values to other category names.
  
