#import necessary modules
import numpy as np
import json
import sys
import os
import argparse
import tensorflow as tf
import tensorflow_hub as hub 
from PIL import Image

#definition of different function bodies namely:
#    1. process_image
#    2. predict function
#    3. main function


def process_image(img):
    img=tf.convert_to_tensor(img)
    img = tf.image.resize(img, (224, 224))
    img/=255.0
    return img.numpy()

def predict(img_pt,mdl,t_k):
    '''
    function predict takes an image from an image path
    takes a model
    and returns the top k most likely labels and
    the correct probabilities
    '''
    img=Image.open(img_pt)
    img=np.asarray(img)
    #pass the image to the process image function
    img=process_image(img)
    #dimension adjustment to cater for the batch size
    dim_adj=np.expand_dims(img,axis=0)
    #takes a model
    pred=mdl.predict(dim_adj)
    #part for the returning the top k most likely label and probability
    proba,clas=tf.nn.top_k(pred,t_k)
    proba=proba.numpy().squeeze()
    cls_l=clas.numpy().squeeze()
    clas=[class_names[str(i+1)] for i in cls_l]
    return proba, clas

#definition of the main function
if __name__ == '__main__':
       
    #initialize the parser
    parser=argparse.ArgumentParser()
    #parse the different parameters now
    parser.add_argument('img_pt')
    parser.add_argument('--t_k',default=5,type=int)
    parser.add_argument('--cls_num',default='label_map.json')
    parser.add_argument('testing_mdl')
    args=parser.parse_args()
    #define image path in the arg mode
    img_pt=args.img_pt
    #define top_k in the arg mode
    t_k=args.t_k
    #from the label mapping code
    with open(args.cls_num, 'r') as f:
        class_names = json.load(f)
                        
    #bring now the trained model
    mdl=tf.keras.models.load_model(args.testing_mdl,custom_objects={'KerasLayer':hub.KerasLayer},compile=False)
    
    #from the model training section
    clas,proba=predict(img_pt,mdl,t_k)
    print('Names of flowers-',clas)
    print('Probability',proba)