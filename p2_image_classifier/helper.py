
from PIL import Image
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def process_image(image):
    image_size = 224
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image,[image_size,image_size])
    image /= 255
    return image.numpy()    

def predict(image, model,k):
    image = Image.open(image)
    image = np.asarray(image)
    processed_image = process_image(image)
    to_predict = np.expand_dims(processed_image,axis=0)    
    pred = model.predict(to_predict)[0]            
    i = np.argpartition(pred, -k)[-k:]
    classes = i[np.argsort((-pred)[i])]    
    probs = pred[classes]        
    return probs,[str(int+1) for int in classes]
    

