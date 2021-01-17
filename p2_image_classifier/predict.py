from PIL import Image
import helper
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path')
    parser.add_argument('model_path')
    # require top_k to be 1 <= k <= len(...something ha)
    parser.add_argument('--top_k', action='store', default=1, type=int)
    parser.add_argument('--category_names', action='store')
    return parser.parse_args()


args = get_args()
model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer':hub.KerasLayer})

prediction, labels = helper.predict(args.image_path, model, args.top_k)

if(args.category_names):
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
        labels = [class_names[label] for label in labels]

print(list(zip(labels, prediction)))