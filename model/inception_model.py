import tensorflow as tf
from PIL import Image
import numpy as np
from utils.preprocess import preprocess_image

model = tf.keras.applications.InceptionV3(weights='imagenet')

def generate_caption_inception(image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = preprocess_image(image, target_size=(299, 299))
    image = np.expand_dims(image, axis=0)
    
    predictions = model.predict(image)
    decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(predictions, top=1)[0]
    caption = decoded_predictions[0][1]
    return caption
