import tensorflow as tf

def preprocess_image(image, target_size=(224, 224)):
    image = tf.image.resize(image, target_size)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image
