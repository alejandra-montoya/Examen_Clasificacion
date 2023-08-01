import numpy as np
import tensorflow as tf
from PIL import Image

class Model:
    def __init__(self):
        self.model = tf.keras.models.load_model('SavedModel101')

    def predict(self, data):
        image = Image.open(data)
        resized_image = image.resize((224,224), Image.LANCZOS)
        pred = self.model.predict(tf.constant([np.array(img) for img in [resized_image]])).argmax(axis=1)
        return pred


