import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
model=tf.keras.models.load_model('model_ex-024_acc-0.996875.h5')

# imp => tensorflow version 1.15.0
import cv2
import matplotlib.pyplot as plt
img_array = cv2.imread('30.jpg')  # convert to array


img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

img_rgb = cv2.resize(img_rgb,(224,224),3)  # resize
img_rgb = np.array(img_rgb).astype(np.float32)/255.0  # scaling
img_rgb = np.expand_dims(img_rgb, axis=0)  # expand dimension
print(model.predict(img_rgb))

array = model.predict(img_rgb)
result = array[0]
answer = np.argmax(result)
if answer == 0:
    result="Not Solan de Cabras"
elif answer == 1:
    result="Solan de Cabras"
else:
    result="Not Sure" 
print(answer)

