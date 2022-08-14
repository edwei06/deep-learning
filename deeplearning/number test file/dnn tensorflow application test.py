import tensorflow as tf
from skimage import io
from skimage.transform import resize
import numpy as np


model = tf.keras.models.load_model(r"G:\vs code file\deep learning file\number test file\model.h5")

for i in range(14):
    uploaded_file = f'G:/vs code file/deep learning file/number test file/{i}.png'
    image1 = io.imread(uploaded_file, as_gray = True)

    image_resized =resize(image1, (28, 28), anti_aliasing = True)
    X1 = image_resized.reshape(1,28,28)
    X1 = np.abs(1-X1)

    predictions = np.argmax(model.predict(X1), axis=1) 
    print(predictions,i)








