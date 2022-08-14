from cProfile import label
from calendar import c
from tabnanny import verbose
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# print(y_train[:10])
# print(x_train[0])
'''
# print first train image
number_image = np.array(x_train[0], dtype= 'float')
pixels = number_image.reshape((28, 28))
plt.imshow(pixels,cmap='gray')
plt.show()

# print train image from 0 to 9
fig, ax = plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(10):
    img = x_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# print 10 same train image
no = 8
fig, ax = plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(10):
    img = x_train[y_train == no][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
'''
# normalization
x_train_norm, x_test_norm = x_train/255.0, x_test/255.0

# build a sequential model

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']
              )

history = model.fit(x_train_norm, y_train,epochs = 25, validation_split = 0.2)

# print(history.history.keys())

plt.figure(figsize = (8,6))
plt.plot(history.history['accuracy'], 'r', label = 'training accuracy')
plt.plot(history.history['val_accuracy'],'g', label = 'validation accuracy')
plt.legend()

plt.figure(figsize = (8 ,6))
plt.plot(history.history['loss'],'r', label = 'training loss')
plt.plot(history.history['val_loss'],'g', label = 'validation_loss')
plt.legend()


score = model.evaluate(x_test_norm, y_test, verbose = 0)

for i, x in enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]:.4f}')

predictions = np.argmax(model.predict(x_test_norm), axis= -1)

'''
print(f'actual : {y_test[0:20]}')
print(f'predictions : {predictions[0:20]}')
predictions = model.predict(x_test_norm[8:9])
print(f'the odds of the prediction of 0~9: {np.around(predictions, 2)}')
plt.figure(figsize = (8 ,6))
X2 = x_test[8,:,:]
plt.imshow(X2.reshape(28, 28), cmap = 'gray')
plt.show()
'''

model.save('model.h5')
'''
print(model.summary())
print(model.get_config())
print(model.get_weights())
print(model.count_params())
'''


