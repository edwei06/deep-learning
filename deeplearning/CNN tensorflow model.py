from sklearn import metrics
import tensorflow as tf
from matplotlib import pyplot as plt
#load mnist hand written number file
cifar10 = tf.keras.datasets.cifar10
#train/ test data X/y dimensions
(x_train, y_train), (x_test, y_test) =  cifar10.load_data()
# normalization
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
x_train_norm, x_test_norm = x_train/255.0 , x_test/255.0
x_train_norm[0]

#build a model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer = 'adam',
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
      metrics = ['accuracy'])

#model training

history = model.fit(x_train_norm, y_train, epochs = 10, validation_split = 0.2)
print(history.history.keys())

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'],'r',label = 'training accuracy')
plt.plot(history.history['val_accuracy'],'g',label = 'validation accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'],'b',label = 'training loss')
plt.plot(history.history['val_loss'],'g',label = 'validation loss')
plt.legend()
plt.show()

#score model

score = model.evaluate(x_test_norm, y_test, verbose = 0)
for i, x in  enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]:.4f}')

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc = 'lower right')
plt.show()

#save the model

model.save('model_CNN.h5')

print(model.summary())
print(model.get_config())
w = model.get_weights()
print(model.count_params())








