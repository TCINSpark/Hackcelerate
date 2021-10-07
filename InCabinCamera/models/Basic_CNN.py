import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from keras.layers.normalization import layer_normalization
from tensorflow.keras.utils import to_categorical

import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

num_class = 5 

X = X/255.0
y = tf.keras.utils.to_categorical(y, num_classes=num_class)


#print(X[100])
#print(y[1000])
print(X.shape[1:])

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.BatchNormalization()

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.BatchNormalization()

# model.BatchNormalization(
#         momentum=0.95, 
#         epsilon=0.005,
#         beta_initializer=RandomNormal(mean=0.0, stddev=0.05), 
#         gamma_initializer=Constant(value=0.9)
#     )
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(num_class))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



model.fit(X, y, batch_size=32, epochs=15, validation_split=0.3)

model.save('emotion_detection_CNN.model')


