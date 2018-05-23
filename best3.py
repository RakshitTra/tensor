

import tensorflow as tf 
import keras
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from keras import backend as K
K.tensorflow_backend._get_available_gpus()


import numpy as np

import cv2
import os
l=np.array()
img1 = []
for filename in [os.system("cd /home/ubuntu/test1/test1 && ls")]:
    l=str(filename)
    img = cv2.imread(os.path.join(os.path.expanduser('~'),'test1','test1',l))
    img1.append(img)
img=[]
for filename in [os.system("cd /home/ubuntu/train1/train1 && ls")]:
    l=str(filename)
    img = cv2.imread(os.path.join(os.path.expanduser('~'),'train1','train1',l))
    img2.append(img)
img1=load_images_from_folder('test1')
img2=load_images_from_folder('train1')
x_test=np.array(img1)
x_train=np.array(img2)



import numpy as np

l1=((np.ones(618))*0).tolist()
l2=((np.ones(618))*1).tolist()
l3=((np.ones(618))*2).tolist()
l4=((np.ones(618))*3).tolist()
m1=((np.ones(1978))*0).tolist()
m2=((np.ones(1979))*1).tolist()
m3=((np.ones(1978))*2).tolist()
m4=((np.ones(1453))*3).tolist()


y_test=l1+l2+l3+l4

y_train=m1+m2+m3+m4
y_train=np.array(y_train)
y_test=np.array(y_test)
batch_size = 64
num_classes = 10
epochs = 5

# input image dimensions
img_rows, img_cols = 320, 240

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
input_shape = (img_rows, img_cols, 3)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
