#project is handwritten character recognition using neural networks

#step 1 initialise project and collect data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy

#(train x&Y),(test x&y) = src(data.load_data())
(x_train,y_train),(x_test,y_test)=mnist.load_data()

 #try to visualise the data you are dealing with to understand it better 
 #in this case the optical characters
print(np.size(x_train))
print(np.size(x_test))

print(np.min(x_train))
print(np.max(x_train))


#our values range between 0 and 255 which is too big a range
#we need to bring this data to a normalised range
#we divide by value the will shrink each item and put it between 0-1 which is 255
x_train=x_train/255.0



print(np.max(x_train[0]))
print(np.min(x_train[0]))

#now our data is normalised 
print(np.size(y_train))
print(np.min(y_train))
print(np.max(y_train))
#checking the current shape of the data since cnn requires(num_sample,height,width,) as shape
print(np.shape(x_train[0]))
print(np.shape(x_test[0]))
#current shape is (28,28)which is in pixels
#we want (28,28,1) so we need to add that 1 dimension that represents channel(greyscale)
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test =x_test.reshape(x_test.shape[0],28,28,1)

#next issue is labels are still 0-9
#One-hot encoding the labels to make the categorical
#to_categorically(labels,n)
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)
#preproccesing is done as now the features and labels are good to be analysed


#defining the model
#our model will have sequntial and dense layers
model = Sequential([
      Conv2D(32,(3,3), activation ='relu',input_shape=(28,28,1)),
      MaxPooling2D(2,2),
      Dropout(0.35),
      Conv2D(64,(3,3),activation ='relu'),
      MaxPooling2D(2,2),
      Dropout(0.35),
      #passing our output from con layers to dense or FC
      Flatten(),
      Dense(128,activation ='relu'),
      Dropout(0.35),
      Dense(10,activation='softmax')
])

#compiling our model
model.compile(
  optimizer ='adam',
  loss='categorical_crossentropy',
  metrics=['Accuracy']

 )

#training the model
history = model.fit(
    x_train,
    y_train,
    batch_size =128,
    epochs=10,
    verbose=2,
    validation_split=0.2
)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('mnist_cnn_model.h5')