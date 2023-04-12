import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()
classifier.add(Conv2D(32, (3,3), input_shape = (64, 64, 3)))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
               metrics = ['accuracy'])

