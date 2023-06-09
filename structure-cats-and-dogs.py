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

generator_training = ImageDataGenerator(rescale = 1./255,
                                        rotation_range = 7,
                                        horizontal_flip = True,
                                        shear_range = 0.2,
                                        height_shift_range = 0.07,
                                        zoom_range = 0.2)

generator_test= ImageDataGenerator(rescale = 1./255)

base_training = generator_training.flow_from_directory('dataset/training_set', 
                                                       target_size = (64, 64),
                                                       batch_size = 32,
                                                       class_mode = 'binary')

base_test = generator_test.flow_from_directory('dataset/test_set', 
                                                       target_size = (64, 64),
                                                       batch_size = 32,
                                                       class_mode = 'binary')

classifier.fit_generator(base_training, steps_per_epoch = 4000 / 32,
                         epochs = 5, validation_data = base_test,
                         validation_steps = 1000 / 128)

