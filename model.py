import csv
import cv2
import numpy as np
from keras.models import Sequential

from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D

lines = []

data_location = 'data/'

with open(data_location + 'driving_log.csv')as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
break_line = '\\'  # for windows \\, linux use /

steering_correction = 0.1

with open(file_location + 'driving_log.csv')as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        steering_center = float(line[3])

        steering_left = steering_center + steering_correction
        steering_right = steering_center - steering_correction

        source_path = line[0]
        filename = source_path.split(break_line)[-1]
        current_path = file_location + '/IMG/' + filename
        image_center = cv2.imread(current_path)

        source_path = line[1]
        filename = source_path.split(break_line)[-1]
        current_path = file_location + '/IMG/' + filename
        image_left = cv2.imread(current_path)

        source_path = line[2]
        filename = source_path.split(break_line)[-1]
        current_path = file_location + '/IMG/' + filename
        image_right = cv2.imread(current_path)

        images.extend(image_center, image_left, image_right)
        measurements.extend((steering_center, steering_left, steering_right))

# for line in lines:
#     source_path = line[0]
#     filename = source_path.split(break_line)[-1]
#     current_path = file_location + '/IMG/' + filename
#     image = cv2.imread(current_path)
#     images.append(image)
#     measurement = float(line[3])
#     measurements.append(measurement)


# Image processing
augmented_images, augmented_measurements = [], []
# image flipping
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
# Cropping the image
model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=(160, 320, 3)))
# Normalize
model.add(Lambda(lambda x: x / 255. - 0.5))


# Use of ElU: http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf
model.add(Conv2D(24, 5, strides=(2, 2), activation='elu'))
model.add(Conv2D(36, 5, strides=(2, 2), activation='elu'))
model.add(Conv2D(48, 5, strides=(2, 2), activation='elu'))
model.add(Conv2D(64, 3, activation='elu'))
model.add(Conv2D(64, 3, activation='elu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1, activation='elu'))


# model.summary()
# model.compile(loss='mse', optimizer='adam')
# history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)
# model.save('model_muti_v3.h5')

#
# from sklearn.model_selection import train_test_split
# import sklearn
# # train_samples, validation_samples = train_test_split(samples, test_size=0.2)
#
# # Generator Slows down the training process
# def generator(samples, batch_size=32):
#     num_samples = len(samples)
#     break_line = '\\'
#     correction = 0.35
#
#     while 1:  # Loop forever so the generator never terminates
#         samples = sklearn.utils.shuffle(samples)
#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset:offset + batch_size]
#
#             images = []
#             measurements = []
#
#             for line in batch_samples:
#                 if line[0] != '':
#                     steering_center = float(line[3])
#                     steering_left = steering_center + correction
#                     steering_right = steering_center - correction
#
#                     # center image
#                     source_path = line[0]
#                     filename = source_path.split(break_line)[-1]
#                     current_path = data_location + 'IMG/' + filename
#                     image = cv2.imread(current_path)
#                     images.append(image)
#                     measurement = steering_center
#                     measurements.append(measurement)
#
#                     # left image
#                     source_path = line[1]
#                     filename = source_path.split(break_line)[-1]
#                     current_path = data_location + 'IMG/' + filename
#                     image = cv2.imread(current_path)
#                     images.append(image)
#                     measurement = steering_left
#                     measurements.append(measurement)
#
#                     # center image
#                     source_path = line[2]
#                     filename = source_path.split(break_line)[-1]
#                     current_path = data_location + 'IMG/' + filename
#                     image = cv2.imread(current_path)
#                     images.append(image)
#                     measurement = steering_right
#                     measurements.append(measurement)
#
#             augmented_images, augmented_measurements = [], []
#             # image flipping
#             for image, measurement in zip(images, measurements):
#                 augmented_images.append(image)
#                 augmented_measurements.append(measurement)
#                 augmented_images.append(cv2.flip(image, 1))
#                 augmented_measurements.append(measurement * -1)
#
#             X_train = np.array(augmented_images)
#             y_train = np.array(augmented_measurements)
#             yield sklearn.utils.shuffle(X_train, y_train)
# # # Set our batch size
# # batch_size=50
#
# # # compile and train the model using the generator function
# # train_generator = generator(train_samples, batch_size=batch_size)
# # validation_generator = generator(validation_samples, batch_size=batch_size)
# # print(train_generator)
#
# # model.fit_generator(train_generator,
# #             steps_per_epoch=math.ceil(len(train_samples)/batch_size),
# #             validation_data=validation_generator,
# #             validation_steps=math.ceil(len(validation_samples)/batch_size),
# #             epochs=8)
# model.fit(train_generator, validation_data = validation_generator, epochs= 5)

model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4)
model.save('model_sample.h5')

