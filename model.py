import os
import csv 
import cv2
import numpy as np 
import sklearn
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import math

#Read in data that I collected myself: 
liner = []
with open('driving_log.csv') as f:
	reader = csv.reader(f)
	for line in reader:
		liner.append(line)

lines = []
straight_angle =[]

#Preprocessing - remove a percentage of the data with 0-degree steering angle, 
# This is so that these images don't dominate the model
for line in liner:
	if math.fabs(float(line[3])) <= 0.01:
		straight_angle.append(line)
	else:
		lines.append(line)

for i in range(0, len(straight_angle), 14):
	lines.append(straight_angle[i])

'''
#Use small subset just to make sure it runs/ trains
#So, I had to cchnage, lines above to liner, DONT forget to change back!!
lines = []
for e in range(0, len(lineys), 6):
	lines.append(lineys[e])
#above needs to be taken out when full test set model is run!!!!!'''

#Method for resizing
def resized(img):
	return cv2.resize(img, (128, 64))

#Training and testing data separated
train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

#Method for adding all 3 camera angles
def random_cam_image(li, imgs, angl):
	for line in li:
		correction_param = 0.12
		cam = np.random.choice(['center', 'left', 'right'])
		if cam == 'center':
			path=line[0]
			image = cv2.imread(path)
			imgs.append(resized(image))
			angl.append(float(line[3]))
		elif cam == 'left':
			path=line[1]
			image = cv2.imread(path)
			imgs.append(resized(image))
			angl.append(float(line[3])+correction_param)
		else:
			path=line[2]
			image = cv2.imread(path)
			imgs.append(resized(image))
			angl.append(float(line[3])-correction_param)
#Figuring out varying correction parameter: Currently unsuccesful, may revisit
'''
		angle = math.fabs(float(line[3]))
		if angle <= 0.01:
			correction_param = 0.15
		elif angle > 0.01 and angle <= 0.15:
			correction_param = 0.17
		elif angle > 0.15 and angle <= 0.3:
			correction_param = 0.19
		elif angle > 0.3 and angle <= 0.45:
			correction_param = 0.21
		elif angle > 0.45 and angle <= 0.6:
			correction_param = 0.23
		elif angle > 0.6 and angle <= 0.75:
			correction_param = 0.25
		elif angle > 0.75 and angle <= 0.9:
			correction_param = 0.27
		else:
			correction_param = 0.28
'''


def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		random.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

#Make two arrays with the data for our model:
			images = []
			angles=[]

#Randomizing which camera is used for image and angle, per batch
			random_cam_image(batch_samples,images, angles)
			'''
			for line in batch_samples:
				path=line[0]
				image = cv2.imread(path)
				images.append(resized(image))
				angles.append(float(line[3]))'''
			#plt.imshow(images[6])
			#plt.show()

#Adding some flipped images:
			for i in range(0, len(images), 8):
				images[i], angles[i] = cv2.flip(images[i],1), (angles[i]*-1)


			#print(current_img_count)
			X_train, y_train = np.array(images), np.array(angles)

			yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((22, 10),(0,0)), input_shape=(64,128,3)))
model.add(Lambda(lambda x: x/27.5 - 0.5))
#model.add(Convolution2D(12, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(12, 3,3,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(48, 3, 3,activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(128, 1,1, activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(128, 1,1, activation='relu'))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1100, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

#Adam optimizer was made use of and mean squared error was used for this Keras Sequential model. 
model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, \
	nb_val_samples = len(validation_samples), nb_epoch=4)

model.save('model.h5')





#Keep for later improvement and development:
'''
Second useles model, cannot get right 
model.add(Cropping2D(cropping=((20, 8),(0,0)), input_shape=(64,128,3)))
model.add(Lambda(lambda x: x/2705 - 0.5))
model.add(Convolution2D(12, 5, 5, subsample=(2,2), activation='relu', name='Conv1'))
model.add(Convolution2D(32, 5, 5, subsample=(2,2), activation='relu', name='Conv2'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu', name='Conv3'))
model.add(Convolution2D(64, 1,1, activation='relu', name='Conv4'))
model.add(Convolution2D(128, 1,1, activation='relu', name='Conv5'))
model.add(Flatten())
model.add(Dense(1164, activation='relu')) # , name='FullyConLayer-1'
model.add(Dropout(0.25))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()
'''
'''
#model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.3))
model.add(Dense(84))
model.add(Dropout(0.4))
model.add(Dense(1))


Basic model before nvidia implementation: 
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))'''