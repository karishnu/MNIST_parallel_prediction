import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt
import multiprocessing as mp
import datetime

import imgproc as imgproc

number_of_tests = 8

seed = 7
np.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

X_test_samp = X_test

testing_db = imgproc.process_image()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def new_inc_model(weights):
	# create model
	new_model = Sequential()
	new_model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	new_model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	new_model.set_weights(weights)
	new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return new_model

model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model

#Create new model fpr new batch size
new_model = new_inc_model(model.get_weights())
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

fig = plt.figure()

def run(segment, n):

	image_array = testing_db[segment]

	print("Random testing input for " + str(segment) + ": \n\n" + str(image_array) + "\n")

	image_array = image_array.ravel()

	values = new_model.predict(np.array([image_array]), batch_size=1)
	print("Prediction Probability for "+ str(segment) + ": \n\n" + str(values)+ "\n")
	print("Predicted value for " + str(segment) + ": " + str(np.argmax(values)) + "\n")
	#print("Actual value for " + str(segment) + ": " + str(y_test[n]) + "\n")
	print("Finished: " + str(segment) + "\n")

p = [None]*number_of_tests
rand_arr = [None]*number_of_tests

for i in range(0, number_of_tests):
	rand_arr[i] = np.random.randint(0, 10000)

a_1 = datetime.datetime.now()

for i in range(0, number_of_tests):
	run(i, rand_arr[i])

b_1 = datetime.datetime.now()

a = datetime.datetime.now()

for i in range(0, number_of_tests):
	p[i] = mp.Process(target=run, args=(i,rand_arr[i], ))
	p[i].start()

for i in range(0, number_of_tests):
	p[i].join()

b = datetime.datetime.now()

print("Time taken in series: " + str(b_1-a_1))
print("Time taken in parallel: " + str(b-a))




