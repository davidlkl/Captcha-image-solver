# Built-in Lib
import os
import string

# External Lib
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU
from keras.layers import BatchNormalization, RepeatVector
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed
from seq2seq.models import Seq2Seq
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Global Variables
TRAIN_DIR = 'train'
IMAGE_SHAPE = (80,200,3)
MAX_CHAR_NUM = 6
MIN_CHAR_NUM = 4
# Classification of individual character
LABEL_ENCODER = LabelEncoder()
LABEL_ENCODER.fit(list(string.ascii_lowercase + string.ascii_uppercase + ' '))
# Total number of possible characters
NUM_CHAR_CLASS = len(LABEL_ENCODER.classes_)

# Vectorize a string with classification
def text_to_vector(text):
    # Example: 'abcd' --> ['a','b','c','d',' ',' ']
    textVector = list(text) + (MAX_CHAR_NUM - len(text)) * [' ']
    # Transform the list of char into list of one-hot vectors
    return LABEL_ENCODER.transform(textVector)

def make_predictions(model, X_test):
    prediction_lists = LABEL_ENCODER.inverse_transform(model.predict_classes(X_test))
    prediction_strings = []    
    for prediction in prediction_lists:
        prediction_strings.append(''.join(prediction))
    return prediction_strings
        
# Load image data from directory
# Return image and label arrays
def load_data():
	import cv2
    images, labels = [], []

    for (dirpath, dirnames, filenames) in os.walk(TRAIN_DIR):
        # Read the .png files only
        filenames = [x for x in filenames if x.split('.')[1]=='png']
        for filename in filenames:
            # Get the english word
            label = filename.split('.')[0]
	    # Encode the string into one-hot vectors
            labels.append(text_to_vector(label))
	    # Read image
            image = cv2.imread(os.path.join(TRAIN_DIR,filename))
            images.append(image)

    images = np.array(images)
    labels = np.asarray([to_categorical(label, NUM_CHAR_CLASS) for label in labels])
    return images, labels


# Return a rnn+cnn model instance
def create_model(image_shape, max_caption_len, vocab_size):
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=image_shape, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3),kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3),kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(Dropout(0.4))
    model.add(RepeatVector(1))
    model.add(Seq2Seq(input_dim=256, input_length=1, hidden_dim=256,
	      output_length=max_caption_len, output_dim=256, peek=True))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    # sgd seems work well in this model (the val_loss decreases smoothly)
    sgd = SGD(lr=0.002, momentum=0.9, nesterov=True, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def train():
	# Randomization
	seed = 7
	np.random.seed(seed)

	# Load data from directory
	X, Y = load_data()

	# train : val : test = 0.68 : 0.17 : 0.15
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.15, random_state=seed)
	X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size=0.2, random_state=seed)

	# Create the defined model
	model = create_model(IMAGE_SHAPE, MAX_CHAR_NUM, NUM_CHAR_CLASS)

	# Callback for visualization of training progress
	tensorboard =  TensorBoard(log_dir="./logs")
	# Callback for saving the model with lowest validation loss
	modelcheckpoint= ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5',
					monitor='val_loss',
					verbose=0,
					save_best_only=True)

	# Fit the model with mini-batch
	history = model.fit(X_train,Y_train,
				batch_size=64,
				validation_data=(X_val,Y_val),
				epochs=400,
				verbose=2,
				shuffle=True,
				callbacks=[tensorboard, modelcheckpoint])


	predictions = model.predict(X_test)
	results = []
	targets = []
	for prediction in predictions:
		results.append(
		"".join(LABEL_ENCODER.inverse_transform([x.argmax() for x in prediction])).replace(" ","")
		)
	for target in Y_test:
		targets.append(
		"".join(LABEL_ENCODER.inverse_transform([x.argmax() for x in target])).replace(" ","")
		)
	import pandas as pd
	df = pd.DataFrame([results,targets]).T
	df.to_csv('results.csv')

