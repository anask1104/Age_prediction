import cv2
import pandas as pd
import numpy as np
import os
import keras
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten,InputLayer, LeakyReLU


TRAIN_DIR = 'D:\project\Ageprediction\TRAIN'
TEST_DIR = 'D:\project\Ageprediction\TEST'
IMG_SIZE = 32
LR = 1e-3

train=pd.read_csv("D:\project\Ageprediction\\train.csv")
test=pd.read_csv("D:\project\Ageprediction\\test.csv")


print(test.shape)

def create_train_data():
    training_data = []
    for image in train.ID:
        path = os.path.join(TRAIN_DIR, image)
        img = cv2.imread(path,-1)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append(img.astype('float32'))

    return training_data


def create_test_data():
    testing_data = []
    for image in test.ID:
        path = os.path.join(TEST_DIR, image)
        img = cv2.imread(path,-1)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append(img.astype('float32'))


    return testing_data



training_set=create_train_data()
testing_set=create_test_data()

training_set=np.stack(training_set)/255;
testing_set_norm=np.stack(testing_set)/255;


lbenc = LabelEncoder()
train_y = lbenc.fit_transform(train.Class)
train_y = keras.utils.np_utils.to_categorical(train_y)


shape=(IMG_SIZE,IMG_SIZE,3)

def create():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',activation = 'relu', input_shape=shape))
    #model.add(LeakyReLU(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.20))

    model.add(Conv2D(32, (3, 3), padding='same',activation = 'relu'))
    #model.add(LeakyReLU(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.20))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    return model


model1 = create()
batch_size = 256
epochs = 50
model1.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model1.fit(training_set, train_y, batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.2)

y_pred = model1.predict_classes(testing_set_norm)

y_pred_label=lbenc.inverse_transform(y_pred)
# for i in range(len(testing_set)):
#     cv2.imwrite('predicted\img' + str(i) + '_' + str(y_pred[i]) + '.jpg', testing_set[i])

df = pd.DataFrame(test,y_pred_label)
df.to_csv("result.csv")

