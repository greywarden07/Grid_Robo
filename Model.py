import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten
from keras.models import Sequential

import cv2

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




path="/kaggle/input/fresh-and-stale-images-of-fruits-and-vegetables"

data_gen=ImageDataGenerator(
rescale=1.0/255,
rotation_range=20,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
fill_mode="nearest",
horizontal_flip=True
)
data=data_gen.flow_from_directory(
path,
target_size=(256,256),
batch_size=32,
class_mode="categorical"
)


model=Sequential()
model.add(Conv2D(256,(3,3),activation="relu",input_shape=(256,256,3)))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dense(32,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(16,activation="relu"))

model.add(Dense(12,activation="softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy","mae"])


model.summary()

model.fit(data,batch_size=32,epochs=10)
model.save('/kaggle/working/fresh_and_stale_fruit_detector_mode.h5')


model_path = '/kaggle/working/fresh_and_stale_fruit_detector_model.h5'
model = tf.keras.models.load_model(model_path)


image_path="/kaggle/input/fresh-and-stale-images-of-fruits-and-vegetables/stale_tomato/Copy of IMG_20200727_223202.jpg_0_1614.jpg"
img=image.load_img(image_path,target_size=(256,256))
img_array=image.img_to_array(img)
img_array=np.expand_dims(img_array,axis=0)
img_array=img_array/255.0

tahmin=model.predict(img_array)

siniflar=[    
"fresh_apple",
"fresh_banana",
"fresh_bitter_gourd",
"fresh_capsicum",
"fresh_orange",
"fresh_tomato",
"stale_apple",
"stale_banana",
"stale_bitter_gourd",
"stale_capsicum",
"stale_orange",
"stale_tomato"
]

enyuksek=np.argmax(tahmin[0])
sonuc=siniflar[enyuksek]

print(sonuc)



