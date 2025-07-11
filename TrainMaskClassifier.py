import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

DATADIR = r"C:\Users\Hp\Desktop\Internship\MaskDetection\data"
CATEGORIES = ["with_mask", "without_mask"]
IMG_SIZE = 100
data = []

print("Info loading and preprocessing images...")

for category in CATEGORIES:
    path=os.path.join(DATADIR, category)
    label= CATEGORIES.index(category)       # 0 for with_mask, 1 for without_mask
    for img_name in os.listdir(path):
        img_path=os.path.join(path, img_name)
        try:
            img=cv2.imread(img_path)
            img=cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append([img, label])
        except:
            pass                           #skip images that cannot be read

X=[]
Y=[]
for features, label in data:
    X.append(features)                 #pixels
    Y.append(label)                    #labels

X=np.array(X).reshape(-1, IMG_SIZE,IMG_SIZE, 3)                 # Reshape to (num_samples[-1 means auto], IMG_SIZE, IMG_SIZE, 3)
X=X/255.0                              # Normalize pixel values from [0, 255] to [0, 1]
Y= to_categorical(Y)                   # Convert labels to one-hot encoding

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, random_state=42)
#X_train  -  Train CNN
#Y_train  -  0s and 1s for training imgs
#X_test   -  Never shown to model before
#Y_test   -  Used to evaluate accuracy

model= Sequential()

#1st Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))       
model.add(MaxPooling2D(pool_size=(2, 2)))           

#2nd Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))        
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())              # Flatten the 2D output to 1D for the Dense layer

# Fully Connected Layer
model.add(Dense(128, activation='relu'))    
model.add(Dropout(0.5))            # prevent overfitting, drops 50% of neurons randomly

# Output Layer
model.add(Dense(2, activation='softmax'))   



#Compiling model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#Training the model
model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))

# Saving the model
model.save(r"C:\Users\Hp\Desktop\Internship\MaskDetection\mask_classifier.h5")
print("Model trained and saved successfully.")