# Initializare pentru a rula pe GPU
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Citirea datelor si preprocesarea de imagini
import cv2
import numpy as np
import os

# Impartirea imaginilor in seturi de date TRAIN/VALIDATION/TEST

# Lista cu numele tuturor imaginilor din folderul data
path = 'data'
img_names = np.array(os.listdir(path))

# Numarul de imagini din fiecare set
n_train = 15000
n_validation = 2000
n_test = 5149

# Incarcam label-urile din fisierele *.txt pentru fiecare set de date

# TRAIN
train_img_names = img_names[:n_train]
with open('train_labels.txt', 'r') as f:
    f.readline()
    train_labels = np.array([int(line.strip().split(',')[1]) for line in f], dtype=np.uint8)

# VALIDATION
validation_img_names = img_names[n_train:n_train+n_validation]
with open('validation_labels.txt', 'r') as f:
    f.readline()
    validation_labels = np.array([int(line.strip().split(',')[1]) for line in f], dtype=np.uint8)

# TEST
test_img_names = img_names[n_train+n_validation:n_train+n_validation+n_test]

# Incarcam imaginile din fisier, le prelucram, iar apoi le memoram sub forma de np array in fisiere de tip *.npy
# Pentru split in seturi de date si pentru a fi citite mai rapid si mai usor la urmatoarea rulare
# Folosit in mod special pentru *.ipynb, unde rulam in mare parte doar modelul de antrenare

for set_name, set_img_names in [('train', train_img_names), ('validation', validation_img_names), ('test', test_img_names)]:
    set_images = []
    for img_name in set_img_names:

        # citim imaginile
        img = cv2.imread(os.path.join(path, img_name))

        # Preprocesare basic - resize si grayscale
        img = cv2.resize(img, (200, 200))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        set_images.append(img)

    # transformam in np array
    set_images = np.array(set_images, dtype=np.uint8)

    np.save(f'{set_name}_images.npy', set_images)

# verificam daca citirea a fost facuta corect
print(train_img_names.shape, train_labels.shape)
print(validation_img_names.shape, validation_labels.shape)
print(test_img_names.shape)


# Incarcam datele din fisierele *.npy
# Diferite celule in *.ipynb, in mod obisnuit nu este necesar
train_images = np.load('train_images.npy')
with open('train_labels.txt', 'r') as f:
    f.readline()
    train_labels = np.array([int(line.strip().split(',')[1]) for line in f], dtype=np.uint8)

validation_images = np.load('validation_images.npy')
with open('validation_labels.txt', 'r') as f:
    f.readline()
    validation_labels = np.array([int(line.strip().split(',')[1]) for line in f], dtype=np.uint8)


# Convolutional Neural Network (CNN)
from tensorflow import keras
from keras import layers

# Definirea retelei neuronale convolutionale
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(200, 200, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    # layers.Dropout(0.25),

    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    # layers.Dropout(0.25),

    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Antrenarea modelului
model.fit(train_images, train_labels, epochs=100, batch_size=64, validation_data=(validation_images, validation_labels))

# Evaluarea modelului pe setul de validare
from sklearn.metrics import confusion_matrix, classification_report, f1_score

validation_pred = model.predict(validation_images)
validation_pred_labels = np.round(validation_pred).astype(int).flatten()
F1 = f1_score(validation_labels, validation_pred_labels)
print("F1 score:", F1)

# matrice de confuzii
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(validation_labels, validation_pred_labels)
print(cm)

sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# raport de clasificare
cr = classification_report(validation_labels, validation_pred_labels)
print(cr)


# Scrierea fisierului de submisie
import csv

test_images = np.load('test_images.npy')

# Predictiile pe setul de test
# Face predictiile, rotunjeste probabilitatile la clasa 0 sau 1 si aplatizeaza array-ul
test_pred = model.predict(test_images)
test_pred_labels = np.round(test_pred).astype(int).flatten()

with open('sample_submissionCNN.csv', mode='w') as submission_file:
    writer = csv.writer(submission_file, delimiter=',')
    writer.writerow(['id', 'class'])
    for i, img_name in enumerate(test_img_names):
        writer.writerow([img_name[:-4], test_pred_labels[i]])


# Test

# 100 epoci

# imag 128, 128 layer - 32 si 32, 64, 64 dropout 0.25; dense 64 dropout 0.5; batch size 64 ~ 56

# imag 128, 128 layer - 32 si 32, 64, 64, 64 dropout 0.25; dense 64 dropout 0.5; batch size 64 ~ 577
# imag 128, 128 layer - 32, 32, 64, 64, 64 dropout 0.25; dense 64 dropout 0.5; batch size 64 ~ 30
# imag 128, 128 layer - 32 si 32, 64, 64, 64 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 571
# imag 128, 128 layer - 32, 32 si 64, 64, 64 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 587  !!! Kaggle 63
# imag 128, 128 layer - 32, 32, 64, 64, 128 dropout 0.25; dense 64 dropout 0.5; batch size 64 ~ 41
# imag 128, 128 layer - 32, 32 si 64, 64, 128 dropout 0.25; dense 64 dropout 0.5; batch size 64 ~ 52
# imag 128, 128 layer - 32, 32 si 64, 64, 128 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 568
# imag 128, 128 layer - 32, 32, 64 si 64, 128 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 50
# imag 128, 128 layer - 32 si 32, 64, 64, 128 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 55
# imag 128, 128 layer - 32, 32, 64, 64, 128 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 50

# imag 150, 150 layer - 32, 32 si 64, 64, 64 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 57

# imag 200, 200 layer - 32, 32 si 32, 64, 64, 128 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 592 !!! Kaggle 64
# imag 200, 200 layer - 32, 32, 32 si 64, 64, 128 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 61
# imag 200, 200 layer - 32, 32 si 64, 64, 64, 128 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 53
# imag 200, 200 layer - 32, 32 si 64, 64, 128, 128 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 585
# imag 200, 200 layer - 16, 16 si 32, 32, 64, 64 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 576
# imag 200, 200 layer - 16, 32 si 32, 64, 64, 128 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 590 !!! Kaggle 656
# imag 200, 200 layer - 16 si 32, 32, 64, 64, 128 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 53
# imag 200, 200 layer - 32, 32 si 64, 64, 64, 64 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 602 !!! Kaggle 67

# 150, 150 layer - 32, 32 si 64, 64, 64 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 57


# 200 epoci

# imag 128, 128 layer - 32, 32 si 64, 64, 128 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 55


# 300 epoci

# imag 128, 128 layer - 32, 32 si 64, 64, 64 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 60
# imag 128, 128 layer - 32, 32, 64, 64, 128 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 53
# imag 128, 128 layer - 32 si 32, 64, 64, 64 dropout 0.25; dense 64 dropout 0; batch size 64 ~ 58

# imag 128, 128 layer - 32, 64, 64, 128, 128 dropout 0.25; dense 64 dropout 0.5; batch size 64 epochs 10 ~ 21
# imag 128, 128 layer - 16, 32, 64, 64, 64 dropout 0.25; dense 64 dropout 0.5; batch size 64 epochs 10 ~ 37
# imag 128, 128 layer - 16, 32, 64, 64, 128 dropout 0.25; dense 64 dropout 0.5; batch size 64 epochs 10 ~ 45
# imag 128, 128 layer - 16, 32, 64, 128, 256 dropout 0.25; dense 64 dropout 0.5; batch size 64 epochs 10 ~ 39
# imag 128, 128 layer - 32, 32, 64, 64, 64 dropout 0.25; dense 64 dropout 0.5; batch size 64 epochs 100 ~ 45
# imag 128, 128 layer - 32, 32, 64, 64, 64 dropout 0.25; dense 64 dropout 0.5; batch size 64 epochs 100 ~ 513
# imag 128, 128 layer - 32, 32, 64, 64, 128 dropout 0.25; dense 64 dropout 0.5; batch size 64 epochs 500 ~ 517