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

# Undersampling
# Selectearea unui numar dat de imagini cu label 0
zero_idx = np.where(train_labels == 0)[0]
np.random.shuffle(zero_idx)
zero_idx = zero_idx[:3500]

# Selectarea tuturor imaginilor cu label 1
one_idx = np.where(train_labels == 1)[0]

# Concatenam np array urile
train_idx = np.concatenate([zero_idx, one_idx])

# Amestecarea indicilor pentru a evita o anumita ordine
np.random.shuffle(train_idx)

# Load the selected training images and labels
train_img_names = img_names[train_idx]
train_labels = train_labels[train_idx]
np.save('train_labels.npy', train_labels)

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
import numpy as np

train_images = np.load('train_images.npy')
train_images = train_images.reshape(train_images.shape[0], -1)
train_labels = np.load('train_labels.npy')

validation_images = np.load('validation_images.npy')
validation_images = validation_images.reshape(validation_images.shape[0], -1)
with open('validation_labels.txt', 'r') as f:
    f.readline()  # skip header
    validation_labels = np.array([int(line.strip().split(',')[1]) for line in f], dtype=np.uint8)

# Random Forrest Classifier
from sklearn.ensemble import RandomForestClassifier

# Definirea „padurii” arborilor de decizie: 100 de arbori de decizie, de inaltime/adancime 10
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Antrenarea modelului
rf.fit(train_images, train_labels)

# Evaluarea modelului pe setul de validare
from sklearn.metrics import confusion_matrix, classification_report, f1_score

validation_pred = rf.predict(validation_images)
F1 = f1_score(validation_labels, validation_pred)
print("F1 score:", F1)

# matrice de confuzii
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(validation_labels, validation_pred)
print(cm)

sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# raport de clasificare
cr = classification_report(validation_labels, validation_pred)
print(cr)

# Scrierea fisierului de submisie
import csv

# Incarcarea imaginilor de test
test_images = np.load('test_images.npy')
test_images = test_images.reshape(test_images.shape[0], -1)

# Predictii pe setul de test
test_pred = rf.predict(test_images)

with open('sample_submissionRFC.csv', mode='w') as submission_file:
    writer = csv.writer(submission_file, delimiter=',')
    writer.writerow(['id', 'class'])
    for i, img_name in enumerate(test_img_names):
        writer.writerow([img_name[:-4], test_pred[i]])
