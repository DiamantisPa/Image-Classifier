import os
import cv2 as cv
import numpy as np
import json

train_folders = ['imagedb']

sift = cv.xfeatures2d_SIFT.create()

def extract_local_features(path):
    img = cv.imread(path)

    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc

# Extract Database
print('Extracting features...')
train_descs = np.zeros((0, 128))
for folder in train_folders:
    subfolders = os.listdir(folder)
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder, subfolder)
        files = os.listdir(subfolder_path)
        for file in files:
            file_path = os.path.join(subfolder_path, file)
            desc = extract_local_features(file_path)
            train_descs = np.concatenate((train_descs, desc), axis=0)

# Create vocabulary
print('Creating vocabulary...')
term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
trainer = cv.BOWKMeansTrainer(50, term_crit, 1, cv.KMEANS_PP_CENTERS)
vocabulary = trainer.cluster(train_descs.astype(np.float32))
np.save('vocabulary.npy', vocabulary)