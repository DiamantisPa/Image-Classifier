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

def encode_bovw_descriptor(desc, vocabulary):
    bow_desc = np.zeros((1, vocabulary.shape[0]))
    for d in range(desc.shape[0]):
        distances = np.sum((desc[d, :] - vocabulary) ** 2, axis=1)
        mini = np.argmin(distances)
        bow_desc[0, mini] += 1
    return bow_desc / np.sum(bow_desc)

# Load vocabulary
vocabulary = np.load('vocabulary.npy')

print('Creating index...')
img_paths = []

bow_descs = np.zeros((0, vocabulary.shape[0]))
for folder in train_folders:
    subfolders = os.listdir(folder)
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder, subfolder)
        files = os.listdir(subfolder_path)
        for file in files:
            file_path = os.path.join(subfolder_path, file)
            desc = extract_local_features(file_path)
            if desc is None:
                continue
            bow_desc = encode_bovw_descriptor(desc, vocabulary)

            img_paths.append(file_path)
            bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)

np.save('index.npy', bow_descs)