import os
import cv2 as cv
import numpy as np
import json

train_folders = ['imagedb']

# Load index
bow_descs = np.load('index.npy')
with open('index_paths.txt', mode='r') as file:
  img_paths = json.load(file)

print('Training SVMs...')
for folder in train_folders:
    subfolders = os.listdir(folder)
    for subfolder in subfolders:
        svm = cv.ml.SVM_create()
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setKernel(cv.ml.SVM_LINEAR)
        # svm.setDegree(2)
        svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))

        svm_labels = np.array([subfolder in p for p in img_paths], np.int32)
        svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, svm_labels)
        svm.save('svm_' + subfolder)