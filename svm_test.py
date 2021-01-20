import os
import cv2 as cv
import numpy as np
import json

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

# Search
test_folder = 'imagedb_test'

# Load index
bow_descs = np.load('index.npy')
with open('index_paths.txt', mode='r') as file:
  img_paths = json.load(file)

# Load vocabulary
vocabulary = np.load('vocabulary.npy')

# accuracy parameters
images = 0
succesfull_classification = 0
accuracy_dic = {}

respones = []
test_folder = 'imagedb_test'
folders = os.listdir(test_folder)
labels = []

for p in img_paths:
    labels.append(os.path.split(os.path.split(p)[0])[1])
labels = np.array(labels, np.int32)

print('Result from SVMs...')
folders = os.listdir(test_folder)
for class_folder in folders:
    files = os.listdir(test_folder + "/" + class_folder)
    accuracy_dic[class_folder] = [0, 0, 0];
    for file in files:

        query_img_path = test_folder + "/" + class_folder + "/" + file
        query_img = cv.imread(query_img_path)
        desc = extract_local_features(query_img_path)
        bow_desc = encode_bovw_descriptor(desc, vocabulary)

        distances = np.sum((bow_desc - bow_descs) ** 2, axis=1)
        retrieved_ids = np.argsort(distances)
        retrieved_ids = retrieved_ids.tolist()

        responses = {}
        for folder in os.listdir(test_folder):
            # Load SVM
            svm = cv.ml.SVM_create()
            svm = svm.load('svm_' + folder)

            response = svm.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
            responses[folder] = response[1][0][0]

        result = min(responses, key=responses.get)
        if(result == class_folder):
            succesfull_classification += 1
            accuracy_dic[class_folder][0] += 1
        images += 1
        accuracy_dic[class_folder][1] += 1
    accuracy_dic[class_folder][2] = (accuracy_dic[class_folder][0] / accuracy_dic[class_folder][1]) * 100
for label in accuracy_dic:
    print("For label: " + label + " there were " + str(accuracy_dic[label][0]) + " successful implementations out of " + str(accuracy_dic[label][1]) + " images")
    print("The classification accuracy for the class was: " + str(np.round(accuracy_dic[label][2], 2)) + "%")
print("The total number of successful classifications are " + str(succesfull_classification) + " out of " + str(images) + " images")
print("with total accuracy of " + str(np.round(((succesfull_classification/images)*100), 2)) + "%")