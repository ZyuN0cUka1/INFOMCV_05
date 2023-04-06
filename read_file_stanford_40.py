from sklearn.model_selection import train_test_split
import cv2 as cv
import numpy as np
from sklearn.preprocessing import LabelEncoder

keep_stanford40 = ["applauding", "climbing", "drinking", "jumping", "pouring_liquid", "riding_a_bike", "riding_a_horse",
                   "running", "shooting_an_arrow", "smoking", "throwing_frisby", "waving_hands"]
labels = len(keep_stanford40)
with open('data/Stanford40/ImageSplits/train.txt', 'r') as f:
    # We won't use these splits but split them ourselves
    train_files = [file_name for file_name in list(map(str.strip, f.readlines())) if
                   '_'.join(file_name.split('_')[:-1]) in keep_stanford40]
    train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]

with open('data/Stanford40/ImageSplits/test.txt', 'r') as f:
    # We won't use these splits but split them ourselves
    test_files = [file_name for file_name in list(map(str.strip, f.readlines())) if
                  '_'.join(file_name.split('_')[:-1]) in keep_stanford40]
    test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]

# Combine the splits and split for keeping more images in the training set than the test set.
all_files = train_files + test_files
all_labels = train_labels + test_labels
train_files, test_files = train_test_split(all_files, test_size=0.1, random_state=0, stratify=all_labels)
train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
img_path = 'data/Stanford40/JPEGImages/'
action_categories = sorted(list(set(train_labels)))

train_set = []
for file in train_files:
    img = cv.imread(img_path + file)
    img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
    img = img.astype('float32')/255.0
    train_set.append(np.array(img))
train_set = np.array(train_set)

test_set = []
for file in test_files:
    img = cv.imread(img_path + file)
    img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
    img = img.astype('float32')/255.0
    test_set.append(np.array(img))
test_set = np.array(test_set)

le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
test_labels = le.fit_transform(test_labels)
