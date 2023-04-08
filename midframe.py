import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse",
               "run", "shoot_bow", "smoke", "throw", "wave"]

def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = frame_count // 2

    while cap.isOpened():
        ret, frame = cap.read()
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if current_frame == middle_frame_index:
            frame = cv2.resize(frame, (224, 224))
            return frame

        if not ret:
            break

    cap.release()

def load_data(hmdb51_folder, keep_hmdb51, resize_shape=(224, 224)):
    data = []
    labels = []

    for class_name in keep_hmdb51:
        class_folder = os.path.join(hmdb51_folder, class_name)
        videos = os.listdir(class_folder)

        for video in videos:
            video_path = os.path.join(class_folder, video)
            middle_frame = extract_middle_frame(video_path)

            if middle_frame is not None:
                data.append(middle_frame)
                labels.append(class_name)

    data = np.array(data)
    labels = np.array(labels)

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1, random_state=42)

    return train_data, test_data, train_labels, test_labels, le

hmdb51_folder = "C:/INFOMCV_05/data/hmdb51/hmdb51_org"

train_data, test_data, train_labels, test_labels, label_encoder = load_data(hmdb51_folder, keep_hmdb51)

print(train_data.shape)
print(test_data.shape)
