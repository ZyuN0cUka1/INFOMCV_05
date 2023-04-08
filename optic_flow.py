import os
import cv2
import numpy as np
import tarfile

def calculate_optical_flow(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames // num_frames

    prev_frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame_idx)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    optical_flow_frames = []

    for i in range(1, num_frames):
        next_frame_idx = i * frame_step
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_idx)
        ret, next_frame = cap.read()

        if not ret:
            break

        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        optical_flow_frames.append(flow)

        prev_gray = next_gray

    cap.release()
    return np.array(optical_flow_frames)

root_folder = "C:\INFOMCV_05\data\hmdb51\hmdb51_org"
optical_flow_folder = "optical_flow_data"

for label in os.listdir(root_folder):
    label_folder = os.path.join(root_folder, label)
    optical_flow_label_folder = os.path.join(optical_flow_folder, label)
    os.makedirs(optical_flow_label_folder, exist_ok=True)

    for video_file in os.listdir(label_folder):
        video_path = os.path.join(label_folder, video_file)
        optical_flow_frames = calculate_optical_flow(video_path)

        # Save the optical flow frames
        optical_flow_file = os.path.splitext(video_file)[0] + ".npy"
        optical_flow_path = os.path.join(optical_flow_label_folder, optical_flow_file)
        np.save(optical_flow_path, optical_flow_frames)
