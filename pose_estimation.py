import os
import cv2
import numpy as np
import pandas as pd
import pyopenpose as op
from parameters import *
from keras.utils import np_utils

print("Scenario", scenario)
print("Frame size", frame_height, frame_widht)

"""## Load dataset"""

print("Load dataset")

points = []
dataset_temp = []
dataset_raw = []
labels_raw = []
dataset = []
labels = []

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

for idx, action in enumerate(actions):
    path_dir = 'dataset/{}'.format(action)
    vids = os.listdir(path_dir)

    # iterates over all data automatically
    for idx2, vid in enumerate(vids):
        if idx2 == 4:
            break
        if vid.endswith(".avi"):
            path_file = '{}/{}'.format(path_dir, vid)
            print(path_file)

            frames = []
            cap = cv2.VideoCapture(path_file)
            while True:
                ret, frame = cap.read()
                if ret:
                    # resize frame
                    frame = cv2.resize(frame, (frame_widht, frame_height))

                    # extract pose features
                    datum = op.Datum()
                    datum.cvInputData = frame
                    opWrapper.emplaceAndPop([datum])
                    posekeypoints = datum.poseKeypoints
                    sp = np.shape(posekeypoints)

                    if len(sp) > 0:
                        points.append(posekeypoints)
                else:
                    break
            cv2.destroyAllWindows()
            dataset_raw.append(points)
            labels_raw.append(idx)

print("Datsaset raw", np.shape(dataset_raw))
print("Labels raw", np.shape(labels_raw))
opWrapper.stop()

print("Generate pose feature sequences")
for i, frames in enumerate(dataset_raw):
  tempframe = []
  action = labels_raw[i]
  print("Preprocess {} {}".format(actions[action], i))
  for j in range(len(frames)):
      if len(tempframe) < frame_sequences:
          tempframe.append(frames[j])
      elif len(tempframe) >= frame_sequences:
          tempp = np.array(tempframe)
          dataset_temp.append(tempp)
          labels.append(action)
          del tempframe[:]

print("Dataset temp", np.shape(dataset_temp))
print("Labels temp", np.shape(labels))


for points in dataset_temp:
    # Remove confidence value
    points = np.delete(points, 2, axis=3)

    # squeeze the points into matrix shape (1, frames, features)
    points = np.squeeze(points)
    sp = points.shape
    points = np.reshape(points, (sp[0], sp[1] * sp[2]))

    y = pd.DataFrame(points)
    x = pd.DataFrame()

    # adding lag to the file, if set the lag to 2, then get:
    # get current frame, previous frame, and previous of previous frame
    for i in range(time_lag + 1):
        x = pd.concat([x, y.shift(i)], axis=1)
    z = x.dropna()
    dataset.append(np.array(z))

labels = np_utils.to_categorical(labels, len(actions))

print("Dataset shape", np.shape(dataset))
print("Labels shape", np.shape(labels))

np.save('dataset-{}.npy'.format(scenario), dataset)
np.save('label-{}.npy'.format(scenario), labels)
