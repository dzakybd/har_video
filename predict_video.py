# predict.py ini adalah uji coba model dalam menprediksi aktivitas pada video

import cv2
import numpy as np
from keras.models import load_model

scenario = 1 # 1 (3D-CNN) / 2 (C-RNN) / 3 (Pose-RNN)
frame_sequences = 20
actions = ['walking', 'handwaving', 'boxing']

# membuka data video
cap = cv2.VideoCapture('dataset/walking/person01_walking_d1_uncomp.avi')

# mendapatkan fps untuk memutar video
fps = int(cap.get(cv2.CAP_PROP_FPS))
fps2 = int(1000/fps)

# membuka model
model = load_model('model-{}.hdf5'.format(scenario))

# inisiasi variabel
frames = []
label = ''

# memutar video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        # mengubah ukuran resolusi video
        tempframe = cv2.resize(frame, (80, 45))

        # mengumpulkan frame hingga 16 frame pada C3D, 8 frame pada R3D
        if len(frames) < frame_sequences:
            frames.append(tempframe)

        # memprediksi aktivitas pada video
        if len(frames) >= frame_sequences:
            data = np.array([frames])
            output = model.predict_on_batch(data)
            del frames[:]

            # membuat label berisi nama kelas dan tingkat keyakinan
            label = actions[output[0].argmax()] + " " + str(max(output[0]))


        # memasang label di dalam video
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label, (30, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # menampilkan video
        cv2.imshow('frame', frame)
        if cv2.waitKey(fps2) & 0xFF == ord('q'):
            break

    # memutar ulang video
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# menutup video
if cap.isOpened():
    cap.release()
    cv2.destroyAllWindows()