"""## Experiment scenario & parameter"""

scenario = 3 # 1 (3D-CNN) / 2 (C-RNN) / 3 (Pose-RNN)

frame_sequences = 20
actions = ['walking', 'handwaving', 'boxing']
random_state = 1
batch_size = 5
number_epoch = 10
test_size = 0.2
original_height = 120
original_width = 160
original_fps = 40
channel = 1
time_lag = 2

# parameter for OpenPose
params = dict()
params["model_folder"] = "C:/Users/HANSUNG/Downloads/openpose/models/"
params["number_people_max"] = 1
params["keypoint_scale"] = 3
# params["net_resolution"] = "-1x300"
params["model_pose"] = "BODY_25"


if scenario == 1:
    resize_scale = 1/3
elif scenario == 2:
    resize_scale = 1
else:
    resize_scale = 1

frame_widht = int(original_width * resize_scale)
frame_height = int(original_height * resize_scale)