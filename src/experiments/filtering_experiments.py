# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from PIL import Image
from pose_engine import PoseEngine
import cv2 as cv
import argparse
import time
import collections
import svgwrite
import pandas as pd
import math
from scipy.signal import savgol_filter, medfilt, cheby1, sosfilt, sosfiltfilt
from scipy.ndimage import gaussian_filter1d
from filters.kalman import EKF
import matplotlib.pyplot as plt
from databases import mpii_3dhp
import pandas as pd
import h5py

activation_threshold = 0.0

parts_to_compare = [
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('right shoulder', 'right elbow'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle')]

result_set = []
filtered_result_set = []
result_dataframe = []
filtered_result_dataframe = []
gt_result_set = []
gt_result_dataframe = []

def pad_img(img, mult=16):
    h, w, _ = img.shape

    h_pad = 0
    w_pad = 0
    if (h - 1) % mult > 0:
        h_pad = mult - ((h - 1) % mult)
    if (w - 1) % mult > 0:
        w_pad = mult - ((w - 1) % mult)
    return np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), 'constant')

def save_keypoints(keypoints):
    result = []
    for keypoint in keypoints:
        result.append(keypoints[keypoint].k)
        result.append(int(round(keypoints[keypoint].yx[1])))
        result.append(int(round(keypoints[keypoint].yx[0])))
        result.append(keypoints[keypoint].score)
    result_set.append(np.array(result))

def save_gt_points(keypoints):
    result = []
    for i, point in enumerate(keypoints):
        result.append(mpii_mapper[i])
        result.append(int(round(point[0]) / 2048 * 720))
        result.append(int(round(point[1]) / 2048 * 720))
    gt_result_set.append(np.array(result))

def draw_gt(img, keypoints, frame_count):
    result = []
    for i, kp in enumerate(keypoints):
        result.append(mpii_mapper[i])
        x, y = keypoints[kp]['X'][frame_count-1], keypoints[kp]['Y'][frame_count-1]
        cv.circle(img, (int(round(x)), int(round(y))), 3, (0, 255, 0), 2, cv.LINE_AA)

        result.append(int(round(x)))
        result.append(int(round(y)))
    gt_result_set.append(np.array(result))
    #print(result)
    return img

def save_gt_points2(keypoints, frame_count):
    result = []
    for i, point in enumerate(keypoints):
        result.append(mpii_mapper[i])
        result.append(int(round(point[0]) / 2048 * 720))
        result.append(int(round(point[1]) / 2048 * 720))
    gt_result_set.append(np.array(result))


def save_smoothed_keypoints(keypoints):
    for i in range(0, len(keypoints['left knee']['X'])):
        result = []
        for keypoint in keypoints:
            result.append(keypoint)
            result.append(int(round(keypoints[keypoint]['X'][i])))
            result.append(int(round(keypoints[keypoint]['Y'][i])))
        filtered_result_set.append(np.array(result))


def create_dataframe():
    temp = np.array(result_set)
    result_dataframe = pd.DataFrame(temp, columns=['nose', 'nose x coord', ' nose y coord', ' nose confidence',
                                                   'left eye', 'left eye x coord', 'left eye y coord',
                                                   'left eye confidence',
                                                   'right eye', 'right eye x coord', 'right eye y coord',
                                                   'right eye confidence',
                                                   'left ear', 'left ear x coord', 'left ear y coord',
                                                   'left ear confidence',
                                                   'right ear', 'right ear x coord', 'right ear y coord',
                                                   'right ear confidence',
                                                   'left shoulder', 'left shoulder x coord', 'left shoulder y coord',
                                                   'left shoulder confidence',
                                                   'right shoulder', 'right shoulder x coord', 'right shoulder y coord',
                                                   'right shoulder confidence',
                                                   'left elbow', 'left elbow x coord', 'left elbow y coord',
                                                   'left elbow confidence',
                                                   'right elbow', 'right elbow x coord', 'right elbow y coord',
                                                   'right elbow confidence',
                                                   'left wrist', 'left wrist x coord', 'left wrist y coord',
                                                   'left wrist confidence',
                                                   'right wrist', 'right wrist x coord', 'right wrist y coord',
                                                   'right wrist confidence',
                                                   'left hip', 'left hip x coord', 'left hip y coord',
                                                   'left hip confidence',
                                                   'right hip', 'right hip x coord', 'right hip y coord',
                                                   'right hip confidence',
                                                   'left knee', 'left knee x coord', 'left knee y coord',
                                                   'left knee confidence',
                                                   'right knee', 'right knee x coord', 'right knee y coord',
                                                   'right knee confidence',
                                                   'left ankle', 'left ankle x coord', 'left ankle y coord',
                                                   'left ankle confidence',
                                                   'right ankle', 'right ankle x coord', 'right ankle y coord',
                                                   'right ankle confidence'])
    return result_dataframe

mpii_mapper = {
    0 : 'upper neck',
    1 : 'right shoulder',
    2 : 'right elbow',
    3 : 'right wrist',
    4 : 'left shoulder',
    5 : 'left elbow',
    6 : 'left wrist',
    7 : 'right hip',
    8: 'right knee',
    9: 'right ankle',
    10: 'left hip',
    11: 'left knee',
    12: 'left ankle',
    13: 'head',
    14: 'pelvis',
    15: 'thorax',
    16 : 'head top'
}

def create_gt_dataframe():
    temp = np.array(gt_result_set)
    gt_result_dataframe = pd.DataFrame(temp, columns=[
                                                            'upper neck', 'upper neck x coord', 'upper neck y coord',
                                                            'right shoulder', 'right shoulder x coord', 'right shoulder y coord',
                                                            'right elbow', 'right elbow x coord', 'right elbow y coord',
                                                            'right wrist', 'right wrist x coord', 'right wrist y coord',
                                                            'left shoulder', 'left shoulder x coord', 'left shoulder y coord',
                                                            'left elbow', 'left elbow x coord', 'left elbow y coord',
                                                            'left wrist', 'left wrist x coord', 'left wrist y coord',
                                                            'right hip', 'right hip x coord', 'right hip y coord',
                                                            'right knee', 'right knee x coord', 'right knee y coord',
                                                            'right ankle', 'right ankle x coord', 'right ankle y coord',
                                                            'left hip', 'left hip x coord', 'left hip y coord',
                                                            'left knee', 'left knee x coord', 'left knee y coord',
                                                            'left ankle', 'left ankle x coord', 'left ankle y coord',
                                                            'head', 'head x coord', 'head y coord',
                                                            'pelvis', 'pelvis x coord', 'pelvis y coord',
                                                            'thorax', 'thorax x coord', 'thorax y coord',
                                                            'head top', 'head top x coord', ' head top y coord',
                                                            ])
    return gt_result_dataframe

def create_filtered_dataframe():
    temp = np.array(filtered_result_set)
    filtered_result_dataframe = pd.DataFrame(temp, columns=['nose', 'nose x coord', ' nose y coord',
                                                            'left eye', 'left eye x coord', 'left eye y coord',
                                                            'right eye', 'right eye x coord', 'right eye y coord',
                                                            'left ear', 'left ear x coord', 'left ear y coord',
                                                            'right ear', 'right ear x coord', 'right ear y coord',
                                                            'left shoulder', 'left shoulder x coord',
                                                            'left shoulder y coord',
                                                            'right shoulder', 'right shoulder x coord',
                                                            'right shoulder y coord',
                                                            'left elbow', 'left elbow x coord', 'left elbow y coord',
                                                            'right elbow', 'right elbow x coord', 'right elbow y coord',
                                                            'left wrist', 'left wrist x coord', 'left wrist y coord',
                                                            'right wrist', 'right wrist x coord', 'right wrist y coord',
                                                            'left hip', 'left hip x coord', 'left hip y coord',
                                                            'right hip', 'right hip x coord', 'right hip y coord',
                                                            'left knee', 'left knee x coord', 'left knee y coord',
                                                            'right knee', 'right knee x coord', 'right knee y coord',
                                                            'left ankle', 'left ankle x coord', 'left ankle y coord',
                                                            'right ankle', 'right ankle x coord',
                                                            'right ankle y coord'])
    return filtered_result_dataframe


def draw_pose(img, keypoints, pairs, text, frame):
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, text, (30, 30), font, 0.5, (50, 50, 155), 2, cv.LINE_AA)
    cv.putText(img, f'frame count: {frame}', (30, 45), font, 0.5, (50, 50, 155), 2, cv.LINE_AA)
    # return 'Keypoint(<{}>, {}, {})'.format(self.k, self.yx, self.score)

    # for keypoint in keypoints:
    #     # print(f'keypoint name: {keypoints[keypoint].k} - coord: {keypoints[keypoint].yx[0]} - {keypoints[keypoint].yx[1]} - confidence - {keypoints[keypoint].score}')
    #     if keypoints[keypoint].score < activation_threshold: continue
    #     cv.putText(img, "{:.3f}".format(keypoints[keypoint].score), (int(keypoints[keypoint].yx[1]), int(keypoints[keypoint].yx[0])), font, 0.4, (50, 50, 155), 2, cv.LINE_AA)

    # for i, pair in enumerate(pairs):
    #     color = (0, 255, 0)
    #     cv.line(img, (keypoints[pair[0]].yx[1], keypoints[pair[0]].yx[0]),
    #             (keypoints[pair[1]].yx[1], keypoints[pair[1]].yx[0]), color=color, lineType=cv.LINE_AA, thickness=1)

    for keypoint in keypoints:
        cv.circle(img, (int(keypoints[keypoint].yx[1]), int(keypoints[keypoint].yx[0])), 4, (0, 0, 255), 2, cv.LINE_AA)


def draw_pose_filtered(pose_img, smoothed_points, color=(0, 0, 255)):
    element = len(points_smoothed['left knee']['X']) - 3
    for key in smoothed_points.keys():
        # if key == 'left knee'
        #     print(f'left knee X: {smoothed_points[key]["X"][element]} Y: {smoothed_points[key]["Y"][element]}')
        x = smoothed_points[key]['X'][element]
        y = smoothed_points[key]['Y'][element]
        cv.circle(pose_img,
                  (int(round(x)), int(round(y))),
                  3,
                  color=color,
                  thickness=2)


def draw_pose_GT(pose_img, GT_coords, color=(0, 255, 0)):
    #joint id
    # 0 - head top,
    # 1 - upper neck,
    # 2 - r shoulder,
    # 3 - r elbow,
    # 4 - r wrist,
    # 5 - l shoulder,
    # 6 - l elbow,
    # 7 - l wrist,
    # 8 - r hip,
    # 9 - r knee,
    # 10 - r ankle,
    # 11 - l hip,
    # 12 - l knee,
    # 13 - l ankle,
    # 14 - pelvis,
    # 15 - thorax
    # 16 - head / nose?
    for i, pair in enumerate(GT_coords):
        if i in (0,1,14,15,16):
            continue
        # if i == 12:
        #     print(f'GT left knee X: {pair[0] / 2048 * 1280} Y: {pair[1] / 2048 * 720}')
        x = pair[0] / 2048 * 720
        y = pair[1] / 2048 * 720
        cv.circle(pose_img,
                  (int(round(x)), int(round(y))),
                  3,
                  color=color,
                  thickness=2)


def draw_filter(img, predmatrix):
    font = cv.FONT_HERSHEY_SIMPLEX
    for predcoord in predmatrix:
        cv.putText(img, 'o', (int(predcoord[0]), int(predcoord[1])), font, 0.5, (255, 0, 0), 2, cv.LINE_AA)

def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)


def shadow_text(dwg, x, y, text, font_size=12):
    dwg.add(dwg.text(text, insert=(x + 1, y + 1), fill='black',
                     font_size=font_size, style='font-family:sans-serif'))
    dwg.add(dwg.text(text, insert=(x, y), fill='white',
                     font_size=font_size, style='font-family:sans-serif'))

def get_GT_values(mat, input_resolution_x, input_resolution_y, output_x, output_y):
    # joint id # mpii, hrnet
    # 0 - head top, nose
    # 1 - upper neck, left eye
    # 2 - r shoulder, right eye
    # 3 - r elbow, left ear
    # 4 - r wrist, right ear
    # 5 - l shoulder, left shoulder
    # 6 - l elbow, right shoulder
    # 7 - l wrist, left elbow
    # 8 - r hip, right elbow
    # 9 - r knee, left wrist
    # 10 - r ankle, right wrist
    # 11 - l hip, left hip
    # 12 - l knee, right hip
    # 13 - l ankle, left knee
    # 14 - pelvis, right knee
    # 15 - thorax, left ankle
    # 16 - head / nose? right ankle
    mapping_FPoseNet = { #MPI INF key --> FPOSENET value
        '0' : '16', # not mapped properly
        '1' : '13', # not mapped properly
        '2' : '1',
        '3' : '2',
        '4' : '3',
        '5' : '4',
        '6' : '5',
        '7' : '6',
        '8' : '7',
        '9' : '8',
        '10': '9',
        '11': '10',
        '12': '11',
        '13': '12',
        '14': '14', # not mapped properly
        '15': '15', # not mapped properly
        '16': '0' # not mapped properly / not obvious
    }
    gt_storage = {
        '0': {'X': [], 'Y': []}, # head top
        '1': {'X': [], 'Y': []}, # upper neck
        '2': {'X': [], 'Y': []}, # r shoulder
        '3': {'X': [], 'Y': []}, # r elbow
        '4': {'X': [], 'Y': []}, # r wrist
        '5': {'X': [], 'Y': []}, # l shoulder
        '6': {'X': [], 'Y': []}, # l elbow
        '7': {'X': [], 'Y': []}, # l wrist
        '8': {'X': [], 'Y': []}, # r hip
        '9': {'X': [], 'Y': []}, # r knee
        '10': {'X': [], 'Y': []},# r ankle
        '11': {'X': [], 'Y': []},# l hip
        '12': {'X': [], 'Y': []},# l knee
        '13': {'X': [], 'Y': []},# l ankle
        '14': {'X': [], 'Y': []},# pelvis
        '15': {'X': [], 'Y': []},# thorax
        '16': {'X': [], 'Y': []},# head/nose
    }
    for frame in mat['annot2']:
        for i, xy in enumerate(frame[0]):
            #print('{}. kp coords: {}'.format(i, xy))
            gt_storage[mapping_FPoseNet[str(i)]]['X'].append(int(round(xy[0]))*output_x/input_resolution_x) # double check if x comes first
            gt_storage[mapping_FPoseNet[str(i)]]['Y'].append(int(round(xy[1]))*output_y/input_resolution_y) # double check if x comes first

    # validate GT storage:
    #for key_ in gt_storage.keys():
    #    print(len(gt_storage[key_]['X']))
    return gt_storage

#  Filters

#  sav-gol
def calculate_savgol(points5, window_size=5, poly_degree=2):
    points5_smoothed = {}
    for key in points5.keys():
        points5_smoothed[key] = {
            'X': savgol_filter(points5[key]['X'], window_size, poly_degree, mode='nearest'),
            'Y': savgol_filter(points5[key]['Y'], window_size, poly_degree, mode='nearest')
        }
    return points5_smoothed


#  gaussian 1d
def calculate_gaussian1d(points5, sigma=2):
    points5_smoothed = {}
    for key in points5.keys():
        points5_smoothed[key] = {
            'X': gaussian_filter1d(points5[key]['X'], sigma),
            'Y': gaussian_filter1d(points5[key]['Y'], sigma)
        }
    return points5_smoothed


#  median 1d
def calculate_median1d(points5, kernel=5):
    points5_smoothed = {}
    for key in points5.keys():
        # print(f'before: {points5[key]["X"]}')
        points5_smoothed[key] = {
            'X': medfilt(points5[key]['X'], kernel),
            'Y': medfilt(points5[key]['Y'], kernel)
        }
        # print(f'after: {points5_smoothed[key]["X"]}')
    return points5_smoothed


def calculate_cheby1(points5, n=3):
    points5_smoothed = {}

    for key in points5.keys():
        sos = cheby1(n, 1, 0.2, 'lp', output='sos')
        if len(points5['left knee']['X']) < 13:
            points5_smoothed[key] = {
                'X': sosfilt(sos, points5[key]['X']),
                'Y': sosfilt(sos, points5[key]['Y']),
            }
        else:
            points5_smoothed[key] = {
                'X': sosfiltfilt(sos, points5[key]['X']),
                'Y': sosfiltfilt(sos, points5[key]['Y']),
            }
        # print(f'original x: {points5[key]["X"]} \nafter filtering: {points5_smoothed[key]["X"]}')
    return points5_smoothed


def plot_graphs(keypoint, predicted_data, filtered_data, ground_truth):
    f2, ax2 = plt.subplots()
    time = np.linspace(0, len(predicted_data[f'{keypoint} x coord']), len(predicted_data[f'{keypoint} x coord']))
    plt.figure(figsize=(38.40, 21.60))
    ax2.scatter(time, np.array(predicted_data[f'{keypoint} x coord']).astype(int), s=2, label='Predicted', color='red')
    ax2.scatter(time, np.array(filtered_data[f'{keypoint} x coord']).astype(int), s=2, label='Filtered', color='blue')
    ax2.scatter(time, np.array(ground_truth[f'{keypoint} x coord']).astype(int), s=2, label='Ground Truth', color='green')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='best')
    f2.savefig(f'temp_plots/{keypoint}_X_plot.jpeg', bbox_inches='tight', pil_kwargs={'progressive': True}, dpi=1000)

    f3, ax3 = plt.subplots()
    plt.figure(figsize=(38.40, 21.60))
    ax3.scatter(time, np.array(predicted_data[f'{keypoint} y coord']).astype(int), s=2, label='Predicted', color='red')
    ax3.scatter(time, np.array(filtered_data[f'{keypoint} y coord']).astype(int), s=2, label='Filtered', color='blue')
    ax3.scatter(time, np.array(ground_truth[f'{keypoint} y coord']).astype(int), s=2, label='Ground Truth', color='green')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='best')
    f3.savefig(f'temp_plots/{keypoint}_Y_plot.jpeg', bbox_inches='tight', pil_kwargs={'progressive': True}, dpi=1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="tflite path")
    parser.add_argument("--video", help="video path")
    parser.add_argument("--height", help="height")
    parser.add_argument("--width", help="width")
    parser.add_argument("--seconds", help="second of the video to be played", default=0.1)
    parser.add_argument("--fps", help="frames per sec of the video", default=50)
    args = parser.parse_args()

    # model path
    model_path = args.model
    # mobilenet - posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite
    # resnet50 - posenet_resnet_50_960_736_32_quant_edgetpu_decoder.tflite
    #model_path = 'posenet_resnet_50_960_736_32_quant_edgetpu_decoder.tflite'
    #model_path = 'models/output_model.tflite'
    # video path
    video_path = args.video
    #video_path = 'video_8.avi'
    video_path = 'TS1.mp4'

    video = cv.VideoCapture(video_path)

    #video fps
    video_fps = args.fps
    width = int(args.width)
    height = int(args.height)

    #GT values
    dataset = 1
    sequence = 2
    camera = 8
    # mat = h5py.File('/home/gables/Practice/mpii_inf_3dhp/mpi_inf_3dhp/S1/Seq2/annot.mat', 'r')
    # mat = scipy.io.loadmat('/home/gables/Practice/mpii_inf_3dhp/mpi_inf_3dhp/S1/Seq2/annot.mat')

    mat = h5py.File('annot_data.mat', 'r')
    gt = get_GT_values(mat, 2048, 2048, 720, 720)

    #gt = mpii_3dhp.train_ground_truth(dataset, sequence)['annot2'][camera]
    #print(gt.shape)

    # pose engine
    engine = PoseEngine(model_path)

    # 1 second
    n_sec = int(args.seconds)

    t_curr = t_start = time.time()
    t_end = t_curr + n_sec

    # frames counter
    i = 0
    # inference time
    inf_times = []

    fps_counter = avg_fps_counter(30)
    sum_inference_time = 0

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('output.mp4', fourcc, 30.0, (481, 353))

    # EKF part
    X_0 = np.array([width / 2, height / 2, 0.0, 0.0])

    # R = np.array([[1000, 0.0],
    #                [0, 1000]])
    R = np.zeros((2, 2))

    Q = np.array([[np.power(0.1, 2), 0.0, 0.0, 0.0],
                  [0.0, np.power(0.1, 2), 0.0, 0.0],
                  [0.0, 0.0, np.power((0.5 * 180.0 / math.pi), 2), 0.0],
                  [0.0, 0.0, 0.0, np.power(0.5, 2)]])
    # Q = np.zeros((4, 4))

    P_0 = np.array([[1000.0, 0.0, 0.0, 0.0],
                    [0.0, 1000.0, 0, 0.0],
                    [0.0, 0.0, 1000.0, 0.0],
                    [0.0, 0.0, 0.0, 1000.0]])


    def motion_model(x, u, delta_t, Q):
        """Implement the motion model here"""

        F = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]])

        theta = x[2]  # x = [x, y, theta, v]

        B = np.array([[delta_t * math.cos(theta), 0.0],
                      [delta_t * math.sin(theta), 0.0],
                      [0.0, delta_t],
                      [1.0, 0.0]])

        # print(f'u: {u}')
        process_noise = np.random.multivariate_normal([0, 0, 0, 0], Q)
        return F @ x + B @ u + process_noise


    def measurement_function(x):
        """Implement the measurement function here"""
        _H = np.eye(2, 4, 0)
        _v = np.random.multivariate_normal([0, 0], R)

        return _H @ x + _v


    def posenet_single_predicted_data(keypoint):
        name = keypoint.k
        score = keypoint.score
        coords = keypoint.yx

        # print(f'{name} posenet measured pos x: {int(coords[1])} y: {int(coords[0])}')
        return np.array([int(coords[1]), int(coords[0]), 0, 0])


    # delta_t = inference based?
    # EKF_filter = EKF(motion_model, measurement_function, posenet_single_predicted_data, X_0=X_0, P_0=P_0, Q=Q, R=R,
    #                  delta_t=1)

    all_left_knee_X = []
    all_left_knee_Y = []

    # end EKF

    frames = []
    # store 5 points
    # we will be always 3 frames late
    points5 = {
        'nose': {},
        'left eye': {},
        'right eye': {},
        'left ear': {},
        'right ear': {},
        'left shoulder': {},
        'right shoulder': {},
        'left elbow': {},
        'right elbow': {},
        'left wrist': {},
        'right wrist': {},
        'left hip': {},
        'right hip': {},
        'left knee': {},
        'right knee': {},
        'left ankle': {},
        'right ankle': {}
    }

    # to store keypoint over the whole video
    # except first two and last two frames
    all_points = {
        'nose': {},
        'left eye': {},
        'right eye': {},
        'left ear': {},
        'right ear': {},
        'left shoulder': {},
        'right shoulder': {},
        'left elbow': {},
        'right elbow': {},
        'left wrist': {},
        'right wrist': {},
        'left hip': {},
        'right hip': {},
        'left knee': {},
        'right knee': {},
        'left ankle': {},
        'right ankle': {}
    }
    for key in points5.keys():
        points5[key] = {
            'X': [],
            'Y': []
        }
        all_points[key] = {
            'X': [],  # original x-position
            'Y': [],  # original y
            'sX': [],  # smoothed and outlier-filtered x-position
            'sY': [],  # smoothed and outlier-filtered y
            'mag-org': [],  # original magnitude sqrt(x^2+y^2)
            'mag-smoothed': []  # # sqrt(sx^2+sy^2)
        }

    skip_counter = 0
    real_frame_number = 0

    fps_list = []
    fps_list_without_filtering = []

    while video.isOpened():
        # if skip_counter < 4150:
        #     video.read()
        #     real_frame_number +=1
        #     skip_counter += 1
        #     continue
        i += 1

        #if t_curr > t_end:

        # process the frame here
        ret, frame = video.read()
        real_frame_number += 1
        if ret == False:
            break
        frame = cv.resize(frame, (481, 353))
        frame = np.pad(frame, ((0,0),(0,560),(0,0)))
        # print(frame.shape)
        t_curr = time.time()
        poses, inference_time = engine.DetectPosesInImage(np.uint8(frame))
        # INFERENCE TIME
        inf_times.append(inference_time)
        # print('Inference time: %.fms' % inference_time)

        src_size = (width, height)
        svg_canvas = svgwrite.Drawing('', size=src_size)

        sum_inference_time += inference_time
        avg_inference_time = sum_inference_time / n_sec
        text_line = 'PoseNet: %.1fms FPS: %.2f' % (
            inference_time, next(fps_counter))

        template_show = np.array(frame)
        pose_img = np.array(template_show)

        # frames.append(pose_img)

        # for pose in poses: ----> SIMPLIFY to first person only
        #     if pose.score < activation_threshold: continue
        #     # print('\nPose Score: ', pose.score)
        #     posenet_single_predicted_data(pose.keypoints['left knee'])
        #     draw_pose(template_show, pose.keypoints, parts_to_compare, text_line, i)
        #     save_keypoints(pose.keypoints)

        if len(poses) < 1:
            #poses = []
            continue
        else:
            print("Poses before:", poses)
            poses = [poses[0]]
            print("Poses after:", poses)

        exit()

        for pose in poses:
            # if pose.score < 0.4: continue
            for key in pose.keypoints.keys():
                # if key in ['nose', 'left eye', 'right eye', 'left ear', 'right ear', ]: continue
                # replacing the low-scored keypoints with the last corrected keypoint in t-1
                # add True to the if condition to eliminate this effect
                if i <= 1 or pose.keypoints[key].score > activation_threshold:
                    points5[key]['X'].append(pose.keypoints[key].yx[1])
                    points5[key]['Y'].append(pose.keypoints[key].yx[0])
                else:
                    points5[key]['X'].append(points5[key]['X'][-1] if i < 6 else all_points[key]['sX'][-1])
                    points5[key]['Y'].append(points5[key]['Y'][-1] if i < 6 else all_points[key]['sY'][-1])
        if len(poses) < 1:
            i-=1
            continue
        pose = poses[0]
        all_left_knee_X.append(int(pose.keypoints['left knee'].yx[1]))
        all_left_knee_Y.append(int(pose.keypoints['left knee'].yx[0]))

        # if pose.score < activation_threshold:
        #     continue
        #EKF_filter.predict()

        draw_pose(template_show, pose.keypoints, parts_to_compare, text_line, i)
        # draw_filter(template_show, EKF_filter.pred_x_dict.values())
        #EKF_filter.update(pose.keypoints)
        save_keypoints(pose.keypoints)
        #save_gt_points(gt[real_frame_number-1])
        #save_gt_points2(gt,real_frame_number - 1)

        draw_pose(template_show, pose.keypoints, parts_to_compare, text_line, i)
        frames.append(template_show)
        ## other filters
        if i < 3: continue
        frames.append(template_show)
        if i < 5: continue

        ## filtering comes here
        filter_data_length = 100

        # for key in points5.keys():
            # if i > filter_data_length:
            #     points_smoothed[key]['X'] = points_smoothed[key]['X'][1:]
            #     points_smoothed[key]['Y'] = points_smoothed[key]['Y'][1:]
        # points_smoothed = calculate_savgol(points5, 15, 3)

        fps_wf = 1. / (time.time() - t_curr)
        fps_list_without_filtering.append(fps_wf)

        points_smoothed = calculate_median1d(points5, 15)
        points_smoothed = calculate_savgol(points_smoothed, 15, 3)
        points_smoothed = calculate_cheby1(points_smoothed, 3)
        #points_smoothed = calculate_cheby1(points_smoothed, 3)
        element = len(points_smoothed['left knee']['X']) - 3

        for key in all_points.keys():
            # if i > filter_data_length:
            #     all_points[key]['sX'] = all_points[key]['sX'][1:]
            #     all_points[key]['sY'] = all_points[key]['sY'][1:]
            #     all_points[key]['X'] = all_points[key]['X'][1:]
            #     all_points[key]['Y'] = all_points[key]['Y'][1:]
            #     all_points[key]['mag-org'] = all_points[key]['mag-org'][1:]
            #     all_points[key]['mag-smoothed'] = all_points[key]['mag-smoothed'][1:]
            all_points[key]['sX'].append(points_smoothed[key]['X'][element])
            all_points[key]['sY'].append(points_smoothed[key]['Y'][element])
            all_points[key]['X'].append(points5[key]['X'][element])
            all_points[key]['Y'].append(points5[key]['Y'][element])
            all_points[key]['mag-org'].append(
                np.sqrt(np.power(all_points[key]['X'][-1], 2) + np.power(all_points[key]['Y'][-1], 2)))
            all_points[key]['mag-smoothed'].append(
                np.sqrt(np.power(all_points[key]['sX'][-1], 2) + np.power(all_points[key]['sY'][-1], 2)))



        #latest_frame = frames[0]
        # draw_pose_filtered(template_show, points_smoothed, color=(255, 0, 0))  # blue smoothed
        # draw_pose_filtered(latest_frame, points5, color=(0, 0, 255))  # red

        # delete the oldest frame
        frames = frames[1:]

        # for key in points5.keys():
        #     points5[key]['X'] = points5[key]['X'][1:]
        #     points5[key]['Y'] = points5[key]['Y'][1:]

        # cv.imshow('Frame', latest_frame)
        # cv.imshow('Frame', template_show)
        # draw_pose(frames[len(frames)-1], pose.keypoints, parts_to_compare, text_line, i)
        fps_filtering = 1. / (time.time() - t_curr)
        fps_list.append(fps_filtering)

        draw_pose_filtered(frames[len(frames) - 4], points_smoothed, color=(255, 0, 0)) # blue-green-red
        #draw_pose_GT(frames[len(frames)-4], gt[real_frame_number-4])
        draw_gt(frames[len(frames)-4], gt, real_frame_number-4)
        cv.imshow('Frame', frames[len(frames) - 4])

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

        out.write(frames[len(frames)-4])

    save_smoothed_keypoints(points_smoothed)

    df = create_dataframe()
    df.to_csv('results.csv', sep='\t')

    filtered_df = create_filtered_dataframe()
    filtered_df.to_csv('filtered_results.csv', sep='\t')

    gt_df = create_gt_dataframe()
    gt_df.to_csv('gt_values.csv', sep='\t')


    print('Frames per second for the video: %.2f fps' % (video.get(cv.CAP_PROP_FPS)))
    avg_inf = sum_inference_time / len(inf_times)

    print('average inference: %.2f ms' % (sum_inference_time / len(inf_times)))
    print('inference performance for the model: %.2f fps' % (i / (t_curr - t_start)))

    # cv.waitKey(0)
    video.release()
    cv.destroyAllWindows()

    f, ax = plt.subplots()
    # magnitude
    ax.plot(all_left_knee_X, label='left knee X', )
    ax.plot(all_left_knee_Y, label='left knee Y', )
    # ax.plot(t, all_points[key]['mag-smoothed'], label='filtered magnitude')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='best')
    f.savefig('result_plot.jpeg', bbox_inches='tight', pil_kwargs={'progressive': True}, dpi=1000)

    data = pd.read_csv('results.csv', sep='\t')
    sensor_data = data[['left knee x coord']]
    filtdata = pd.read_csv('filtered_results.csv', sep='\t')
    filtsensor_data = filtdata[['left knee x coord']]
    gt_data = pd.read_csv('gt_values.csv', sep='\t')
    gt_coords = gt_data['left knee x coord']


    alt_1 = []
    alt_2 = []
    for keypoint in mpii_mapper.values():
        predicted_distances = []
        filtered_distances = []
        if keypoint in df.columns:
            for i, x in enumerate(gt_df[f'{keypoint} x coord']):
                pred_dist_x = int(x) - int(df[f'{keypoint} x coord'].iloc[[i]])
                y = gt_df[f'{keypoint} y coord'].iloc[[i]]
                pred_dist_y = int(y) - int(df[f'{keypoint} y coord'].iloc[[i]])
                predicted_distances.append(math.sqrt((pred_dist_x**2 + pred_dist_y**2)))

                filt_dist_x = int(x) - int(filtered_df[f'{keypoint} x coord'].iloc[[i]])
                y = gt_df[f'{keypoint} y coord'].iloc[[i]]
                filt_dist_y = int(y) - int(filtered_df[f'{keypoint} y coord'].iloc[[i]])
                filtered_distances.append(math.sqrt((filt_dist_x ** 2 + filt_dist_y ** 2)))

            pred_mean = round(np.asarray(predicted_distances).mean()/353*224, 3)
            alt_1.append(pred_mean)
            filt_mean = round(np.asarray(filtered_distances).mean()/353*224, 3)
            alt_2.append(filt_mean)
            print(f'RMSE between predicted PoseNet {keypoint} and Ground Truth keypoint: {pred_mean}')
            print(f'RMSE between filtered PoseNet {keypoint} and Ground Truth keypoint: {filt_mean}, improvement: {round(pred_mean-filt_mean,3)}')
            #plot_graphs(keypoint, df, filtered_df, gt_df)
    print(f'AVG RMSE between predicted PoseNet {round(np.asarray(alt_1).mean(),3)}')
    print(f'AVG RMSE between filtered PoseNet {round(np.asarray(alt_2).mean(),3)}')

    print(f'unfiltered avg fps: {round(np.asarray(fps_list_without_filtering).mean()),3}')
    print(f'filtered avg fps: {round(np.asarray(fps_list).mean()),3}')
    print(f'avg inference: {round(np.asarray(inf_times).mean()),3}')