import json
import math
import os
import random

import numpy as np

import cv2
import torch
import torch.nn as nn

import utils

from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class Sign_Dataset(Dataset):

    def __init__(self, file_name_index, split, pose_directory, sampling_method='random', num_samples=25, num_copies=4,
                 img_transforms=None, video_transforms=None, test_index_file=None):
        
        assert os.path.exists(file_name_index), "File path does not exist...check...: {}.".format(file_name_index)
        assert os.path.exists(pose_directory), "File to pose does not exist...check... {}.".format(pose_directory)

        self.data = []
        self.label_encoder, self.onehot_encoder = LabelEncoder(), OneHotEncoder(categories='auto')

        if type(split) == 'str':
            split = [split]

        self.test_index_file = test_index_file
        self.init_dataset(file_name_index, split)

        self.file_name_index = file_name_index
        self.pose_directory = pose_directory
        self.framename = 'image_{}_keypoints.json'
        self.sampling_method = sampling_method
        self.num_samples = num_samples

        self.img_transforms = img_transforms
        self.video_transforms = video_transforms

        self.num_copies = num_copies

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_id, label_number, frame_start, frame_end = self.data[index]
        # frames of dimensions (T, H, W, C)
        x = self._load_poses(video_id, frame_start, frame_end, self.sampling_method, self.num_samples)

        if self.video_transforms:
            x = self.video_transforms(x)

        y = label_number

        return x, y, video_id

    def init_dataset(self, file_name_index, split):
        with open(file_name_index, 'r') as f:
            content = json.load(f)

        # create label encoder
        labels = sorted([label_entry['gloss'] for label_entry in content])

        self.label_encoder.fit(labels)
        self.onehot_encoder.fit(self.label_encoder.transform(self.label_encoder.classes_).reshape(-1, 1))

        if self.test_index_file is not None:
            print('Trained on {}, tested on {}'.format(file_name_index, self.test_index_file))
            with open(self.test_index_file, 'r') as f:
                content = json.load(f)

        # make dataset
        for label_entry in content:
            gloss, instances = label_entry['gloss'], label_entry['instances']
            label_number = utils.labels2cat(self.label_encoder, [gloss])[0]

            for instance in instances:
                if instance['split'] not in split:
                    continue

                frame_end = instance['frame_end']
                frame_start = instance['frame_start']
                video_id = instance['video_id']

                instance_entry = video_id, label_number, frame_start, frame_end
                self.data.append(instance_entry)

    def _load_poses(self, video_id, frame_start, frame_end, sampling_method, num_samples):
        """ Load frames of a video. Start and end indices are provided just to avoid listing and sorting the directory unnecessarily.
         """
        poses = []

        if sampling_method == 'random':
            frames_to_sample = rand_start_sampling(frame_start, frame_end, num_samples)
        elif sampling_method == 'seq':
            frames_to_sample = sequential_sampling(frame_start, frame_end, num_samples)
        elif sampling_method == 'k_copies':
            frames_to_sample = k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples,
                                                                         self.num_copies)
        
        else:
            raise NotImplementedError('Unimplemented sample strategy found: {}.'.format(sampling_method))

        for i in frames_to_sample:
            pose_path = os.path.join(self.pose_directory, video_id, self.framename.format(str(i).zfill(5)))
            # pose = cv2.imread(frame_path, cv2.COLOR_BGR2RGB)
            pose = read_pose_file(pose_path)

            if pose is not None:
                if self.img_transforms:
                    pose = self.img_transforms(pose)

                poses.append(pose)
            else:
                try:
                    poses.append(poses[-1])
                except IndexError:
                    print(pose_path)

        pad = None

        # if len(frames_to_sample) < num_samples:
        if len(poses) < num_samples:
            num_padding = num_samples - len(frames_to_sample)
            last_pose = poses[-1]
            pad = last_pose.repeat(1, num_padding)

        poses_across_time = torch.cat(poses, dim=1)
        if pad is not None:
            poses_across_time = torch.cat([poses_across_time, pad], dim=1)

        return poses_across_time


def rand_start_sampling(frame_start, frame_end, num_samples):
    """Randomly select a starting point and return the continuous ${num_samples} frames."""
    num_frames = frame_end - frame_start + 1

    if num_frames > num_samples:
        select_from = range(frame_start, frame_end - num_samples + 1)
        sample_start = random.choice(select_from)
        frames_to_sample = list(range(sample_start, sample_start + num_samples))
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample


def sequential_sampling(frame_start, frame_end, num_samples):
    """Keep sequentially ${num_samples} frames from the whole video sequence by uniformly skipping frames."""
    num_frames = frame_end - frame_start + 1

    frames_to_sample = []
    if num_frames > num_samples:
        frames_skip = set()

        num_skips = num_frames - num_samples
        interval = num_frames // num_skips

        for i in range(frame_start, frame_end + 1):
            if i % interval == 0 and len(frames_skip) <= num_skips:
                frames_skip.add(i)

        for i in range(frame_start, frame_end + 1):
            if i not in frames_skip:
                frames_to_sample.append(i)
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample

def k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples, num_copies):
    num_frames = frame_end - frame_start + 1

    frames_to_sample = []

    if num_frames <= num_samples:
        num_pads = num_samples - num_frames

        frames_to_sample = list(range(frame_start, frame_end + 1))
        frames_to_sample.extend([frame_end] * num_pads)

        frames_to_sample *= num_copies

    elif num_samples * num_copies < num_frames:
        mid = (frame_start + frame_end) // 2
        half = num_samples * num_copies // 2

        frame_start = mid - half

        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * num_samples,
                                               frame_start + i * num_samples + num_samples)))

    else:
        stride = math.floor((num_frames - num_samples) / (num_copies - 1))
        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * stride,
                                               frame_start + i * stride + num_samples)))

    return frames_to_sample


def read_pose_file(filepath):
    
    body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}

    try:
        content = json.load(open(filepath))["people"][0]
    except IndexError:
        return None

    path_parts = os.path.split(filepath)

    frame_id = path_parts[1][:11]
    vid = os.path.split(path_parts[0])[-1]

    save_to = os.path.join('/home/jovyan/Documents/DL/DL_Project/WLASL/code/Pose-GCN/posegcn/features', vid)

    try:
        ft = torch.load(os.path.join(save_to, frame_id + '_ft.pt'))

        xy = ft[:, :2]
        # angles = torch.atan(ft[:, 110:]) / 90
        # ft = torch.cat([xy, angles], dim=1)
        return xy
    except FileNotFoundError:
        
        # print(filepath)
        body_pose = content["pose_keypoints_2d"]
        left_hand_pose = content["hand_left_keypoints_2d"]
        right_hand_pose = content["hand_right_keypoints_2d"]

        body_pose.extend(left_hand_pose)
        body_pose.extend(right_hand_pose)

        x = [v for i, v in enumerate(body_pose) if i % 3 == 0 and i // 3 not in body_pose_exclude]
        y = [v for i, v in enumerate(body_pose) if i % 3 == 1 and i // 3 not in body_pose_exclude]
        # conf = [v for i, v in enumerate(body_pose) if i % 3 == 2 and i // 3 not in body_pose_exclude]

        x = 2 * ((torch.FloatTensor(x) / 256.0) - 0.5)
        y = 2 * ((torch.FloatTensor(y) / 256.0) - 0.5)
        # conf = torch.FloatTensor(conf)

        x_diff = torch.FloatTensor(get_difference(x)) / 2
        y_diff = torch.FloatTensor(get_difference(y)) / 2

        zero_indices = (x_diff == 0).nonzero()

        orient = y_diff / x_diff
        orient[zero_indices] = 0

        xy = torch.stack([x, y]).transpose_(0, 1)

        ft = torch.cat([xy, x_diff, y_diff, orient], dim=1)

        path_parts = os.path.split(filepath)

        frame_id = path_parts[1][:11]
        vid = os.path.split(path_parts[0])[-1]

        save_to = os.path.join('/home/jovyan/Documents/DL/DL_Project/WLASL/code/Pose-GCN/posegcn/features', vid)
        if not os.path.exists(save_to):
            os.mkdir(save_to)
        torch.save(ft, os.path.join(save_to, frame_id + '_ft.pt'))

        xy = ft[:, :2]
        # angles = torch.atan(ft[:, 110:]) / 90
        # ft = torch.cat([xy, angles], dim=1)
        #
        return xy
    

def get_difference(x):
    diff = []

    for i, j in enumerate(x):
        temp = []
        for j, k in enumerate(x):
            if i != j:
                temp.append(j - k)

        diff.append(temp)

    return diff