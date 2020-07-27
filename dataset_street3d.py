import os
import os.path as osp

import pandas as pd
import numpy as np
import pickle

from utils.utils.ply import read_ply, write_ply
from utils import provider
from utils.point_cloud_util import load_labels

from open3d import open3d

class FileData:
    def __init__(self, pcd_file_path, box_size_x, box_size_y, use_color=False):
        
        self.pcd_file_path = pcd_file_path
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        
        # load points
        # data = read_ply(self.ply_file_path)
        # self.points = np.vstack((data['x'], data['y'], data['z'])).T
        # self.labels = data['label']
        self.points = np.asarray(open3d.io.read_point_cloud(self.pcd_file_path).points)
        self.labels = load_labels(self.pcd_file_path[:-4]+'.labels')
        
        if use_color:
            pass
        else:
            self.colors = np.zeros_like(self.points)
        
        # Sort according to x to speed up computation of boxes and z-boxes
        sort_idx = np.argsort(self.points[:, 0])
        self.points = self.points[sort_idx]
        self.labels = self.labels[sort_idx]
        self.colors = self.colors[sort_idx]

    def _get_fix_sized_sample_mask(self, points, num_points_per_sample):
        """
        Get down-sample or up-sample mask to sample points to num_points_per_sample
        """
        # Shuffling or up-sampling if needed
        if len(points) - num_points_per_sample > 0:
            true_array = np.ones(num_points_per_sample, dtype=bool)
            false_array = np.zeros(len(points) - num_points_per_sample, dtype=bool)
            sample_mask = np.concatenate((true_array, false_array), axis=0)
            np.random.shuffle(sample_mask)
        else:
            # Not enough points, recopy the data until there are enough points
            sample_mask = np.arange(len(points))
            while len(sample_mask) < num_points_per_sample:
                sample_mask = np.concatenate((sample_mask, sample_mask), axis=0)
            sample_mask = sample_mask[:num_points_per_sample]
        return sample_mask
    def _center_box(self, points):
        # Shift the box so that z = 0 is the min and x = 0 and y = 0 is the box center
        # E.g. if box_size_x == box_size_y == 10, then the new mins are (-5, -5, 0)
        box_min = np.min(points, axis=0)
        shift = np.array(
            [
                box_min[0] + self.box_size_x / 2,
                box_min[1] + self.box_size_y / 2,
                box_min[2],
            ]
        )
        points_centered = points - shift
        return points_centered
    def _extract_z_box(self, center_point):
        """"Crop along z axis (vertical) from the center_point."""
        scene_z_size = np.max(self.points, axis=0)[2] - np.min(self.points, axis=0)[2]
        box_min = center_point - [
            self.box_size_x / 2,
            self.box_size_y / 2,
            scene_z_size,
        ]
        box_max = center_point + [
            self.box_size_x / 2,
            self.box_size_y / 2,
            scene_z_size,
        ]

        i_min = np.searchsorted(self.points[:, 0], box_min[0])
        i_max = np.searchsorted(self.points[:, 0], box_max[0])
        mask = (
            np.sum(
                (self.points[i_min:i_max, :] >= box_min)
                * (self.points[i_min:i_max, :] <= box_max),
                axis=1,
            )
            == 3
        )
        mask = np.hstack(
            (
                np.zeros(i_min, dtype=bool),
                mask,
                np.zeros(len(self.points) - i_max, dtype=bool),
            )
        )

        # mask = np.sum((points>=box_min)*(points<=box_max),axis=1) == 3
        assert np.sum(mask) != 0
        return mask
    
    def sample(self, num_points_per_sample):
        points = self.points
        # Pick a point, and crop a z-box around
        center_point = points[np.random.randint(0, len(points))]
        scene_extract_mask = self._extract_z_box(center_point)
        points = points[scene_extract_mask]
        labels = self.labels[scene_extract_mask]
        colors = self.colors[scene_extract_mask]

        sample_mask = self._get_fix_sized_sample_mask(points, num_points_per_sample)
        points = points[sample_mask]
        labels = labels[sample_mask]
        colors = colors[sample_mask]

        # Shift the points, such that min(z) == 0, and x = 0 and y = 0 is the center
        # This canonical column is used for both training and inference
        points_centered = self._center_box(points)

        return points_centered, points, labels, colors
    
    def sample_batch(self, batch_size, num_points_per_sample):
        batch_points_centered = []
        batch_points_raw = []
        batch_labels = []
        batch_colors = []

        for _ in range(batch_size):
            points_centered, points_raw, gt_labels, colors = self.sample(
                num_points_per_sample
            )
            batch_points_centered.append(points_centered)
            batch_points_raw.append(points_raw)
            batch_labels.append(gt_labels)
            batch_colors.append(colors)

        return (
            np.array(batch_points_centered),
            np.array(batch_points_raw),
            np.array(batch_labels),
            np.array(batch_colors),
        )
    

class Street3D:
    def __init__(self, num_points_per_sample, split, box_size_x, box_size_y, voxel_size):
        self.num_points_per_sample = num_points_per_sample
        self.split = split
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.num_classes = 6
        self.path = '/home/kt/cyclomedia_disk/kt/3D_Point_Cloud_Semantic_Segmentation_for_StreetScenes/'
        self.label_to_names = {0: 'undefined',
                               1: 'building',
                               2: 'car',
                               3: 'ground',
                               4: 'pole',
                               5: 'vegetation'}
        # self.label_to_names = {
        #                        0: 'building',
        #                        1: 'car',
        #                        2: 'ground',
        #                        3: 'pole',
        #                        4: 'vegetation'}
        self.init_labels()
        self.train_list_file = osp.join(self.path, 'train_list.txt')
        self.test_list_file = osp.join(self.path, 'test_list.txt')
        self.train_names = [line.rstrip() for line in open(self.train_list_file)]
        self.test_names = [line.rstrip() for line in open(self.test_list_file)]
        
        # self.origin_ply_path = osp.join(self.path, 'origin_ply')
        self.downsampled_pcd_path = osp.join(self.path, 'downsampled_pcd_{:.2f}'.format(voxel_size))
        
        self.train_ply_files = [osp.join(self.downsampled_pcd_path, f+'.pcd') for f in self.train_names]
        self.test_ply_files = [osp.join(self.downsampled_pcd_path, f+'.pcd') for f in self.test_names]
        
        # load files
        self.list_file_data = []
        if self.split == 'train':
            ply_files = self.train_ply_files
        elif self.split == 'test':
            ply_files = self.test_ply_files
        else:
            raise "Unkonwn split {}".format(self.split)
        for f in ply_files:
            file_data = FileData(f, self.box_size_x, self.box_size_y)
            self.list_file_data.append(file_data)
        
        self.total_num_points = self.get_total_num_points()
            
        # Pre-compute the probability of picking a scene
        self.num_scenes = len(self.list_file_data)
        self.scene_probas = [
            len(fd.points) / self.total_num_points for fd in self.list_file_data
        ]
        
        # Pre-compute the point weights if it is a training set
        if self.split == 'train':
            label_weights = np.zeros(self.num_classes)
            for labels in [fd.labels for fd in self.list_file_data]:
                tmp, _ = np.histogram(labels, range(self.num_classes+1))
                label_weights += tmp
            
            # Then, a heuristic gives the weights
            # 1 / log(1.2 + probability of occurrence)
            label_weights = label_weights.astype(np.float32)
            label_weights = label_weights / np.sum(label_weights)
            self.label_weights = 1 / np.log(1.2 + label_weights)
        else:
            self.label_weights = np.zeros(self.num_classes)
    
    def sample_batch_in_all_files(self, batch_size, augment=True):
        batch_data = []
        batch_label = []
        batch_weights = []

        for _ in range(batch_size):
            points, labels, weights = self.sample_in_all_files(is_training=True)
            # if use_color:
                # batch_data.append(np.hstack((points, colors)))
            batch_data.append(points)
            batch_label.append(labels)
            batch_weights.append(weights)
        
        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)
        batch_weights = np.array(batch_weights)

        if augment:
            if 1 == 2: # use_color
                pass
            else:
                batch_data = provider.rotate_point_cloud(batch_data)
        
        return batch_data, batch_label, batch_weights            

            
    def sample_in_all_files(self, is_training):
        scene_idx = np.random.choice(np.arange(0, len(self.list_file_data)), p=self.scene_probas)

        # Sample from the selected scene
        points_centered, points_raw, labels, colors = self.list_file_data[scene_idx].sample(num_points_per_sample=self.num_points_per_sample)

        if is_training:
            weights = self.label_weights[labels]
            return points_centered, labels, weights
        else:
            return scene_idx, points_centered, points_raw, labels, colors
         
    
    def get_total_num_points(self):
        list_num_points = [len(fd.points) for fd in self.list_file_data]
        return np.sum(list_num_points)
    
    def get_num_batches(self, batch_size):
        return int(self.total_num_points / (batch_size * self.num_points_per_sample))
    
    def init_labels(self):
        # Initiate all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    
if __name__ == '__main__':
    
    # train_dataset = Street3D()
        
    TRAIN_DATASET = Street3D(num_points_per_sample=8960,
                           split="test",
                           box_size_x=10,
                           box_size_y=10,
                           voxel_size=0.3
                           )

    a, b, c = TRAIN_DATASET.sample_batch_in_all_files(4, augment=True)

