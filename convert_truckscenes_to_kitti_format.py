# Nuscenes dev-kit.
# Code written by Holger Caesar, 2019.
# Code adapted by Ludmila Dias, 2023.

"""
This script converts truckscenes data to KITTI format.
It is used for compatibility with software that uses KITTI-style annotations.

Main function:
- truckscenes_gt_to_kitti(): Converts truckscenes GT annotations to KITTI format.

Command to run:
- python export_kitti.py truckscenes_gt_to_kitti --trucksc_kitti_dir ~/trucksc_kitti
--trucksc_dir data/man-truckscenes --split training
"""

import os
from typing import List
from tqdm import tqdm
from pyquaternion import Quaternion
import numpy as np
import fire
from PIL import Image

from truckscenes.truckscenes import TruckScenes
from truckscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from truckscenes.utils.data_classes import LidarPointCloud
from truckscenes.eval.detection.utils import category_to_detection_name

class KittiConverter:

    def __init__(self,
                 trucksc_kitti_dir: str = '~/trucksc_kitti', # Directory where KITTI-style dataset will be stored.
                 trucksc_dir: str = 'data/man-truckscenes', # Directory where truckscenes data is stored.
                 cam_name: str = 'CAMERA_LEFT_FRONT', # Name of the camera sensor to be used.
                 lidar_name: str = 'LIDAR_LEFT', # Name of the lidar sensor to be used.
                 image_count: int = 20542, # Number of images in the dataset.
                 trucksc_version: str = 'v1.0-trainval', # Version of the truckscenes dataset.
                 split: str = 'train'): # Split of the dataset to be converted (train, val or test).


        self.trucksc_kitti_dir = os.path.expanduser(trucksc_kitti_dir)
        self.cam_name = cam_name
        self.lidar_name = lidar_name
        self.image_count = image_count
        self.trucksc_version = trucksc_version
        self.split = split

        os.makedirs(self.trucksc_kitti_dir, exist_ok=True)
        self.trucksc = TruckScenes(trucksc_version, trucksc_dir)

    def truckscenes_gt_to_kitti(self) -> None:
        """
        Convert TruckScenes GT to KITTI format.
        """
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse

        token_idx = 0
        split_logs = self._create_splits_logs(self.split, self.trucksc)

        folders = ['label_2', 'calib', 'image_2', 'velodyne']
        output_paths = {f: os.path.join(self.trucksc_kitti_dir, self.split, f) for f in folders}
        for path in output_paths.values():
            os.makedirs(path, exist_ok=True)

        sample_tokens = self._split_to_samples(split_logs)[:self.image_count]
        tokens = []

        for sample_token in tqdm(sample_tokens):
            sample = self.trucksc.get('sample', sample_token)
            lidar_token = sample['data'][self.lidar_name]
            cam_token = sample['data'][self.cam_name]
            sd_cam = self.trucksc.get('sample_data', cam_token)
            sd_lid = self.trucksc.get('sample_data', lidar_token)
            cs_cam = self.trucksc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
            cs_lid = self.trucksc.get('calibrated_sensor', sd_lid['calibrated_sensor_token'])

            # Transform from lidar to camera
            lid_to_ego = transform_matrix(cs_lid['translation'], Quaternion(cs_lid['rotation']), inverse=False)
            ego_to_cam = transform_matrix(cs_cam['translation'], Quaternion(cs_cam['rotation']), inverse=True)
            velo_to_cam = np.dot(ego_to_cam, lid_to_ego)
            velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

            # Calibration
            imu_to_velo_kitti = np.zeros((3, 4))
            r0_rect = Quaternion(axis=[1, 0, 0], angle=0)
            p_left_kitti = np.zeros((3, 4))
            p_left_kitti[:3, :3] = cs_cam['camera_intrinsic']

            # Transform points from lidar to camera
            velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
            velo_to_cam_trans = velo_to_cam_kitti[:3, 3]
            assert (velo_to_cam_trans[1:3] < 0).all()

            src_im = os.path.join(self.trucksc.dataroot, sd_cam['filename'])
            src_lid = os.path.join(self.trucksc.dataroot, sd_lid['filename'])
            dst_im = os.path.join(output_paths['image_2'], sample_token + '.png')
            dst_lid = os.path.join(output_paths['velodyne'], sample_token + '.bin')

            if not os.path.exists(src_im):
                print(f"⚠️ Image not found: {src_im}")
                continue

            if not os.path.exists(dst_im):
                Image.open(src_im).save(dst_im, "PNG")

            pcl = LidarPointCloud.from_file(src_lid)
            pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)
            points = pcl.points.T.astype(np.float32)

            # Filter points
            mask = (
                (points[:, 0] >= -100) & (points[:, 0] <= 100) &
                (points[:, 1] >= -100) & (points[:, 1] <= 100) &
                (points[:, 2] >= -4) & (points[:, 2] <= 4)
            )
            points = points[mask]
            points = points[np.all(np.isfinite(points), axis=1)]

            with open(dst_lid, "wb") as f:
                points.astype(np.float32).tofile(f)

            tokens.append(sample_token)

            # Define Calib file data
            kitti_transforms = { 
                'P0': np.zeros((3, 4)),
                'P1': np.zeros((3, 4)),
                'P2': p_left_kitti,
                'P3': np.zeros((3, 4)),
                'R0_rect': r0_rect.rotation_matrix,
                'Tr_velo_to_cam': np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1))),
                'Tr_imu_to_velo': imu_to_velo_kitti
            }
            
            calib_path = os.path.join(output_paths['calib'], sample_token + '.txt')
            with open(calib_path, "w") as f:
                for k, v in kitti_transforms.items():
                    v = v.flatten()
                    f.write(f"{k}: {' '.join(f'{num:.12e}' for num in v)}\n")

            # Labels
            label_path = os.path.join(output_paths['label_2'], sample_token + '.txt')
            if os.path.exists(label_path):
                continue

            with open(label_path, "w") as label_file:
                for ann_token in sample['anns']:
                    ann = self.trucksc.get('sample_annotation', ann_token)
                    _, boxes, _ = self.trucksc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE, selected_anntokens=[ann_token])
                    if not boxes:
                        continue

                    box = boxes[0]
                    detection_name = category_to_detection_name(ann['category_name'])
                    if detection_name is None:
                        continue

                    box.rotate(kitti_to_nu_lidar_inv)
                    h, w, l = box.wlh[2], box.wlh[0], box.wlh[1]
                    x, y, z = box.center
                    yaw = box.orientation.yaw_pitch_roll[0]

                    label_file.write(f"{detection_name} 0.0 0 -1 -1 -1 -1 -1 {h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {yaw:.2f}\n")

    def _split_to_samples(self, split_logs: List[str]) -> List[str]:
        """
        Get the sample tokens from the given logs.
        """
        return [sample['token'] for sample in self.trucksc.sample if self.trucksc.get('scene', sample['scene_token'])['name'] in split_logs]

    def _create_splits_logs(self, split: str, trucksc: TruckScenes) -> List[str]:
        """
        Get the logs from the given split.
        """
        return [scene['name'] for scene in trucksc.scene] if split.startswith('mini') else [scene['name'] for scene in trucksc.scene]

if __name__ == '__main__':
    fire.Fire(KittiConverter)

