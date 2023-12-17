from torch.utils.data import Dataset
from CUDE.utils.image import load_image
import os
import numpy as np
import json
from cv2 import imread, IMREAD_UNCHANGED

class CITYSCAPESDataset(Dataset):
    """
    CITYSCAPES dataset class

    Parameters
    ----------

    """

    def __init__(self, root_dir, file_list, train=True,
                 data_transform=None, depth_type=None, with_pose=False,
                 back_context=0, forward_context=0, strides=(1,)
                 ):
        # Assertions
        backward_context = back_context
        assert backward_context >= 0 and forward_context >= 0, "Invalid contexts"

        # For joint training with DGP dataset
        self.dataset_idx = 0

        self.backward_context = backward_context
        self.backward_context_paths = []
        self.forward_context = forward_context
        self.forward_context_paths = []

        self.with_context = (backward_context != 0 or forward_context != 0)
        self.split = file_list.split(".")[0].split("/")[-1]
        if "train" in self.split:
            self.split = "train"
        elif "test" in self.split:
            self.split = "test"
        elif "val" in self.split:
            self.split = "val"

        self.train = train
        self.root_dir = root_dir
        self.data_transform = data_transform

        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None
        self.with_pose = with_pose

        self._cache = {}
        self.pose_cache = {}
        self.oxts_cache = {}
        self.calibration_cache = {}
        self.imu2velo_calib_cache = {}
        self.sequence_origin_cache = {}

        with open(file_list, "r") as f:
            data = f.readlines()

        self.paths = []
        # Get file list from data
        for i, fname in enumerate(data):
            path = os.path.join(root_dir, fname.split()[0])
            if not self.with_depth:
                self.paths.append(path)
            else:
                # Check if the depth file exists
                parent_folder = self._get_parent_folder(path)
                depth = self._get_depth_file(parent_folder, path, self.split)
                if depth is not None and os.path.exists(depth):
                    self.paths.append(path)

        # If using context, filter file list
        if self.with_context:
            paths_with_context = []
            for stride in strides:
                for idx, file in enumerate(self.paths):
                    backward_context_idxs, forward_context_idxs = \
                        self._get_sample_context(
                            file, backward_context, forward_context, stride)
                    if backward_context_idxs is not None and forward_context_idxs is not None:
                        paths_with_context.append(self.paths[idx])
                        self.forward_context_paths.append(forward_context_idxs)
                        self.backward_context_paths.append(backward_context_idxs[::-1])
            self.paths = paths_with_context

    @staticmethod
    def _get_next_file(next_idx, file):
        """Get next file given next idx and current file."""
        base, ext = os.path.splitext(os.path.basename(file))
        city, seq, file_idx, img_type = base.split("_")
        return os.path.join(
            os.path.dirname(file), f"{city}_{seq}_{str(next_idx).zfill(len(file_idx))}_{img_type}" + ext
        )

    @staticmethod
    def _get_parent_folder(image_file):
        """Get the parent folder from image_file."""
        return os.path.abspath(os.path.join(image_file, "../../../.."))

    @staticmethod
    def _get_calibration_file(parent_folder, image_file, split):
        """Get the parent folder from image_file."""
        # parent_folder = os.path.abspath(os.path.join(image_file, "../../../.."))
        camera_main_folder = os.path.join(parent_folder, "camera")
        city, seq, idx, _ = os.path.basename(image_file).split("_")
        idxs_in_folder = [int(f.split("_")[2]) for f in os.listdir(os.path.join(camera_main_folder, split, city)) if f"{city}_{seq}" in f]
        camera_id = min(idxs_in_folder, key=lambda x: abs(x - int(idx)))
        camera_file = os.path.join(camera_main_folder, split, city, f"{city}_{seq}_{camera_id:06}_camera.json")
        assert os.path.exists(camera_file), f"Camera file {camera_file} not found!"
        return camera_file


    @staticmethod
    def _read_raw_calib_file(file):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        c_data = data['intrinsic']
        baseline = data['extrinsic']['baseline']
        return c_data, baseline

    @staticmethod
    def _get_intrinsics(calib_data):
        camera_matrix = np.array(
            [[calib_data['fx'], 0, calib_data['u0']], [0, calib_data['fy'], calib_data['v0']],
             [0, 0, 1]],
            dtype=np.float32)
        return camera_matrix


########################################################################################################################
#### DEPTH
########################################################################################################################

    def _read_depth(self, depth_file, focal_length_x, baseline):
        """Get the depth map from a file."""
        if self.depth_type in ['stereo']:
            disparity = imread(depth_file, IMREAD_UNCHANGED).astype(np.float32)
            disparity[disparity > 0] = (disparity[disparity > 0] - 1) / 256  # According to README
            # Reconstructing raw data values
            depth = np.zeros_like(disparity)
            depth[disparity > 0] = (baseline * focal_length_x) / disparity[disparity > 0]
            return depth
        else:
            raise NotImplementedError(
                'Depth type {} not implemented'.format(self.depth_type))


    def _get_depth_file(self, parent_folder, image_file, split):
        """Get the corresponding depth file from an image file."""
        # parent_folder = os.path.abspath(os.path.join(image_file, "../../../.."))
        depth_main_folder = os.path.join(parent_folder, "disparity_sequence")
        city, seq, idx, _ = os.path.basename(image_file).split("_")
        depth_file = os.path.join(depth_main_folder, split, city, f"{city}_{seq}_{idx}_disparity.png")
        assert os.path.exists(depth_file), f"Depth file {depth_file} not found!"
        return depth_file



    def _get_sample_context(self, sample_name,
                            backward_context, forward_context, stride=1):
        """
        Get a sample context

        Parameters
        ----------
        sample_name : str
            Path + Name of the sample
        backward_context : int
            Size of backward context
        forward_context : int
            Size of forward context
        stride : int
            Stride value to consider when building the context

        Returns
        -------
        backward_context : list of int
            List containing the indexes for the backward context
        forward_context : list of int
            List containing the indexes for the forward context
        """
        base, ext = os.path.splitext(os.path.basename(sample_name))
        city, seq, idx, img_type = base.split("_")
        f_idx = int(idx)

        # Each sequence has 30 training samples
        max_num_files = 30
        min_idx = max(0, round(f_idx - max_num_files/2))
        max_idx = round(f_idx + max_num_files/2)


        # Check bounds
        if (f_idx - backward_context * stride) < min_idx or  \
            (f_idx + forward_context * stride) > max_idx:
            return None, None

        # Backward context
        c_idx = f_idx
        backward_context_idxs = []
        while len(backward_context_idxs) < backward_context and c_idx > min_idx:
            c_idx -= stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                backward_context_idxs.append(c_idx)
        if c_idx < 0 or c_idx < min_idx:
            return None, None

        # Forward context
        c_idx = f_idx
        forward_context_idxs = []
        while len(forward_context_idxs) < forward_context and c_idx < max_idx:
            c_idx += stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                forward_context_idxs.append(c_idx)
        if c_idx > max_idx:
            return None, None

        return backward_context_idxs, forward_context_idxs

    def _get_context_files(self, sample_name, idxs):
        """
        Returns image and depth context files

        Parameters
        ----------
        sample_name : str
            Name of current sample
        idxs : list of idxs
            Context indexes

        Returns
        -------
        image_context_paths : list of str
            List of image names for the context
        depth_context_paths : list of str
            List of depth names for the context
        """
        image_context_paths = [self._get_next_file(i, sample_name) for i in idxs]
        if self.with_depth:
            parent_folder = self._get_parent_folder(f)
            depth_context_paths = [self._get_depth_file(parent_folder, f, self.split) for f in image_context_paths]
            return image_context_paths, depth_context_paths
        else:
            return image_context_paths, None

    ########################################################################################################################

    def __len__(self):
        """Dataset length."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Get dataset sample given an index."""
        # Add image information
        sample = {
            'idx': idx,
            'dataset_idx': self.dataset_idx, # For joint training with DGP
            'sensor_name': 'CAMERA', # For joint training with DGP
            'splitname': self.split, # For joint training with DGP
            'filename': '%s_%s' % (self.split, os.path.basename(self.paths[idx]).split(".")[0]),
            'rgb': load_image(self.paths[idx]),
        }

        # Add intrinsics
        parent_folder = self._get_parent_folder(self.paths[idx])
        camera_file = self._get_calibration_file(parent_folder, self.paths[idx], self.split)
        if camera_file in self.calibration_cache:
            c_data, baseline = self.calibration_cache[camera_file]
        else:
            c_data, baseline = self._read_raw_calib_file(camera_file)
            self.calibration_cache[camera_file] = [c_data, baseline]
        sample.update({
            'intrinsics': self._get_intrinsics(c_data),
        })

        # Add depth information if requested
        if self.with_depth:
            sample.update({
                'depth': self._read_depth(self._get_depth_file(parent_folder, self.paths[idx],
                                                               self.split),
                                                               focal_length_x=c_data['fx'],
                                                               baseline=baseline),
            })

        # Add context information if requested
        if self.with_context:
            # Add context images
            all_context_idxs = self.backward_context_paths[idx] + \
                               self.forward_context_paths[idx]
            image_context_paths, _ = \
                self._get_context_files(self.paths[idx], all_context_idxs)
            image_context = [load_image(f) for f in image_context_paths]
            sample.update({
                'rgb_context': image_context
            })

        # Apply transformations
        if self.data_transform:
            sample = self.data_transform(sample)

        # Return sample
        return sample

########################################################################################################################
