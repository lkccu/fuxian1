# Copied and edited from https://github.com/princeton-vl/RAFT/blob/master/core/datasets.py
# which is under BSD 3-Clause License

# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import tensorflow as tf

import os
import math
import copy
import random
from glob import glob
import os.path as osp

from . import frame_utils
from .augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset:
    ''' Base class for optical flow dataset iterations '''
    def __init__(self, aug_params=None, sparse=False):
        ''' 
        Args:
          aug_params (dict): A dict containing augmentation parameters
            treated as arguments for FlowAugmentor.
            - crop_size (tuple<int>): Spartial crop size
            - min_scale (float): minimum scale factor
            - max_scale (float): maximum scale factor
            - do_flip (bool): flip flag
          sparse (bool): Whether the flow data is annotated sparse or not 
        '''
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            # img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            # img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        # if not self.init_seed:
        #     # worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         # torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        # img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        # img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        # flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            # valid = torch.from_numpy(valid)
            pass
        else:
            # valid = (flow[:, :, 0].abs() < 1000) & (flow[:, :, 1].abs() < 1000)
            valid = (np.abs(flow[:, :, 0]) < 1000) * (np.abs(flow[:, :, 1]) < 1000)

        # return img1, img2, flow, valid.float()
        return img1, img2, flow, valid

    def __rmul__(self, v):
        self_copy = copy.deepcopy(self)
        self_copy.flow_list *= v
        self_copy.image_list *= v
        return self_copy

    def __add__(self, other):
        copied = copy.deepcopy(self)
        copied.flow_list += other.flow_list
        copied.image_list += other.image_list
        return copied
        
    def __len__(self):
        return len(self.image_list)

    def __call__(self):
        for sample in self:
            yield sample

    def shuffle(self):
        perm = np.random.permutation(len(self))
        self.flow_list = [self.flow_list[i] for i in perm]
        self.image_list = [self.image_list[i] for i in perm]
        

class MpiSintel(FlowDataset):
    '''MPI Sintel dataset: http://sintel.is.tue.mpg.de/ '''
    def __init__(self,
                 aug_params=None,
                 split='training',
                 root='datasets/MPI-Sintel-complete',
                 dstype='clean'):
        ''' 
        Args:
          aug_params (dict): A dict containing augmentation parameters
            treated as arguments for FlowAugmentor.
            - crop_size (tuple<int>): Spartial crop size
            - min_scale (float): minimum scale factor
            - max_scale (float): maximum scale factor
            - do_flip (bool): flip flag
          split (str): training/validation split
          root (str): path to the dataset directory
          dstype (str): clean/final path (difficulty)
        '''        
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    ''' FlyingChairs dataset:
     https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html
     '''
    def __init__(self,
                 aug_params=None,
                 split='training',
                 split_txt='FlyingChairs_train_val.txt',
                 root='datasets/FlyingChairs_release/data'):
        ''' 
        Args:
          aug_params (dict): A dict containing augmentation parameters
            treated as arguments for FlowAugmentor.
            - crop_size (tuple<int>): Spartial crop size
            - min_scale (float): minimum scale factor
            - max_scale (float): maximum scale factor
            - do_flip (bool): flip flag
          split (str): training/validation split
          split_txt (str): path to the textfile indicating train/val split
          root (str): path to the dataset directory
        '''        
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt(split_txt, dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self,
                 aug_params=None,
                 root='datasets/FlyingThings3D',
                 dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self,
                 aug_params=None,
                 split='training',
                 root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


def ShapeSetter(batch_size, image_size):
    def f(image1, image2, flow, valid):
        image1.set_shape((batch_size, *image_size, 3))
        image2.set_shape((batch_size, *image_size, 3))
        flow.set_shape((batch_size, *image_size, 2))
        valid.set_shape((batch_size, *image_size))
        return image1, image2, flow, valid
    return f
    

def as_supervised(image1, image2, flow, valid):
    return (image1, image2), (flow, valid)


def CropOrPadder(target_size):
    def f(image1, image2, flow, valid):
        image1 = tf.image.resize_with_crop_or_pad(image1, *target_size)
        image2 = tf.image.resize_with_crop_or_pad(image2, *target_size)
        
        flow = tf.image.resize_with_crop_or_pad(flow, *target_size)
        valid = tf.expand_dims(valid, axis=-1)
        valid = tf.image.resize_with_crop_or_pad(valid, *target_size)
        valid = tf.squeeze(valid, axis=-1)        

        return image1, image2, flow, valid
    return f
