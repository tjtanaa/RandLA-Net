from __future__ import division

import numpy as np
from os.path import join
from copy import deepcopy
import time
import multiprocessing
from tqdm import tqdm
from numpy.linalg import multi_dot, inv
from augmentation import gauss_dist, rotate, scale, flip, shear, shuffle, transform, \
    point_feature_value_perturbation
from normalization import length_normalize, feature_normalize, bboxes_normalization, convert_threejs_coors, convert_threejs_bbox

class KITTI_PC(object):
    def __init__(self,
                 phase,
                 npoint,
                 batch_size,
                 rotate_range=0.,
                 rotate_mode='g',
                 scale_range=0.,
                 scale_mode='g',
                 flip=False,
                 gauss_drop=False,
                 normalization='channel_std',
                 nbbox=64,
                 use_refined_bbox=False,
                 queue_size=20,
                 hvd_id=0,
                 hvd_size=1,
                 num_worker=1,
                 home='/media/data1/detection_reforge/dataset'):
        self.home = home
        assert phase in ["training", "validation"]
        self.phase = phase
        self.npoint = npoint
        self.batch_size = batch_size
        self.rotate_range = rotate_range
        self.rotate_mode = rotate_mode
        self.scale_range = scale_range
        self.scale_mode = scale_mode
        self.flip = flip
        self.shuffle = shuffle
        self.normalization = normalization
        self.nbbox = nbbox
        self.gauss_drop = gauss_drop
        self.points = np.load(join(home, '{}_lidar_points.npy'.format(self.phase)))
        self.attentions = np.load(join(home, '{}_pred_foreground_points.npy'.format(self.phase)))
        if use_refined_bbox:
            self.bboxes = np.load(join(home, '{}_bbox_refined.npy'.format(self.phase)), allow_pickle=True)
        else:
            self.bboxes = np.load(join(home, '{}_bbox.npy'.format(self.phase)), allow_pickle=True)
        self.queue_size = queue_size
        self.num_worker = num_worker
        self.hvd_id = hvd_id
        self.hvd_size = hvd_size
        self.total_data_length = int(len(self.points) * 1.0)
        self.hvd_data_length = self.total_data_length // self.hvd_size
        self.batch_sum = int(np.ceil(self.hvd_data_length / batch_size))
        self.test_start_id = self.hvd_data_length * self.hvd_id
        self.idx = self.test_start_id
        self.threads = []
        self.q = multiprocessing.Queue(maxsize=self.queue_size)
        if self.hvd_id == 0:
            print("==========Generator Configurations===========")
            print("Using Configurations:")
            print("Dataset home: {}".format(self.home))
            print("Phase: {}".format(self.phase))
            print("Use refined bbox: {}".format(use_refined_bbox))
            print("Total Dataset length: {}".format(self.total_data_length))
            print("Horovod Dataset length: {}".format(self.hvd_data_length))
            print("Number of input points: {}".format(self.npoint))
            print("Batch size: {}".format(self.batch_size))
            print("Shuffle: {}".format(self.shuffle))
            print("Normalization method: {}".format(self.normalization))
            print("    Rotate range: {}, mode: {}".format(self.rotate_range, self.rotate_mode))
            print("    Scale range: {}, mode: {}".format(self.scale_range, self.scale_mode))
            print("    Flip: {}".format(self.flip))
            print("    Shuffle: {}".format(self.shuffle))
            print("    Guass dropout: {}".format(self.gauss_drop))
            print("Queue size: {}".format(self.queue_size))
            print("Horovod node id: {}".format(self.hvd_id))
            print("Horovod node number: {}".format(self.hvd_size))
            print("Number of threads per node: {}".format(self.num_worker))
            print("==============================================")
            self.start()

    def start(self):
        for i in range(self.num_worker):
            thread = multiprocessing.Process(target=self.aug_process)
            thread.daemon = True
            self.threads.append(thread)
            thread.start()

    def stop(self):
        for thread in self.threads:
            thread.terminate()
            thread.join()
            self.q.close()

    def aug_process(self):
        np.random.seed(int(time.time()*1e3-int(time.time())*1e3))
        while True:
            if self.q.qsize() < self.queue_size:
                sample_npoint = self.npoint if not self.gauss_drop else int(np.floor(gauss_dist(self.npoint, self.npoint // 4)))
                batch_coors = np.zeros((self.batch_size, self.npoint, 3), dtype=np.float32)
                batch_features = np.zeros((self.batch_size, self.npoint, 1), dtype=np.float32)
                batch_attentions = np.zeros((self.batch_size, self.npoint, 1), dtype=np.int32)
                batch_bbox = np.zeros((self.batch_size, self.nbbox, 9), dtype=np.float32)

                for i in range(self.batch_size):
                    idx = np.random.randint(self.hvd_data_length)
                    points = deepcopy(self.points[idx])
                    attentions = deepcopy(self.attentions[idx])
                    bboxes = deepcopy(self.bboxes[idx])
                    point_data = length_normalize(np.concatenate([points, attentions], axis=-1), length=sample_npoint)
                    point_data = shuffle(point_data)
                    coors = point_data[:, :3]
                    features = point_data[:, 3:-1]
                    attentions = point_data[:, -1:]


                    T_rotate, angle = rotate(self.rotate_range, self.rotate_mode)
                    T_scale, scale_xyz = scale(self.scale_range, self.scale_mode)
                    T_flip, flip_y = flip(flip=self.flip)


                    T_coors = multi_dot([T_scale, T_flip, T_rotate])
                    coors = transform(coors, T_coors)

                    features = feature_normalize(features, method=self.normalization)
                    ret_bboxes = []
                    for box in bboxes:
                        w, l, h, x, y, z, r = box[:7]
                        x, y, z = transform(np.array([x, y, z]), T_coors)
                        w, l, h = transform(np.array([w, l, h]), T_scale)
                        r += angle
                        if flip_y == -1:
                            r = r + 2 * (np.pi / 2 - r)
                        if np.abs(r) > np.pi:
                            r = (2 * np.pi - np.abs(r)) * ((-1)**(r//np.pi))
                        category = box[-2]
                        difficulty = box[-1]
                        ret_bboxes.append([w, l, h, x, y, z, r, category, difficulty])

                    batch_coors[i] = coors
                    batch_features[i] = features
                    batch_attentions[i] = np.round(attentions).astype(np.int32)
                    batch_bbox[i] = bboxes_normalization(ret_bboxes, length=self.nbbox)
                self.q.put([batch_coors, batch_features, batch_attentions, batch_bbox])
            else:
                time.sleep(0.05)

    def train_generator(self):
        while True:
            if self.q.qsize() != 0:
                yield self.q.get()
            else:
                time.sleep(0.05)

    def valid_generator(self, start_idx=None):
        if start_idx is not None:
            self.idx = start_idx
        while True:
            stop_idx = int(np.min([self.idx + self.batch_size, self.hvd_data_length * (self.hvd_id + 1)]))
            batch_size = stop_idx - self.idx
            batch_coors = np.zeros((batch_size, self.npoint, 3), dtype=np.float32)
            batch_features = np.zeros((batch_size, self.npoint, 1), dtype=np.float32)
            batch_attentions = np.zeros((batch_size, self.npoint, 1), dtype=np.int32)
            batch_bbox = np.zeros((batch_size, self.nbbox, 9), dtype=np.float32)
            for i in range(batch_size):
                points = deepcopy(self.points[self.idx])
                attentions = deepcopy(self.attentions[self.idx])
                point_data = length_normalize(np.concatenate([points, attentions], axis=-1), length=self.npoint)
                coors = point_data[:, :3]
                features = point_data[:, 3:-1]
                attentions = point_data[:, -1:]

                if self.phase != 'testing':
                    bboxes = deepcopy(self.bboxes[self.idx])
                    batch_bbox[i] = deepcopy(bboxes_normalization(bboxes, length=64))
                batch_coors[i] = coors
                batch_features[i] = feature_normalize(features, method=self.normalization)
                batch_attentions[i] = np.round(attentions).astype(np.int32)
                self.idx += 1
            self.idx = self.test_start_id if stop_idx == self.hvd_data_length * (self.hvd_id + 1) else stop_idx
            yield batch_coors, batch_features, batch_attentions, batch_bbox


# Below is the testing script.
if __name__ == '__main__':
    dataset = KITTI_PC(phase='training',
                       batch_size=16,
                       npoint=20000,
                       rotate_range=np.pi/4,
                       rotate_mode='u',
                       scale_range=0.5,
                       scale_mode='u',
                       flip=True,
                       use_refined_bbox=False,
                       normalization='channel_std',
                       hvd_size=5,
                       hvd_id=2)
    for i in tqdm(range(1000)):
        coors, ref, attention, bboxes = next(dataset.valid_generator())
    # coors, ref, attention, bboxes = next(dataset.train_generator())
    dataset.stop()
    coors = coors[0]
    bboxes = bboxes[0]
    attention = attention[0]
    rgbs = np.zeros(coors.shape) + 255.
    rgbs[attention[:, 0]==1, :] = [255., 0, 0]


