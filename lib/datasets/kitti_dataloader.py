import numpy as np
# from scipy.misc import imread
import imageio
import os
import sys
import json
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
# print(BASE_DIR)
sys.path.append('/home/tan/tjtanaa/Detect_RandLA-Net')

from helper_tool import DataProcessing as DP
from helper_tool import ConfigKitti as cfg
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
import tensorflow as tf



class KittiLoader(object):
    
    def __init__(self):
        # with open('./lib/datasets/dataloader_config.json', 'r') as f:
        #     json_obj = json.load(f)
        # self.root = json_obj['TRAINING_DATA_PATH']
        # self.num_pts = json_obj['NUM_POINTS']
        # self.pc_path = os.path.join(self.root, 'point_cloud')
        # self.label_path = os.path.join(self.root, 'labels')
        # self.frames = [s.split('.')[0] for s in os.listdir(self.pc_path) if '.bin' in s ]
        self.name = 'KITTI'
        self.root = cfg.train_data_path
        self.num_pts = cfg.num_points
        self.pc_path = os.path.join(self.root, 'point_cloud')
        self.label_path = os.path.join(self.root, 'labels')
        self.frames = [s.split('.')[0] for s in os.listdir(self.pc_path) if '.bin' in s ]
        self.num_classes = cfg.num_classes
        if self.num_classes ==1:
            self.num_classes = 0
        self.num_features = cfg.num_features
        self.num_target_attributes = cfg.num_target_attributes
        self.split_ratio = cfg.split_ratio
        self.num_samples = len(self.frames)
        assert np.abs(np.sum(self.split_ratio) - 1.0) < 1e-5
        train_split = int(self.num_samples * self.split_ratio[0])
        val_split = int(self.num_samples * np.sum(self.split_ratio[:2]))

        self.frames_indices = np.arange(len(self.frames))
        # self.train_list = self.frames[:train_split]
        # self.val_list = self.frames[train_split:val_split]
        # self.test_list = self.frames[val_split:]
        self.train_list = self.frames_indices[:train_split]
        self.val_list = self.frames_indices[train_split:val_split]
        self.test_list = self.frames_indices[val_split:]

        self.train_list = DP.shuffle_list(self.train_list)
        self.val_list = DP.shuffle_list(self.val_list)
        self.num_points_left = int(cfg.num_points * np.prod(cfg.sub_sampling_ratio))


    def resampling_pc_indices(self, pc):
        # resampling the point cloud to NUM_POINTS
        indices = np.arange(pc.shape[0])
        if self.num_pts > pc.shape[0]:
            num_pad_pts = self.num_pts - pc.shape[0]
            pad_indices = np.random.choice(indices, size=num_pad_pts)
            indices = np.hstack([indices, pad_indices])
            np.random.shuffle(indices)
        else:
            np.random.shuffle(indices)
            indices = indices[:self.num_pts]

        return indices


    def get_sample(self, idx):
        _idx = idx
        while(True):
            with open(os.path.join(self.pc_path, self.frames[_idx] + '.bin'), 'rb') as f:
                pc = np.fromfile(f, dtype=np.float32)
                pc = pc.reshape(pc.shape[0]//4, 4)
            
            # with open(os.path.join(self.label_path, self.frames[idx] + '.pkl'), 'rb') as f:
            #     label = pickle.load(f)

            with open(os.path.join(self.label_path, self.frames[_idx] + '.bin'), 'rb') as f:
                
                target = np.fromfile(f, dtype=np.float32)
                target = target.reshape(target.shape[0]//(self.num_target_attributes)
                                , self.num_target_attributes)
            

            # start to precompute the randlanet input

            # Random resampling the points to NUM_POINTS self.num_pts
            indices = self.resampling_pc_indices(pc)

            resampled_pc = pc[indices,:3]
            resampled_features = pc[indices,:4]
            resampled_bboxes = target[indices, :self.num_target_attributes - 2] # [x,y,z,h,w,l,ry]
            resampled_fgbg = target[indices, self.num_target_attributes-2] #.reshape(-1,1) # [fgbg]
            resampled_cls = target[indices, self.num_target_attributes-1] - 1.0 #.reshape(-1,1) # [cls]
            # resampled_cls_one_hot = None
            # if self.num_classes > 1:
            #     # print("num class > 1")
            #     resampled_cls_one_hot = target[indices, self.num_target_attributes:self.num_target_attributes + self.num_classes]
            
            # print(resampled_pc.shape)
            # print(resampled_features.shape)
            # print(resampled_target.shape)
            # print(resampled_fgbg.shape)
            # print(resampled_cls_one_hot.shape)
            if np.sum(resampled_fgbg[:self.num_points_left]) > 0:
            # return resampled_pc, resampled_features, resampled_target, resampled_fgbg, resampled_cls_one_hot
                return resampled_pc, resampled_features, resampled_bboxes, resampled_fgbg, resampled_cls
            else:
                if _idx -1 < 0:
                    _idx = _idx + 1
                else:
                    _idx = _idx - 1
        

    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = int(len(self.train_list) / cfg.batch_size) * cfg.batch_size
            path_list = self.train_list
        elif split == 'validation':
            num_per_epoch = int(len(self.val_list) / cfg.val_batch_size) * cfg.val_batch_size
            cfg.val_steps = int(len(self.val_list) / cfg.batch_size)
            path_list = self.val_list
        elif split == 'test':
            num_per_epoch = int(len(self.test_list) / cfg.val_batch_size) * cfg.val_batch_size * 4
            path_list = self.test_list        

        def spatially_regular_gen():
            for i in range(num_per_epoch):
                sample_id = path_list[i]
                pc, features, bboxes, fgbg, cls_label = self.get_sample(sample_id)
                yield (pc.astype(np.float32), 
                    features.astype(np.float32), 
                    bboxes.astype(np.float32), 
                    fgbg.astype(np.int32), 
                    cls_label.astype(np.int32))

        gen_func = spatially_regular_gen            
        gen_types = (tf.float32, tf.float32, tf.float32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 4], [None, 7], [None], [None])
        return gen_func, gen_types, gen_shapes
    
    @staticmethod
    def get_tf_mapping():

        def tf_map(batch_pc, batch_features, batch_bboxes, batch_fgbg, batch_cls):
            features = batch_features
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                print("Number of layer: ", i)
                neighbour_idx = tf.py_func(DP.knn_search, [batch_pc, batch_pc, cfg.k_n], tf.int32)
                index_limit = tf.cast(tf.cast(tf.shape(batch_pc)[1], dtype=tf.float32) * cfg.sub_sampling_ratio[i], dtype = tf.int32)
                print("index_limit ", index_limit)
                sub_points = batch_pc[:, :index_limit, :]
                pool_i = neighbour_idx[:, :index_limit, :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_pc, 1], tf.int32)
                input_points.append(batch_pc)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_pc = sub_points

            batch_bboxes = batch_bboxes[:, :int(tf.shape(batch_pc)[1]),:]
            batch_fgbg = batch_fgbg[:, :int(tf.shape(batch_pc)[1])]
            batch_cls = batch_cls[:, :int(tf.shape(batch_pc)[1])]

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [features, batch_pc, batch_bboxes, batch_fgbg, batch_cls]

            print('len(input_list): ', len(input_list))

            return input_list

        return tf_map


    def init_input_pipeline(self):
        print('Initiating input pipelines')
        # cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        gen_function_test, _, _ = self.get_batch_gen('test')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)
        self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)
        self.test_init_op = iter.make_initializer(self.batch_test_data)



if __name__ == '__main__':

    dataset = KittiLoader()
    dataset.init_input_pipeline()
    
    with tf.Session() as sess:
        one_batch = sess.run(dataset.train_init_op)
        one_batch = sess.run(dataset.flat_inputs)
        print("len(one_batch): ", len(one_batch))
        for tensor in one_batch:
            print(tensor.shape)
        print("tensor")
        print(np.sum(one_batch[-2]))
        print(np.sum(one_batch[-1]))
        fgbg_value_set = set()
        for i in range(len(one_batch[-2][0])):
            fgbg_value_set.update([one_batch[-2][0][i]])
        print("fgbg_value_set: ", fgbg_value_set)
        
        cls_value_set = set()
        for j in range(len(one_batch[-1])):
            per_scene_cls_value_set = set()
            for i in range(len(one_batch[-1][0])):
                cls_value_set.update([one_batch[-1][j][i]])
                per_scene_cls_value_set.update([one_batch[-1][j][i]])
        print("cls_value_set: ", cls_value_set)
        print(tensor)

    # print(len(dataset.frames))
    # pc, features, target, fgbg, cls_one_hot = dataset.get_sample(0)
    # print(pc.shape)
    # print(pc[0])

