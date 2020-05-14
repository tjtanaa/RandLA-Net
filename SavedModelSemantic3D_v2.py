from os.path import join, exists
from RandLANet import Network
from tester_Semantic3D import ModelTester
from helper_ply import read_ply
from helper_tool import Plot
from helper_tool import DataProcessing as DP
from helper_tool import ConfigSemantic3D as cfg
import tensorflow as tf
import numpy as np
import pickle, argparse, os
from inference_RandLANet_v2 import Inference_Network
from inference_tester_Semantic3D_v2 import InferenceModelTester
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# tf.enable_eager_execution()

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


class Semantic3D:
    def __init__(self):
        self.name = 'Semantic3D'
        self.path = '/media/data3/tjtanaa/semantic3d'
        self.label_to_names = {0: 'unlabeled',
                               1: 'man-made terrain',
                               2: 'natural terrain',
                               3: 'high vegetation',
                               4: 'low vegetation',
                               5: 'buildings',
                               6: 'hard scape',
                               7: 'scanning artefacts',
                               8: 'cars'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])

        self.original_folder = join(self.path, 'original_data')
        self.full_pc_folder = join(self.path, 'original_ply')
        self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))

        # Following KPConv to do the train-validation split
        self.all_splits = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        self.val_split = 1

        # Initial training-validation-testing files
        self.train_files = []
        self.val_files = []
        self.test_files = []
        cloud_names = [file_name[:-4] for file_name in os.listdir(self.original_folder) if file_name[-4:] == '.txt']
        for pc_name in cloud_names:
            if exists(join(self.original_folder, pc_name + '.labels')):
                self.train_files.append(join(self.sub_pc_folder, pc_name + '.ply'))
            else:
                self.test_files.append(join(self.full_pc_folder, pc_name + '.ply'))

        self.train_files = np.sort(self.train_files)
        self.test_files = np.sort(self.test_files)

        for i, file_path in enumerate(self.train_files):
            if self.all_splits[i] == self.val_split:
                self.val_files.append(file_path)

        self.train_files = np.sort([x for x in self.train_files if x not in self.val_files])

        # Initiate containers
        self.val_proj = [] # not needed
        self.val_labels = [] # not needed
        self.test_proj = [] # not needed
        self.test_labels = [] # not needed

        self.possibility = {}
        self.min_possibility = {}
        self.class_weight = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}

        # Ascii files dict for testing
        self.ascii_files = {
            'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply': 'marketsquarefeldkirch4-reduced.labels',
            'sg27_station10_rgb_intensity-reduced.ply': 'sg27_10-reduced.labels',
            'sg28_Station2_rgb_intensity-reduced.ply': 'sg28_2-reduced.labels',
            'StGallenCathedral_station6_rgb_intensity-reduced.ply': 'stgallencathedral6-reduced.labels',
            'birdfountain_station1_xyz_intensity_rgb.ply': 'birdfountain1.labels',
            'castleblatten_station1_intensity_rgb.ply': 'castleblatten1.labels',
            'castleblatten_station5_xyz_intensity_rgb.ply': 'castleblatten5.labels',
            'marketplacefeldkirch_station1_intensity_rgb.ply': 'marketsquarefeldkirch1.labels',
            'marketplacefeldkirch_station4_intensity_rgb.ply': 'marketsquarefeldkirch4.labels',
            'marketplacefeldkirch_station7_intensity_rgb.ply': 'marketsquarefeldkirch7.labels',
            'sg27_station10_intensity_rgb.ply': 'sg27_10.labels',
            'sg27_station3_intensity_rgb.ply': 'sg27_3.labels',
            'sg27_station6_intensity_rgb.ply': 'sg27_6.labels',
            'sg27_station8_intensity_rgb.ply': 'sg27_8.labels',
            'sg28_station2_intensity_rgb.ply': 'sg28_2.labels',
            'sg28_station5_xyz_intensity_rgb.ply': 'sg28_5.labels',
            'stgallencathedral_station1_intensity_rgb.ply': 'stgallencathedral1.labels',
            'stgallencathedral_station3_intensity_rgb.ply': 'stgallencathedral3.labels',
            'stgallencathedral_station6_intensity_rgb.ply': 'stgallencathedral6.labels'}

        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    @tf.function
    def knn_search(self, support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        # neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        # return neighbor_idx.astype(np.int32)
        # print("support_pts.shape: ", support_pts.shape, "\tquery_pts: ", query_pts.shape, "k: ", k)
        # print("tf.expand_dims(query_pts, 1).shape", tf.expand_dims(support_pts, 1).shape)
        # print("tf.expand_dims(query_pts, 1).shape", tf.expand_dims(query_pts, 2).shape)
        # distance = tf.reduce_sum(tf.abs(tf.subtract(tf.expand_dims(support_pts, 1), tf.expand_dims(query_pts, 2))), axis=3)
        # print("distance: ", distance)
        # _, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
        # print("top_k_indices: ", top_k_indices)
        # print("neighbor_idx[0]: ", neighbor_idx[0])
        # print("top_k_indices[0]: ", top_k_indices[0])
        _, top_k_indices = tf.nn.top_k(tf.negative(tf.reduce_sum(tf.abs(tf.subtract(tf.expand_dims(support_pts, 1), tf.expand_dims(query_pts, 2))), axis=3)), k=k)
        return top_k_indices


    def load_sub_sampled_clouds(self, sub_grid_size):

        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        files = np.hstack((self.train_files, self.val_files, self.test_files))

        for i, file_path in enumerate(files):
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)
            if file_path in self.val_files:
                cloud_split = 'validation'
            elif file_path in self.train_files:
                cloud_split = 'training'
            else:
                cloud_split = 'test'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            if cloud_split == 'test':
                sub_labels = None
            else:
                sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)
                # print(dir(search_tree))
                print(len(search_tree.get_arrays()[0]))
                # with open('')

            self.input_trees[cloud_split] += [search_tree] # check whether we need this subsample tree
            self.input_colors[cloud_split] += [sub_colors] # need to remove color
            if cloud_split in ['training', 'validation']:  # do not need this label
                self.input_labels[cloud_split] += [sub_labels]

        # Get validation and test re_projection indices
        print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(files):

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if file_path in self.val_files:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]

            # Test projection
            if file_path in self.test_files:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                    
                self.test_proj += [proj_idx]
                self.test_labels += [labels]
        print('finished')
        return


    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        elif split == 'test':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        # Reset possibility
        self.possibility[split] = []
        self.min_possibility[split] = []
        self.class_weight[split] = []

        # Random initialize
        for i, tree in enumerate(self.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3] # this could be replaced by the input point clouds
                                                                                    # random functionshould be replaced by tf.random.normal()
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]  # replace by tf.math.reduce_min(input_tensor, axis=None, keepdims=False, name=None)

        if split != 'test':
            _, num_class_total = np.unique(np.hstack(self.input_labels[split]), return_counts=True)
            self.class_weight[split] += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]

        def spatially_regular_gen():

            # Generator loop
            for i in range(num_per_epoch):  # num_per_epoch

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)
                query_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                query_idx = DP.shuffle_idx(query_idx)
                # query_idx = DP.tf_shuffle_idx(query_idx)

                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[query_idx]
                queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                queried_pc_colors = self.input_colors[split][cloud_idx][query_idx]
                if split == 'test':
                    queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
                    queried_pt_weight = 1
                else:
                    queried_pc_labels = self.input_labels[split][cloud_idx][query_idx]
                    queried_pc_labels = np.array([self.label_to_idx[l] for l in queried_pc_labels])
                    queried_pt_weight = np.array([self.class_weight[split][0][n] for n in queried_pc_labels])

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
                self.possibility[split][cloud_idx][query_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))
                # print("XYZ: ",queried_pc_xyz.shape)
                # print("Color: ",queried_pc_colors.shape)
                if True:
                    yield (queried_pc_xyz,
                           queried_pc_colors,
                           queried_pc_labels,
                           query_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))
                    # yield (queried_pc_xyz,
                    #        queried_pc_colors.astype(np.float32),
                    #        queried_pc_labels,
                    #        query_idx.astype(np.int32),
                    #        np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    def get_tf_mapping(self):
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            # print("Batch_Cloud_Idx: ", batch_cloud_idx)
            # batch_features = tf.map_fn(self.tf_augment_input, [batch_xyz, batch_features], dtype=tf.float32)
            batch_features = tf.map_fn(self.tf_augment_input, [batch_xyz], dtype=tf.float32)
            # batch_features = batch_xyz
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []
            print("batch_xyz.shape", batch_xyz.shape)
            print("batch_features", batch_features.shape)

            for i in range(cfg.num_layers):
                # neigh_idx = self.knn_search(batch_xyz, batch_xyz, cfg.k_n)
                neigh_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                # neigh_idx = DP.tf_knn_search(batch_xyz, batch_xyz, cfg.k_n)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neigh_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                # up_i = DP.tf_knn_search(sub_points, batch_xyz, 1)
                # up_i = self.knn_search(sub_points, batch_xyz, 1)
                # print("neigh_idx: ", neigh_idx.shape, "sub_points: ", sub_points.shape , "pool_i: ", pool_i.shape, "up_i: ", up_i.shape)
                input_points.append(batch_xyz)
                input_neighbors.append(neigh_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points
                print("sub_points.shape", sub_points.shape)
                print("pool_i.shape", pool_i.shape)
                print("up_i.shape", up_i.shape)

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]
            print("len(input_points): ", len(input_points))
            print("len(input_neighbors): ", len(input_neighbors))
            print("len(input_pools): ", len(input_pools))
            print("len(input_up_samples): ", len(input_up_samples))
            print("type(input_points): ", type(input_points))
            print("type(input_neighbors): ", type(input_neighbors))
            print("type(input_pools): ", type(input_pools))
            print("type(input_up_samples): ", type(input_up_samples))
            print("batch_features.shape", batch_features.shape)            
            return input_list

        return tf_map

    # data augmentation
    @staticmethod
    def tf_augment_input(inputs):
        xyz = inputs[0]
        # features = inputs[1]
        theta = tf.random_uniform((1,), minval=0, maxval=2 * np.pi)
        # Rotation matrices
        c, s = tf.cos(theta), tf.sin(theta)
        cs0 = tf.zeros_like(c)
        cs1 = tf.ones_like(c)
        R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
        stacked_rots = tf.reshape(R, (3, 3))

        # Apply rotations
        transformed_xyz = tf.reshape(tf.matmul(xyz, stacked_rots), [-1, 3])
        # Choose random scales for each example
        min_s = cfg.augment_scale_min
        max_s = cfg.augment_scale_max
        if cfg.augment_scale_anisotropic:
            s = tf.random_uniform((1, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random_uniform((1, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if cfg.augment_symmetries[i]:
                symmetries.append(tf.round(tf.random_uniform((1, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([1, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)

        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.tile(s, [tf.shape(transformed_xyz)[0], 1])

        # Apply scales
        transformed_xyz = transformed_xyz * stacked_scales

        noise = tf.random_normal(tf.shape(transformed_xyz), stddev=cfg.augment_noise)
        transformed_xyz = transformed_xyz + noise
        # rgb = features[:, :3]
        # stacked_features = tf.concat([transformed_xyz, rgb, features[:, 3:4]], axis=-1)
        # return stacked_features
        # generic_feature = tf.ones([tf.shape(transformed_xyz)[0], 1], dtype=tf.float32)
        # print("generic_feature.shape: ", tf.shape(generic_feature))
        # print("transformed_xyz.shape: ", tf.shape(transformed_xyz))
        # print("xyz.shape: ", tf.shape(xyz))
        # return generic_feature
        # return xyz
        return transformed_xyz

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
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


def gen_one_sample(point_cloud):
    point_cloud_input = tf.random.shuffle(point_cloud, seed=None, name=None)
    batch_xyz = tf.expand_dims(point_cloud_input, axis=0) # [Npts,3] => [1,Npts,3]
    point_cloud = point_cloud.reshape(1, *point_cloud.shape)
    batch_feature_input = point_cloud
    # batch_xyz = tf.expand_dims(batch_xyz, axis=2) 
    print("batch_xyz.shape", batch_xyz.shape)   
    input_points = []
    input_neighbors = []
    input_pools = []
    input_up_samples = []
    # batch_feature = tf.concat([batch_xyz, batch_xyz], axis=-1)
    batch_feature = batch_xyz


    # with open("./BATCH_FEATURE.txt", 'w') as f:
    #     for p in range(batch_feature.shape[1]):
    #         f.write(str(batch_feature_input[0,p,0]) + " ")
    #         f.write(str(batch_feature_input[0,p,1]) + " ")
    #         f.write(str(batch_feature_input[0,p,2]) + " ")
    #         if(p < batch_feature.shape[1] - 1):
    #             f.write('\n')
    tf_holder_batch_feature = tf.compat.v1.placeholder(tf.float32, shape=batch_xyz.shape, name='Batch_Feature')
    
    # tf_holder_batch_feature = tf.compat.v1.placeholder(tf.float32, shape=(batch_xyz.shape[0], batch_xyz.shape[1] , batch_xyz.shape[2] ), name='Batch_Feature')
    # tf_batch_feature = tf.compat.v1.placeholder(tf.float32, shape=(batch_xyz.shape[0], batch_xyz.shape[1] , batch_xyz.shape[2] ), name='Batch_Feature')
    # tf_batch_feature_map = Semantic3D.tf_augment_input([tf.squeeze(tf_batch_feature,axis=0)])
    # tf_holder_batch_feature = tf.expand_dims(tf_batch_feature_map, axis=0)
    # print("tf_batch_feature.size: ", tf_batch_feature.shape) 
    # print("tf_holder_batch_size: ", tf_holder_batch_feature.shape)
    tf_holder_input_points = []
    tf_holder_input_neighbors = []
    tf_holder_input_pools = []
    tf_holder_input_up_samples = []

    for i in range(cfg.num_layers):
        # neigh_idx = self.knn_search(batch_xyz, batch_xyz, cfg.k_n)
        # neigh_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
        # neigh_idx = tf.compat.v1.placeholder(tf.int64, shape=(batch_xyz.shape[0], batch_xyz.shape[1], cfg.k_n), name='Neighbor_Index_' + str(i))
        # # sub_points = tf.compat.v1.placeholder(tf.float32, shape=(batch_xyz.shape[0], batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], batch_xyz.shape[2]), name='Subpoints_' + str(i))
        # pool_i = tf.compat.v1.placeholder(tf.int64, shape=(batch_xyz.shape[0], batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], cfg.k_n), name='Pool_I_' + str(i))
        # up_i = tf.compat.v1.placeholder(tf.int64, shape=(batch_xyz.shape[0], batch_xyz.shape[1], 1), name='Up_I_' + str(i))
        
        neigh_idx = tf.compat.v1.placeholder(tf.int64, shape=(point_cloud.shape[0], point_cloud.shape[1], cfg.k_n), name='Neighbor_Index_' + str(i))
        # sub_points = tf.compat.v1.placeholder(tf.float32, shape=(batch_xyz.shape[0], batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], batch_xyz.shape[2]), name='Subpoints_' + str(i))
        pool_i = tf.compat.v1.placeholder(tf.int64, shape=(point_cloud.shape[0], point_cloud.shape[1] // cfg.sub_sampling_ratio[i], cfg.k_n), name='Pool_I_' + str(i))
        up_i = tf.compat.v1.placeholder(tf.int64, shape=(point_cloud.shape[0], point_cloud.shape[1], 1), name='Up_I_' + str(i))
        
        # neigh_idx = DP.tf_knn_search(batch_xyz, batch_xyz, cfg.k_n)
        neigh_idx_np = DP.knn_search(point_cloud, point_cloud, cfg.k_n)
        # sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
        sub_points_np = point_cloud[:, :point_cloud.shape[1] // cfg.sub_sampling_ratio[i], :]
        # pool_i = neigh_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
        pool_i_np = neigh_idx_np[:, :point_cloud.shape[1]  // cfg.sub_sampling_ratio[i], :]
        # # up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
        # up_i = DP.tf_knn_search(sub_points, batch_xyz, 1)
        up_i_np = DP.knn_search(sub_points_np, point_cloud, 1)
        # up_i = self.knn_search(sub_points, batch_xyz, 1)
        # print("neigh_idx: ", neigh_idx.shape, "sub_points: ", sub_points.shape , "pool_i: ", pool_i.shape, "up_i: ", up_i.shape)
        
        # tf_holder_batch_xyz =  tf.compat.v1.placeholder(tf.float32, shape=(batch_xyz.shape[0], batch_xyz.shape[1] , batch_xyz.shape[2]), name='Batch_XYZ_' +str(i))
        tf_holder_batch_xyz =  tf.compat.v1.placeholder(tf.float32, shape=(point_cloud.shape[0], point_cloud.shape[1] , point_cloud.shape[2]), name='Batch_XYZ_' +str(i))
        tf_holder_input_points.append(tf_holder_batch_xyz)

        # input_points.append(batch_xyz)
        # input_neighbors.append(neigh_idx)
        # input_pools.append(pool_i)
        # input_up_samples.append(up_i)
        input_points.append(point_cloud)
        input_neighbors.append(neigh_idx_np)
        input_pools.append(pool_i_np)
        input_up_samples.append(up_i_np)
        point_cloud = sub_points_np
        # batch_xyz = sub_points
        # print("sub_points.shape", sub_points.shape)
        # print("neigh_idx.shape", neigh_idx.shape)
        # print("pool_i.shape", pool_i.shape)
        # print("up_i.shape", up_i.shape)
        # print("sub_points.shape", sub_points.shape)
        # print("neigh_idx.shape", neigh_idx.shape)
        # print("pool_i.shape", pool_i.shape)
        # print("up_i.shape", up_i.shape)
        print("neigh_idx_np.shape", neigh_idx_np.shape)
        print("sub_points_np.shape", sub_points_np.shape)
        print("pool_i_np.shape", pool_i_np.shape)
        print("up_i_np.shape", up_i_np.shape)
        tf_holder_input_neighbors.append(neigh_idx)
        tf_holder_input_pools.append(pool_i)
        tf_holder_input_up_samples.append(up_i)

    input_list_data = input_points + input_neighbors + input_pools + input_up_samples
    input_list_data += [batch_feature_input]

    input_list = tf_holder_input_points + tf_holder_input_neighbors + tf_holder_input_pools + tf_holder_input_up_samples
    input_list += [tf_holder_batch_feature]
    for i, tf_holder in enumerate(input_list):
        print("tf_holder " , i, " ", tf_holder.shape)
    return input_list, input_list_data
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis, save')
    parser.add_argument('--testdir', type=str, default='modeldirectory', help='test model')
    # Log_2020-04-05_05-03-01 65536
    # Log_2020-04-06_05-49-40
    # Log_2020-04-22_06-56-56 
    # Log_2020-04-27_03-43-49 65536//2
    # Log_2020-04-30_02-13-19 65536//16
    FLAGS = parser.parse_args()

    GPU_ID = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    Mode = FLAGS.mode
    if not(FLAGS.mode == 'save' or FLAGS.mode == 'tflite'):
        print("Mode ", FLAGS.mode)
        dataset = Semantic3D()
        dataset.init_input_pipeline()

    if Mode == 'train':
        model = Network(dataset, cfg)
        model.train(dataset)
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        chosen_snapshot = -1
        # logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
        # chosen_folder = logs[-1]
        chosen_folder = FLAGS.testdir
        snap_path = join(chosen_folder, 'snapshots')
        snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)

    elif Mode == 'save':
        point_cloud  = np.around(np.random.randn(65536//4, 3), 5).astype(np.float32)
        # with open("./sample.txt", 'w') as f:
        #     for p in range(point_cloud.shape[0]):
        #         f.write(str(point_cloud[p,0]) + " ")
        #         f.write(str(point_cloud[p,1]) + " ")
        #         f.write(str(point_cloud[p,2]))
        #         if p < point_cloud.shape[0]-1:
        #             f.write("\n")
        # load from sample.txt and double check the compute indices with the one found in bazel
        # this is to ensure that the knn-search algorithm in the bazel has been implemented
        # correctly

        r = 0
        with open('./sample.txt', 'r') as f:
            for i, line in enumerate(f):
                # if i >= 65536//2:
                #     break
                values = line.rstrip('\n').split(" ")
                point_cloud[r, 0] = np.around(float(values[0]),5)
                point_cloud[r, 1] = np.around(float(values[1]),5)
                point_cloud[r, 2] = np.around(float(values[2]),5)
                r += 1 

        with open("./sample.txt", 'w') as f:
            for p in range(point_cloud.shape[0]):
                f.write(str(point_cloud[p,0]) + " ")
                f.write(str(point_cloud[p,1]) + " ")
                f.write(str(point_cloud[p,2])+ " ")
                if p < point_cloud.shape[0]-1:
                    f.write("\n")

        input_to_model, input_list_data = gen_one_sample(point_cloud)
        cfg.saving = False
        model = Inference_Network(input_to_model, cfg)
        chosen_snapshot = -1
        chosen_folder = FLAGS.testdir
        snap_path = join(chosen_folder, 'snapshots')
        snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        tester = InferenceModelTester(model, restore_snap=chosen_snap)
        feed_dict = {}
        for i in range(len(input_to_model)):
            feed_dict[input_to_model[i]] = input_list_data[i]
        segmentation_mask, encoder_features, encoder_sub_indices, decoder_features, decoder_sub_indices \
            = tester.sess.run([tester.prob_logits, tester.encoder_features, tester.encoder_sub_indices , tester.decoder_features, tester.decoder_sub_indices], feed_dict)
        print(segmentation_mask.shape)  

        with open("./results.txt", 'w') as f:
            for p in range(segmentation_mask.shape[0]):
                f.write(str(segmentation_mask[p,0]) + " ")
                f.write(str(segmentation_mask[p,1]) + " ")
                f.write(str(segmentation_mask[p,2]) + " ")
                f.write(str(segmentation_mask[p,3]) + " ")
                f.write(str(segmentation_mask[p,4]) + " ")
                f.write(str(segmentation_mask[p,5]) + " ")
                f.write(str(segmentation_mask[p,6]) + " ")
                f.write(str(segmentation_mask[p,7]) + " ")
                if p < segmentation_mask.shape[0]-1:
                    f.write("\n")
        # load from sample.txt and double check the compute indices with the one found in bazel
        # this is to ensure that the knn-search algorithm in the bazel has been implemented
        # correctly
        # r = 0
        # with open('./results.txt', 'r') as f:
        #     for i, line in enumerate(f):
        #         values = line.rstrip('\n').split(" ")
        #         flag = True
        #         flag = flag & (np.abs(segmentation_mask[r, 0] - float(values[0])) < 1e-7)
        #         flag = flag & (np.abs(segmentation_mask[r, 1] - float(values[1])) < 1e-7)
        #         flag = flag & (np.abs(segmentation_mask[r, 2] - float(values[2])) < 1e-7)
        #         flag = flag & (np.abs(segmentation_mask[r, 3] - float(values[3])) < 1e-7)
        #         flag = flag & (np.abs(segmentation_mask[r, 4] - float(values[4])) < 1e-7)
        #         flag = flag & (np.abs(segmentation_mask[r, 5] - float(values[5])) < 1e-7)
        #         flag = flag & (np.abs(segmentation_mask[r, 6] - float(values[6])) < 1e-7)
        #         flag = flag & (np.abs(segmentation_mask[r, 7] - float(values[7])) < 1e-7)
        #         if not flag:
        #             print("error at point ", i)
        #         r += 1 

        for i in range(len(encoder_features)): 
            print(encoder_features[i].shape) 

        for i in range(len(encoder_sub_indices)):
            print(encoder_sub_indices[i].shape)  
            with open("./encoder_sub_indices_" + str(i) + ".txt", 'w') as f:
                for p in range(encoder_sub_indices[i].shape[1]):
                    for c in range(encoder_sub_indices[i].shape[2]):
                        f.write(str(encoder_sub_indices[i][0, p,c]) + " ")
                    if p < encoder_sub_indices[i].shape[1]-1:
                        f.write("\n")
            # r = 0
            # with open("./encoder_sub_indices_" + str(i) + ".txt", 'r') as f:
            #     for p, line in enumerate(f):
            #         values = line.rstrip('\n').split(" ")
            #         # print(values)
            #         # exit()
            #         flag = True
            #         for c in range(len(values)-1):
            #             flag = flag & (np.abs(encoder_sub_indices[i][0, p,c] - float(values[c])) < 1e-7)
            #         if not flag:
            #             print("error at point ", i)
            #         r += 1 

        for i in range(len(decoder_features)): 
            print(decoder_features[i].shape) 

        for i in range(len(decoder_sub_indices)):
            print(decoder_sub_indices[i].shape)  
            with open("./decoder_sub_indices_" + str(i) + ".txt", 'w') as f:
                for p in range(decoder_sub_indices[i].shape[1]):
                    for c in range(decoder_sub_indices[i].shape[2]):
                        f.write(str(decoder_sub_indices[i][0, p,c]) + " ")
                    if p < decoder_sub_indices[i].shape[1]-1:
                        f.write("\n")
            # r = 0
            # with open("./decoder_sub_indices_" + str(i) + ".txt", 'r') as f:
            #     for p, line in enumerate(f):
            #         values = line.rstrip('\n').split(" ")
            #         flag = True
            #         for c in range(len(values)-1):
            #             flag = flag & (np.abs(decoder_sub_indices[i][0, p,c] - float(values[c])) < 1e-7)
            #         if not flag:
            #             print("error at point ", i)
            #         r += 1 

    elif Mode == "tflite":
        converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.testdir)
        tflite_model = converter.convert()
        open("converted_model.tflite", "wb").write(tflite_model)


    else:
        ##################
        # Visualize data #
        ##################

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            
            while True:
                flat_inputs = sess.run(dataset.flat_inputs)
                pc_xyz = flat_inputs[0]
                sub_pc_xyz = flat_inputs[1]
                labels = flat_inputs[21]
                Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :], cfg.num_classes + 1)
                Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]], cfg.num_classes + 1)
