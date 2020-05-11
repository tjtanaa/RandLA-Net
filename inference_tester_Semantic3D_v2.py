from os import makedirs
from os.path import exists, join
from helper_ply import read_ply, write_ply
import tensorflow as tf
import numpy as np
import time


def log_string(out_str, log_out):
    log_out.write(out_str + '\n')
    log_out.flush()
    print(out_str)


class InferenceModelTester:
    def __init__(self, model, restore_snap=None):
        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)


        # for op in tf.get_default_graph().get_operations():
        #     print(str(op.name))

        output_tensor=self.sess.graph.get_tensor_by_name('results/Softmax:0')

        # is_training_tensor = self.sess.graph.get_tensor_by_name('IS_TRAINING:0')

        Neighbor_Index_0 = self.sess.graph.get_tensor_by_name('Neighbor_Index_0:0')
        Neighbor_Index_1 = self.sess.graph.get_tensor_by_name('Neighbor_Index_1:0')
        Neighbor_Index_2 = self.sess.graph.get_tensor_by_name('Neighbor_Index_2:0')
        Neighbor_Index_3 = self.sess.graph.get_tensor_by_name('Neighbor_Index_3:0')
        Neighbor_Index_4 = self.sess.graph.get_tensor_by_name('Neighbor_Index_4:0')

        # Subpoints_0 = self.sess.graph.get_tensor_by_name('Subpoints_0:0')
        # Subpoints_1 = self.sess.graph.get_tensor_by_name('Subpoints_1:0')
        # Subpoints_2 = self.sess.graph.get_tensor_by_name('Subpoints_2:0')
        # Subpoints_3 = self.sess.graph.get_tensor_by_name('Subpoints_3:0')
        # Subpoints_4 = self.sess.graph.get_tensor_by_name('Subpoints_4:0')

        Pool_I_0 = self.sess.graph.get_tensor_by_name('Pool_I_0:0')
        Pool_I_1 = self.sess.graph.get_tensor_by_name('Pool_I_1:0')
        Pool_I_2 = self.sess.graph.get_tensor_by_name('Pool_I_2:0')
        Pool_I_3 = self.sess.graph.get_tensor_by_name('Pool_I_3:0')
        Pool_I_4 = self.sess.graph.get_tensor_by_name('Pool_I_4:0')

        Up_I_0 = self.sess.graph.get_tensor_by_name('Up_I_0:0')
        Up_I_1 = self.sess.graph.get_tensor_by_name('Up_I_1:0')
        Up_I_2 = self.sess.graph.get_tensor_by_name('Up_I_2:0')
        Up_I_3 = self.sess.graph.get_tensor_by_name('Up_I_3:0')
        Up_I_4 = self.sess.graph.get_tensor_by_name('Up_I_4:0')

        Batch_XYZ_0 = self.sess.graph.get_tensor_by_name('Batch_XYZ_0:0')
        Batch_XYZ_1 = self.sess.graph.get_tensor_by_name('Batch_XYZ_1:0')
        Batch_XYZ_2 = self.sess.graph.get_tensor_by_name('Batch_XYZ_2:0')
        Batch_XYZ_3 = self.sess.graph.get_tensor_by_name('Batch_XYZ_3:0')
        Batch_XYZ_4 = self.sess.graph.get_tensor_by_name('Batch_XYZ_4:0')

        Batch_Feature = self.sess.graph.get_tensor_by_name('Batch_Feature:0')

        builder = tf.saved_model.builder.SavedModelBuilder('./RandLA-Net_builder_v2')

        # tensor_is_training_tensor = tf.compat.v1.saved_model.utils.build_tensor_info(is_training_tensor)
        
        tensor_Neighbor_Index_0 = tf.compat.v1.saved_model.utils.build_tensor_info(Neighbor_Index_0)
        tensor_Neighbor_Index_1 = tf.compat.v1.saved_model.utils.build_tensor_info(Neighbor_Index_1)
        tensor_Neighbor_Index_2 = tf.compat.v1.saved_model.utils.build_tensor_info(Neighbor_Index_2)
        tensor_Neighbor_Index_3 = tf.compat.v1.saved_model.utils.build_tensor_info(Neighbor_Index_3)
        tensor_Neighbor_Index_4 = tf.compat.v1.saved_model.utils.build_tensor_info(Neighbor_Index_4)

        # tensor_Subpoints_0 = tf.compat.v1.saved_model.utils.build_tensor_info(Subpoints_0)
        # tensor_Subpoints_1 = tf.compat.v1.saved_model.utils.build_tensor_info(Subpoints_1)
        # tensor_Subpoints_2 = tf.compat.v1.saved_model.utils.build_tensor_info(Subpoints_2)
        # tensor_Subpoints_3 = tf.compat.v1.saved_model.utils.build_tensor_info(Subpoints_3)
        # tensor_Subpoints_4 = tf.compat.v1.saved_model.utils.build_tensor_info(Subpoints_4)

        tensor_Pool_I_0 = tf.compat.v1.saved_model.utils.build_tensor_info(Pool_I_0)
        tensor_Pool_I_1 = tf.compat.v1.saved_model.utils.build_tensor_info(Pool_I_1)
        tensor_Pool_I_2 = tf.compat.v1.saved_model.utils.build_tensor_info(Pool_I_2)
        tensor_Pool_I_3 = tf.compat.v1.saved_model.utils.build_tensor_info(Pool_I_3)
        tensor_Pool_I_4 = tf.compat.v1.saved_model.utils.build_tensor_info(Pool_I_4)

        tensor_Up_I_0 = tf.compat.v1.saved_model.utils.build_tensor_info(Up_I_0)
        tensor_Up_I_1 = tf.compat.v1.saved_model.utils.build_tensor_info(Up_I_1)
        tensor_Up_I_2 = tf.compat.v1.saved_model.utils.build_tensor_info(Up_I_2)
        tensor_Up_I_3 = tf.compat.v1.saved_model.utils.build_tensor_info(Up_I_3)
        tensor_Up_I_4 = tf.compat.v1.saved_model.utils.build_tensor_info(Up_I_4)

        tensor_Batch_XYZ_0 = tf.compat.v1.saved_model.utils.build_tensor_info(Batch_XYZ_0)
        tensor_Batch_XYZ_1 = tf.compat.v1.saved_model.utils.build_tensor_info(Batch_XYZ_1)
        tensor_Batch_XYZ_2 = tf.compat.v1.saved_model.utils.build_tensor_info(Batch_XYZ_2)
        tensor_Batch_XYZ_3 = tf.compat.v1.saved_model.utils.build_tensor_info(Batch_XYZ_3)
        tensor_Batch_XYZ_4 = tf.compat.v1.saved_model.utils.build_tensor_info(Batch_XYZ_4)

        tensor_Batch_Feature = tf.compat.v1.saved_model.utils.build_tensor_info(Batch_Feature)

        tensor_output_tensor = tf.compat.v1.saved_model.utils.build_tensor_info(output_tensor)

        prediction_signature = (
            tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    'Neighbor_Index_0': tensor_Neighbor_Index_0,
                    'Neighbor_Index_1': tensor_Neighbor_Index_1,
                    'Neighbor_Index_2': tensor_Neighbor_Index_2,
                    'Neighbor_Index_3': tensor_Neighbor_Index_3,
                    'Neighbor_Index_4': tensor_Neighbor_Index_4,
                    # 'Subpoints_0' : tensor_Subpoints_0,
                    # 'Subpoints_1' : tensor_Subpoints_1,
                    # 'Subpoints_2' : tensor_Subpoints_2,
                    # 'Subpoints_3' : tensor_Subpoints_3,
                    # 'Subpoints_4' : tensor_Subpoints_4,
                    'Pool_I_0' : tensor_Pool_I_0,
                    'Pool_I_1' : tensor_Pool_I_1,
                    'Pool_I_2' : tensor_Pool_I_2,
                    'Pool_I_3' : tensor_Pool_I_3,
                    'Pool_I_4' : tensor_Pool_I_4,
                    'Up_I_0' : tensor_Up_I_0,
                    'Up_I_1' : tensor_Up_I_1,
                    'Up_I_2' : tensor_Up_I_2,
                    'Up_I_3' : tensor_Up_I_3,
                    'Up_I_4' : tensor_Up_I_4,
                    'Batch_XYZ_0' : tensor_Batch_XYZ_0,
                    'Batch_XYZ_1' : tensor_Batch_XYZ_1,
                    'Batch_XYZ_2' : tensor_Batch_XYZ_2,
                    'Batch_XYZ_3' : tensor_Batch_XYZ_3,
                    'Batch_XYZ_4' : tensor_Batch_XYZ_4,
                    'Batch_Feature' : tensor_Batch_Feature
                },
                outputs={'prob_logits': tensor_output_tensor},
                method_name=tf.compat.v1.saved_model.signature_constants
                .PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            self.sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    prediction_signature,
            })

        builder.save()






        # Add a softmax operation for predictions
        self.prob_logits = tf.nn.softmax(model.logits)
        # self.test_probs = [np.zeros((l.data.shape[0], model.config.num_classes), dtype=np.float16)
        #                    for l in dataset.input_trees['test']]

        # self.log_out = open('log_test_' + str(dataset.val_split) + '.txt', 'a')

    # def test_single_sample(self, pointcloud, num_votes=100):
    #     feed_dict = {: [your_image]}
    #     classification = self.sess.run(self.prob_logits, feed_dict)
    #     print classification 

    def test(self, model, dataset, num_votes=100):

        # Smoothing parameter for votes
        test_smooth = 0.98

        # Initialise iterator with train data
        self.sess.run(dataset.test_init_op)

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'predictions')) if not exists(join(test_path, 'predictions')) else None
        makedirs(join(test_path, 'probs')) if not exists(join(test_path, 'probs')) else None

        #####################
        # Network predictions
        #####################

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_votes:

            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],)

                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    inds = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                step_id += 1
                log_string('Epoch {:3d}, step {:3d}. min possibility = {:.1f}'.format(epoch_id, step_id, np.min(
                    dataset.min_possibility['test'])), self.log_out)

            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_possibility['test'])
                log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.log_out)

                if last_min + 4 < new_min:

                    print('Saving clouds')

                    # Update last_min
                    last_min = new_min

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    files = dataset.test_files
                    i_test = 0
                    for i, file_path in enumerate(files):
                        # Get file
                        points = self.load_evaluation_points(file_path)
                        points = points.astype(np.float16)

                        # Reproject probs
                        probs = np.zeros(shape=[np.shape(points)[0], 8], dtype=np.float16)
                        proj_index = dataset.test_proj[i_test]

                        probs = self.test_probs[i_test][proj_index, :]

                        # Insert false columns for ignored labels
                        probs2 = probs
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs2 = np.insert(probs2, l_ind, 0, axis=1)

                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.uint8)

                        # Save plys
                        cloud_name = file_path.split('/')[-1]

                        # Save ascii preds
                        ascii_name = join(test_path, 'predictions', dataset.ascii_files[cloud_name])
                        np.savetxt(ascii_name, preds, fmt='%d')
                        log_string(ascii_name + 'has saved', self.log_out)
                        i_test += 1

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))
                    self.sess.close()
                    import sys
                    sys.exit()

                self.sess.run(dataset.test_init_op)
                epoch_id += 1
                step_id = 0
                continue
        return

    @staticmethod
    def load_evaluation_points(file_path):
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T
