from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
import time


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        self.config = config
        self.num_classes = config.num_classes
        # if self.num_classes ==1:
        #     self.num_classes = 0
        self.num_features = self.config.num_features + 1 # (not available)
        self.num_target_attributes = self.config.num_target_attributes
        self.num_bboxes_attributes = 7
        self.num_fgbg_attributes = 2
        self.num_output_attributes = self.config.num_output_attributes
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            self.inputs['features'] = flat_inputs[4 * num_layers]
            self.inputs['coors'] = flat_inputs[4 * num_layers + 1]
            self.inputs['bboxes'] = flat_inputs[4 * num_layers + 2]
            self.inputs['fgbg'] = flat_inputs[4 * num_layers + 3]
            self.inputs['class_label'] = flat_inputs[4 * num_layers + 4]

            # self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            # self.class_weights = DP.get_class_weights(dataset.name)
            self.Log_file = open('log_train_' + dataset.name + '.txt', 'a')

        with tf.variable_scope('layers'):
            # self.bboxes_pred , self.fgbg_logits , self.cls_logits = self.inference(self.inputs, self.is_training)
            self.logits = self.inference(self.inputs, self.is_training)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss'):
            # self.logits = tf.reshape(self.logits, [config.batch_size, -1, self.num_output_attributes])

            self.num_fg = tf.math.reduce_sum(self.inputs['fgbg']) # (B, N) => ()
            self.alpha = tf.divide(tf.cast(self.num_fg, dtype=tf.float32), 
                    tf.cast(
                        tf.shape(self.inputs['fgbg'])[0]* tf.shape(self.inputs['fgbg'])[1]
                        , dtype=tf.float32))
            
            # compute the loss of the fgbg
            # self.pred_fgbg = self.logits[:,:,
            #     self.num_bboxes_attributes:self.num_bboxes_attributes+ self.num_fgbg_attributes]
            
            
            # focal loss takes in [batch_size, num_anchors, num_classes]

            self.pred_fgbg = tf.reshape(self.logits, [config.batch_size, -1, self.num_fgbg_attributes])


            self.mask_loss =  helper_tf_util.focal_loss(self.pred_fgbg, 
                                        tf.one_hot(self.inputs['fgbg'], depth=self.num_fgbg_attributes), 
                                        weights=None, 
                                        alpha=0.75, 
                                        gamma=2)
            # helper_tf_util.focal_loss(self.reshaped_pred_fgbg, 
            #                     tf.cast(self.reshaped_fgbg, dtype=tf.float32), 
            #                     weights=None, 
            #                     alpha=0.25, 
            #                     gamma=2)
                
 

            # compute the loss for the bounding boxes
            # self.mask = tf.reshape(tf.equal(self.inputs['fgbg'], 1), shape=(-1,1))
            # # self.mask = tf.reshape(tf.equal(self.inputs['fgbg'], 1), shape=(-1, tf.shape(self.inputs['fgbg'])[-1]))
            # self.mask = tf.squeeze(self.mask, -1)
            # self.num_fg = tf.math.reduce_sum(tf.cast(self.mask, dtype=tf.int32))
            # self.reshaped_input = tf.reshape(self.inputs['coors'], shape=(-1, tf.shape(self.inputs['coors'])[-1]))
            # self.reshaped_bboxes = tf.reshape(self.inputs['bboxes'], shape=(-1, tf.shape(self.inputs['bboxes'])[-1]))
            
            # self.masked_input = tf.boolean_mask(self.reshaped_input, self.mask)
            # self.masked_bboxes = tf.boolean_mask(self.reshaped_bboxes, self.mask)

            # self.pred_bboxes = tf.reshape(self.logits[:,:, :self.num_bboxes_attributes], [-1, self.num_bboxes_attributes])
            # self.masked_pred_bboxes = tf.boolean_mask(self.pred_bboxes, self.mask)
            # pad_zero = tf.zeros([tf.shape(self.masked_input)[0], 1])
            # pad_anchor_size = tf.convert_to_tensor(self.config.anchor_size)
            # pad_anchor_size = tf.tile(pad_anchor_size, [tf.shape(self.masked_input)[0] ,1])
            # self.masked_input = tf.concat([self.masked_input,pad_anchor_size,  pad_zero],-1)
            # self.actual_pred_bboxes = self.masked_input + self.masked_pred_bboxes
            # self.bbox_loss = 0
            # for i in range(7):
            #     self.bbox_loss += tf.losses.huber_loss(
            #         self.masked_bboxes[:,i],
            #         self.actual_pred_bboxes[:,i],
            #         weights=1.0,
            #         delta=1.0,
            #     )

            # compute the loss for classification
            # self.pred_cls = tf.reshape(self.logits[:,:, -self.num_classes:], shape=[-1, self.num_classes])
            # self.reshaped_cls = tf.one_hot(self.inputs['class_label'], depth=self.num_classes)
            # self.reshaped_cls = tf.reshape(self.reshaped_cls, shape=(-1, self.num_classes))
            
            # self.masked_pred_cls = tf.boolean_mask(self.pred_cls, self.mask)
            # self.masked_cls = tf.boolean_mask(self.reshaped_cls, self.mask)
            # self.unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.masked_pred_cls, labels=self.masked_cls)
            # self.cls_loss = tf.reduce_mean(self.unweighted_losses)

            self.loss = self.mask_loss # + self.bbox_loss #+ self.cls_loss



        #     # self.labels = tf.reshape(self.labels, [-1])

        #     # Boolean mask of points that should be ignored
        #     ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
        #     for ign_label in self.config.ignored_label_inds:
        #         ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

        #     # Collect logits and labels that are not ignored
        #     valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
        #     valid_logits = tf.gather(self.logits, valid_idx, axis=0)
        #     valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

        #     # Reduce label values in the range of logit shape
        #     reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
        #     inserted_value = tf.zeros((1,), dtype=tf.int32)
        #     for ign_label in self.config.ignored_label_inds:
        #         reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
        #     valid_labels = tf.gather(reducing_list, valid_labels_init)

        #     self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            # self.class_label = tf.cast(self.inputs['class_label'],dtype=tf.int32)
            # self.class_label = tf.reshape(self.class_label, shape=(-1,1))
            # self.class_label = tf.squeeze(self.class_label,-1)
            # self.correct_prediction = tf.nn.in_top_k(self.pred_cls, self.class_label, 1)
            # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.pred_fgbg = tf.reshape(self.pred_fgbg , shape=(-1, self.num_fgbg_attributes))
            self.gt_fgbg = tf.cast(self.inputs['fgbg'],dtype=tf.int32)
            self.gt_fgbg = tf.reshape(self.gt_fgbg, [-1])
            # self.gt_fgbg = tf.squeeze(self.gt_fgbg,-1)
            self.fgbg_correct_prediction = tf.nn.in_top_k(self.pred_fgbg, self.gt_fgbg, 1)
            self.fgbg_accuracy = tf.reduce_mean(tf.cast(self.fgbg_correct_prediction, tf.float32))
            # self.prob_logits = tf.nn.softmax(self.pred_cls)
            self.prob_logits = self.pred_fgbg
            
            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            # tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('fgbg_accuracy', self.fgbg_accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def inference(self, inputs, is_training):

        d_out = self.config.d_out
        feature = inputs['features']
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)


        # # bboxes head
        # bboxes_layer_fc1 = helper_tf_util.conv2d(f_encoder_list[-1], 64, [1, 1], 'bboxes_fc1', [1, 1], 'VALID', True, is_training)
        # bboxes_layer_fc2 = helper_tf_util.conv2d(bboxes_layer_fc1, 32, [1, 1], 'bboxes_fc2', [1, 1], 'VALID', True, is_training)
        # bboxes_layer_drop = helper_tf_util.dropout(bboxes_layer_fc2, keep_prob=0.5, is_training=is_training, scope='bboxes_dp1')
        # bboxes_layer_fc3 = helper_tf_util.conv2d(bboxes_layer_drop, self.num_target_attributes-1, [1, 1], 'bboxes_fc', [1, 1], 'VALID', False,
        #                                     is_training, activation_fn=None)
        # bboxes_out = tf.squeeze(bboxes_layer_fc3, [2])


        # # fgbg head
        # fgbg_layer_fc1 = helper_tf_util.conv2d(f_encoder_list[-1], 64, [1, 1], 'fgbg_fc1', [1, 1], 'VALID', True, is_training)
        # fgbg_layer_fc2 = helper_tf_util.conv2d(fgbg_layer_fc1, 32, [1, 1], 'fgbg_fc2', [1, 1], 'VALID', True, is_training)
        # fgbg_layer_drop = helper_tf_util.dropout(fgbg_layer_fc2, keep_prob=0.5, is_training=is_training, scope='fgbg_dp1')
        # fgbg_layer_fc3 = helper_tf_util.conv2d(fgbg_layer_drop, 1, [1, 1], 'fgbg_fc', [1, 1], 'VALID', False,
        #                                     is_training, activation_fn=None)
        # fgbg_out = tf.squeeze(fgbg_layer_fc3, [2])


        # # classification head

        # cls_layer_fc1 = helper_tf_util.conv2d(f_encoder_list[-1], 64, [1, 1], 'cls_fc1', [1, 1], 'VALID', True, is_training)
        # cls_layer_fc2 = helper_tf_util.conv2d(cls_layer_fc1, 32, [1, 1], 'cls_fc2', [1, 1], 'VALID', True, is_training)
        # cls_layer_drop = helper_tf_util.dropout(cls_layer_fc2, keep_prob=0.5, is_training=is_training, scope='cls_dp1')
        # cls_layer_fc3 = helper_tf_util.conv2d(cls_layer_drop, self.num_classes, [1, 1], 'cls_fc', [1, 1], 'VALID', False,
        #                                     is_training, activation_fn=None)
        # cls_out = tf.squeeze(cls_layer_fc3, [2])
        # return bboxes_out, fgbg_out, cls_out

        f_layer_fc1 = helper_tf_util.conv2d(f_encoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        # f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.num_output_attributes, [1, 1], 'fc', [1, 1], 'VALID', False,
        #                                     is_training, activation_fn=None)
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.num_fgbg_attributes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                    is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out

    def unit_test(self, dataset, cfg):
        ops = [self.logits, self.prob_logits, self.gt_fgbg]
        # , self.mask, self.masked_input, self.masked_bboxes, self.bbox_loss, self.unweighted_losses,
        #         self.gt_fgbg, self.pred_fgbg]  
                # self.pad_anchor_size]
        #  #, self.num_fg, self.mask_loss, self.reshaped_input, 
        # # self.reshaped_bboxes, self.masked_input, self.masked_bboxes, self.pred_bboxes, self.masked_pred_bboxes, 
        # # self.bbox_loss, self.actual_pred_bboxes, self.masked_pred_cls, self.unweighted_losses, self.loss,
        # # self.pred_cls, self.reshaped_cls, self.accuracy]
        self.sess.run(dataset.train_init_op)
        logits = self.sess.run(ops, {self.is_training: True})
        print("single inference")
        print("fgbg shape: ", logits[0].shape)
        for i in range(10):
            print(logits[0][0][i])
        print("fgbg shape: ", logits[1].shape)
        print("fgbg shape: ", logits[2].shape)
        for i in range(10):
            print(logits[1][i])
            print(logits[2][i])
        # print("mask shape: ", logits[1].shape)
        # # for i in range(logits[1].shape[1]):
        #     # print(logits[1][0][i])
        #     # if logits[1][0][i][0]:
        #     #     print(logits[1][0][i])
        # # print("num fg pts:", logits[2].shape)
        # # print("anchor_bbox: ", logits[2][0])
        # # print("mask loss focal loss: ", logits[3])
        # print("reshaped_input: ", logits[4])
        # print("reshaped_bboxes: ", logits[5].shape)
        # print("masked_input: ", logits[6].shape)
        # # print("masked_input: ", logits[2][0])
        # print("masked_bboxes: ", logits[7].shape)
        # # print("logits[7]", logits[7][0])
        # print("pred_bboxes: ", logits[8].shape)
        # print("masked_pred_bboxes: ", logits[9].shape)
        # print("masked_pred_bboxes: ", logits[9][0])
        # print("bbox_loss: ", logits[10])
        # print("actual_pred_bboxes: ", logits[11].shape)
        # print("actual_pred_bboxes: ", logits[11][0])
        # print("masked_pred_cls: ", logits[12].shape)
        # print("masked_pred_cls: ", logits[12][0])
        # print("unweighted_losses: ", logits[13].shape)
        # print("loss: ", logits[14])
        # print("pred_cls: ", logits[15].shape)
        # print("reshaped_cls: ", logits[16].shape)
        # print("accuracy: ", logits[17])
        # print(logits[1][0][0])
        # print(logits[1][1][0])
        # print(logits[1][2][0])
        # print(logits[1][3][0])
        self.sess.run(dataset.train_init_op)

        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                        self.extra_update_ops,
                        self.merged,
                        self.loss,
                        self.logits,
                        self.alpha,
                        self.prob_logits, self.gt_fgbg,
                        self.fgbg_accuracy]
                _, _, summary, l_out, probs, alpha, prob_logits, gt_fgbg, fgbg_accuracy= self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    # for i in range(10):
                    #     print(prob_logits[i])
                    #     print(gt_fgbg[i])
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} FgbgAcc={:4.2f} FgbgPer={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, 0, fgbg_accuracy, alpha, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1
            except tf.errors.OutOfRangeError:
                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                print("Epoch ". self.training_epoch)
        # while self.training_epoch < self.config.max_epoch:
        #     t_start = time.time()
        #     try:
        #         ops = [self.train_op,
        #                 self.extra_update_ops,
        #                 self.merged,
        #                 self.loss,
        #                 self.logits,
        #                 self.accuracy,
        #                 self.fgbg_accuracy]
        #         _, _, summary, l_out, probs, acc , fgbg_accuracy= self.sess.run(ops, {self.is_training: True})
        #         self.train_writer.add_summary(summary, self.training_step)
        #         t_end = time.time()
        #         if self.training_step % 50 == 0:
        #             message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} FgbgAcc={:4.2f} ''---{:8.2f} ms/batch'
        #             log_out(message.format(self.training_step, l_out, acc, fgbg_accuracy, 1000 * (t_end - t_start)), self.Log_file)
        #         self.training_step += 1
        #     except tf.errors.OutOfRangeError:
        #         self.training_epoch += 1
        #         self.sess.run(dataset.train_init_op)


    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:

                m_iou = self.evaluate(dataset)
                if m_iou > np.max(self.mIou_list):
                    # Save the best model
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg
