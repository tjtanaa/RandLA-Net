from os.path import exists, join
from os import makedirs
import os
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
import time
import io
import matplotlib.pyplot as plt
import sklearn
import itertools
from datetime import datetime

from tensorboard.plugins.mesh import summary as mesh_summary

def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def plot_confusion_matrix(cm, class_names=None):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure()

    if class_names is None:
        class_names = [i for i in range(len(cm))]
    figure = plt.figure(figsize=(8, 8))
    canvas = FigureCanvas(fig)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    canvas.draw()       # draw the canvas, cache the renderer

    s, (width, height) = canvas.print_to_buffer()

    # Option 2a: Convert to a NumPy array.
    image = np.fromstring(s, np.uint8).reshape((height, width, 4))  #.astype('float32')
    # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    # image[:,:,0] *= (1/255.0)
    # image[:,:,1] *= (1/255.0)
    # image[:,:,2] *= (1/255.0)
    return image[np.newaxis,:,:,:]
    # return figure


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
        self.num_points_left = int(config.num_points * np.prod(config.sub_sampling_ratio))
        self.num_points_to_pad = config.num_points - self.num_points_left
        # self.global_step = 0
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
            self.validation_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            # self.class_weights = DP.get_class_weights(dataset.name)
            self.Log_file = open('log_train_' + dataset.name + '.txt', 'a')

        # with tf.variable_scope('layers'):
        #     # self.bboxes_pred , self.fgbg_logits , self.cls_logits = self.inference(self.inputs, self.is_training)
        #     self.logits = self.inference(self.inputs, self.is_training)


        #####################################################################
        # single head predictor: 1. fgbg classifation + class classification #
        #####################################################################
        # with tf.variable_scope('loss'):
        #     # self.logits = tf.reshape(self.logits, [config.batch_size, -1, self.num_output_attributes])
        #     self.logits = tf.reshape(self.logits, [config.batch_size, -1, self.num_fgbg_attributes + self.num_classes])

        #     self.num_fg = tf.math.reduce_sum(self.inputs['fgbg']) # (B, N) => ()
        #     self.alpha = tf.divide(tf.cast(self.num_fg, dtype=tf.float32), 
        #             tf.cast(
        #                 tf.shape(self.inputs['fgbg'])[0]* tf.shape(self.inputs['fgbg'])[1]
        #                 , dtype=tf.float32))
            
        #     # compute the loss of the fgbg
        #     # self.pred_fgbg = self.logits[:,:,
        #     #     self.num_bboxes_attributes:self.num_bboxes_attributes+ self.num_fgbg_attributes]
            
        #     # self.logits = tf.reshape(self.logits, [config.batch_size, -1, self.num_fgbg_attributes + self.num_classes])

            
        #     # focal loss takes in [batch_size, num_anchors, num_classes]

        #     self.pred_fgbg = self.logits[:, :, :self.num_fgbg_attributes]
            

        #     self.mask_loss =  helper_tf_util.focal_loss(self.pred_fgbg, 
        #                                 tf.one_hot(self.inputs['fgbg'], depth=self.num_fgbg_attributes), 
        #                                 weights=None, 
        #                                 alpha=0.25, 
        #                                 gamma=2)

        #     self.interested_pc = self.inputs['xyz'][0]        
        #     self.pred_fgbg_label = tf.equal(tf.cast(tf.argmax(tf.nn.sigmoid(self.logits),axis=-1),dtype=tf.int32), tf.cast(self.inputs['fgbg'],dtype=tf.int32))          
 
        #     # compute the loss for classification
        #     self.mask = tf.reshape(tf.equal(self.inputs['fgbg'], 1), shape=[-1])
        #     self.pred_cls = tf.reshape(self.logits[:,:, -self.num_classes:], shape=[-1, self.num_classes])
        #     self.reshaped_target_cls = tf.reshape(self.inputs['class_label'], shape=[-1])
        #     self.masked_pred_cls = tf.boolean_mask(self.pred_cls, self.mask)
        #     self.masked_target_cls = tf.boolean_mask(self.reshaped_target_cls, self.mask)
        #     self.masked_one_hot_cls = tf.one_hot(self.masked_target_cls, depth=self.num_classes)
        #     self.unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.masked_pred_cls, labels=self.masked_one_hot_cls)
        #     self.cls_loss = tf.cond(self.num_fg > 0, lambda: tf.reduce_mean(self.unweighted_losses), 
        #         lambda: tf.convert_to_tensor(0.0)* tf.reduce_mean(self.unweighted_losses))
            
        #     # self.mask = tf.reshape(self.pred_fgbg_label,shape=[-1])
        #     # self.pred_cls = tf.reshape(self.logits[:,:, -self.num_classes:], shape=[-1, self.num_classes])
        #     # self.valid_gt_label = tf.boolean_mask(tf.reshape(self.inputs['class_label'],shape=[-1]), self.mask)
        #     # self.one_hot_cls = tf.one_hot(self.valid_gt_label, depth=self.num_classes)
        #     # self.masked_pred_cls = tf.boolean_mask(self.pred_cls, self.mask)
        #     # self.num_active_points = tf.reduce_sum(tf.cast(self.mask, dtype=tf.int32))
        #     # self.unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.masked_pred_cls, labels=self.one_hot_cls)
        #     # self.cls_loss = tf.cond(self.num_active_points > 0, lambda: tf.reduce_mean(self.unweighted_losses), 
        #     #     lambda: tf.convert_to_tensor(0.0)* tf.reduce_mean(self.unweighted_losses))
            

            


        #     # compute the loss for the bounding boxes
        #     # self.mask = tf.reshape(tf.equal(self.inputs['fgbg'], 1), shape=(-1,1))
        #     # # self.mask = tf.reshape(tf.equal(self.inputs['fgbg'], 1), shape=(-1, tf.shape(self.inputs['fgbg'])[-1]))
        #     # self.mask = tf.squeeze(self.mask, -1)
        #     # self.num_fg = tf.math.reduce_sum(tf.cast(self.mask, dtype=tf.int32))
        #     # self.reshaped_input = tf.reshape(self.inputs['coors'], shape=(-1, tf.shape(self.inputs['coors'])[-1]))
        #     # self.reshaped_bboxes = tf.reshape(self.inputs['bboxes'], shape=(-1, tf.shape(self.inputs['bboxes'])[-1]))
            
        #     # self.masked_input = tf.boolean_mask(self.reshaped_input, self.mask)
        #     # self.masked_bboxes = tf.boolean_mask(self.reshaped_bboxes, self.mask)

        #     # self.pred_bboxes = tf.reshape(self.logits[:,:, :self.num_bboxes_attributes], [-1, self.num_bboxes_attributes])
        #     # self.masked_pred_bboxes = tf.boolean_mask(self.pred_bboxes, self.mask)
        #     # pad_zero = tf.zeros([tf.shape(self.masked_input)[0], 1])
        #     # pad_anchor_size = tf.convert_to_tensor(self.config.anchor_size)
        #     # pad_anchor_size = tf.tile(pad_anchor_size, [tf.shape(self.masked_input)[0] ,1])
        #     # self.masked_input = tf.concat([self.masked_input,pad_anchor_size,  pad_zero],-1)
        #     # self.actual_pred_bboxes = self.masked_input + self.masked_pred_bboxes
        #     # self.bbox_loss = 0
        #     # for i in range(7):
        #     #     self.bbox_loss += tf.losses.huber_loss(
        #     #         self.masked_bboxes[:,i],
        #     #         self.actual_pred_bboxes[:,i],
        #     #         weights=1.0,
        #     #         delta=1.0,
        #     #     )

            

        #     self.loss = self.mask_loss # + self.bbox_loss #+ self.cls_loss
        #     # self.loss = self.mask_loss +  self.cls_loss
        #     # self.loss = self.cls_loss

        #####################################################################
        # two different heads: 1. fgbg classifation 2. class classification #
        #####################################################################

        with tf.variable_scope('layers'):
            # self.bboxes_pred , self.fgbg_logits , self.cls_logits = self.inference(self.inputs, self.is_training)
            self.pred_fgbg, self.cls_logits = self.inference(self.inputs, self.is_training)


        with tf.variable_scope('loss'):
            # self.logits = tf.reshape(self.logits, [config.batch_size, -1, self.num_output_attributes])
            # self.logits = tf.reshape(self.logits, [config.batch_size, -1, self.num_classes])

            self.num_fg_points = tf.math.reduce_sum(self.inputs['fgbg']) # (B, N) => ()
            self.fgbg_ratio = tf.divide(tf.cast(self.num_fg_points, dtype=tf.float32), 
                    tf.cast(
                        tf.shape(self.inputs['fgbg'])[0]* tf.shape(self.inputs['fgbg'])[1]
                        , dtype=tf.float32))



            # compute the loss of the fgbg
            # self.pred_fgbg = self.logits[:,:,
            #     self.num_bboxes_attributes:self.num_bboxes_attributes+ self.num_fgbg_attributes]
            
            # self.logits = tf.reshape(self.logits, [config.batch_size, -1, self.num_fgbg_attributes + self.num_classes])

            
            # focal loss takes in [batch_size, num_anchors, num_classes]

            self.mask_loss =  helper_tf_util.focal_loss(self.pred_fgbg, 
                                        tf.one_hot(self.inputs['fgbg'], depth=self.num_fgbg_attributes), 
                                        weights=None, 
                                        alpha=self.config.alpha, 
                                        gamma=self.config.gamma)

            self.interested_pc = self.inputs['xyz'][0]
            self.gt_output_interested_point_cloud = self.inputs['class_label']       
            self.pred_fgbg_label = tf.equal(tf.cast(tf.argmax(tf.nn.sigmoid(self.pred_fgbg),axis=-1)
                                    ,dtype=tf.int32), tf.cast(self.inputs['fgbg'],dtype=tf.int32))          
 
            # compute the loss for multiclassification
            # self.mask = tf.reshape(tf.equal(self.inputs['fgbg'], 1), shape=[-1])
            # self.pred_cls = tf.reshape(self.cls_logits, shape=[-1, self.num_classes])
            # self.reshaped_target_cls = tf.reshape(self.inputs['class_label'], shape=[-1])
            # self.masked_pred_cls = tf.boolean_mask(self.pred_cls, self.mask)
            # self.masked_target_cls = tf.boolean_mask(self.reshaped_target_cls, self.mask)
            # self.masked_one_hot_cls = tf.one_hot(self.masked_target_cls, depth=self.num_classes)
            # self.unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.masked_pred_cls, labels=self.masked_one_hot_cls)
            # self.cls_loss = tf.cond(self.num_fg_points > 0, lambda: tf.reduce_mean(self.unweighted_losses), 
            #     lambda: tf.convert_to_tensor(0.0)* tf.reduce_mean(self.unweighted_losses))

            # compute loss for binary classification
            self.mask = tf.reshape(tf.equal(self.inputs['fgbg'], 1), shape=[-1])
            self.pred_cls = tf.reshape(self.cls_logits, shape=[-1])
            self.reshaped_target_cls = tf.reshape(self.inputs['class_label'], shape=[-1])
            self.masked_pred_cls = tf.boolean_mask(self.pred_cls, self.mask)
            self.masked_target_cls = tf.cast(tf.boolean_mask(self.reshaped_target_cls, self.mask), dtype=tf.float32)
            # self.masked_one_hot_cls = tf.one_hot(self.masked_target_cls, depth=self.num_classes)
            # self.unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.masked_pred_cls, labels=self.masked_one_hot_cls)
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.masked_target_cls, logits=self.masked_pred_cls)
            self.cls_loss = tf.reduce_mean(self.cross_entropy)
            # self.cls_loss = tf.cond(self.num_fg_points > 0, lambda: tf.reduce_mean(self.unweighted_losses), 
            #     lambda: tf.convert_to_tensor(0.0)* tf.reduce_mean(self.unweighted_losses))
            # self.cls_loss = tf.losses.mean_squared_error(tf.cast(self.reshaped_target_cls, dtype=tf.float32), self.pred_cls)


            # compute percentage of points which are not class 0
            self.num_non_zero_cls = tf.math.reduce_sum(tf.cast(tf.greater(self.masked_target_cls, 0), dtype=tf.float32))
            # self.num_non_zero_cls = tf.math.reduce_sum(tf.cast(tf.greater(self.reshaped_target_cls, 0), dtype=tf.float32))
            self.non_zero_cls_ratio = tf.divide(self.num_non_zero_cls, 
                    tf.cast(
                        tf.shape(self.inputs['class_label'])[0]* tf.shape(self.inputs['class_label'])[1]
                        , dtype=tf.float32))
            # self.confusion = tf.confusion_matrix(labels=self.masked_target_cls, 
            #     predictions=self.masked_pred_cls, num_classes=self.num_classes)

            # self.mask = tf.reshape(self.pred_fgbg_label,shape=[-1])
            # self.pred_cls = tf.reshape(self.logits[:,:, -self.num_classes:], shape=[-1, self.num_classes])
            # self.valid_gt_label = tf.boolean_mask(tf.reshape(self.inputs['class_label'],shape=[-1]), self.mask)
            # self.one_hot_cls = tf.one_hot(self.valid_gt_label, depth=self.num_classes)
            # self.masked_pred_cls = tf.boolean_mask(self.pred_cls, self.mask)
            # self.num_active_points = tf.reduce_sum(tf.cast(self.mask, dtype=tf.int32))
            # self.unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.masked_pred_cls, labels=self.one_hot_cls)
            # self.cls_loss = tf.cond(self.num_active_points > 0, lambda: tf.reduce_mean(self.unweighted_losses), 
            #     lambda: tf.convert_to_tensor(0.0)* tf.reduce_mean(self.unweighted_losses))
            

            

            # self.loss = self.mask_loss # + self.bbox_loss #+ self.cls_loss
            self.loss = self.mask_loss +  self.cls_loss
            # self.loss = self.cls_loss

        def _compute_gradients(tensor, var_list, opt):
            # gvs = opt_func.compute_gradients(self.loss, tvars)
            grads = opt_func.compute_gradients(tensor, var_list)
            # return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]
            return [grad for grad in grads if grad[0] is not None]

        with tf.variable_scope('optimizer'):
            # self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            # self.decayed_lr = tf.train.exponential_decay(self.learning_rate,
            #                             self.training_step, 10000,
            #                             0.98, staircase=True)
            opt_func = tf.train.AdamOptimizer(self.learning_rate) # .minimize(self.loss)
            tvars = tf.trainable_variables()

            accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvars]
            self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

            # grads = tf.gradients(self.loss, tvars)
            # self.grads = [tf.copy(grad) for grad in grads]
            # grads, _ = tf.clip_by_global_norm(grads, 1)
            
            # gvs = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 1)
            self.gvs = opt_func.compute_gradients(self.loss, tvars)
            # self.gvs = tf.clip_by_global_norm(_compute_gradients(self.loss, tvars, opt_func),1)
            # gvs = self.gvs
            ## Adds to each element from the list you initialized earlier with zeros its gradient (works because accum_vars and gvs are in the same order)

            self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.gvs) if gv[0] is not None]


            ## Define the training step (part with variable value update)
            self.train_op = opt_func.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(self.gvs)])
            # grads, _ = self.gvs

            # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 1)
            # self.train_op = opt_func.apply_gradients(zip(grads, tvars))
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # self.global_step += 1

        with tf.variable_scope('results'):
            # self.class_label = tf.cast(self.inputs['class_label'],dtype=tf.int32)
            # self.class_label = tf.reshape(self.class_label, shape=(-1,1))
            # self.class_label = tf.squeeze(self.class_label,-1)
            # self.accuracy = tf.convert_to_tensor(-1.0)
            # self.correct_prediction = tf.nn.in_top_k(self.masked_pred_cls, self.masked_target_cls, 1)
            # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            # self.pred_cls_label = tf.cast(tf.argmax(tf.nn.softmax(self.masked_pred_cls),-1), dtype=tf.int32)
            # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred_cls_label, self.valid_gt_label),dtype=tf.float32))

            # self.classification_accuracy = tf.cond(self.num_fg_points > 0, 
            # lambda:  tf.reduce_mean(tf.cast(
            #     tf.nn.in_top_k(self.masked_pred_cls, self.masked_target_cls, 1)
            #     , tf.float32)), lambda: tf.convert_to_tensor(-1.0))

            # # self.accuracy = tf.cond(self.num_active_points > 0, 
            # # lambda:  tf.reduce_mean(tf.cast(
            # #     tf.nn.in_top_k(self.masked_pred_cls, self.valid_gt_label, 1)
            # #     , tf.float32)), lambda: tf.convert_to_tensor(-1.0))

            # self.pred_fgbg = tf.reshape(self.pred_fgbg , shape=(-1, self.num_fgbg_attributes))
            # self.gt_fgbg = tf.cast(self.inputs['fgbg'],dtype=tf.int32)
            # self.gt_fgbg = tf.reshape(self.gt_fgbg, [-1])
            # # self.gt_fgbg = tf.squeeze(self.gt_fgbg,-1)
            # self.fgbg_correct_prediction = tf.nn.in_top_k(self.pred_fgbg, self.gt_fgbg, 1)
            # self.fgbg_accuracy = tf.reduce_mean(tf.cast(self.fgbg_correct_prediction, tf.float32))
            # # self.prob_logits = tf.nn.softmax(self.pred_cls)
            # self.prob_fgbg = tf.nn.sigmoid(self.pred_fgbg)
            # self.prob_cls = tf.nn.softmax(self.masked_pred_cls)
            
            # self.classification_accuracy = tf.cond(self.num_fg_points > 0, 
            # lambda:  tf.reduce_mean(tf.cast(
            #     tf.nn.in_top_k(self.masked_pred_cls, self.masked_target_cls, 1)
            #     , tf.float32)), lambda: tf.convert_to_tensor(-1.0))
# 
            # self.classification_accuracy = tf.reduce_sum(tf.cast(tf.cast(self.pred_cls > 0.5, dtype=tf.int32) == self.masked_target_cls , dtype=tf.float32) ) \
            #                                 / tf.cast(tf.shape(self.pred_cls)[0], dtype=tf.float32)
            # self.classification_accuracy = tf.reduce_sum(tf.cast(tf.cast(self.masked_pred_cls > 0.5, dtype=tf.int32) == self.masked_target_cls , dtype=tf.float32) ) \
            #                                 / tf.cast(tf.shape(self.masked_pred_cls)[0], dtype=tf.float32)
            
            predicted = tf.nn.sigmoid(self.masked_pred_cls)
            correct_pred = tf.equal(tf.round(predicted), self.masked_target_cls)
            self.classification_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            # self.accuracy = tf.cond(self.num_active_points > 0, 
            # lambda:  tf.reduce_mean(tf.cast(
            #     tf.nn.in_top_k(self.masked_pred_cls, self.valid_gt_label, 1)
            #     , tf.float32)), lambda: tf.convert_to_tensor(-1.0))

            self.pred_fgbg = tf.reshape(self.pred_fgbg , shape=(-1, self.num_fgbg_attributes))
            self.gt_fgbg = tf.cast(self.inputs['fgbg'],dtype=tf.int32)
            self.gt_fgbg = tf.reshape(self.gt_fgbg, [-1])
            # self.gt_fgbg = tf.squeeze(self.gt_fgbg,-1)
            self.fgbg_correct_prediction = tf.nn.in_top_k(tf.nn.sigmoid(self.pred_fgbg), self.gt_fgbg, 1)
            self.fgbg_accuracy = tf.reduce_mean(tf.cast(self.fgbg_correct_prediction, tf.float32))
            # self.prob_logits = tf.nn.softmax(self.pred_cls)
            self.prob_fgbg = tf.nn.sigmoid(self.pred_fgbg)
            # self.prob_cls = tf.nn.softmax(self.masked_pred_cls)
            self.prob_cls = predicted
            self.pred_cls_label = tf.round(predicted)
            

            # material_config = {
            # 'cls': 'PointsMaterial',
            # 'size': 1,
            # }

            # tf.summary.scalar('learning_rate', self.learning_rate)
            # tf.summary.scalar('fgbg_loss', self.mask_loss)
            # tf.summary.scalar('classification_loss', self.cls_loss)
            # tf.summary.scalar('loss', self.loss)
            # tf.summary.scalar('classification_accuracy', self.classification_accuracy)
            # tf.summary.scalar('fgbg_accuracy', self.fgbg_accuracy)
            # tf.summary.scalar('num_fg_points', self.num_fg_points)
            # tf.summary.scalar('fgbg_ratio', self.fgbg_ratio)
            # tf.summary.scalar('num_non_zero_cls', self.num_non_zero_cls)
            # tf.summary.scalar('non_zero_cls_ratio', self.non_zero_cls_ratio)
            

            train_summary = []
            val_summary = []

            # create a mesh point
            # self.point_cloud = \
            #     tf.concat([tf.cast(self.fgbg_correct_prediction, tf.float32),  tf.ones(self.num_points_to_pad) * 2.0],
            #                 axis = 0)

            # self.color = tf.expand_dims(
            #     tf.tile(
            #                                 tf.expand_dims(
            #                                     tf.cast(self.point_cloud, tf.float32), -1), [1,3]),0) * 30 + 100
            train_summary.append(tf.summary.scalar('train_learning_rate', self.learning_rate))
            train_summary.append(tf.summary.scalar('train_fgbg_loss', self.mask_loss))
            train_summary.append(tf.summary.scalar('train_classification_loss', self.cls_loss))
            train_summary.append(tf.summary.scalar('train_loss', self.loss))
            train_summary.append(tf.summary.scalar('train_classification_accuracy', self.classification_accuracy))
            train_summary.append(tf.summary.scalar('train_fgbg_accuracy', self.fgbg_accuracy))
            train_summary.append(tf.summary.scalar('train_num_fg_points', self.num_fg_points))
            train_summary.append(tf.summary.scalar('train_fgbg_ratio', self.fgbg_ratio))
            train_summary.append(tf.summary.scalar('train_num_non_zero_cls', self.num_non_zero_cls))
            train_summary.append(tf.summary.scalar('train_non_zero_cls_ratio', self.non_zero_cls_ratio))
            # train_summary.append(mesh_summary.op('train_point_cloud', vertices=self.inputs['xyz'][0],
            #                         colors = tf.cast(self.color, tf.int32)))
                                    #             ,
                                    # config_dict={"material": material_config}))

            val_summary.append(tf.summary.scalar('val_learning_rate', self.learning_rate))
            val_summary.append(tf.summary.scalar('val_fgbg_loss', self.mask_loss))
            val_summary.append(tf.summary.scalar('val_classification_loss', self.cls_loss))
            val_summary.append(tf.summary.scalar('val_loss', self.loss))
            val_summary.append(tf.summary.scalar('val_classification_accuracy', self.classification_accuracy))
            val_summary.append(tf.summary.scalar('val_fgbg_accuracy', self.fgbg_accuracy))
            val_summary.append(tf.summary.scalar('val_num_fg_points', self.num_fg_points))
            val_summary.append(tf.summary.scalar('val_fgbg_ratio', self.fgbg_ratio))
            val_summary.append(tf.summary.scalar('val_num_non_zero_cls', self.num_non_zero_cls))
            val_summary.append(tf.summary.scalar('val_non_zero_cls_ratio', self.non_zero_cls_ratio))
            
            # self.mean_val_loss_ph = tf.placeholder(dtype=tf.float32, shape = (),
            #     name='mean_validation_loss')

            # self.mean_val_fgbg_accuracy_ph = tf.placeholder(dtype=tf.float32, shape = (),
            #     name='mean_validation_fgbg_accuracy')
            # self.mean_val_classification_accuracy_ph = tf.placeholder(dtype=tf.float32, shape = (),
            #     name='mean_val_classification_accuracy')
            # self.mean_val_summary = tf.summary.merge([tf.summary.scalar("mean_validation_loss", self.mean_val_loss_ph),
            #     tf.summary.scalar("mean_validation_fgbg_accuracy", self.mean_val_fgbg_accuracy_ph),
            #     tf.summary.scalar("mean_val_classification_accuracy", self.mean_val_classification_accuracy_ph)
            # ])

            self.mean_train_fgbg_loss_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_train_fgbg_loss')

            self.mean_train_cls_loss_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_train_cls_loss')

            self.mean_train_loss_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_train_loss')

            self.mean_train_fgbg_accuracy_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_train_fgbg_accuracy')
            self.mean_train_cls_accuracy_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_train_cls_accuracy')
            self.mean_train_fgbg_iou_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_train_fgbg_iou')
            self.mean_train_cls_iou_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_train_cls_iou')
            self.mean_train_fgbg_precision_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_train_fgbg_precision')
            self.mean_train_cls_precision_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_train_cls_precision')
            self.mean_train_fgbg_recall_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_train_fgbg_recall')
            self.mean_train_cls_recall_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_train_cls_recall')

            self.mean_train_summary = tf.summary.merge([tf.summary.scalar("mean_train_fgbg_loss", self.mean_train_fgbg_loss_ph),
                tf.summary.scalar("mean_train_cls_loss", self.mean_train_cls_loss_ph),
                tf.summary.scalar("mean_train_loss", self.mean_train_loss_ph),
                tf.summary.scalar("mean_train_fgbg_accuracy", self.mean_train_fgbg_accuracy_ph),
                tf.summary.scalar("mean_train_cls_accuracy", self.mean_train_cls_accuracy_ph),
                tf.summary.scalar("mean_train_fgbg_iou", self.mean_train_fgbg_iou_ph),
                tf.summary.scalar("mean_train_cls_iou", self.mean_train_cls_iou_ph),
                tf.summary.scalar("mean_train_fgbg_precision", self.mean_train_fgbg_precision_ph),
                tf.summary.scalar("mean_train_cls_precision", self.mean_train_cls_precision_ph),
                tf.summary.scalar("mean_train_fgbg_recall", self.mean_train_fgbg_recall_ph),
                tf.summary.scalar("mean_train_cls_recall", self.mean_train_cls_recall_ph)
            ])

            self.train_cm_image_ph = tf.placeholder(dtype=tf.uint8, shape = (1, None,None,4),
                name='mean_val_cls_recall')
            self.train_cm_image = tf.summary.image("Train Confusion Matrix", self.train_cm_image_ph, family="confusion_matrix_" + str(self.config.alpha))
                                


            self.mean_val_fgbg_loss_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_val_fgbg_loss')

            self.mean_val_cls_loss_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_val_cls_loss')

            self.mean_val_loss_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_val_loss')

            self.mean_val_fgbg_accuracy_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_val_fgbg_accuracy')
            self.mean_val_cls_accuracy_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_val_cls_accuracy')
            self.mean_val_fgbg_iou_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_val_fgbg_iou')
            self.mean_val_cls_iou_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_val_cls_iou')
            self.mean_val_fgbg_precision_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_val_fgbg_precision')
            self.mean_val_cls_precision_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_val_cls_precision')
            self.mean_val_fgbg_recall_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_val_fgbg_recall')
            self.mean_val_cls_recall_ph = tf.placeholder(dtype=tf.float32, shape = (),
                name='mean_val_cls_recall')

            self.mean_val_summary = tf.summary.merge([tf.summary.scalar("mean_val_fgbg_loss", self.mean_val_fgbg_loss_ph),
                tf.summary.scalar("mean_val_cls_loss", self.mean_val_cls_loss_ph),
                tf.summary.scalar("mean_val_loss", self.mean_val_loss_ph),
                tf.summary.scalar("mean_val_fgbg_accuracy", self.mean_val_fgbg_accuracy_ph),
                tf.summary.scalar("mean_val_cls_accuracy", self.mean_val_cls_accuracy_ph),
                tf.summary.scalar("mean_val_fgbg_iou", self.mean_val_fgbg_iou_ph),
                tf.summary.scalar("mean_val_cls_iou", self.mean_val_cls_iou_ph),
                tf.summary.scalar("mean_val_fgbg_precision", self.mean_val_fgbg_precision_ph),
                tf.summary.scalar("mean_val_cls_precision", self.mean_val_cls_precision_ph),
                tf.summary.scalar("mean_val_fgbg_recall", self.mean_val_fgbg_recall_ph),
                tf.summary.scalar("mean_val_cls_recall", self.mean_val_cls_recall_ph)
            ])

            self.val_cm_image_ph = tf.placeholder(dtype=tf.uint8, shape = (1, None,None,4),
                name='mean_val_cls_recall')
            self.val_cm_image = tf.summary.image("Val Confusion Matrix", self.val_cm_image_ph, family="confusion_matrix_" + str(self.config.alpha))
                    



        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        # self.train_summary = tf.summary.merge_all()
        self.train_summary = tf.summary.merge(train_summary)
        self.val_summary = tf.summary.merge(val_summary)
        logdir = ( datetime.now().strftime("%Y%m%d_%H%M%S") \
        + '_lr_' + str(self.config.learning_rate) \
        + '_max_epoch_' + str(self.config.max_epoch) \
        + '_num_points_' + str(self.config.num_points) \
        + '_num_classes_' + str(self.config.num_classes) \
        + '_alpha_' + str(self.config.alpha) \
        + '_gamma_' + str(self.config.gamma))
        self.logdir = os.path.join(self.config.train_sum_dir, logdir)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        import shutil
        # make a copy of the current config file to the train_log
        shutil.copy('helper_tool.py', os.path.join(self.logdir, 'helpter_tool.py'))
        shutil.copy('DetectRandLANet.py', os.path.join(self.logdir, 'DetectRandLANet.py'))

        self.config.visual_log_path = os.path.join(self.config.visual_log_path, logdir)
        if not os.path.exists(self.config.visual_log_path):
            os.makedirs(self.config.visual_log_path)
        self.train_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        
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
        self.feature = f_encoder_list[-1]
        # feature = f_encoder_list[-1]
        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)
                # print("cls_cm_cumulative: ", cls_cm_cboxes_layer_drop, self.num_target_attributes-1, [1, 1], 'bboxes_fc', [1, 1], 'VALID', False,
        #                                     is_training, activation_fn=None)
        # bboxes_out = tf.squeeze(bboxes_layer_fc3, [2])


        # fgbg head
        fgbg_layer_fc1 = helper_tf_util.conv2d(feature, 64, [1, 1], 'fgbg_fc1', [1, 1], 'VALID', True, is_training)
        fgbg_layer_fc2 = helper_tf_util.conv2d(fgbg_layer_fc1, 32, [1, 1], 'fgbg_fc2', [1, 1], 'VALID', True, is_training)
        fgbg_layer_drop = helper_tf_util.dropout(fgbg_layer_fc2, keep_prob=0.5, is_training=is_training, scope='fgbg_dp1')
        fgbg_layer_fc3 = helper_tf_util.conv2d(fgbg_layer_drop, self.num_fgbg_attributes, [1, 1], 'fgbg_fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        fgbg_out = tf.squeeze(fgbg_layer_fc3, [2])


        # classification head

        cls_layer_fc1 = helper_tf_util.conv2d(feature, 64, [1, 1], 'cls_fc1', [1, 1], 'VALID', True, is_training)
        cls_layer_fc2 = helper_tf_util.conv2d(cls_layer_fc1, 32, [1, 1], 'cls_fc2', [1, 1], 'VALID', True, is_training)
        cls_layer_drop = helper_tf_util.dropout(cls_layer_fc2, keep_prob=0.5, is_training=is_training, scope='cls_dp1')
        cls_layer_fc3 = helper_tf_util.conv2d(cls_layer_drop, self.num_classes, [1, 1], 'cls_fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        cls_out = tf.squeeze(cls_layer_fc3, [2])


        # num_bboxes_attributes head
        # centers and dimensions

        # bbox_layer_fc1 = helper_tf_util.conv2d(feature, 64, [1, 1], 'bbox_fc1', [1, 1], 'VALID', True, is_training)
        # bbox_layer_fc2 = helper_tf_util.conv2d(bbox_layer_fc1, 32, [1, 1], 'bbox_fc2', [1, 1], 'VALID', True, is_training)
        # # cls_layer_drop = helper_tf_util.dropout(bbox_layer_fc2, keep_prob=0.5, is_training=is_training, scope='cls_dp1')
        # bbox_layer_fc3 = helper_tf_util.conv2d(bbox_layer_fc2, self.num_bboxes_attributes-1, [1, 1], 'bbox_fc', [1, 1], 'VALID', False,
        #                                     is_training, activation_fn=None)
        # bbox_out = tf.squeeze(bbox_layer_fc3, [2])

        return fgbg_out, cls_out #, bbox_out

        # f_layer_fc1 = helper_tf_util.conv2d(f_encoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        # f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        # f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        # # f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.num_output_attributes, [1, 1], 'fc', [1, 1], 'VALID', False,
        # #                                     is_training, activation_fn=None)
        # f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.num_fgbg_attributes + self.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
        #                             is_training, activation_fn=None)
        # f_out = tf.squeeze(f_layer_fc3, [2])
        # return f_out

    def unit_test(self, dataset, cfg):
        # ops = [self.num_active_points, self.cls_loss]
        # # ops = [self.logits, self.prob_logits, self.gt_fgbg, self.interested_pc, self.pred_fgbg_label, self.fgbg_accuracy]
        # ops = [self.mask, self.pred_fgbg_label, self.effective_mask, self.cls_loss, self.one_hot_cls, self.valid_gt_label]
        # # , self.mask, self.masked_input, self.masked_bboxes, self.bbox_loss, self.unweighted_losses,
        # #         self.gt_fgbg, self.pred_fgbg]  
        #         # self.pad_anchor_size]
        # #  #, self.num_fg, self.mask_loss, self.reshaped_input, 
        # # # self.reshaped_bboxes, self.masked_input, self.masked_bboxes, self.pred_bboxes, self.masked_pred_bboxes, 
        # # # self.bbox_loss, self.actual_pred_bboxes, self.masked_pred_cls, self.unweighted_losses, self.loss,
        # # # self.pred_cls, self.reshaped_cls, self.accuracy]
        
        # ops = [self.point_cloud, self.interested_pc, self.color]
        # self.sess.run(dataset.train_init_op)
        # logits = self.sess.run(ops, {self.is_training: True})
        # print("single inference")
        # print("fgbg shape: ", logits[0].shape)
        # # print("fgbg shape: ", logits[0])
        # print("interested_pc shape: ", logits[1].shape)
        # print("color shape: ", logits[2].shape)
        # print("color shape: ", logits[2])
        

        self.sess.run(dataset.train_init_op)
        previous_epoch = self.training_epoch
        fgbg_loss_array = []
        cls_loss_array = []
        overall_loss_array = []
        fgbg_acc_array = []
        cls_acc_array = []
        fgbg_iou_array = []
        cls_iou_array = []
        fgbg_cm_cumulative = np.array([[0,0],[0,0]])
        cls_cm_cumulative = np.array([[0,0],[0,0]])
        self.sess.run(self.zero_ops)
        while self.training_epoch < self.config.max_epoch:
            if self.training_epoch > previous_epoch:
                previous_epoch = self.training_epoch
                fgbg_loss_array = []
                cls_loss_array = []
                overall_loss_array = []
                fgbg_acc_array = []
                cls_acc_array = []
                fgbg_iou_array = []
                cls_iou_array = []
                fgbg_cm_cumulative = np.array([[0,0],[0,0]])
                cls_cm_cumulative = np.array([[0,0],[0,0]])

            t_start = time.time()
            try:
                ops = [self.accum_ops,
                        self.extra_update_ops,
                        self.train_summary,
                        self.mask_loss,
                        self.cls_loss,
                        self.loss,
                        # self.logits,
                        # self.pred_fgbg,
                        self.prob_fgbg,
                        self.prob_cls,
                        self.interested_pc,
                        self.gt_output_interested_point_cloud,
                        self.fgbg_ratio,
                        self.non_zero_cls_ratio,
                        # self.gt_fgbg,
                        # self.pred_fgbg_label,
                        self.fgbg_accuracy,
                        self.classification_accuracy,
                        # self.masked_pred_cls,
                        # self.masked_target_cls,
                        self.pred_cls,
                        self.reshaped_target_cls,
                        self.pred_cls_label,
                        self.masked_target_cls,
                        self.gt_fgbg,
                        self.feature,
                        # self.grads
                        # self.gvs
                        ]
                _, _, train_summary, fgbg_loss, cls_loss, l_out, prob_fgbg, prob_cls  ,interested_pc, \
                    gt_output_interested_point_cloud, fgbg_ratio, non_zero_cls_ratio,\
                        fgbg_accuracy, cls_acc, pred_cls, target_cls , pred_cls_label, masked_target_cls, gt_fgbg, \
                        feature = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(train_summary, self.training_step)
                # print("Mask.sum: ", np.sum(mask))
                # # Calculate the confusion matrix.
                # cm = sklearn.metrics.confusion_matrix(target_cls, np.argmax(pred_cls,axis=-1))
                # # Log the confusion matrix as an image summary.
                # figure = plot_confusion_matrix(cm, class_names=['Car', 'Pedestrian', 'Cyclist', 'Van'])
                # cm_image = plot_to_image(figure)
                # tf_cm_summary = tf.summary.image("Confusion Matrix", cm_image, family="confusion_matrix")
                # tf_cm_image = self.sess.run(tf_cm_summary)
                # self.train_writer.add_summary(tf_cm_image, self.training_step)
                
                t_end = time.time()

                # for grad in grads:
                #     print("grad[0].shape: ", grad[0].shape, "\t grad[1].shape: ", grad[1].shape)                    

                # for i, gv in enumerate(gvs):
                #     print("\t grad[0].shape: ", grads[i][0].shape, "\t grad[1].shape: ", grads[i][1].shape, "\t gv[0].shape: ", gv[0].shape, "\t gv[1].shape: ", gv[1].shape)
                    

                # print("feature.shape: ", feature.shape)

                pred_fgbg_label = np.argmax(prob_fgbg, 1)
                # print("pred_fgbg_label: ", set(pred_fgbg_label))
                # print("pred_fgbg_label.shape: ", pred_fgbg_label.shape)
                fgbg_cm = [] # [[TN, FP], [FN, TP]]
                for label_idx in range(2):
                    row = []
                    for pred_idx in range(2):
                        mask1 = (gt_fgbg == label_idx)
                        mask2 = (pred_fgbg_label == pred_idx)
                        # print(mask1)
                        # print(mask2)
                        correct = mask1 & mask2
                        row.append(np.sum(correct))
                    fgbg_cm.append(row)
                # print(fgbg_cm)
                fgbg_cm_cumulative =  fgbg_cm_cumulative + np.array(fgbg_cm)
                # print(fgbg_cm_cumulative)
                # fgbg_iou = fgbg_cm[1][1]/ (np.sum(fgbg_cm) - fgbg_cm[0][0])
                # fgbg_precision = fgbg_cm[1][1] / (fgbg_cm[0][1] + fgbg_cm[1][1])
                # fgbg_recall = fgbg_cm[1][1] / (fgbg_cm[1][0] + fgbg_cm[1][1])
                # print("fgbg iou: ", fgbg_iou, "\t fgbg precision: ", fgbg_precision, "\t fgbg recall: ", fgbg_recall)
                cls_cm = [] # [[TN, FP], [FN, TP]]
                for label_idx in range(2):
                    row = []
                    for pred_idx in range(2):
                        mask1 = (masked_target_cls == label_idx)
                        mask2 = (pred_cls_label == pred_idx)
                        # print(mask1)
                        # print(mask2)
                        correct = mask1 & mask2
                        row.append(np.sum(correct))
                    cls_cm.append(row)
                cls_cm_cumulative = cls_cm_cumulative + np.array(cls_cm)
                # cls_iou = cls_cm[1][1]/ (np.sum(cls_cm) - cls_cm[0][0])
                # cls_precision = cls_cm[1][1] / (cls_cm[0][1] + cls_cm[1][1])
                # cls_recall = cls_cm[1][1] / (cls_cm[1][0] + cls_cm[1][1])
                # print("cls iou: ", cls_iou, "\t cls precision: ", cls_precision, "\t cls recall: ", cls_recall)
                
                if self.training_step % 50 == 0:
                    self.sess.run(self.train_op)
                    self.sess.run(self.zero_ops)
                    # for i in range(10):
                    #     print(prob_logits[i])
                    #     print(gt_fgbg[i])
                    message = 'Step {:08d} Pred_fgbg_sum {:08d} L_out={:5.9f} Acc={:4.5f} FgbgAcc={:4.5f} FgbgPer={:4.5f} NonZeroClsPer={:4.5f} ''---{:8.2f} ms/batch {}' 
                    log_out(message.format(self.training_step, np.sum(np.argmax(prob_fgbg, axis=1)), l_out, cls_acc, fgbg_accuracy, fgbg_ratio, non_zero_cls_ratio, 1000 * (t_end - t_start), str(set(target_cls))), self.Log_file)
                    # print(prob_fgbg.shape)
                    # if (self.training_step % 200 == 0) and (self.training_epoch % 5 == 0):
                    interested_pc_output_path = os.path.join(self.config.visual_log_path, 'interested_pc_' + str(self.training_step) + '.bin')
                    gt_output_interested_pc_output_path = os.path.join(self.config.visual_log_path, 'gt_output_interested_pc_' + str(self.training_step) + '.bin')
                    pred_fgbg_label_output_path = os.path.join(self.config.visual_log_path, 'pred_fgbg_label_' + str(self.training_step) + '.bin')
                    pred_class_label_output_path = os.path.join(self.config.visual_log_path, 'pred_class_label_' + str(self.training_step) + '.bin')
                    interested_pc.astype('float32').tofile(interested_pc_output_path)
                    gt_output_interested_point_cloud.astype('float32').tofile(gt_output_interested_pc_output_path)
                    prob_fgbg.astype('float32').tofile(pred_fgbg_label_output_path)
                    prob_cls.astype('float32').tofile(pred_class_label_output_path)
                    
                    # Calculate the confusion matrix.
                    # fgbg_cm = sklearn.metrics.confusion_matrix(gt_fgbg, pred_fgbg_label, np.arange(0,self.num_fgbg_attributes))
                    # cls_cm = sklearn.metrics.confusion_matrix(masked_target_cls, pred_cls_label, np.arange(0,self.num_fgbg_attributes))
                    # print("train_confusion_matrix")
                    # print("fgbg_cm: \n", fgbg_cm)
                    # print("cls_cm: \n", cls_cm)
                    # print("fgbg iou: ", fgbg_iou, "\t fgbg precision: ", fgbg_precision, "\t fgbg recall: ", fgbg_recall)
                    # print("cls iou: ", cls_iou, "\t cls precision: ", cls_precision, "\t cls recall: ", cls_recall)
                    # cm = sklearn.metrics.confusion_matrix(masked_target_cls, pred_cls_label)
                    # print("cm: ", cm)
                    # print("cls iou: ", cls_iou)
                    # tn, fp, fn, tp = sklearn.metrics.confusion_matrix(masked_target_cls, pred_cls_label).ravel()
                    # print(tn, fp, fn, tp)
                    # Log the confusion matrix as an image summary.
                    # figure = plot_confusion_matrix(cm, class_names=['Car', 'Pedestrian', 'Cyclist', 'Van'])
                    # figure = plot_confusion_matrix(np.array(cls_cm), class_names=["not car", "car"])
                    # image_cm = plot_to_image(figure)

                    # figure = plot_confusion_matrix(np.array(fgbg_cm), class_names=["bg", "fg"])
                    # print("figure.shape: ", figure.shape)
                    # # image_cm = plot_to_image(figure)
                    # tf_cm_image = self.sess.run(self.train_cm_image, {self.train_cm_image_ph: figure})
                    # self.train_writer.add_summary(tf_cm_image, self.training_step)


                fgbg_loss_array.append(fgbg_loss)
                cls_loss_array.append(cls_loss)
                overall_loss_array.append(l_out)
                fgbg_acc_array.append(fgbg_accuracy)
                cls_acc_array.append(cls_acc)
                # fgbg_iou_array.append(fgbg_iou)
                # cls_iou_array.append(cls_iou)

                self.training_step += 1

            except tf.errors.OutOfRangeError:
                # print("fgbg_cm_cumulative: ", fgbg_cm_cumulative)
                # print("cls_cm_cumulative: ", cls_cm_cumulative)
                self.sess.run(self.train_op)
                self.sess.run(self.zero_ops)

                fgbg_iou = fgbg_cm_cumulative[1][1]/ (np.sum(fgbg_cm_cumulative) - fgbg_cm_cumulative[0][0])
                fgbg_precision = fgbg_cm_cumulative[1][1] / (fgbg_cm_cumulative[0][1] + fgbg_cm_cumulative[1][1])
                fgbg_recall = fgbg_cm_cumulative[1][1] / (fgbg_cm_cumulative[1][0] + fgbg_cm_cumulative[1][1])
                cls_iou = cls_cm_cumulative[1][1]/ (np.sum(cls_cm_cumulative) - cls_cm_cumulative[0][0])
                cls_precision = cls_cm_cumulative[1][1] / (cls_cm_cumulative[0][1] + cls_cm_cumulative[1][1])
                cls_recall = cls_cm_cumulative[1][1] / (cls_cm_cumulative[1][0] + cls_cm_cumulative[1][1])

                figure = plot_confusion_matrix(fgbg_cm_cumulative, class_names=["bg", "fg"])
                # print("figure.shape: ", figure.shape)
                # image_cm = plot_to_image(figure)
                tf_cm_image = self.sess.run(self.train_cm_image, {self.train_cm_image_ph: figure})
                self.train_writer.add_summary(tf_cm_image, self.training_epoch)


                ops = self.mean_train_summary
                print("Epoch: ", self.training_epoch,
                    "\n mean_overall_loss: ", np.mean(overall_loss_array),
                    "\n mean_fgbg_loss: ", np.mean(fgbg_loss_array),
                    "\n mean_cls_loss_array: ", np.mean(cls_loss_array),
                    "\n fgbg_cm_cumulative: ", fgbg_cm_cumulative,
                    "\n cls_cm_cumulative: ", cls_cm_cumulative,
                    "\n fgbg iou: ", fgbg_iou, "\n fgbg precision: ", fgbg_precision, "\n fgbg recall: ", fgbg_recall,
                    "\n cls iou: ", cls_iou, "\n cls precision: ", cls_precision, "\n cls recall: ", cls_recall)

                mean_train_summary = self.sess.run(ops, {self.mean_train_fgbg_loss_ph: np.mean(fgbg_loss_array),
                                    self.mean_train_cls_loss_ph: np.mean(cls_loss_array),
                                    self.mean_train_loss_ph: np.mean(overall_loss_array),
                                    self.mean_train_fgbg_accuracy_ph: np.mean(fgbg_acc_array),
                                    self.mean_train_cls_accuracy_ph: np.mean(cls_acc_array),
                                    self.mean_train_fgbg_iou_ph: np.mean(fgbg_iou),
                                    self.mean_train_cls_iou_ph: np.mean(cls_iou),
                                    self.mean_train_fgbg_precision_ph: np.mean(fgbg_precision),
                                    self.mean_train_cls_precision_ph: np.mean(cls_precision),
                                    self.mean_train_fgbg_recall_ph: np.mean(fgbg_recall),
                                    self.mean_train_cls_recall_ph: np.mean(cls_recall)})
                self.train_writer.add_summary(mean_train_summary, self.training_epoch)

                # summary_list = tf.summary.merge([tf.summary.scalar('mean_fgbg_loss_per_epoch', np.mean(fgbg_loss_array)),
                # tf.summary.scalar('mean_cls_loss_per_epoch',  np.mean(cls_loss_array)),
                # tf.summary.scalar('mean_overall_loss_per_epoch',  np.mean(overall_loss_array)),
                # tf.summary.scalar('mean_fgbg_acc_per_epoch',  np.mean(fgbg_acc_array)),
                # tf.summary.scalar('mean_cls_acc_per_epoch',  np.mean(cls_acc_array)),
                # tf.summary.scalar('mean_fgbg_iou_per_epoch',  np.mean(fgbg_iou_array)),
                # tf.summary.scalar('mean_cls_iou_per_epoch',  np.mean(cls_iou_array))])
                # tf.summary.scalar("mean_train_fgbg_precision", self.mean_train_fgbg_precision_ph),
                # tf.summary.scalar("mean_train_cls_precision", self.mean_train_cls_precision_ph),
                # tf.summary.scalar("mean_train_fgbg_recall", self.mean_train_fgbg_recall_ph),
                # tf.summary.scalar("mean_train_cls_recall", self.mean_train_cls_recall_ph)
                # tf_summary_list = self.sess.run(summary_list)
                # self.train_writer.add_summary(tf_summary_list, self.training_epoch)
                
                m_iou, _ = self.evaluate(dataset)
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

                m_iou = np.sum(self.evaluate(dataset))
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

        fgbg_gt_classes = [0 for _ in range(self.num_fgbg_attributes)]
        fgbg_positive_classes = [0 for _ in range(self.num_fgbg_attributes)]
        fgbg_true_positive_classes = [0 for _ in range(self.num_fgbg_attributes)]
        fgbg_val_total_correct = 0
        fgbg_val_total_seen = 0

        cls_gt_classes = [0 for _ in range(self.config.num_classes)]
        cls_positive_classes = [0 for _ in range(self.config.num_classes)]
        cls_true_positive_classes = [0 for _ in range(self.config.num_classes)]
        cls_val_total_correct = 0
        cls_val_total_seen = 0

        val_loss = []
        val_fgbg_accuracy = []
        val_classification_accuracy = []

        fgbg_loss_array = []
        cls_loss_array = []
        overall_loss_array = []
        fgbg_acc_array = []
        cls_acc_array = []
        fgbg_iou_array = []
        cls_iou_array = []
        fgbg_cm_cumulative = np.zeros((2,2))
        cls_cm_cumulative = np.zeros((2,2))

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = [self.val_summary,
                        self.mask_loss,
                        self.cls_loss,
                        self.loss,
                        self.prob_fgbg,
                        self.gt_fgbg,
                        self.prob_cls,
                        self.interested_pc,
                        self.gt_output_interested_point_cloud,
                        self.fgbg_ratio,
                        self.non_zero_cls_ratio,
                        self.fgbg_accuracy,
                        self.classification_accuracy,
                        self.masked_pred_cls,
                        self.reshaped_target_cls,
                        self.pred_cls_label,
                        self.masked_target_cls,
                        self.gt_fgbg
                        ]
                val_summary, fgbg_loss, cls_loss, l_out, prob_fgbg, gt_fgbg, prob_cls, \
                interested_pc, gt_output_interested_point_cloud, fgbg_ratio, non_zero_cls_ratio,\
                    fgbg_accuracy, cls_acc,\
                        pred_cls, reshaped_target_cls, pred_cls_label, masked_target_cls, gt_fgbg = self.sess.run(ops, {self.is_training: False})
                self.train_writer.add_summary(val_summary, self.validation_step)
                self.validation_step += 1
                # val_loss.append(loss)
                # val_fgbg_accuracy.append(fgbg_accuracy)
                # val_classification_accuracy.append(classification_accuracy)


                pred_fgbg_label = np.argmax(prob_fgbg, 1)
                fgbg_correct = np.sum(pred_fgbg_label == gt_fgbg)
                fgbg_val_total_correct += fgbg_correct
                fgbg_val_total_seen += len(prob_fgbg)

                fgbg_cm = [] # [[TN, FP], [FN, TP]]
                for label_idx in range(2):
                    row = []
                    for pred_idx in range(2):
                        mask1 = (gt_fgbg == label_idx)
                        mask2 = (pred_fgbg_label == pred_idx)
                        # print(mask1)
                        # print(mask2)
                        correct = mask1 & mask2
                        row.append(np.sum(correct))
                    fgbg_cm.append(row)
                # print(fgbg_cm)
                fgbg_cm_cumulative =  fgbg_cm_cumulative + np.array(fgbg_cm)
                fgbg_conf_matrix = np.array(fgbg_cm)
                # fgbg_conf_matrix = sklearn.metrics.confusion_matrix(gt_fgbg, pred_fgbg_label, np.arange(0, self.num_fgbg_attributes, 1))
                
                # conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                fgbg_gt_classes += np.sum(fgbg_conf_matrix, axis=1)
                fgbg_positive_classes += np.sum(fgbg_conf_matrix, axis=0)
                fgbg_true_positive_classes += np.diagonal(fgbg_conf_matrix)

                # print("np.sum(fgbg_conf_matrix, axis=1): ", np.sum(fgbg_conf_matrix, axis=1))
                # print("np.sum(fgbg_conf_matrix, axis=0): ", np.sum(fgbg_conf_matrix, axis=0))
                # print("np.diagonal(fgbg_conf_matrix): ", np.diagonal(fgbg_conf_matrix))
                
                cls_correct = np.sum(pred_cls_label == masked_target_cls)
                cls_val_total_correct += cls_correct
                cls_val_total_seen += len(masked_target_cls)
                
                cls_cm = [] # [[TN, FP], [FN, TP]]
                for label_idx in range(2):
                    row = []
                    for pred_idx in range(2):
                        mask1 = (masked_target_cls == label_idx)
                        mask2 = (pred_cls_label == pred_idx)
                        # print(mask1)
                        # print(mask2)
                        correct = mask1 & mask2
                        row.append(np.sum(correct))
                    cls_cm.append(row)
                cls_cm_cumulative = cls_cm_cumulative + np.array(cls_cm)
                cls_conf_matrix = np.array(cls_cm)
                # cls_conf_matrix = sklearn.metrics.confusion_matrix(masked_target_cls, pred_cls_label, np.arange(0, self.num_fgbg_attributes, 1))
                # conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                cls_gt_classes += np.sum(cls_conf_matrix, axis=1)
                cls_positive_classes += np.sum(cls_conf_matrix, axis=0)
                cls_true_positive_classes += np.diagonal(cls_conf_matrix)

                # print("val confusion matrix")
                # print("fgbg: ")
                # print(fgbg_conf_matrix)
                # print("cls: ")
                # print(cls_conf_matrix)
                # print("np.sum(cls_conf_matrix, axis=1): ", np.sum(cls_conf_matrix, axis=1))
                # print("np.sum(cls_conf_matrix, axis=0): ", np.sum(cls_conf_matrix, axis=0))
                # print("np.diagonal(cls_conf_matrix): ", np.diagonal(cls_conf_matrix))
                # fgbg_conf_matrix_sum += fgbg_conf_matrix
                # cls_conf_matrix_sum += cls_conf_matrix

                interested_pc_output_path = os.path.join(self.config.visual_log_path, 'val_interested_pc_' + str(self.training_step) + '.bin')
                gt_output_interested_point_cloud_output_path = os.path.join(self.config.visual_log_path, 'val_gt_output_interested_pc_' + str(self.training_step) + '.bin')
                pred_fgbg_label_output_path = os.path.join(self.config.visual_log_path, 'val_pred_fgbg_label_' + str(self.training_step) + '.bin')
                pred_class_label_output_path = os.path.join(self.config.visual_log_path, 'val_pred_class_label_' + str(self.training_step) + '.bin')
                interested_pc.astype('float32').tofile(interested_pc_output_path)
                gt_output_interested_point_cloud.astype('float32').tofile(gt_output_interested_point_cloud_output_path)
                prob_fgbg.astype('float32').tofile(pred_fgbg_label_output_path)
                prob_cls.astype('float32').tofile(pred_class_label_output_path)


                fgbg_loss_array.append(fgbg_loss)
                cls_loss_array.append(cls_loss)
                overall_loss_array.append(l_out)
                fgbg_acc_array.append(fgbg_accuracy)
                cls_acc_array.append(cls_acc)
            

            except tf.errors.OutOfRangeError:
                break
        
        # print("fgbg_cm_cumulative: ", fgbg_cm_cumulative)
        # print("cls_cm_cumulative: ", cls_cm_cumulative)

        figure = plot_confusion_matrix(np.array(fgbg_cm_cumulative), class_names=["bg", "fg"])
        # image_cm = plot_to_image(figure)
        tf_cm_image = self.sess.run(self.val_cm_image, {self.val_cm_image_ph: figure})
        self.train_writer.add_summary(tf_cm_image, self.training_epoch)

        fgbg_iou = fgbg_cm_cumulative[1][1]/ (np.sum(fgbg_cm_cumulative) - fgbg_cm_cumulative[0][0])
        fgbg_precision = fgbg_cm_cumulative[1][1] / (fgbg_cm_cumulative[0][1] + fgbg_cm_cumulative[1][1])
        fgbg_recall = fgbg_cm_cumulative[1][1] / (fgbg_cm_cumulative[1][0] + fgbg_cm_cumulative[1][1])
        cls_iou = cls_cm_cumulative[1][1]/ (np.sum(cls_cm_cumulative) - cls_cm_cumulative[0][0])
        cls_precision = cls_cm_cumulative[1][1] / (cls_cm_cumulative[0][1] + cls_cm_cumulative[1][1])
        cls_recall = cls_cm_cumulative[1][1] / (cls_cm_cumulative[1][0] + cls_cm_cumulative[1][1])
        ops = self.mean_train_summary
        print("Validation Epoch: ", self.training_epoch,
            "\n mean_overall_loss: ", np.mean(overall_loss_array),
            "\n mean_fgbg_loss: ", np.mean(fgbg_loss_array),
            "\n mean_cls_loss_array: ", np.mean(cls_loss_array),
            "\n fgbg_cm_cumulative: ", fgbg_cm_cumulative,
            "\n cls_cm_cumulative: ", cls_cm_cumulative,
            "\n fgbg iou: ", fgbg_iou, "\n fgbg precision: ", fgbg_precision, "\n fgbg recall: ", fgbg_recall,
            "\n cls iou: ", cls_iou, "\n cls precision: ", cls_precision, "\n cls recall: ", cls_recall)
            
        ops = self.mean_val_summary
        

        mean_val_summary = self.sess.run(ops, {self.mean_val_fgbg_loss_ph: np.mean(fgbg_loss_array),
                            self.mean_val_cls_loss_ph: np.mean(cls_loss_array),
                            self.mean_val_loss_ph: np.mean(overall_loss_array),
                            self.mean_val_fgbg_accuracy_ph: np.mean(fgbg_acc_array),
                            self.mean_val_cls_accuracy_ph: np.mean(cls_acc_array),
                            self.mean_val_fgbg_iou_ph: np.mean(fgbg_iou),
                            self.mean_val_cls_iou_ph: np.mean(cls_iou),
                            self.mean_val_fgbg_precision_ph: np.mean(fgbg_precision),
                            self.mean_val_cls_precision_ph: np.mean(cls_precision),
                            self.mean_val_fgbg_recall_ph: np.mean(fgbg_recall),
                            self.mean_val_cls_recall_ph: np.mean(cls_recall)})
        self.train_writer.add_summary(mean_val_summary, self.training_epoch)
        # self.train_writer.add_summary(mean_val_summary, self.training_epoch)

        fgbg_iou_list = []
        for n in range(0, self.num_fgbg_attributes, 1):
            fgbg_iou = fgbg_true_positive_classes[n] / float(fgbg_gt_classes[n] + fgbg_positive_classes[n] - fgbg_true_positive_classes[n])
            fgbg_iou_list.append(fgbg_iou)
        fgbg_mean_iou = sum(fgbg_iou_list) / float(self.num_fgbg_attributes)

        log_out('fgbg_eval accuracy: {}'.format(fgbg_val_total_correct / float(fgbg_val_total_seen)), self.Log_file)
        log_out('fgbg_mean IOU:{}'.format(fgbg_mean_iou), self.Log_file)

        fgbg_mean_iou = 100 * fgbg_mean_iou
        log_out('fgbg_Mean IoU = {:.1f}%'.format(fgbg_mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(fgbg_mean_iou)
        for fgbg_IoU in fgbg_iou_list:
            s += '{:5.2f} '.format(100 * fgbg_IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)

        cls_iou_list = []
        for n in range(0, self.num_classes, 1):
            cls_iou = cls_true_positive_classes[n] / float(cls_gt_classes[n] + cls_positive_classes[n] - cls_true_positive_classes[n])
            cls_iou_list.append(cls_iou)
        cls_mean_iou = sum(cls_iou_list) / float(self.num_classes)

        log_out('cls_eval accuracy: {}'.format(cls_val_total_correct / float(cls_val_total_seen)), self.Log_file)
        log_out('cls_mean IOU:{}'.format(cls_mean_iou), self.Log_file)

        cls_mean_iou = 100 * cls_mean_iou
        log_out('cls_Mean IoU = {:.1f}%'.format(cls_mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(cls_mean_iou)
        for cls_IoU in cls_iou_list:
            s += '{:5.2f} '.format(100 * cls_IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)


        return (fgbg_mean_iou, cls_mean_iou)

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

    # tensorflow v1 style
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

    # @tf.function
    # def gather_neighbour(pc, neighbor_idx):
    #         # gather the coordinates or features of neighboring points
    #         return helper_tf_util.batch_gather(pc, neighbor_idx, axis=1)

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3]
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg
