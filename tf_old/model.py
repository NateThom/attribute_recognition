import numpy as np
import tensorflow as tf
import pandas as pd

import utils
import os
import time

from natsort import natsorted

class MiniNet():
    def __init__(self, sess, args):
        self.sess = sess

        self.load = bool(int(args.load))
        self.training = bool(int(args.training))

        self.input_width = int(args.input_width)
        self.input_height = int(args.input_height)
        self.col_dim = int(args.col_dim)
        self.crop_width = int(args.crop_width)
        self.crop_height = int(args.crop_height)
        self.attribute_count = int(args.attribute_count)
        self.attribute_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                                'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
                                'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                                'Eyeglasses', 'Goatee', 'Grey_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                                'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                                'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Side_Burns',
                                'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                                'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        self.attr_train_size = int(args.attr_train_size)
        self.attr_val_size = int(args.attr_val_size)
        self.attr_test_size = int(args.attr_test_size)
        self.attr_all_size = int(args.attr_all_size)

        self.mask_train_size = int(args.mask_train_size)
        self.mask_val_size = int(args.mask_val_size)
        self.mask_test_size = int(args.mask_test_size)
        self.mask_all_size = int(args.mask_all_size)

        self.batch_size = int(args.batch_size)
        self.balancing = bool(int(args.balancing))
        self.train_epoch = int(args.train_epoch)
        self.lr = float(args.lr)
        self.dropout = bool(int(args.dropout))
        self.dropout_rate = float(args.dropout_rate)

        self.image_path = args.image_path
        self.attr_label_path = args.attr_label_path
        self.mask_image_path = args.mask_image_path
        self.mask_label_path = args.mask_label_path

        self.load_path = args.load_path
        self.model_number = int(args.model_number)
        self.save_path = args.save_path

        self.save = bool(int(args.save))
        self.gpu = int(args.gpu)

        self.load_labels()

        self.image_names = natsorted(os.listdir(self.image_path))

        self.build_model()

    def load_labels(self):
        print("Loading labels for attributes and masks...")
        attr_labels_df = pd.read_csv(self.attr_label_path, sep=',', skiprows=0, usecols=[n for n in range(1, 41)])
        attr_labels_df_filenames = pd.read_csv(self.attr_label_path, sep=',', skiprows=0, usecols=[0])
        self.attr_names = np.asarray(list(attr_labels_df.columns.str.lower()))
        self.attr_file_names = np.asarray(list(attr_labels_df_filenames))
        attr_labels_df_filenames = np.asarray(attr_labels_df_filenames)
        for i in range(len(attr_labels_df_filenames)):
            for j in range(len(attr_labels_df_filenames[i])):
                attr_labels_df_filenames[i, j] = self.image_path + attr_labels_df_filenames[i, j]
        # attr_labels_df_filenames = np.asarray(attr_labels_df_filenames)

        attr_labels_df = np.asarray(attr_labels_df)
        attr_labels_df[attr_labels_df < 1] = 0

        self.attr_val_labels = attr_labels_df[: self.attr_val_size]
        self.attr_val_filenames = attr_labels_df_filenames[: self.attr_val_size]

        self.attr_train_labels = attr_labels_df[self.attr_val_size:self.attr_train_size + self.attr_val_size]
        self.attr_train_filenames = attr_labels_df_filenames[self.attr_val_size:self.attr_train_size + self.attr_val_size]

        self.attr_test_labels = attr_labels_df[self.attr_train_size + self.attr_val_size:]
        self.attr_test_filenames = attr_labels_df_filenames[self.attr_train_size + self.attr_val_size:]

        self.attr_all_labels = attr_labels_df
        self.attr_all_filenames = attr_labels_df_filenames

        print(f"Attribute validation labels: {self.attr_val_labels.shape}")
        print(f"Attribute training labels: {self.attr_train_labels.shape}")
        print(f"Attribute testing labels: {self.attr_test_labels.shape}")
        print(f"All attribute labels: {self.attr_all_labels.shape}")
        print("attributes loaded...")

        mask_labs = pd.read_csv(self.mask_label_path, sep=',', usecols=self.attribute_names)
        mask_labs_filenames = pd.read_csv(self.mask_label_path, sep=',', usecols=["image_name"])
        mask_labs = np.asarray(mask_labs)
        mask_labs_filenames = np.asarray(mask_labs_filenames)
        for i in range(len(mask_labs_filenames)):
            for j in range(len(mask_labs_filenames[i])):
                mask_labs_filenames[i, j] = self.image_path + mask_labs_filenames[i, j]


        self.mask_val_labels = mask_labs[: self.mask_val_size]
        self.mask_val_filenames = mask_labs_filenames[: self.mask_val_size]

        self.mask_train_labels = mask_labs[self.mask_val_size:self.mask_train_size + self.mask_val_size]
        self.mask_train_filenames = mask_labs_filenames[self.mask_val_size:self.mask_train_size + self.mask_val_size]

        self.mask_test_labels = mask_labs[self.mask_train_size + self.mask_val_size:]
        self.mask_test_filenames = mask_labs_filenames[self.mask_train_size + self.mask_val_size:]

        self.mask_all_labels = mask_labs
        self.mask_all_filenames = mask_labs_filenames

        print(f"Mask validation labels: {self.mask_val_labels.shape}")
        print(f"Mask training labels: {self.mask_train_labels.shape}")
        print(f"Mask testing labels: {self.mask_test_labels.shape}")
        print(f"All mask labels: {self.mask_all_labels.shape}")

        print("masks loaded...")
        print("Done!")

    def build_model(self):
        if self.training:
            self.build_train_dataset()
            self.build_val_dataset()

            self.mask_acc_logits, self.attr_acc_logits = self.CNN(self.val_dataset_images, training=False)
            self.attr_acc_probs = tf.nn.sigmoid(self.attr_acc_logits)
            self.attr_acc_preds = tf.round(self.attr_acc_probs)

            # self.mask_acc_probs = tf.nn.l2_loss(self.mask_acc_logits)
            # self.mask_acc_preds = tf.round(self.mask_acc_probs)

            # print(f"output: {self.mask_acc_logits}")
            # print(f"prob: {self.mask_acc_probs}")
            # print(f"pred: {self.mask_acc_preds}")

            self.attr_pos_preds = tf.equal(self.attr_acc_preds, 1)
            self.attr_neg_preds = tf.logical_not(self.attr_pos_preds)
            self.attr_pos_labels = tf.equal(self.val_dataset_attr_labels, 1)
            self.attr_neg_labels = tf.logical_not(self.attr_pos_labels)

            self.attr_tp = tf.count_nonzero(tf.logical_and(self.attr_pos_preds, self.attr_pos_labels), 0, dtype=tf.float32)
            self.attr_fp = tf.count_nonzero(tf.logical_and(self.attr_pos_preds, self.attr_neg_labels), 0, dtype=tf.float32)
            self.attr_tn = tf.count_nonzero(tf.logical_and(self.attr_neg_preds, self.attr_neg_labels), 0, dtype=tf.float32)
            self.attr_fn = tf.count_nonzero(tf.logical_and(self.attr_neg_preds, self.attr_pos_labels), 0, dtype=tf.float32)

            self.attr_acc = tf.divide(tf.add(self.attr_tp, self.attr_tn), tf.add_n([self.attr_tp, self.attr_fp, self.attr_tn, self.attr_fn]))
            self.pos_acc = tf.divide(self.attr_tp, tf.add(self.attr_tp, self.attr_fn))
            self.pos_acc = tf.where(tf.is_nan(self.pos_acc), tf.zeros_like(self.pos_acc, dtype=tf.float32),
                                    self.pos_acc)
            self.neg_acc = tf.divide(self.attr_tn, tf.add(self.attr_fp, self.attr_tn))
            self.neg_acc = tf.where(tf.is_nan(self.neg_acc), tf.zeros_like(self.neg_acc, dtype=tf.float32),
                                    self.neg_acc)

            self.mask_logits, self.logits = self.CNN(self.train_dataset_images, training=True)
            self.sig_cross = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.train_dataset_attr_labels)
            #self.euclidean = tf.norm(self.mask_logits - tf.squeeze(self.train_dataset_mask_labels), ord='euclidean')

            # if self.balancing:
            #     self.SL_mask = self.get_SL_recognition_mask(self.attr_train_labels)
            #     self.sig_cross = self.SL_mask * self.sig_cross

            # self.loss = tf.reduce_mean(self.sig_cross) + tf.reduce_mean(self.euclidean)

            #self.loss = (tf.reduce_mean(self.sig_cross) + (self.euclidean / 350))
            self.loss = tf.reduce_mean(self.sig_cross)
            # self.loss = self.euclidean

            # self.mini_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CNN')
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=self.mini_variables)

            # self.saver = tf.train.Saver(self.mini_variables, max_to_keep=5)

        else:
            self.build_test_dataset()

            self.mask_acc_logits, self.attr_acc_logits = self.CNN(self.test_dataset_images, training=False)
            self.attr_acc_probs = tf.nn.sigmoid(self.attr_acc_logits)
            self.attr_acc_preds = tf.round(self.attr_acc_probs)

            self.attr_pos_preds = tf.equal(self.attr_acc_preds, 1)
            self.attr_neg_preds = tf.logical_not(self.attr_pos_preds)
            self.attr_pos_labels = tf.equal(self.test_dataset_attr_labels, 1)
            self.attr_neg_labels = tf.logical_not(self.attr_pos_labels)

            self.attr_tp = tf.count_nonzero(tf.logical_and(self.attr_pos_preds, self.attr_pos_labels), 0,
                                            dtype=tf.float32)
            self.attr_fp = tf.count_nonzero(tf.logical_and(self.attr_pos_preds, self.attr_neg_labels), 0,
                                            dtype=tf.float32)
            self.attr_tn = tf.count_nonzero(tf.logical_and(self.attr_neg_preds, self.attr_neg_labels), 0,
                                            dtype=tf.float32)
            self.attr_fn = tf.count_nonzero(tf.logical_and(self.attr_neg_preds, self.attr_pos_labels), 0,
                                            dtype=tf.float32)

            self.attr_acc = tf.divide(tf.add(self.attr_tp, self.attr_tn),
                                      tf.add_n([self.attr_tp, self.attr_fp, self.attr_tn, self.attr_fn]))
            self.pos_acc = tf.divide(self.attr_tp, tf.add(self.attr_tp, self.attr_fn))
            self.pos_acc = tf.where(tf.is_nan(self.pos_acc), tf.zeros_like(self.pos_acc, dtype=tf.float32),
                                    self.pos_acc)
            self.neg_acc = tf.divide(self.attr_tn, tf.add(self.attr_fp, self.attr_tn))
            self.neg_acc = tf.where(tf.is_nan(self.neg_acc), tf.zeros_like(self.neg_acc, dtype=tf.float32),
                                    self.neg_acc)

        self.mini_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CNN')
        self.saver = tf.train.Saver(self.mini_variables, max_to_keep=5)

        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            self.sess.run(tf.initialize_all_variables())

        if self.load:
            self.load_model()

    def build_train_dataset(self):
        print("Building training dataset made up of input images, attribute labels, and masks...")

        # attr_train_labels_tens = tf.convert_to_tensor(self.attr_train_labels[:100], dtype=tf.float32)
        attr_train_labels_tens = tf.convert_to_tensor(self.attr_train_labels, dtype=tf.float32)

        # mask_train_labels_tens = tf.convert_to_tensor(self.mask_train_labels[:100])
        mask_train_labels_tens = tf.convert_to_tensor(self.mask_train_labels)

        # train_filenames_1d = self.attr_train_filenames[:100].flatten()
        train_filenames_1d = self.attr_train_filenames.flatten()

        # mask_train_filenames_tens = tf.convert_to_tensor(self.mask_train_filenames)
        # mask_train_labels = self.mask_parse_fn()
        # mask_train_labels_tens = tf.convert_to_tensor(self.mask_train_labels, dtype=tf.float32)

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_filenames_1d, attr_train_labels_tens, mask_train_labels_tens))
        self.train_dataset = self.train_dataset.shuffle(len(train_filenames_1d))
        self.train_dataset = self.train_dataset.map(self.parse_fn, num_parallel_calls=6)
        self.train_dataset = self.train_dataset.batch(self.batch_size)
        self.train_dataset = self.train_dataset.map(self.train_preproc, num_parallel_calls=6)
        self.train_dataset = self.train_dataset.prefetch(self.batch_size)

        self.train_iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                                   self.train_dataset.output_shapes)

        self.train_dataset_images, self.train_dataset_attr_labels, self.train_dataset_mask_labels = self.train_iterator.get_next()

        self.train_init_op = self.train_iterator.make_initializer(self.train_dataset)

        print("Done!")

    def parse_fn(self, attr_filename, attr_label, mask_label):
        image_string = tf.read_file(attr_filename)
        image = tf.image.decode_jpeg(image_string, channels=self.col_dim)
        image = tf.image.convert_image_dtype(image, tf.float32)

        mil = []

        for mask in range(0, self.attribute_count):
            mask_string = tf.read_file(self.mask_image_path + mask_label[mask])
            mask_image = tf.image.decode_and_crop_jpeg(mask_string, [0, 0, 204, 164], channels=1)
            mask_image = tf.image.convert_image_dtype(mask_image, tf.float32)
            mil.append(mask_image)
            # mask_images = tf.concat([mask_images, mask_image], 0)
            # mask_images = tf.stack([mask_images, mask_image])
        mask_labels = tf.stack(
            [mil[0], mil[1], mil[2], mil[3], mil[4], mil[5], mil[6], mil[7], mil[8], mil[9], mil[10],
             mil[11], mil[12], mil[13], mil[14], mil[15], mil[16], mil[17], mil[18], mil[19],
             mil[20], mil[21], mil[22], mil[23], mil[24], mil[25], mil[26], mil[27], mil[28],
             mil[29], mil[30], mil[31], mil[32], mil[33], mil[34], mil[35], mil[36], mil[37],
             mil[38], mil[39]], axis=2)

        # mask_labels = tf.squeeze(mask_labels)

        # mask_labels = tf.image.central_crop(mask_labels, central_fraction=.5)

        return image, attr_label, mask_labels

    def train_preproc(self, images, attr_labels, mask_labels):
        images = tf.random_crop(images, [self.batch_size, self.crop_height, self.crop_width, self.col_dim])

        # images = tf.subtract(images, tf.reduce_mean(images, [0, 1, 2]))

        # mask_labels = tf.random_crop(mask_labels, [self.batch_size, self.crop_height, self.crop_width, 1])

        return images, attr_labels, mask_labels



    def build_val_dataset(self):
        print("Building validation dataset made up of input images, attribute labels, and masks...")

        # attr_val_labels_tens = tf.convert_to_tensor(self.attr_val_labels[:100], dtype=tf.float32)
        attr_val_labels_tens = tf.convert_to_tensor(self.attr_val_labels, dtype=tf.float32)

        # mask_val_labels_tens = tf.convert_to_tensor(self.mask_val_labels[:100])
        mask_val_labels_tens = tf.convert_to_tensor(self.mask_val_labels)

        # val_filenames_1d = self.attr_val_filenames[:100].flatten()
        val_filenames_1d = self.attr_val_filenames.flatten()

        self.val_dataset = tf.data.Dataset.from_tensor_slices(
            (val_filenames_1d, attr_val_labels_tens, mask_val_labels_tens))
        self.val_dataset = self.val_dataset.shuffle(len(val_filenames_1d))
        self.val_dataset = self.val_dataset.map(self.val_parse_preproc, num_parallel_calls=6)
        self.val_dataset = self.val_dataset.batch(self.batch_size)
        self.val_dataset = self.val_dataset.prefetch(self.batch_size)

        self.val_iterator = tf.data.Iterator.from_structure(self.val_dataset.output_types, self.val_dataset.output_shapes)

        self.val_dataset_images, self.val_dataset_attr_labels, self.val_dataset_mask_labels = self.val_iterator.get_next()

        self.val_init_op = self.val_iterator.make_initializer(self.val_dataset)

        print("Done!")

    def val_parse_preproc(self, filename, label, mask_filename):
        image_string = tf.read_file(filename)
        image = tf.image.decode_and_crop_jpeg(image_string, [0, 0, self.crop_height, self.crop_width],
                                              channels=self.col_dim)
        image = tf.image.convert_image_dtype(image, tf.float32)

        mil = []

        for mask in range(0, self.attribute_count):
            mask_string = tf.read_file(self.mask_image_path + mask_filename[mask])
            mask_image = tf.image.decode_and_crop_jpeg(mask_string, [0, 0, 204, 164], channels=1)
            mask_image = tf.image.convert_image_dtype(mask_image, tf.float32)
            mil.append(mask_image)

        mask_images = tf.stack(
            [mil[0], mil[1], mil[2], mil[3], mil[4], mil[5], mil[6], mil[7], mil[8], mil[9], mil[10],
             mil[11], mil[12], mil[13], mil[14], mil[15], mil[16], mil[17], mil[18], mil[19],
             mil[20], mil[21], mil[22], mil[23], mil[24], mil[25], mil[26], mil[27], mil[28],
             mil[29], mil[30], mil[31], mil[32], mil[33], mil[34], mil[35], mil[36], mil[37],
             mil[38], mil[39]], axis=2)

        return image, label, mask_images

    def build_test_dataset(self):
        print("Building test dataset made up of input images, attribute labels, and masks...")

        # attr_test_labels_tens = tf.convert_to_tensor(self.attr_val_labels[:100], dtype=tf.float32)
        attr_test_labels_tens = tf.convert_to_tensor(self.attr_test_labels, dtype=tf.float32)

        # mask_test_labels_tens = tf.convert_to_tensor(self.mask_val_labels[:100])
        mask_test_labels_tens = tf.convert_to_tensor(self.mask_test_labels)

        # test_filenames_1d = self.attr_val_filenames[:100].flatten()
        test_filenames_1d = self.attr_test_filenames.flatten()

        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_filenames_1d, attr_test_labels_tens, mask_test_labels_tens))
        self.test_dataset = self.test_dataset.shuffle(len(test_filenames_1d))
        self.test_dataset = self.test_dataset.map(self.test_parse_preproc, num_parallel_calls=6)
        self.test_dataset = self.test_dataset.batch(self.batch_size)
        self.test_dataset = self.test_dataset.prefetch(self.batch_size)

        self.test_iterator = tf.data.Iterator.from_structure(self.test_dataset.output_types, self.test_dataset.output_shapes)

        self.test_dataset_images, self.test_dataset_attr_labels, self.test_dataset_mask_labels = self.test_iterator.get_next()

        self.test_init_op = self.test_iterator.make_initializer(self.test_dataset)

        print("Done!")

    def test_parse_preproc(self, filename, label, mask_filename):
        image_string = tf.read_file(filename)
        image = tf.image.decode_and_crop_jpeg(image_string, [0, 0, self.crop_height, self.crop_width],
                                              channels=self.col_dim)
        image = tf.image.convert_image_dtype(image, tf.float32)

        mil = []

        for mask in range(0, self.attribute_count):
            mask_string = tf.read_file(self.mask_image_path + mask_filename[mask])
            mask_image = tf.image.decode_and_crop_jpeg(mask_string, [0, 0, 204, 164], channels=1)
            mask_image = tf.image.convert_image_dtype(mask_image, tf.float32)
            mil.append(mask_image)

        mask_images = tf.stack(
            [mil[0], mil[1], mil[2], mil[3], mil[4], mil[5], mil[6], mil[7], mil[8], mil[9], mil[10],
             mil[11], mil[12], mil[13], mil[14], mil[15], mil[16], mil[17], mil[18], mil[19],
             mil[20], mil[21], mil[22], mil[23], mil[24], mil[25], mil[26], mil[27], mil[28],
             mil[29], mil[30], mil[31], mil[32], mil[33], mil[34], mil[35], mil[36], mil[37],
             mil[38], mil[39]], axis=2)

        return image, label, mask_images


    def CNN(self, inputs, training=True):
        with tf.variable_scope('CNN', reuse=tf.AUTO_REUSE):
            print('Input: ', inputs.shape)
            conv1 = tf.layers.conv2d(inputs, 75, [7, 7], strides=(1, 1), padding='valid', name='conv1')
            print('Conv 1: ', conv1.shape)
            temp = conv1.shape
            relu1 = tf.nn.relu(tf.layers.batch_normalization(conv1, training=training))
            pool1 = tf.layers.max_pooling2d(relu1, [2, 2], strides=(1, 1), padding='same', name='pool1')
            print('Pool 1: ', pool1.shape)
            temp = pool1.shape
            conv2 = tf.layers.conv2d(pool1, 200, [3, 3], strides=(1, 1), padding='valid', name='conv2')
            print('Conv 2: ', conv2.shape)
            relu2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=training))
            # pool2 = tf.layers.max_pooling2d(relu2, [3, 3], strides=(2, 2), padding='same', name='pool2')
            # print('Pool 2: ', pool2.shape)
            temp = relu2.shape
            conv3 = tf.layers.conv2d(relu2, 300, [3, 3], strides=(1, 1), padding='valid', name='conv3')
            relu3 = tf.nn.relu(tf.layers.batch_normalization(conv3, training=training))
            print('Conv3: ', conv3.shape)
            temp = relu3.shape
            conv4 = tf.layers.conv2d(relu3, 512, [3, 3], strides=(1, 1), padding='valid', name='conv4')
            relu4 = tf.nn.relu(tf.layers.batch_normalization(conv3, training=training))
            print('Conv4: ', conv4.shape)
            temp = relu4.shape
            conv5 = tf.layers.conv2d(relu4, 512, [3, 3], strides=(1, 1), padding='valid', name='conv5')
            relu5 = tf.nn.relu(tf.layers.batch_normalization(conv5, training=training))
            print('Conv5: ', conv5.shape)
            temp = relu5.shape
            conv6 = tf.layers.conv2d(relu5, 40, [3, 3], strides=(1, 1), padding='valid', name='conv6')
            print('Conv6: ', conv6.shape)
            temp1 = conv6.shape
            flat1 = tf.layers.flatten(conv6, name='flat1', data_format='channels_last')
            temp2 = flat1.shape
            temp3 = tf.squeeze(flat1).shape
            fc1 = tf.layers.dense(flat1, units=self.attribute_count, name='fc1')
            # relu6 = tf.nn.relu(tf.layers.batch_normalization(fc1, training=training))
            # drop1 = tf.layers.dropout(relu6, rate=self.dropout_rate, training=training)

            # fc2 = tf.layers.dense(drop1, units=self.attribute_count, name='fc2')

            return conv6, fc1

    def load_model(self):
        print('Loading model...')
        try:
            self.saver.restore(self.sess, os.path.join(self.load_path, '-%d' % self.model_number))
        except:
            raise Exception('Error loading model.')

    def train(self):
        train_start_time = time.time()
        epoch_start_time = time.time()

        epoch = 0
        iteration = 0
        print_iter = int((self.attr_train_size / self.batch_size) / 141.202)
        print('\nEpoch: %d' % epoch)

        self.sess.run(self.train_init_op)
        self.sess.run(self.val_init_op)

        train_acc_flag = 1

        train_accs = []
        train_pos_accs = []
        train_neg_accs = []
        train_tp = []
        train_fp = []
        train_tn = []
        train_fn = []

        while True:
            try:
                if train_acc_flag is 1:
                        train_accs = []
                        train_pos_accs = []
                        train_neg_accs = []
                        train_tp = []
                        train_fp = []
                        train_tn = []
                        train_fn = []
                        train_acc_flag = 0
                tp, fp, tn, fn, acc, pos, neg, _, loss = self.sess.run([self.attr_tp, self.attr_fp, self.attr_tn,
                                                                        self.attr_fn, self.attr_acc, self.pos_acc,
                                                                        self.neg_acc, self.optim, self.loss])
                # acc, pos, neg = self.sess.run([self.attr_acc, self.pos_acc, self.neg_acc, self.optim, self.loss])

                train_accs.append(acc)
                train_pos_accs.append(pos)
                train_neg_accs.append(neg)
                train_tp.append(tp)
                train_fp.append(fp)
                train_tn.append(tn)
                train_fn.append(fn)

                if iteration % print_iter == 0:
                    print('Training Iteration: {0:5d}    Training Loss: {1:8.6f}'.format(iteration, np.mean(loss)))
                iteration += 1

            except (tf.errors.OutOfRangeError, tf.errors.InvalidArgumentError) as e:
                epoch_length = time.time() - epoch_start_time
                print('Total epoch time: %.2f' % epoch_length)

                self.write_verification_results(epoch, np.mean(train_accs, 0), np.mean(train_pos_accs, 0),
                                                np.mean(train_neg_accs, 0), np.mean(train_tp, 0), np.mean(train_fp, 0),
                                                np.mean(train_tn, 0), np.mean(train_fn, 0))
                train_acc_flag = 1

                # self.test_recognition_model(epoch)
                self.sess.run(self.train_init_op)
                self.sess.run(self.val_init_op)

                if self.save:
                    print("saving")
                    self.saver.save(self.sess, './%s/' % self.save_path, global_step=epoch)

                epoch += 1

                iteration = 0
                if epoch == self.train_epoch:
                    break

                print('\nEpoch: %d' % epoch)
                epoch_start_time = time.time()

        print('Total training time: %.2f' % (time.time() - train_start_time))

    def test(self):
        test_start_time = time.time()

        iteration = 0
        print_iter = int((self.attr_test_size / self.batch_size) / 10)

        self.sess.run(self.test_init_op)

        test_accs = []
        test_pos_accs = []
        test_neg_accs = []
        test_tp = []
        test_fp = []
        test_tn = []
        test_fn = []

        while True:
            try:

                tp, fp, tn, fn, acc, pos, neg = self.sess.run([self.attr_tp, self.attr_fp, self.attr_tn,
                                                                        self.attr_fn, self.attr_acc, self.pos_acc,
                                                                        self.neg_acc])

                test_accs.append(acc)
                test_pos_accs.append(pos)
                test_neg_accs.append(neg)
                test_tp.append(tp)
                test_fp.append(fp)
                test_tn.append(tn)
                test_fn.append(fn)

                if iteration % print_iter == 0:
                    print('Testing Iteration: {0:5d}'.format(iteration))
                iteration += 1


            except:
                test_length = time.time() - test_start_time
                print('Total test time: %.2f' % test_length)

                self.write_test_results(np.mean(test_accs, 0), np.mean(test_pos_accs, 0),
                                                np.mean(test_neg_accs, 0), np.mean(test_tp, 0), np.mean(test_fp, 0),
                                                np.mean(test_tn, 0), np.mean(test_fn, 0))
                break

    def write_verification_results(self, epoch, acc, acc_1, acc_0, tp, fp, tn, fn):
        print('Epoch: %d' % epoch)

        print('{0:29} {1:12} {2:13} {3:13}'.format('\nAttributes', 'Acc', 'Acc_1', 'Acc_0'))
        print('-' * 103)
        for attr, a, a_1, a_0 in zip(self.attr_names, acc, acc_1, acc_0):
            print('{0:19} {1:13.3f} {2:13.3f} {3:13.3f}'.format(attr, a, a_1, a_0))
        print('-' * 103)
        print('{0:19} {1:13.3f} {2:13.3f} {3:13.3f}'.format('', np.mean(acc), np.mean(acc_1), np.mean(acc_0)))

        print('-' * 103)

        print('{0:29} {1:12} {2:13} {3:13} {4:13}'.format('\nAttributes', 'TP', 'FP', 'TN', 'FN'))
        print('-' * 103)
        for attr, true_p, false_p, true_n, false_n in zip(self.attr_names, tp, fp, tn, fn):
            print('{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f}'.format(attr, true_p, false_p, true_n, false_n))
        print('-' * 103)
        print('{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f}'.format('', np.mean(tp), np.mean(fp), np.mean(tn), np.mean(fn)))

        f = open("verification.txt", 'a+')
        f.write('Epoch: %d\n' % epoch)

        f.write('{0:29} {1:12} {2:13} {3:13}\n'.format('\nAttributes', 'Acc', 'Acc_1', 'Acc_0'))
        f.write('-' * 103)
        f.write('\n')
        for attr, a, a_1, a_0 in zip(self.attr_names, acc, acc_1, acc_0):
            f.write('{0:19} {1:13.3f} {2:13.3f} {3:13.3f}\n'.format(attr, a, a_1, a_0))
        f.write('-' * 103)
        f.write('\n')
        f.write('{0:19} {1:13.3f} {2:13.3f} {3:13.3f}\n'.format('', np.mean(acc), np.mean(acc_1), np.mean(acc_0)))

        f.write('-' * 103)
        f.write('\n')

        f.write('{0:29} {1:12} {2:13} {3:13} {4:13}\n'.format('\nAttributes', 'TP', 'FP', 'TN', 'FN'))
        f.write('-' * 103)
        f.write('\n')
        for attr, true_p, false_p, true_n, false_n in zip(self.attr_names, tp, fp, tn, fn):
            f.write('{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f}\n'.format(attr, true_p, false_p, true_n, false_n))
        f.write('-' * 103)
        f.write('\n')
        f.write('{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f}\n'.format('', np.mean(tp), np.mean(fp), np.mean(tn),
                                                                      np.mean(fn)))

        f.close()

    def write_test_results(self, acc, acc_1, acc_0, tp, fp, tn, fn):
        print('{0:29} {1:12} {2:13} {3:13}'.format('\nAttributes', 'Acc', 'Acc_1', 'Acc_0'))
        print('-' * 103)
        for attr, a, a_1, a_0 in zip(self.attr_names, acc, acc_1, acc_0):
            print('{0:19} {1:13.3f} {2:13.3f} {3:13.3f}'.format(attr, a, a_1, a_0))
        print('-' * 103)
        print('{0:19} {1:13.3f} {2:13.3f} {3:13.3f}'.format('', np.mean(acc), np.mean(acc_1), np.mean(acc_0)))

        print('{0:29} {1:12} {2:13} {3:13} {4:13}'.format('\nAttributes', 'TP', 'FP', 'TN', 'FN'))
        print('-' * 103)
        for attr, true_p, false_p, true_n, false_n in zip(self.attr_names, tp, fp, tn, fn):
            print('{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f}'.format(attr, true_p, false_p, true_n, false_n))
        print('-' * 103)
        print('{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f}'.format('', np.mean(tp), np.mean(fp), np.mean(tn),
                                                                      np.mean(fn)))

        f = open("test_non_euc.txt", 'a+')

        f.write('{0:29} {1:12} {2:13} {3:13}\n'.format('\nAttributes', 'Acc', 'Acc_1', 'Acc_0'))
        f.write('-' * 103)
        f.write('\n')
        for attr, a, a_1, a_0 in zip(self.attr_names, acc, acc_1, acc_0):
            f.write('{0:19} {1:13.3f} {2:13.3f} {3:13.3f}\n'.format(attr, a, a_1, a_0))
        f.write('-' * 103)
        f.write('\n')
        f.write('{0:19} {1:13.3f} {2:13.3f} {3:13.3f}\n'.format('', np.mean(acc), np.mean(acc_1), np.mean(acc_0)))

        f.write('-' * 103)
        f.write('\n')

        f.write('{0:29} {1:12} {2:13} {3:13} {4:13}\n'.format('\nAttributes', 'TP', 'FP', 'TN', 'FN'))
        f.write('-' * 103)
        f.write('\n')
        for attr, true_p, false_p, true_n, false_n in zip(self.attr_names, tp, fp, tn, fn):
            f.write('{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f}\n'.format(attr, true_p, false_p, true_n, false_n))
        f.write('-' * 103)
        f.write('\n')
        f.write('{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f}\n'.format('', np.mean(tp), np.mean(fp), np.mean(tn),
                                                                          np.mean(fn)))

        f.close()

    def get_SL_recognition_mask(self, labels):

        pos_count = tf.reduce_sum(labels, axis=0)
        neg_count = tf.subtract(float(self.batch_size), pos_count)

        check_zeros = tf.multiply(pos_count, neg_count)
        zeros_bools = tf.where(tf.greater(check_zeros, 0), tf.ones_like(check_zeros), tf.zeros_like(check_zeros))

        pos_weights = tf.divide(tf.ones_like(pos_count, dtype=tf.float32), pos_count)
        neg_weights = tf.divide(tf.ones_like(neg_count, dtype=tf.float32), neg_count)

        pos_weights_1 = tf.where(tf.is_inf(pos_weights), tf.zeros_like(pos_weights, dtype=tf.float32), pos_weights)
        neg_weights_1 = tf.where(tf.is_inf(neg_weights), tf.zeros_like(neg_weights, dtype=tf.float32), neg_weights)

        weighted_pos = tf.multiply(labels, pos_weights_1)
        weighted_neg = tf.multiply(tf.subtract(1.0, labels), neg_weights_1)

        full_mask = tf.add(weighted_pos, weighted_neg)
        final_mask = tf.multiply(zeros_bools, full_mask)

        return final_mask

    def test_recognition_model(self, epoch):
        val_accs = []
        val_pos_accs = []
        val_neg_accs = []
        print_iter = int((self.attr_val_size / self.batch_size) / 40.343)
        iteration = 0
        self.sess.run(self.val_init_op)
        while True:
            try:
                acc, pos, neg, loss = self.sess.run([self.attr_acc, self.pos_acc, self.neg_acc, self.loss])

                if iteration % print_iter == 0:
                    print('Validation Iteration: {0:5d}    Validation Loss: {1:8.6f}'.format(iteration, np.mean(loss)))
                iteration += 1

                val_accs.append(acc)
                val_pos_accs.append(pos)
                val_neg_accs.append(neg)
            except (tf.errors.OutOfRangeError, tf.errors.InvalidArgumentError):
                val_accs = np.asarray(val_accs)
                val_pos_accs = np.asarray(val_pos_accs)
                val_neg_accs = np.asarray(val_neg_accs)

                self.write_verification_results(epoch, np.mean(val_accs, 0), np.mean(val_pos_accs, 0), np.mean(val_neg_accs, 0))
