import numpy as np
import tensorflow as tf


class CNNModel:
    def __init__(self, num_weights=32, num_samples=32, num_bands=9,
                 load_path="/content/ee-data/img_full_output/",
                 save_path='/content/datalab/crop_yield_prediction/train_results/final/yearly/'):
        """Creates a TensorFlow CNN model using batch normalization, Adam optimizer, L1 regularization, and L2 loss

        Parameters
        ----------
        config: configuration object for the CNN model including size of parameters, training steps, lr, and weight decay
        """

        # Set variables
        self.load_path = load_path
        self.save_path = save_path
        self.num_weights = num_weights
        self.num_samples = num_samples
        self.num_bands = num_bands

        # Set TF variables
        self.x = tf.placeholder(tf.float32, [None, num_weights, num_samples, num_bands], name="x")
        self.y = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])

        ### Building TF layers
        # Layer 1
        self.conv1_1 = self.conv_relu_batch(self.x, 128, 3, 1, name="conv1_1")
        conv1_1_d = tf.nn.dropout(self.conv1_1, self.keep_prob)
        conv1_2 = self.conv_relu_batch(conv1_1_d, 256, 3, 2, name="conv1_2")
        conv1_2_d = tf.nn.dropout(conv1_2, self.keep_prob)

        # Layer 2
        conv2_1 = self.conv_relu_batch(conv1_2_d, 256, 3, 1, name="conv2_1")
        conv2_1_d = tf.nn.dropout(conv2_1, self.keep_prob)
        conv2_2 = self.conv_relu_batch(conv2_1_d, 512, 3, 2, name="conv2_2")
        conv2_2_d = tf.nn.dropout(conv2_2, self.keep_prob)

        # Layer 3
        conv3_1 = self.conv_relu_batch(conv2_2_d, 512, 3, 1, name="conv3_1")
        conv3_1_d = tf.nn.dropout(conv3_1, self.keep_prob)
        conv3_2 = self.conv_relu_batch(conv3_1_d, 1024, 3, 2, name="conv3_2")
        conv3_2_d = tf.nn.dropout(conv3_2, self.keep_prob)

        # Layer 4
        dim = np.prod(conv3_2_d.get_shape().as_list()[1:])

        # Layer 5
        flattened = tf.reshape(conv3_2_d, [-1, dim])

        # Layer 6
        self.fc6 = self.dense(flattened, 1024, name="fc6")

        # Layer 7
        self.fc7 = self.dense(self.fc6, 1, name="dense")

        # Layer 8
        self.logits = tf.squeeze(self.fc7, name="logits")

        ### End of Layers

        # L2 Loss error
        self.loss_err = tf.nn.l2_loss(self.logits - self.y)

        # TODO: Figure out what this is
        # This might need to be changed since it is never actually being reused
        with tf.variable_scope('dense') as scope:
            scope.reuse_variables()
            self.dense_W = tf.get_variable('W')
            self.dense_B = tf.get_variable('b')
        with tf.variable_scope('conv1_1/conv2d') as scope:
            scope.reuse_variables()
            self.conv_W = tf.get_variable('W')
            self.conv_B = tf.get_variable('b')

        # L1 loss regularizer
        self.loss_reg = tf.abs(tf.reduce_sum(self.logits - self.y))

        # TODO: Figure out what this is
        # soybean
        # alpha = 1.5
        # corn
        alpha = 5

        # loss
        self.loss = self.loss_err + self.loss_reg * alpha

        # training optimizer
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def conv2d(self, input_data, out_channels, filter_size, stride, in_channels=None, name="conv2d"):
        """Perform a 2D convolution on the input_data and return the resulting tensor

        Parameters
        ----------
        input_data : tensor
            input tensor
        out_channels : int
            number of outputs
        filter_size : int
            shared weights size
        stride : int
            stride distance aka pixel step size
        in_channels : int
            number of inputs
        name : str
            layer name

        Returns
        -------
        A TensorFlow convolutional 2D layer
        """
        if not in_channels:
            in_channels = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            weights = tf.get_variable("W", [filter_size, filter_size, in_channels, out_channels],
                                      initializer=tf.contrib.layers.variance_scaling_initializer())
            bias = tf.get_variable("b", [1, 1, 1, out_channels])
            return tf.nn.conv2d(input_data, weights, [1, stride, stride, 1], "SAME") + bias

    def conv_relu_batch(self, input_data, out_channels, filter_size, stride, in_channels=None, name="crb"):
        """Create a batch normalized 2D convolutional tensor and apply RELU activation

        Parameters
        ----------
        input_data : tensor
            input tensor
        out_channels : int
            number of outputs
        filter_size : int
            shared weights size
        stride : int
            stride distance aka pixel step size
        in_channels
            number of inputs
        name
            layer name

        Returns
        -------
        TensorFlow RELU activation layer

        """
        with tf.variable_scope(name):
            a = self.conv2d(input_data, out_channels, filter_size, stride, in_channels)
            b = self.batch_normalization(a, axes=[0, 1, 2])
            r = tf.nn.relu(b)
            return r

    def dense(self, input_data, H, N=None, name="dense"):
        """Dense layer that performs the weight and intput multiplication

        Parameters
        ----------
        input_data : tensor
            input tensor
        H : int
            x dimension in the shape of the weights
        N : int
            y dimension in the shape of the weights
        name : str
            TF object name

        Returns
        -------
        tensor of type input_data that is input_data * W + bias

        """
        if not N:
            N = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            weights = tf.get_variable("W", [N, H], initializer=tf.contrib.layers.variance_scaling_initializer())
            bias = tf.get_variable("b", [1, H])
            return tf.matmul(input_data, weights, name="matmul") + bias

    def batch_normalization(self, input_data, axes=[0], name="batch"):
        """Perform batch normalization on a tensor

        Parameters
        ----------
        input_data : tensor
            input tensor
        axes : int list
            axes to calculate the mean and variance around
        name : str
            TF object name

        Returns
        -------
        TF normalized tensor same shape as input tensor

        """
        with tf.variable_scope(name):
            mean, variance = tf.nn.moments(input_data, axes, keep_dims=True, name="moments")
            return tf.nn.batch_normalization(input_data, mean=mean, variance=variance, offset=None, scale=None,
                                             variance_epsilon=1e-6, name="batch")

    def train(self, training_iterations=25000, learning_rate=1e-3, dropout_rate=0.25, num_bins=32):
        """Train the model

        Returns
        -------
        Trained model
        """

        # Loss / error over run time
        summary_train_loss = []
        summary_eval_loss = []
        summary_RMSE = []
        summary_ME = []

        # load data to memory
        filename = 'histogram_all_full' + '.npz'
        content = np.load(self.load_path + filename)
        image_all = content['output_image']
        yield_all = content['output_yield']
        year_all = content['output_year']
        locations_all = content['output_locations']
        index_all = content['output_index']

        # delete all images which have less than the seasonal number of days in a year before 2016
        # TODO: Why 287?
        list_delete = []
        for i in range(image_all.shape[0]):
            if np.sum(image_all[i, :, :, :]) <= 287:
                if year_all[i] < 2016:
                    list_delete.append(i)
        image_all = np.delete(image_all, list_delete, 0)
        yield_all = np.delete(yield_all, list_delete, 0)
        year_all = np.delete(year_all, list_delete, 0)
        locations_all = np.delete(locations_all, list_delete, 0)
        index_all = np.delete(index_all, list_delete, 0)

        # keep targeted states
        list_keep = []
        for i in range(image_all.shape[0]):
            if (index_all[i, 0] == 5) or (index_all[i, 0] == 17) or (index_all[i, 0] == 18) or (
                        index_all[i, 0] == 19) or (index_all[i, 0] == 20) or (index_all[i, 0] == 27) or (
                        index_all[i, 0] == 29) or (index_all[i, 0] == 31) or (index_all[i, 0] == 38) or (
                        index_all[i, 0] == 39) or (index_all[i, 0] == 46):
                list_keep.append(i)
        image_all = image_all[list_keep, :, :, :]
        yield_all = yield_all[list_keep]
        year_all = year_all[list_keep]
        locations_all = locations_all[list_keep, :]
        index_all = index_all[list_keep, :]

        # training montage
        for predict_year in range(2009, 2016):

            # # split into train and validate
            # index_train = np.nonzero(year_all < predict_year)[0]
            # index_validate = np.nonzero(year_all == predict_year)[0]
            # index_test = np.nonzero(year_all == predict_year+1)[0]

            # random choose validation set
            # TODO: This doesn't actually work. Selected years are not valid sets for training / validating each loop
            # This appears to be a method for splitting training and validation for forecasting
            index_train = np.nonzero(year_all < predict_year)[0]
            index_validate = np.nonzero(year_all == predict_year)[0]
            print 'train size', index_train.shape[0]
            print 'validate size', index_validate.shape[0]

            # # calc train image mean (for each band), and then detract (broadcast)
            # image_mean=np.mean(image_all[index_train],(0,1,2))
            # image_all = image_all - image_mean

            image_validate = image_all[index_validate]
            yield_validate = yield_all[index_validate]

            for time in range(10, 31, 4):
                RMSE_min = 100
                g = tf.Graph()
                with g.as_default():
                    # modify config
                    num_samples = time

                    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)
                    # Launch the graph.
                    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
                    sess.run(tf.initialize_all_variables())
                    saver = tf.train.Saver()
                    for i in range(training_iterations):
                        # train the model
                        if i == 4000:
                            learning_rate /= 10

                        if i == 20000:
                            learning_rate /= 10

                        # index_train_batch = np.random.choice(index_train,size=num_bins)
                        index_validate_batch = np.random.choice(index_validate, size=num_bins)

                        # try data augmentation while training
                        shift = 1
                        index_train_batch_1 = np.random.choice(index_train, size=num_bins + shift * 2)
                        index_train_batch_2 = np.random.choice(index_train, size=num_bins + shift * 2)
                        image_train_batch = (image_all[index_train_batch_1, :, 0:num_samples, :] + image_all[
                                                                                                   index_train_batch_1, :,
                                                                                                   0:num_samples, :]) / 2
                        yield_train_batch = (yield_all[index_train_batch_1] + yield_all[index_train_batch_1]) / 2

                        arg_index = np.argsort(yield_train_batch)
                        yield_train_batch = yield_train_batch[arg_index][shift:-shift]
                        image_train_batch = image_train_batch[arg_index][shift:-shift]

                        _, train_loss, train_loss_reg = sess.run([self.train_op, self.loss_err, self.loss_reg],
                                                                 feed_dict={
                                                                     self.x: image_train_batch,
                                                                     self.y: yield_train_batch,
                                                                     self.lr: learning_rate,
                                                                     self.keep_prob: dropout_rate
                                                                 })

                        if i % 500 == 0:
                            val_loss, val_loss_reg = sess.run([self.loss_err, self.loss_reg], feed_dict={
                                self.x: image_all[index_validate_batch, :, 0:num_samples, :],
                                self.y: yield_all[index_validate_batch],
                                self.keep_prob: 1
                            })

                            print str(time) + 'predict year' + str(predict_year) + 'step' + str(
                                i), train_loss, train_loss_reg, val_loss, val_loss_reg, learning_rate

                        if i % 500 == 0:
                            # do validation
                            pred = []
                            real = []
                            for j in range(image_validate.shape[0] / num_bins):
                                real_temp = yield_validate[j * num_bins:(j + 1) * num_bins]
                                pred_temp = sess.run(self.logits, feed_dict={
                                    self.x: image_validate[j * num_bins:(j + 1) * num_bins, :, 0:num_samples, :],
                                    self.y: yield_validate[j * num_bins:(j + 1) * num_bins],
                                    self.keep_prob: 1
                                })
                                pred.append(pred_temp)
                                real.append(real_temp)
                            pred = np.concatenate(pred)
                            real = np.concatenate(real)
                            RMSE = np.sqrt(np.mean((pred - real) ** 2))
                            ME = np.mean(pred - real)
                            RMSE_ideal = np.sqrt(np.mean((pred - ME - real) ** 2))
                            arg_index = np.argsort(pred)
                            pred = pred[arg_index][50:-50]
                            real = real[arg_index][50:-50]
                            ME_part = np.mean(pred - real)

                            if RMSE < RMSE_min:
                                RMSE_min = RMSE

                            # print 'Validation set','RMSE',RMSE,'ME',ME,'RMSE_min',RMSE_min

                            print 'Validation set', 'RMSE', RMSE, 'RMSE_ideal', RMSE_ideal, 'ME', ME, 'ME_part', ME_part, 'RMSE_min', RMSE_min

                            summary_train_loss.append(train_loss)
                            summary_eval_loss.append(val_loss)
                            summary_RMSE.append(RMSE)
                            summary_ME.append(ME)
                    # save
                    CNNModel.save_path = saver.save(sess, CNNModel.save_path + str(time) + str(
                        predict_year) + 'CNNModel.ckpt')
                    print('save in file: %s' % CNNModel.save_path)

                    # save result
                    # TODO: determine if this is test code
                    pred_out = []
                    real_out = []
                    feature_out = []
                    year_out = []
                    locations_out = []
                    index_out = []
                    for i in range(image_all.shape[0] / num_bins):
                        feature, pred = sess.run(
                            [self.fc6, self.logits], feed_dict={
                                self.x: image_all[i * num_bins:(i + 1) * num_bins, :, 0:num_samples, :],
                                self.y: yield_all[i * num_bins:(i + 1) * num_bins],
                                self.keep_prob: 1
                            })
                        real = yield_all[i * num_bins:(i + 1) * num_bins]

                        pred_out.append(pred)
                        real_out.append(real)
                        feature_out.append(feature)
                        year_out.append(year_all[i * num_bins:(i + 1) * num_bins])
                        locations_out.append(locations_all[i * num_bins:(i + 1) * num_bins])
                        index_out.append(index_all[i * num_bins:(i + 1) * num_bins])
                        # print i
                    weight_out, b_out = sess.run(
                        [self.dense_W, self.dense_B], feed_dict={
                            self.x: image_all[0 * num_bins:(0 + 1) * num_bins, :, 0:num_samples, :],
                            self.y: yield_all[0 * num_bins:(0 + 1) * num_bins],
                            self.keep_prob: 1
                        })
                    pred_out = np.concatenate(pred_out)
                    real_out = np.concatenate(real_out)
                    feature_out = np.concatenate(feature_out)
                    year_out = np.concatenate(year_out)
                    locations_out = np.concatenate(locations_out)
                    index_out = np.concatenate(index_out)

                    np.savez(CNNModel.save_path + str(time) + str(predict_year) + 'result_prediction.npz',
                             pred_out=pred_out, real_out=real_out, feature_out=feature_out,
                             year_out=year_out, locations_out=locations_out, weight_out=weight_out, b_out=b_out,
                             index_out=index_out)
                    np.savez(CNNModel.save_path + str(time) + str(predict_year) + 'result.npz',
                             summary_train_loss=summary_train_loss, summary_eval_loss=summary_eval_loss,
                             summary_RMSE=summary_RMSE, summary_ME=summary_RMSE)


def test(self):
    """Test the model

    Returns
    -------
    Tested model
    """
    pass
