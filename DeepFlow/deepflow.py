# ________                  _______________                
# ___  __ \____________________  ____/__  /________      __
# __  / / /  _ \  _ \__  __ \_  /_   __  /_  __ \_ | /| / /
# _  /_/ //  __/  __/_  /_/ /  __/   _  / / /_/ /_ |/ |/ / 
# /_____/ \___/\___/_  .___//_/      /_/  \____/____/|__/  
#                  /_/                                    
#

import pandas as pd
import numpy as np
import tensorflow as tf
pd.options.mode.chained_assignment = None

__author__ = 'yazan'

class Learn(object):
    def __init__(self, dataset_object):
        print('Initializing DeepFlow: {}'.format(dataset_object.name))
        self.model = None

    @staticmethod
    def _train_input_func(x, y, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((dict(x), y))
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
        return dataset

    @staticmethod
    def _test_input_func(x, y, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((dict(x), y))
        dataset = dataset.shuffle(1000).batch(batch_size)
        return dataset

    @staticmethod
    def _predict_input_func(x,batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((dict(x)))
        dataset = dataset.batch(batch_size)
        return dataset

    @staticmethod
    def rmse(predictions, actual):
        yhat = np.array([p['predictions'] for p in predictions])
        y = np.array(actual.values.tolist())
        return np.mean(np.sqrt((yhat-y)**2))


    def feedforward(self, mode, x, y=None, param=None):
        if mode == 'train':
            if y is None:
                print('Must supply x AND y data pair for training.')
                return

            units = param['units'] if 'units' in param.keys() else [10, 10] 
            batch_size = param['batch_size'] if 'batch_size' in \
                                            param.keys() else 100
            steps = param['steps'] if 'steps' in param.keys() else 100
            
            feature_columns = []
            for key in x.keys():
                feature_columns.append(tf.feature_column.numeric_column(key=key))
            
            print('Creating Deep Neural Network')

            self.model = tf.estimator.DNNRegressor(
                         feature_columns=feature_columns,
                         hidden_units=units)

            print('Training Deep Neural Network')

            self.model.train(
                        input_fn=lambda: self._train_input_func(x,y,batch_size), 
                           steps=steps)
        elif mode == 'test':
            print('Testing Deep Neural Network')
            batch_size = param['batch_size'] if 'batch_size' in \
                                            param.keys() else 100
            eval_result = self.model.evaluate(
                         input_fn=lambda:self._test_input_func(x,y,batch_size))

            print(eval_result)
        elif mode == 'predict':
            if self.model is not None:
                batch_size = param['batch_size'] if 'batch_size' in \
                                            param.keys() else 100
                prediction = self.model.predict(input_fn=
                    lambda:self._predict_input_func(x, batch_size))
                return prediction
            else:
                print('Must train a model before running predictions.')
        else:
            print('Must supply valid mode')


    def _convnet(features, labels, mode):
        """Model function for CNN.
        """
        # Input Layer
        input_layer = tf.reshape(features["x"], [-1, 8, 1])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
           filters=32,
             kernel_size=[5, 5],
             padding="same",
          activation=tf.nn.relu)
        
          # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, 
                                        pool_size=[2, 2], 
                                        strides=2)
        
         # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
             inputs=pool1,
             filters=64,
             kernel_size=[5, 5],
             padding="same",
             activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, 
                                         pool_size=[2, 2], 
                                         strides=2)
        
         # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, 
                                units=1024, 
                                activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, 
                                    rate=0.4, 
                                    training=mode == tf.estimator.ModeKeys.TRAIN)
        
        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)
        
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
          optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
          train_op = optimizer.minimize(
              loss=loss,
              global_step=tf.train.get_global_step())
          return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
             "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def convnet(self, mode, param=None):
        # _num_features = None
        # estimator = tf.estimator.Estimator(
        #             model_fn=self.cnn, model_dir="/tmp/deepflow_deleteme")
        # # Set up logging for predictions
        # # Log the values in the "Softmax" tensor with label "probabilities"
        # tensors_to_log = {"probabilities": "softmax_tensor"}
        # logging_hook = tf.train.LoggingTensorHook(
        #     tensors=tensors_to_log, every_n_iter=100)
        # # Train the model
        # train_input_fn = tf.estimator.inputs.numpy_input_fn(
        #     x={"x": train_data},
        #     y=train_labels,
        #     batch_size=100,
        #     num_epochs=None,
        #     shuffle=True)
        # mnist_classifier.train(
        #     input_fn=train_input_fn,
        #     steps=5000,
        #     hooks=[logging_hook])

        # # Evaluate the model and print results
        # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        #     x={"x": eval_data},
        #     y=eval_labels,
        #     num_epochs=1,
        #     shuffle=False)
        # eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        # print(eval_results)
        pass

    def rnn(self, mode, param=None):
        raise NotImplementedError