import tensorflow as tf
import numpy as np

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

num_classes=10
dropout_rate=0.4
learning_rate=0.001

def cnn_model(features, labels, mode):
    input_data = tf.reshape(features, [-1, 3, 32, 32])
    input_layer = tf.to_float(input_data)
    
    conv1_layer = tf.layers.conv2d(
        inputs=input_layer,
        filters=40,
        kernel_size=(5,5),
        strides=(2,2),
        padding='same',
        data_format='channels_first',
        activation=tf.nn.relu,
        name='conv1')
    
    pool1_layer = tf.layers.max_pooling2d(
        inputs=conv1_layer,
        pool_size=(2,2),
        strides=(2,2),
        padding='same',
        data_format='channels_first',
        name='pool1')
    
    conv2_layer = tf.layers.conv2d(
        inputs=pool1_layer,
        filters=80,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        data_format='channels_first',
        activation=tf.nn.relu,
        name='conv2')
    
    pool2_layer = tf.layers.max_pooling2d(
        inputs=conv2_layer,
        pool_size=(2,2),
        strides=(2,2),
        padding='same',
        data_format='channels_first',
        name='pool2')
    pool2_flat = tf.reshape(pool2_layer, [-1, 80*4*4])
    
    dense1 = tf.layers.dense(
        pool2_flat, 
        units=400, 
        activation=tf.nn.relu,
        name='dense1')
    
    dropout = tf.layers.dropout(
        inputs=dense1, 
        rate=dropout_rate, 
        training=mode==learn.ModeKeys.TRAIN,
        name='dropout')
    
    class_prediction = tf.layers.dense(
        inputs=dropout, 
        units=num_classes,
        name='predict')
    
    training_op = None
    loss = None
    if mode != learn.ModeKeys.INFER: # ground-truth is unknown if we are inferring with new examples
        ground_truth = tf.one_hot(indices=labels, depth=num_classes)
#        print(class_prediction)
        ground_truth = tf.reshape(ground_truth, [-1, 10])
#        print(ground_truth)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=ground_truth,
                                       logits=class_prediction)
    
    if mode == learn.ModeKeys.TRAIN:
        training_op = tf.contrib.layers.optimize_loss(
            loss=loss, 
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=learning_rate,
            optimizer="SGD")
        summary = tf.summary.tensor_summary('loss', loss)
    
    predictions = {'classes': tf.argmax(input=class_prediction, axis=1), # the prediction
                   'probabilties': tf.nn.softmax(class_prediction, name='softmax') }# predicted probabilities
    
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=training_op)


import cPickle
import random

random.seed(0x34c)

label_names = dict()

def unpickle(file_name):
    with open(file_name, 'rb') as fo:
        data = cPickle.load(fo)
    return data

def load_cifar10_data(type=learn.ModeKeys.TRAIN, randomize=False):
    if (type==learn.ModeKeys.TRAIN):
        data1 = unpickle('dataset/cifar-10-batches-py/data_batch_1')
        data2 = unpickle('dataset/cifar-10-batches-py/data_batch_2')
        data3 = unpickle('dataset/cifar-10-batches-py/data_batch_3')
        data4 = unpickle('dataset/cifar-10-batches-py/data_batch_4')
        data5 = unpickle('dataset/cifar-10-batches-py/data_batch_5')

        data = np.concatenate(
            (data1['data'], 
             data2['data'], 
             data3['data'], 
             data4['data'], 
             data5['data']))
        labels = np.concatenate(
            (data1['labels'], 
             data2['labels'], 
             data3['labels'], 
             data4['labels'], 
             data5['labels']))
    
    else:
        data1 = unpickle('dataset/cifar-10-batches-py/test_batch')
        data = data1['data']
        labels = data1['labels']
    
    if randomize:
        c = list(zip(data, labels))
        random.shuffle(c)
        data, labels = zip(*c)
    
    return data, labels

def main(unused_argv):
    global label_names
    log_dir = 'tmp/cnn_convnet_model' 
    label_names = unpickle('dataset/cifar-10-batches-py/batches.meta')['label_names']
    data, labels = load_cifar10_data()
    eval_data, eval_labels = load_cifar10_data(type=learn.ModeKeys.EVAL)
    
    classifier = learn.Estimator(
        model_fn=cnn_model, 
        model_dir=log_dir)
    logging_hook = tf.train.LoggingTensorHook(
        tensors={'probabilities': 'softmax'}, 
        every_n_iter=50)
    classifier.fit(
        x=data,
        y=labels,
        batch_size=100,
        steps=20000,
        #monitors=[logging_hook])
        monitors=None)
    
    metrics={
        'accuracy': learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key='classes'),
    }
    
    eval_results = classifier.evaluate(
        x=eval_data,
        y=eval_labels,
        metrics=metrics)
    
    print(eval_results)

if __name__ == "__main__":
    main(0)
