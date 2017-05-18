import alex
import random

import tensorflow as tf
import numpy as np
import tensorflow.contrib.learn as learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib



data_path = '/mnt/data/datasets/collective-activity/stills/resized'
random.seed(0x58c)

def unpickle(file_name):
    import cPickle
    with open(file_name, 'rb') as fo:
        data = cPickle.load(fo)
    return data

def load_CA_data(data):
    data = unpickle(data_path)

    images = data['images']
    labels = data['labels']

    c = list(zip(images, labels))
    random.shuffle(c)
    imgs, lbls = zip(*c)
    
    dataset_size = len(lbls)

    split = dataset_size * 9 / 10

    train_imgs = imgs[:split]
    train_lbls = lbls[:split]

    test_imgs = imgs[split+1:dataset_size]
    test_lbls = lbls[split+1:dataset_size]

    train_data = {'images': train_imgs,
                  'labels': train_lbls}

    test_data = {'images': test_imgs,
                 'labels': test_lbls}

    return train_data, test_data

def main():
    global input_height
    global input_width
    global channels_first
    global num_classes

    log_dir = 'tmp/alexnet_trained'
    train_data, test_data = load_CA_data(data_path)

    input_height = 259
    input_width = 131
    channels_first = False
    num_chasses = 7
    batch_size=128

    classifier = learn.Estimator(
        model_fn=alex.alex_cnn,
        model_dir=log_dir)

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'probabilities': 'softmax'},
        every_n_iter=50)

    classifier.fit(
        x=train_data['images'],
        y=train_data['labels'],
        batch_size=batch_size,
        steps=20000,
        monitors=None)

    metrics={'accuracy': learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key='classes')}
    eval_results = classifier.evaluate(
        x=test_data['images'],
        y=test_data['labels'],
        metrics=metrics)

    print(eval_results)

if __name__ == '__main__':
    main()