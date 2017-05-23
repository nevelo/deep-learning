import alex
import random

import tensorflow as tf
import numpy as np
import tensorflow.contrib.learn as learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

SEED=0x58c

data_path = '/mnt/data/datasets/collective-activity/stills/resized'
random.seed(SEED)
tf.logging.set_verbosity(tf.logging.INFO)

def unpickle(file_name):
    import cPickle
    with open(file_name, 'rb') as fo:
        data = cPickle.load(fo)
    return data

def load_CA_data(data_path):
    data = unpickle(data_path)

    images = data['images']
    labels = data['labels']
    poses = data['poses']
    meta = data['meta']

    print("Shuffling data, using random seed %d. Input data is of type" %SEED),
    print(type(images)),
    print(type(labels)),
    print("and"),
    print(type(poses))

    c = list(zip(images, labels, poses))
    random.shuffle(c)
    imgs, lbls, ps = zip(*c)
    imgs = np.array(imgs)
    lbls = np.array(lbls)
    ps = np.array(ps)

    dataset_size = len(lbls)

    print("Dataset is size %d. Dataset is of type" %dataset_size),
    print(type(imgs)),
    print(type(lbls)),
    print('and'),
    print(type(ps))


    split = dataset_size * 9 / 10

    train_imgs = imgs[:split]
    train_lbls = lbls[:split]
    train_ps = ps[:split]

    print("Split data. Trainig data is of type"),
    print(type(train_imgs)),
    print(type(train_lbls)),
    print("and"),
    print(type(train_ps))

    test_imgs = imgs[split+1:dataset_size]
    test_lbls = lbls[split+1:dataset_size]
    test_ps = ps[split+1:dataset_size]

    train_data = {'images': train_imgs,
                  'labels': train_lbls,
                  'poses' : train_ps}

    test_data = {'images': test_imgs,
                 'labels': test_lbls,
                 'poses' : test_ps}    

    return train_data, test_data, meta

def main():
    log_dir = 'tmp/alexnet_trained'
    print("Loading CA data...")
    train_data, test_data, meta = load_CA_data(data_path)
    print("Data loaded.")

    alex.input_height = meta['height']
    alex.input_width = meta['width']
    alex.channels_first = meta['channels_first']
    alex.num_classes = meta['num_classes']
    alex.batch_size=128

    print("Train data type:"),
    print(type(train_data))
    print("Test data type:"),
    print(type(test_data))

    print('Train data images key type:'),
    print(type(train_data['images']))
    print('Image data type:'),
    print(type(train_data['images'][0]))

    print('Train data labels key type:'),
    print(type(train_data['labels']))

    classifier = learn.Estimator(
        model_fn=alex.alex_cnn,
        model_dir=log_dir)

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'probabilities': 'softmax'},
        every_n_iter=50)

    classifier.fit(
        x=train_data['images'],
        y=train_data['labels'],
        batch_size=alex.batch_size,
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