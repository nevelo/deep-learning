import tensorflow as tf
import numpy as np
import tensorflow.contrib.learn as learn


CHANNELS_FIRST = True

# The original AlexNet paper claims an image input size of 224x224 -- however, this image size does
# not actually fit with the math! 227x227 with 'valid' convolution (i.e., no padding) and a stride
# of 4 provides the 55x55 output as expected from the first convolution layer.
INPUT_H = 227
INPUT_W = 227

def alex_cnn(features, labels, mode):
    if CHANNELS_FIRST:
        input_data = tf.reshape(features, [-1, 3, INPUT_H, INPUT_W])
        data_format = 'channels_first'
    else:
        input_data = tf.reshape(features, [-1, INPUT_H, INPUT_W, 3])
        data_format = 'channels_last'

    input_layer = tf.to_float(input_data)
    
    # Filter size of 11, stride of 4, as described in Section 3.5.
    # Krizhevsky et. al. needed to split their implementation into two model-parallel networks to
    # efficiently split computations over two GPUs -- at the time, even state-of-the-art GPUs could
    # not fit the entire model in their memory. This network topology is preserved in this
    # implementation, by splitting the output of the first convolution layer and then sending it to
    # each parallel stream.
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=(11,11),
        strides=(4,4),
        padding='valid',
        data_format=data_format,
        activation=tf.nn.relu,
        name='conv1')

    print(conv1)

    conv1a, conv1b = tf.split(
        value=conv1, 
        num_or_size_splits=2, 
        axis=(1 if CHANNELS_FIRST else 3),
        name="split1")

    print(conv1a)
    print(conv1b)

    # Local response normalization as described in Section 3.3. It's unclear if in the original work,
    # this operation occured before or after the split (which would produce slightly different results).
    rnorm1a = tf.nn.local_response_normalization(
        input=conv1a,
        depth_radius=5,
        bias=2,
        alpha=(10**(-4)),
        beta=0.75,
        name='rnorm1a')

    rnorm1b = tf.nn.local_response_normalization(
        input=conv1b,
        depth_radius=5,
        bias=2,
        alpha=(10**(-4)),
        beta=0.75,
        name='rnorm1b')

    print(rnorm1a) 
    print(rnorm1b)

    # The original AlexNet paper semantically considers pooling to the first operation of a new layer,
    # so pooling layers are numbered starting with 2. Max pooling has overlapping windows as described
    # in Section 3.4. The output of the pooling layer is the same size as the output of the following
    # convolution layer, implying that stride=1 in conv2 below.
    pool2a = tf.layers.max_pooling2d(
        inputs=rnorm1a,
        pool_size=(3,3),
        strides=(2,2),
        padding='valid',
        data_format=data_format,
        name='pool2a')

    pool2b = tf.layers.max_pooling2d(
        inputs=rnorm1b,
        pool_size=(3,3),
        strides=(2,2),
        padding='valid',
        data_format=data_format,
        name='pool2b')
    
    print(pool2a)
    print(pool2b)

    # A stride of 1 is required to produce an output of 27x27, see above.
    conv2a = tf.layers.conv2d(
        inputs=pool2a,
        filters=128,
        kernel_size=(5,5),
        strides=(1,1),
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        name='conv2a')

    conv2b = tf.layers.conv2d(
        inputs=pool2b,
        filters=128,
        kernel_size=(11,11),
        strides=(1,1),
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        name='conv2b')

    print(conv2a)
    print(conv2b)

    rnorm2a = tf.nn.local_response_normalization(
        input=conv2a,
        depth_radius=5,
        bias=2,
        alpha=(10**(-4)),
        beta=0.75,
        name='rnorm2a')

    rnorm2b = tf.nn.local_response_normalization(
        input=conv2b,
        depth_radius=5,
        bias=2,
        alpha=(10**(-4)),
        beta=0.75,
        name='rnorm2b')

    print(rnorm2a) 
    print(rnorm2b)

    pool3a = tf.layers.max_pooling2d(
        inputs=rnorm2a,
        pool_size=(3,3),
        strides=(2,2),
        padding='valid',
        data_format=data_format,
        name='pool3a')

    pool3b = tf.layers.max_pooling2d(
        inputs=rnorm2b,
        pool_size=(3,3),
        strides=(2,2),
        padding='valid',
        data_format=data_format,
        name='pool3b')

    print(pool3a)
    print(pool3b)

    # Layer 3 uses 3x3 filters but also invovles "communication between GPUs" (in the original
    # paper) -- here represented as communication between parallel tensor flows. As in the first
    # convolution layer, each input is calculated and then split with tf.split(). Here, the
    # split outputs are then concatinated with tf.concat()
    conv3a = tf.layers.conv2d(
        inputs=pool3a,
        filters=192,
        kernel_size=(5,5),
        strides=(1,1),
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        name='conv3a')

    conv3b = tf.layers.conv2d(
        inputs=pool3b,
        filters=192,
        kernel_size=(11,11),
        strides=(1,1),
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        name='conv3b')

    conv3a_split1, conv3a_split2 = tf.split(
        value=conv3a, 
        num_or_size_splits=2, 
        axis=(1 if CHANNELS_FIRST else 3),
        name="split3a")

    conv3b_split1, conv3b_split2 = tf.split(
        value=conv3b, 
        num_or_size_splits=2, 
        axis=(1 if CHANNELS_FIRST else 3),
        name="split3b")

    print(conv3a_split1)
    print(conv3a_split2)
    print(conv3b_split1)
    print(conv3b_split2)

    conv3a_concat = tf.concat(
        values=[conv3a_split1, conv3b_split1], 
        axis=(1 if CHANNELS_FIRST else 3),
        name='conv3a_concat')

    conv3b_concat = tf.concat(
        values=[conv3a_split2, conv3b_split2], 
        axis=(1 if CHANNELS_FIRST else 3),
        name='conv3b_concat')

    print(conv3a_concat)
    print(conv3b_concat)

def main(unused_argv):
    num_rand_samples = 5
    rand_data = np.random.rand(num_rand_samples, 3, INPUT_H, INPUT_W)
    rand_labels = np.random.randint(low=0, high=1000, size=5, dtype='uint16')
    alex_cnn(rand_data, rand_labels, learn.ModeKeys.TRAIN)

if __name__ == '__main__':
    main(0)