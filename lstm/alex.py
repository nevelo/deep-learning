import tensorflow as tf
import numpy as np
import tensorflow.contrib.learn as learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# The original AlexNet paper claims an image input size of 224x224 -- however, this image size does
# not actually fit with the math! 227x227 with 'valid' convolution (i.e., no padding) and a stride
# of 4 provides the 55x55 output as expected from the first convolution layer.

input_height = 227
input_width = 227
channels_first = False
num_classes = 1000

learning_rate=0.01
weight_decay=0.0005
momentum=0.9
batch_size=128

def alex_cnn(features, labels, mode, num_classes, input_height=227, input_width=227, channels_first=False):
    if channels_first:
        input_data = tf.reshape(features, [-1, 3, input_height, input_width])
        data_format = 'channels_first'
    else:
        input_data = tf.reshape(features, [-1, input_height, input_width, 3])
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
        axis=(1 if channels_first else 3),
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
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        name='conv3a')

    conv3b = tf.layers.conv2d(
        inputs=pool3b,
        filters=192,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        name='conv3b')

    print(conv3a)
    print(conv3b)

    conv3a_split1, conv3a_split2 = tf.split(
        value=conv3a, 
        num_or_size_splits=2, 
        axis=(1 if channels_first else 3),
        name="split3a")

    conv3b_split1, conv3b_split2 = tf.split(
        value=conv3b, 
        num_or_size_splits=2, 
        axis=(1 if channels_first else 3),
        name="split3b")

    print(conv3a_split1)
    print(conv3a_split2)
    print(conv3b_split1)
    print(conv3b_split2)

    conv3a_concat = tf.concat(
        values=[conv3a_split1, conv3b_split1], 
        axis=(1 if channels_first else 3),
        name='conv3a_concat')

    conv3b_concat = tf.concat(
        values=[conv3a_split2, conv3b_split2], 
        axis=(1 if channels_first else 3),
        name='conv3b_concat')

    print(conv3a_concat)
    print(conv3b_concat)

    conv4a = tf.layers.conv2d(
        inputs=conv3a_concat,
        filters=192,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        name='conv4a')

    conv4b = tf.layers.conv2d(
        inputs=conv3b_concat,
        filters=192,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        name='conv4b')

    print(conv4a)
    print(conv4b)

    conv5a = tf.layers.conv2d(
        inputs=conv4a,
        filters=128,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        name='conv5a')

    conv5b = tf.layers.conv2d(
        inputs=conv4b,
        filters=128,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        name='conv5b')

    print(conv5a)
    print(conv5b)

    pool6a = tf.layers.max_pooling2d(
        inputs=conv5a,
        pool_size=(3,3),
        strides=(2,2),
        padding='valid',
        data_format=data_format,
        name='pool6a')

    pool6b = tf.layers.max_pooling2d(
        inputs=conv5b,
        pool_size=(3,3),
        strides=(2,2),
        padding='valid',
        data_format=data_format,
        name='pool6b')

    print(pool6a)
    print(pool6b)

    pool6 = tf.concat(
        values=[pool6a, pool6b], 
        axis=(1 if channels_first else 3),
        name='pool6')

    print(pool6)
    shape = pool6.get_shape().as_list()
    shape[0] = 1
    elements = [-1]
    elements.append(reduce(lambda x, y: x*y, shape))
    print(elements)
    pool6_flat = tf.reshape(pool6, shape=elements, name='pool6_flat')

    print(pool6_flat)

    dense7 = tf.layers.dense(
        inputs=pool6_flat, 
        units=4096, 
        activation=tf.nn.relu,
        name='dense7')

    print(dense7)

    dropout7 = tf.layers.dropout(
        inputs=dense7,
        rate=0.5,
        training= (True if mode is learn.ModeKeys.TRAIN else False),
        name='dropout7')

    print(dropout7)

    dense8 = tf.layers.dense(
        inputs=dropout7, 
        units=4096, 
        activation=tf.nn.relu,
        name='dense8')

    print(dense8)

    dropout8 = tf.layers.dropout(
        inputs=dense8,
        rate=0.5,
        training= (True if mode is learn.ModeKeys.TRAIN else False),
        name='dropout8')

    print(dropout8)

    classification = tf.layers.dense(
        inputs=dropout8, 
        units=num_classes, 
        activation=tf.nn.relu,
        name='classification')

    print(classification)

    # Code for training.
    training_op = None
    loss = None
    # We use the label to create a one-hot vector for training purposes, if we're using a mode (TRAIN or EVAL)
    # in which the ground truth is known.
    if mode != learn.ModeKeys.INFER: 
        ground_truth = tf.one_hot(indices=labels, depth=num_classes)
        ground_truth = tf.reshape(ground_truth, [-1, num_classes])
        loss = tf.losses.softmax_cross_entropy(onehot_labels=ground_truth,
                                       logits=classification)

    if mode == learn.ModeKeys.TRAIN:
        # The training operation attempts to optimize the loss using various algorithms. AlexNet requires
        # stochastic gradient descent, a batch size of 128, momentum of 0.9, and a weight decay of 0.0005.
        # Details of the weight decay function is in section 5 of the paper and impleneted in the function
        # delay_func().
        training_op = tf.contrib.layers.optimize_loss(
            loss=loss, 
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=learning_rate,
            optimizer="SGD")
        summary = tf.summary.tensor_summary('loss', loss)
    
    predictions = {'classes': tf.argmax(input=classification, axis=1), # the prediction
                   'probabilties': tf.nn.softmax(classification, name='softmax') }# predicted probabilities
    
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=training_op)

def decay_func(learning_rate, global_step):
    pass



def main(unused_argv):
    num_rand_samples = 5
    rand_data = np.random.rand(num_rand_samples, 259, 131, 3)
    rand_labels = np.random.randint(low=0, high=1000, size=5, dtype='uint16')
    alex_cnn(rand_data, rand_labels, learn.ModeKeys.TRAIN)

if __name__ == '__main__':
    main(0)