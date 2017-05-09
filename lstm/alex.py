
CHANNELS_FIRST = False
INPUT_H = 224
INPUT_W = 224

def alex_cnn(features, labels, mode):
    if CHANNELS_FIRST:
        input_data = tf.reshape(features, [-1, 3, INPUT_H, INPUT_W])
        data_format = 'channels_first'
    else:
        input_data = tf.reshape(features, [-1, INPUT_H, INPUT_W, 3])
        data_format = 'channels_last'

    input_layer = tf.to_float(input_data)
    
    conv1a = tf.layers.conv2d(
        inputs=input_layer,
        filters=48,
        kernel_size=(11,11),
        strides=(4,4),
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        name='conv1a')

    conv1b = tf.layers.conv2d(
        inputs=input_layer,
        filters=48,
        kernel_size=(11,11),
        strides=(4,4),
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        name='conv1b')

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

    pool1a

    pool1b